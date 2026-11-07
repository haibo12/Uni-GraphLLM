import json
import os
from typing import Any, Optional

import torch
from torch import Tensor
from tqdm import tqdm
import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_self_loop_attr,
    is_torch_sparse_tensor,
    scatter,
    to_edge_index,
    to_torch_coo_tensor,
    to_torch_csr_tensor,
)
from .instruction_pro import preprocess


def add_node_attr(
    data: Data,
    value: Any,
    attr_name: Optional[str] = None,
) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if data.x is not None:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class AddRandomWalkPE_My(BaseTransform):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        N = data.num_nodes
        assert N is not None

        if data.edge_weight is None:
            value = torch.ones(data.num_edges, device=row.device)
        else:
            value = data.edge_weight
        value = scatter(value, row, dim_size=N, reduce='sum').clamp(min=1)[row]
        value = 1.0 / value

        if N <= 2_000:  # Dense code path for faster computation:
            adj = torch.zeros((N, N), device=row.device)
            adj[row, col] = value
            loop_index = torch.arange(N, device=row.device)
        elif torch_geometric.typing.NO_MKL:  # pragma: no cover
            adj = to_torch_coo_tensor(data.edge_index, value, size=data.size())
        else:
            adj = to_torch_csr_tensor(data.edge_index, value, size=data.size())

        def get_pe(out: Tensor) -> Tensor:
            if is_torch_sparse_tensor(out):
                return get_self_loop_attr(*to_edge_index(out), num_nodes=N)
            return out[loop_index, loop_index]

        out = adj
        pe_list = [get_pe(out)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            pe_list.append(get_pe(out))

        pe = torch.stack(pe_list, dim=-1)
        # edge_index = (row, col)
        edge_pe = pe[row] - pe[col]
        data = add_node_attr(data, pe, attr_name=self.attr_name)
        data['edge_pe'] = edge_pe
        return data


# Data root. The datasets live one level above the repo, under ``graphllm/data/gfm``.
# Resolved relative to this file so it works regardless of the current working dir.
# Override with the ``UNIGRAPHLM_DATA_ROOT`` env var if the data lives elsewhere.
DATA_ROOT = os.environ.get(
    'UNIGRAPHLM_DATA_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gfm')),
)
# Released packs use ``summary/``; older local trees still have ``jh_summary/``.
_SUMMARY_DIR = os.path.join(DATA_ROOT, 'summary')
_JH_SUMMARY_DIR = os.path.join(DATA_ROOT, 'jh_summary')
JH_SUMMARY_DIR = _SUMMARY_DIR if os.path.isdir(_SUMMARY_DIR) else _JH_SUMMARY_DIR
INSTRUCTIONS_DIR = os.path.join(DATA_ROOT, 'instructions')


instruction_train_path = {
    'Cora': 'Cora',
    'Pubmed': 'Pubmed',
    'wikics': 'wikics',
    'arxiv': 'arxiv_sampled_train',
    'WN18RR': 'WN18RR',
    'FB15K237': 'FB15K237_sampled_train',
    'chemhiv': 'chemhiv_sampled_train_balanced',
    'chempcba': 'chempcba_sampled_train_balanced',
}

instruction_test_path = {
    'Cora': 'Cora',
    'Pubmed': 'Pubmed',
    'wikics': 'wikics',
    'arxiv': 'arxiv_test',
    'WN18RR': 'WN18RR',
    'FB15K237': 'FB15K237_test',
    'chemhiv': 'chemhiv',
    'chempcba': 'chempcba_test',
}


def chem_feature_ids(jd, graph_id, data, slices, args):
    """Atom / bond feature ids of one molecule, as indices into data.node_embs / data.edge_embs."""
    if slices is None or getattr(args, 'chem_feat_ids', 'processed') == 'jh_summary':
        return jd['x'], jd['xe']
    # summary/chemhiv.json was built against a differently ordered atom/bond
    # vocabulary than GFT-data/chemhiv (same molecules and atom order, permuted ids),
    # so take the ids from the processed file that owns the embedding tables.
    xs, xes = slices['x'], slices['xe']
    node_ids = data.x[xs[graph_id]:xs[graph_id + 1]].tolist()
    edge_ids = data.xe[xes[graph_id]:xes[graph_id + 1]].tolist()
    assert len(node_ids) == len(jd['x']) and len(edge_ids) == len(jd['xe']), f'graph {graph_id} misaligned'
    return node_ids, edge_ids


def parse_source_data(name, data, xe, tokenizer, mask, split='train', args=None, slices=None):
    transform = AddRandomWalkPE_My(walk_length=32, attr_name='pe')
    json_data = []

    with open(f'{JH_SUMMARY_DIR}/{name}.json', 'r') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    instructions_path = instruction_train_path[name] if split == 'train' else instruction_test_path[name]
    with open(f'{INSTRUCTIONS_DIR}/{instructions_path}.json', 'r') as ins_file: # subgraph-summary pair
        instructions_data = json.load(ins_file)
        # json_data = fcc_data

    print(f"[BUILD] Processing graphs for {name}")
    collected_graph_data = []
    if name in ['Cora', 'Pubmed', 'wikics']:
        for id, jd in enumerate(tqdm(json_data)):
            if not mask[id]:
                continue   # 跳过非训练数据
            # debug

            assert id == jd['node_id']
            link_ids = jd['link_index']
            edges = torch.tensor(jd['edge_index'])
            # summary = jd['summary']
            # reindex
            node_idx = torch.unique(edges) if edges.shape[1]>0 else torch.tensor([jd['node_id']])
            node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
            sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
            target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
            edge_index = torch.IntTensor([sources_idx, target_idx]).long()
            edge_attr = data.edge_text_feat[xe[link_ids]]
            assert edge_index.shape[1] == edge_attr.shape[0]
            y = data.y[jd['node_id']].item() if isinstance(data.y[jd['node_id']], torch.Tensor) else data.y[jd['node_id']]
            # add
            instructions = instructions_data[id]['instruction']
            target = instructions_data[id]['label']
            llm_data = preprocess(
                name,
                instructions,
                target,
                tokenizer,
                max_length=args.max_text_length,
                mode=split,
                args=args,
            )
            
            graph = Data(dataname = name, edge_index=edge_index, x=data.x[node_idx].to(dtype=torch.bfloat16), edge_attr = edge_attr.to(dtype=torch.bfloat16), y=y, root_n_index=node_idx_map[jd['node_id']], root_n_index1= node_idx_map[jd['node_id']], task = 0, input_ids = llm_data['input_ids'], target_ids = llm_data['target_ids'], attention_mask = llm_data['attention_mask'], is_node = llm_data['is_node'], text =llm_data['text'])
            graph=transform(graph) # add PE
            collected_graph_data.append(graph)
        return collected_graph_data
    elif name == 'arxiv':
        for id, ijd in enumerate(tqdm(instructions_data)):
            node_id = ijd['node_id']

            
            jd = json_data[node_id]
            link_ids = jd['link_index']
            edges = torch.tensor(jd['edge_index'])
            # summary = jd['summary']
            # reindex
            node_idx = torch.unique(edges) if edges.shape[1]>0 else torch.tensor([node_id])
            node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
            sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
            target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
            edge_index = torch.IntTensor([sources_idx, target_idx]).long()
            edge_attr = data.edge_text_feat[xe[link_ids]]
            assert edge_index.shape[1] == edge_attr.shape[0]
            y = data.y[node_id].item() if isinstance(data.y[node_id], torch.Tensor) else data.y[node_id]
            
            instructions = ijd['instruction']
            target = ijd['label']
            llm_data = preprocess(
                name,
                instructions,
                target,
                tokenizer,
                max_length=args.max_text_length,
                mode=split,
                args=args,
            )
            
            graph = Data(dataname = name, edge_index=edge_index, x=data.x[node_idx].to(dtype=torch.bfloat16), edge_attr = edge_attr.to(dtype=torch.bfloat16), y=y, root_n_index=node_idx_map[node_id], root_n_index1= node_idx_map[node_id], task = 0, input_ids = llm_data['input_ids'], target_ids = llm_data['target_ids'], attention_mask = llm_data['attention_mask'], is_node = llm_data['is_node'], text =llm_data['text'])
            graph=transform(graph) # add PE
            collected_graph_data.append(graph)
        return collected_graph_data
    elif name in ['WN18RR']:
        for id, jd in enumerate(tqdm(json_data)):
            if not mask[id]:
                continue   # 跳过非训练数据

            assert id == jd['link_id']
            link_node_ids = jd['node_link_index']
            link_ids = jd['link_index']
            node_idx = jd['x']
            edges = torch.tensor(jd['edge_index'])
            # summary = jd['summary']
            # reindex
            link_tensor = torch.tensor(link_node_ids)
            merged_tensor = torch.cat([edges.flatten(), link_tensor])
            node_idx = torch.unique(merged_tensor)
            node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
            sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
            target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
            edge_index = torch.IntTensor([sources_idx, target_idx]).long()
            edge_attr = data.edge_text_feat[xe[link_ids]]
            y = data.edge_types[jd['link_id']].item() if isinstance(data.edge_types[jd['link_id']], torch.Tensor) else data.edge_types[jd['link_id']]
                        # add
            instructions = instructions_data[id]['instruction']
            target = instructions_data[id]['label']
            llm_data = preprocess(
                name,
                instructions,
                target,
                tokenizer,
                max_length=args.max_text_length,
                mode=split,
                args=args,
            )
            
            graph = Data(dataname = name, edge_index=edge_index, x=data.x[node_idx].to(dtype=torch.bfloat16), edge_attr = edge_attr.to(dtype=torch.bfloat16), y=y, root_n_index=
                         node_idx_map[link_node_ids[0]], root_n_index1= node_idx_map[link_node_ids[1]], task = 1, input_ids = llm_data['input_ids'], target_ids = llm_data['target_ids'], attention_mask = llm_data['attention_mask'], is_node = llm_data['is_node'], text =llm_data['text'])
            graph=transform(graph) # add PE
            collected_graph_data.append(graph)
        # torch.save(collected_graph_data, cache_path)
        # print(f"[SAVE] Processed graphs saved to {cache_path}")
        return collected_graph_data
    elif name in ['FB15K237']:
        for id, ijd in enumerate(tqdm(instructions_data)):
            link_id = ijd['link_id']

            jd = json_data[link_id]
            link_node_ids = jd['node_link_index']
            link_ids = jd['link_index']
            node_idx = jd['x']
            edges = torch.tensor(jd['edge_index'])
            # summary = jd['summary']
            # reindex
            link_tensor = torch.tensor(link_node_ids)
            merged_tensor = torch.cat([edges.flatten(), link_tensor])
            node_idx = torch.unique(merged_tensor)
            node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
            sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
            target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
            edge_index = torch.IntTensor([sources_idx, target_idx]).long()
            edge_attr = data.edge_text_feat[xe[link_ids]]
            y = data.edge_types[jd['link_id']].item() if isinstance(data.edge_types[jd['link_id']], torch.Tensor) else data.edge_types[jd['link_id']]
                        # add
            instructions = ijd['instruction']
            target = ijd['label']
            llm_data = preprocess(
                name,
                instructions,
                target,
                tokenizer,
                max_length=args.max_text_length,
                mode=split,
                args=args,
            )
            
            graph = Data(dataname = name, edge_index=edge_index, x=data.x[node_idx].to(dtype=torch.bfloat16), edge_attr = edge_attr.to(dtype=torch.bfloat16), y=y, root_n_index=
                         node_idx_map[link_node_ids[0]], root_n_index1= node_idx_map[link_node_ids[1]], task = 1, input_ids = llm_data['input_ids'], target_ids = llm_data['target_ids'], attention_mask = llm_data['attention_mask'], is_node = llm_data['is_node'], text =llm_data['text'])
            graph=transform(graph) # add PE
            collected_graph_data.append(graph)
        return collected_graph_data
    elif name in ['chemhiv'] and split == 'test':
        for id, jd in enumerate(tqdm(json_data)):
            if not mask[id]:
                continue   # 跳过非训练数据
            # if name == 'chempcba' and id >= 40000:  
            #     break
            # if id > 10:
            #     break
            assert id == jd['graph_id']
            link_ids = jd['link_index']
            node_ids, xe = chem_feature_ids(jd, id, data, slices, args)
            edges = torch.tensor(jd['edge_index'])
            # summary = jd['summary']
            # reindex
            # node_idx = torch.unique(edges)
            node_idx = torch.unique(edges) if edges.shape[1]>0 else torch.tensor(range(len(node_ids)))
            node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
            sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
            target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
            edge_index = torch.IntTensor([sources_idx, target_idx]).long()
            edge_attr = data.edge_embs[xe]
            assert edge_index.shape[1] == edge_attr.shape[0]
            # graph = Data(dataname = name, edge_index=edge_index, x=data.x[node_idx], edge_attr = edge_attr, y=data.y[jd['graph_id']], root_n_index=[jd['graph_id']]*2, summary=summary, task = 2)
            y = data.y[jd['graph_id']].item() if isinstance(data.y[jd['graph_id']], torch.Tensor) else data.y[jd['graph_id']]
            
            instructions = instructions_data[id]['instruction']
            target = instructions_data[id]['label']
            llm_data = preprocess(
                name,
                instructions,
                target,
                tokenizer,
                max_length=args.max_text_length,
                mode=split,
                args=args,
            )
            
            graph = Data(dataname = name, edge_index=edge_index, x=data.node_embs[node_ids].to(dtype=torch.bfloat16), edge_attr = edge_attr.to(dtype=torch.bfloat16), y=y, root_n_index=0, root_n_index1=0, task = 2,
                input_ids=llm_data['input_ids'],
                target_ids=llm_data['target_ids'],
                attention_mask=llm_data['attention_mask'],
                is_node=llm_data['is_node'],
                text=llm_data['text'],
            )
            graph=transform(graph) # add PE
            collected_graph_data.append(graph)
        return collected_graph_data
    elif name in ['chempcba','chemhiv']:
        for id, ijd in enumerate(tqdm(instructions_data)):
            graph_id = ijd['graph_id']

            
            jd = json_data[graph_id]
            link_ids = jd['link_index']
            node_ids, xe = chem_feature_ids(jd, graph_id, data, slices, args)
            edges = torch.tensor(jd['edge_index'])

            node_idx = torch.unique(edges) if edges.shape[1] > 0 else torch.tensor(range(len(node_ids)))
            node_idx_map = {j: i for i, j in enumerate(node_idx.numpy().tolist())}
            sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
            target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
            edge_index = torch.IntTensor([sources_idx, target_idx]).long()
            edge_attr = data.edge_embs[xe]
            assert edge_index.shape[1] == edge_attr.shape[0]
            y = data.y[graph_id].item() if isinstance(data.y[graph_id], torch.Tensor) else data.y[graph_id]

            instructions = ijd['instruction']
            target = ijd['label']
            llm_data = preprocess(
                name,
                instructions,
                target,
                tokenizer,
                max_length=args.max_text_length,
                mode=split,
                args=args,
            )

            graph = Data(
                dataname = name, 
                edge_index=edge_index,
                x=data.node_embs[node_ids].to(dtype=torch.bfloat16),
                edge_attr=edge_attr.to(dtype=torch.bfloat16),
                y=y,
                root_n_index=0,
                root_n_index1=0,
                task=2,
                input_ids=llm_data['input_ids'],
                target_ids=llm_data['target_ids'],
                attention_mask=llm_data['attention_mask'],
                is_node=llm_data['is_node'],
                text=llm_data['text'],
            )
            graph = transform(graph) # add PE
            collected_graph_data.append(graph)

        return collected_graph_data
