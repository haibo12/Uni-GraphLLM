import os
import torch
from torch_geometric.loader import DataListLoader

from data.gft_data_utils.process_datasets import get_finetune_graph
from utils.preprocess import pre_node, pre_link, pre_graph
from utils.process import parse_source_data

# Data root. The datasets live one level above the repo, under ``graphllm/data/gfm``.
# Resolved relative to this file so it works regardless of the current working dir.
# Override with the ``UNIGRAPHLM_DATA_ROOT`` env var if the data lives elsewhere.
DATA_ROOT = os.environ.get(
    'UNIGRAPHLM_DATA_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gfm')),
)
GFT_DATA_DIR = os.path.join(DATA_ROOT, 'GFT-data')
SPLITS_DIR = os.path.join(DATA_ROOT, 'splits')

# task mapping
source_name2task = {
    'Cora': 'node',
    'Pubmed': 'node',
    'wikics': 'node',
    'arxiv': 'node',
    'WN18RR': 'link',
    'FB15K237': 'link',
    'chemblpre': 'graph',
    'chempcba': 'graph',
    'chemhiv': 'graph',
}


def parse_dataset_repeat_spec(spec):
    repeat_map = {}
    default_repeat = 1
    raw = '*:1' if spec is None else str(spec).strip()
    if raw == '':
        raw = '*:1'

    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f'Invalid dataset_repeat item "{item}", expected "name:int".')
        key, value = item.split(':', 1)
        key = key.strip()
        value = value.strip()
        if key == '':
            raise ValueError(f'Invalid dataset_repeat item "{item}", dataset name is empty.')
        try:
            repeat = int(value)
        except ValueError as exc:
            raise ValueError(f'Invalid repeat count "{value}" for "{key}", expected positive integer.') from exc
        if repeat <= 0:
            raise ValueError(f'Invalid repeat count {repeat} for "{key}", expected positive integer.')
        if key == '*':
            default_repeat = repeat
        else:
            repeat_map[key] = repeat
    return repeat_map, default_repeat


def get_repeat_for_dataset(dataset_name, repeat_map, default_repeat):
    if dataset_name in repeat_map:
        return repeat_map[dataset_name]
    lowered = dataset_name.lower()
    for key, repeat in repeat_map.items():
        if key.lower() == lowered:
            return repeat
    return default_repeat

def get_preprocess(task):
    if task == 'node':
        return pre_node
    elif task == 'link':
        return pre_link
    elif task == 'graph':
        return pre_graph
    else:
        raise NotImplementedError


def load_data(source_name, task, data_path):
    preprocess = get_preprocess(task)
    dataset, splits, labels, num_classes, num_tasks = get_finetune_graph(data_path, source_name)
    dataset = preprocess(dataset)
    data = dataset[0]
    data.y = labels
    return data


def get_dataloader(source_data, batch_size, tokenizer, split='train', args=None):
    source_name_list = source_data.split("+")
    all_source_graph = []
    repeat_map, default_repeat = parse_dataset_repeat_spec(getattr(args, 'dataset_repeat', '*:1'))
    repeat_stats = {}
    for source_name in source_name_list:
        source_data, slices = torch.load(f"{GFT_DATA_DIR}/{source_name}/processed/geometric_data_processed.pt", weights_only=False)
        if source_name in ['Cora', 'Pubmed']:
            train_mask = source_data.train_masks[0]
            test_mask = source_data.test_masks[0]
        elif source_name in ['wikics']:
            train_mask = source_data.train_mask.T[0]
            test_mask = source_data.test_mask.T
        elif source_name in ['arxiv']:
            train_mask = source_data.train_mask
            test_mask = source_data.test_mask
        elif source_name in ['WN18RR', 'FB15K237']:
            splits = torch.load(f'{SPLITS_DIR}/{source_name}_split.pt', weights_only=False)
            train_mask = torch.zeros(source_data.edge_index.shape[1], dtype=bool)
            train_idx = splits['train']
            train_mask[train_idx] = True
            test_mask = torch.zeros(source_data.edge_index.shape[1], dtype=bool)
            test_idx = splits['test']
            test_mask[test_idx] = True
        elif source_name in ['chempcba', 'chemhiv']:
            splits = torch.load(f'{SPLITS_DIR}/{source_name}_split.pt', weights_only=False)
            train_mask = torch.zeros(source_data.y.shape[0], dtype=bool)
            train_idx = splits['train']
            train_mask[train_idx] = True
            test_mask = torch.zeros(source_data.y.shape[0], dtype=bool)
            test_idx = splits['test']
            test_mask[test_idx] = True

        pretrain_data = load_data(source_name, source_name2task[source_name], GFT_DATA_DIR)

        if source_name2task[source_name] == 'graph':
            source_graph = parse_source_data(
                source_name,
                source_data,
                source_data.xe,
                tokenizer,
                train_mask if split == 'train' else test_mask,
                split,
                args=args,
                slices=slices,
            )
        else:
            source_data.x = source_data.node_text_feat
            source_graph = parse_source_data(
                source_name,
                source_data,
                pretrain_data.xe,
                tokenizer,
                train_mask if split == 'train' else test_mask,
                split,
                args=args,
            )
        repeat_count = 1
        if split == 'train':
            repeat_count = get_repeat_for_dataset(source_name, repeat_map, default_repeat)
        repeat_stats[source_name] = {
            'repeat': repeat_count,
            'raw_samples': len(source_graph),
            'expanded_samples': len(source_graph) * repeat_count,
        }
        for _ in range(repeat_count):
            all_source_graph.extend(source_graph)
    
    if split == 'train':
        print(f"[dataset_repeat] spec={getattr(args, 'dataset_repeat', '*:1')}")
        for name in source_name_list:
            stat = repeat_stats[name]
            print(
                f"[dataset_repeat] {name}: repeat={stat['repeat']}, "
                f"raw={stat['raw_samples']}, expanded={stat['expanded_samples']}"
            )
    print(f"We have {len(all_source_graph)} pretraining graphs")

    dataloader = DataListLoader(all_source_graph, batch_size=batch_size, shuffle=(split == 'train'))

    return dataloader


