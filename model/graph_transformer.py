import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.nn import global_mean_pool

from .layers import SAGEConv


class GraphEncoder(nn.Module):
    """first_model: graph encoder + projector that splices graph tokens into the LLM input embeddings."""

    def __init__(self, args, llama_embed):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.GT = GraphSAGE(args)

        self.embed_tokens = llama_embed   # frozen LLM word embeddings
        self.embed_dim = llama_embed.shape[1]

        self.graph_projector = nn.Sequential(
            nn.Linear(args.gnn_output, self.args.num_token * self.embed_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

    def forward(self, batch):
        input_ids = batch.input_ids
        is_node = batch.is_node

        key_embs, all_node_embs = self.GT(batch.x, batch.edge_index, batch.edge_attr, batch)
        node_embedding = self.graph_projector(key_embs)
        node_embeddings = node_embedding.view(-1, self.embed_dim)  # [bs * num_token, dim]

        safe_ids = input_ids.clone()
        safe_ids[is_node] = 0
        inputs_embeds = self.embed_tokens[safe_ids]
        inputs_embeds[is_node] = node_embeddings

        return inputs_embeds  # [bsz, seq, dim]


class GraphSAGE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        in_channels = args.gnn_input
        n_layers = args.gt_layers
        hidden_channels = args.att_d_model
        out_channels = args.gnn_output

        gnn_conv = SAGEConv
        self.convs = nn.ModuleList()
        if n_layers > 1:
            self.convs.append(gnn_conv(in_channels, hidden_channels))
            for i in range(1, n_layers - 1):
                self.convs.append(gnn_conv(hidden_channels, hidden_channels))
            self.convs.append(gnn_conv(hidden_channels, out_channels))
        else:
            self.convs.append(gnn_conv(in_channels, out_channels))

        self.activation = F.relu
        self.mlp = Sequential(
            Linear(args.gnn_output * 2, args.gnn_output),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)

        all_node_embs = x
        g_x = global_mean_pool(all_node_embs, batch.batch)

        c_x = torch.stack([all_node_embs[batch.root_n_index], all_node_embs[batch.root_n_index1]], dim=0).mean(dim=0)

        c_x = torch.where((batch.task == 2).unsqueeze(-1), g_x, c_x)

        g_x = torch.cat((g_x, c_x), 1)  # cat average and center
        return self.mlp(g_x), all_node_embs
