"""Version 2 R-GCN encoder with a relation-specific DistMult decoder."""
import torch
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    def __init__(
        self,
        metadata,
        num_nodes_dict,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
    ):
        super().__init__()
        self.node_types = list(metadata[0])

        self.embeddings = torch.nn.ModuleDict(
            {
                kind: torch.nn.Embedding(
                    num_nodes_dict[kind],
                    hidden_channels,
                )
                for kind in self.node_types
            }
        )

        self.convs = torch.nn.ModuleList(
            [
                RGCNConv(
                    hidden_channels,
                    (
                        hidden_channels
                        if i < num_layers - 1
                        else out_channels
                    ),
                    num_relations=len(metadata[1]),
                    num_bases=min(30, len(metadata[1])),
                )
                for i in range(num_layers)
            ]
        )

        self.relations = torch.nn.Embedding(
            len(metadata[1]),
            out_channels,
        )
        torch.nn.init.xavier_uniform_(self.relations.weight)

    def forward(self, x_dict, edge_index, edge_type):
        x = torch.cat(
            [
                self.embeddings[kind](x_dict[kind])
                for kind in self.node_types
            ]
        )

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = x.relu()

        return x

    def score(self, embeddings, heads, relations, tails):
        return (
            embeddings[heads]
            * self.relations(relations)
            * embeddings[tails]
        ).sum(dim=-1)