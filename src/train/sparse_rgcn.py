"""Reuse normalized relation adjacencies across full-graph CPU training steps."""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def _tensor(matrix):
    matrix = matrix.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr),
        torch.from_numpy(matrix.indices),
        torch.from_numpy(matrix.data),
        size=matrix.shape,
    )


class _MeanMessages(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, adjacency, transpose):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(adjacency, source)

    @staticmethod
    def backward(ctx, gradient):
        (transpose,) = ctx.saved_tensors
        return torch.sparse.mm(transpose, gradient.contiguous()), None, None


class SparseRGCN(nn.Module):
    """Evaluate the same RGCNConv mean, basis weights, root and bias on CPU."""

    def __init__(self, model, graph, contract):
        super().__init__()
        self.model = model
        self.contract = contract
        self.adjacencies = []
        for source, relation, target in contract["relations"]:
            heads, tails = graph[source, relation, target].edge_index.cpu().numpy()
            count = contract["num_nodes"][target]
            degree = np.bincount(tails, minlength=count)
            weights = np.reciprocal(degree[tails].astype(np.float32))
            adjacency = sp.csr_matrix(
                (weights, (tails, heads)),
                shape=(count, contract["num_nodes"][source]),
            )
            self.adjacencies.append((source, target, _tensor(adjacency), _tensor(adjacency.T)))

    def forward(self, x_dict, edge_index, edge_type):
        node_types = self.contract["node_types"]
        counts = [self.contract["num_nodes"][kind] for kind in node_types]
        features = torch.cat([self.model.embeddings[kind](x_dict[kind]) for kind in node_types])
        for index, layer in enumerate(self.model.convs):
            source_features = dict(zip(node_types, features.split(counts)))
            weights = (layer.comp @ layer.weight.flatten(1)).reshape(
                len(self.adjacencies), layer.in_channels_l, layer.out_channels
            )
            terms = {kind: [] for kind in node_types}
            for weight, (source, target, adjacency, transpose) in zip(weights, self.adjacencies):
                mean = _MeanMessages.apply(source_features[source], adjacency, transpose)
                terms[target].append(mean @ weight)
            messages = [sum(terms[kind], features.new_zeros((count, layer.out_channels)))
                        for kind, count in zip(node_types, counts)]
            features = torch.cat(messages) + features @ layer.root + layer.bias
            if index < len(self.model.convs) - 1:
                features = features.relu()
        return features

    def score(self, embeddings, heads, relations, tails):
        return self.model.score(embeddings, heads, relations, tails)
