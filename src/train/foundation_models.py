"""Load pinned upstream graph models for offline BioKG experiments."""

from functools import partial
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import torch
from torch_geometric.data import Data

from src.utils.graph_contract import flatten_graph

PRETRAIN_GRAPHS = ("FB15k237", "WN18RR", "CoDExMedium")
UPSTREAM = {
    "trix-entity": (
        "7596e14eefefe89e61396205a0550172cadeddb0",
        "entity_prediction.pth",
        "8f6e7266093c2d15ad88d41e9825e4b327171143543cf060cfee907d9a890342",
    ),
    "trix-relation": (
        "7596e14eefefe89e61396205a0550172cadeddb0",
        "relation_prediction.pth",
        "5d15d0b50c9df4e4ffeccda7f30aa941231126fbf660b6daeea1ba96dfb249e5",
    ),
    "ultra": (
        "427966ad8ed60420eef034063d44f3153addff90",
        "ckpts/ultra_3g.pth",
        "fdedc01b0045fc089d2ad5da08569466b7b221691fe978a18e91082bbb133c18",
    ),
}


def _fused_propagate(layer, edge_index, size=None, *, input, relation, boundary,
                     edge_type, edge_weight, kernels):
    # Call the official CPU/CUDA kernel with pre-sorted edges. This bypasses
    # PyG's version-dependent Inspector and the per-layer re-sort.
    if edge_type.numel() == 0:
        return layer.update(boundary, input)
    batch_size, num_nodes = input.shape[:2]
    flat_input = input.transpose(0, 1).flatten(1)
    flat_relation = relation.transpose(0, 1).flatten(1)
    if torch.is_grad_enabled() and (flat_input.requires_grad or flat_relation.requires_grad):
        flat_update = kernels.RSPMMAddMulFunction.apply(
            edge_index, edge_type, edge_weight, flat_relation, flat_input
        )
    else:
        operation = (
            kernels.rspmm.rspmm_add_mul_forward_cuda if input.is_cuda
            else kernels.rspmm.rspmm_add_mul_forward_cpu
        )
        flat_update = operation(edge_index, edge_type, edge_weight, flat_relation, flat_input)
    update = flat_update.view(num_nodes, batch_size, -1).transpose(0, 1)
    return layer.update(update + boundary, input)


def load_foundation(name, checkout, device):
    revision, filename, expected_hash = UPSTREAM[name]
    root = Path(checkout).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if head != revision or changed:
        raise ValueError(f"Use a clean upstream checkout at {revision}")
    path = root / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Unexpected checkpoint bytes in {path}")

    width = 64 if name == "ultra" else 32

    def layers(depth):
        return dict(input_dim=width, hidden_dims=[width] * depth, message_func="distmult",
                    aggregate_func="sum", short_cut=True, layer_norm=True)

    sys.path.insert(0, str(root if name == "ultra" else root / "src"))
    try:
        if name == "ultra":
            model = importlib.import_module("ultra.models").Ultra(
                dict(**layers(6), **{"class": "RelNBFNet"}),
                dict(**layers(6), **{"class": "EntityNBFNet"}),
            )
        elif name == "trix-entity":
            model = importlib.import_module("trix.models_entity").TRIX(
                layers(3), layers(2), layers(4)
            )
        else:
            model = importlib.import_module("trix.models_relation").TRIX(
                layers(2), layers(2),
                SimpleNamespace(num_layer=3, feature_dim=32, num_mlp_layer=2),
            )
        package = "ultra" if name == "ultra" else "trix"
        conv = importlib.import_module(f"{package}.layers").GeneralizedRelationalConv
        kernels = importlib.import_module(f"{package}.rspmm.rspmm")
        for layer in model.modules():
            if isinstance(layer, conv):
                if (layer.message_func, layer.aggregate_func) != ("distmult", "sum"):
                    raise ValueError("The fused adapter requires the pinned DistMult sum configuration")
                layer.propagate = partial(MethodType(_fused_propagate, layer), kernels=kernels)
    finally:
        sys.path.pop(0)
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Upstream checkpoint contains nonfinite parameters")
    return model.to(device), {
        "upstream_revision": revision,
        "upstream_checkpoint": filename,
        "upstream_checkpoint_sha256": expected_hash,
        "pretraining_graphs": PRETRAIN_GRAPHS,
    }


def _incident_relations(nodes, relations, num_nodes, num_relations):
    keys = (nodes * num_relations + relations).unique(sorted=True)
    counts = (keys // num_relations).bincount(minlength=num_nodes)
    return keys % num_relations, counts, counts.cumsum(0) - counts


def _relation_edges(left, right, num_nodes, num_relations, exclude_self):
    left_rel, left_count, left_offset = left
    right_rel, right_count, right_offset = right
    pair_counts = left_count * right_count
    shared = torch.arange(num_nodes, device=pair_counts.device).repeat_interleave(pair_counts)
    position = torch.arange(len(shared), device=shared.device)
    position -= (pair_counts.cumsum(0) - pair_counts)[shared]
    first = left_rel[left_offset[shared] + position // right_count[shared]]
    second = right_rel[right_offset[shared] + position % right_count[shared]]
    if exclude_self:
        keep = first != second
        first, second, shared = first[keep], second[keep], shared[keep]
    return Data(edge_index=torch.stack([first, second]), edge_type=shared,
                num_nodes=num_relations, num_relations=num_nodes)


def _sort_edges(data):
    order = (data.edge_index[0] * data.num_nodes + data.edge_index[1]).argsort(stable=True)
    data.edge_index = data.edge_index[:, order]
    data.edge_type = data.edge_type[order]
    return data


def message_graph(name, graph, contract, device):
    # Upstream checkpoints expect inverse IDs in the second half. These
    # synthetic edges come only from the training graph.
    edge_index, edge_type = flatten_graph(graph, contract, "cpu")
    relation_count = len(contract["relations"])
    data = Data(
        edge_index=torch.cat([edge_index, edge_index.flip(0)], dim=1),
        edge_type=torch.cat([edge_type, edge_type + relation_count]),
        num_nodes=sum(contract["num_nodes"].values()),
        num_relations=2 * relation_count,
    )
    head, tail = [
        _incident_relations(nodes, data.edge_type, data.num_nodes, data.num_relations)
        for nodes in data.edge_index
    ]
    if name == "ultra":
        # ULTRA consumes only nonzero coordinates of each incidence product.
        # Collapse shared entities here to avoid sparse matrix multiplication
        # on the full typed BioKG graph.
        pair_edges, pair_types = [], []
        for kind, (left, right) in enumerate(((head, head), (tail, tail), (head, tail), (tail, head))):
            pairs = _relation_edges(left, right, data.num_nodes, data.num_relations, False)
            keys = (pairs.edge_index[0] * data.num_relations + pairs.edge_index[1]).unique(sorted=True)
            pair_edges.append(torch.stack([keys // data.num_relations, keys % data.num_relations]))
            pair_types.append(torch.full_like(keys, kind))
        data.relation_graph = _sort_edges(Data(edge_index=torch.cat(pair_edges, dim=1),
                                               edge_type=torch.cat(pair_types),
                                               num_nodes=data.num_relations, num_relations=4))
    else:
        data.relation_adj = {
            key: _sort_edges(_relation_edges(
                left, right, data.num_nodes, data.num_relations, key in ("hh", "tt")
            ))
            for key, left, right in (
                ("hh", head, head), ("ht", head, tail),
                ("th", tail, head), ("tt", tail, tail),
            )
        }
    return _sort_edges(data).to(device)
