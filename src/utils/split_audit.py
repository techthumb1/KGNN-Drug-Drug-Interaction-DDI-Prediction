"""Check exact typed triples without using held-out data for message passing."""
import numpy as np

from src.utils.evaluate import split_tensors
from src.utils.graph_contract import flatten_graph


def _keys(heads, relations, tails, contract):
    nodes = sum(contract["num_nodes"].values())
    return (heads * len(contract["relations"]) + relations) * nodes + tails


def audit_split_separation(dataset, graph, contract):
    train = dataset.load_split("train")
    tensors = split_tensors(train, contract, "cpu")
    train_keys = _keys(*(t.numpy() for t in tensors[:3]), contract)
    known = np.unique(train_keys)
    edges, relation_ids = flatten_graph(graph, contract, "cpu")
    graph_keys = _keys(edges[0].numpy(), relation_ids.numpy(), edges[1].numpy(), contract)
    if len(graph_keys) != len(train_keys) or not np.array_equal(
        np.sort(graph_keys), np.sort(train_keys)
    ):
        raise ValueError("Message-passing graph must contain exactly the official training triples")
    report = {"train_triples": len(train_keys), "train_graph_matches": True}
    for name in ("valid", "test"):
        split = dataset.load_split(name)
        heldout = split_tensors(split, contract, "cpu")
        keys = _keys(*(t.numpy() for t in heldout[:3]), contract)
        overlap = int(np.isin(keys, known, assume_unique=False).sum())
        report[name] = {"triples": len(keys), "exact_train_overlap": overlap}
        if overlap:
            raise ValueError(f"{name} contains {overlap} exact typed triples in the training graph")
    # Reversed pairs and equivalent facts can cross a random split. This audit
    # checks exact directed triples, not an entity-disjoint or temporal design.
    return report
