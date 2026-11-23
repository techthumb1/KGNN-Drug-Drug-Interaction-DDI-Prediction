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
        ddi = (np.asarray(split["head_type"]) == "drug") & (np.asarray(split["tail_type"]) == "drug")
        reversed_ddi = _keys(
            heldout[2].numpy()[ddi], heldout[1].numpy()[ddi], heldout[0].numpy()[ddi], contract
        )
        report[name] = {
            "triples": len(keys),
            "exact_train_overlap": overlap,
            "drug_drug_triples": int(ddi.sum()),
            "drug_drug_same_relation_reversed_in_train": int(np.isin(reversed_ddi, known).sum()),
        }
        if overlap:
            raise ValueError(f"{name} contains {overlap} exact typed triples in the training graph")
        del split
    # Related facts can cross the random split even when exact and reversed
    # overlap are zero. These counts do not establish an entity or temporal split.
    return report
