"""Shared indexing contract for training, evaluation and inference."""
import hashlib
import json

import torch


def graph_contract(graph):
    node_types = sorted(graph.node_types)
    counts = {
        kind: int(graph[kind].num_nodes)
        for kind in node_types
    }

    offsets, total = {}, 0
    for kind in node_types:
        offsets[kind] = total
        total += counts[kind]

    relations = [
        list(key)
        for key in sorted(graph.edge_types)
    ]

    digest = hashlib.sha256(
        json.dumps(
            [counts, relations],
            sort_keys=True,
        ).encode()
    )

    for relation in relations:
        edges = (
            graph[tuple(relation)]
            .edge_index.detach()
            .cpu()
            .contiguous()
        )
        digest.update(str(list(edges.shape)).encode())
        digest.update(edges.numpy().tobytes())

    return {
        "node_types": node_types,
        "num_nodes": counts,
        "offsets": offsets,
        "relations": relations,
        "train_graph_sha256": digest.hexdigest(),
    }


def flatten_graph(graph, contract, device):
    indices, types = [], []

    for relation_id, (src, relation, dst) in enumerate(
        contract["relations"]
    ):
        edge = (
            graph[(src, relation, dst)]
            .edge_index.to(device)
            .clone()
        )
        edge[0] += contract["offsets"][src]
        edge[1] += contract["offsets"][dst]

        indices.append(edge)
        types.append(
            torch.full(
                (edge.shape[1],),
                relation_id,
                dtype=torch.long,
                device=device,
            )
        )

    return torch.cat(indices, dim=1), torch.cat(types)


def query_indices(
    contract,
    head,
    relation,
    tail,
    head_type="drug",
    tail_type="drug",
):
    for kind, value in (
        (head_type, head),
        (tail_type, tail),
    ):
        if (
            type(value) is not int
            or not 0 <= value < contract["num_nodes"].get(kind, 0)
        ):
            raise ValueError(f"Invalid local {kind} index")

    if head_type == tail_type and head == tail:
        raise ValueError("Choose distinct entities")

    try:
        relation_id = contract["relations"].index(
            [head_type, str(relation), tail_type]
        )
    except ValueError:
        raise ValueError(
            "Unsupported relation for the selected entity types"
        ) from None

    return (
        contract["offsets"][head_type] + head,
        relation_id,
        contract["offsets"][tail_type] + tail,
    )