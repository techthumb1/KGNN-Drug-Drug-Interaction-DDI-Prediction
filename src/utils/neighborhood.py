"""Bounded recorded relationships, independent of model explanations."""


def source_neighborhood(
    graph,
    mappings,
    first,
    second,
    limit=100,
):
    for value in (first, second):
        if (
            type(value) is not int
            or not 0 <= value < graph["drug"].num_nodes
        ):
            raise ValueError("Invalid local drug index")

    if (
        first == second
        or type(limit) is not int
        or not 1 <= limit <= 100
    ):
        raise ValueError(
            "Choose distinct drugs and an integer "
            "edge limit between 1 and 100"
        )

    nodes = {
        ("drug", first),
        ("drug", second),
    }
    links, truncated = [], False

    for (
        src_type,
        relation,
        dst_type,
    ), edges in graph.edge_index_dict.items():
        if src_type != "drug" and dst_type != "drug":
            continue

        mask = edges[0] < 0

        if src_type == "drug":
            mask = (
                mask
                | (edges[0] == first)
                | (edges[0] == second)
            )

        if dst_type == "drug":
            mask = (
                mask
                | (edges[1] == first)
                | (edges[1] == second)
            )

        for index, selected in enumerate(mask.tolist()):
            if not selected:
                continue

            if len(links) == limit:
                truncated = True
                break

            src = (
                src_type,
                int(edges[0, index]),
            )
            dst = (
                dst_type,
                int(edges[1, index]),
            )

            nodes.update((src, dst))
            links.append(
                {
                    "source": f"{src[0]}:{src[1]}",
                    "target": f"{dst[0]}:{dst[1]}",
                    "relation": relation,
                }
            )

        if truncated:
            break

    return {
        "view_type": "source_neighborhood",
        "edge_limit": limit,
        "truncated": truncated,
        "description": (
            "Recorded edges, not model importance "
            "or evidence of clinical harm."
        ),
        "nodes": [
            {
                "id": f"{kind}:{index}",
                "local_index": index,
                "type": kind,
                "label": str(
                    mappings.get(kind, {}).get(
                        index,
                        f"{kind}:{index}",
                    )
                ),
            }
            for kind, index in sorted(nodes)
        ],
        "links": links,
    }