"""Official typed BioKG candidate ranking."""
import numpy as np
import torch


def split_tensors(split, contract, device):
    size = len(split["head"])

    heads = np.asarray(
        split["head"], dtype=np.int64
    )
    tails = np.asarray(
        split["tail"], dtype=np.int64
    )
    head_types = np.asarray(split["head_type"])
    tail_types = np.asarray(split["tail_type"])
    raw_relations = np.asarray(
        split["relation"], dtype=np.int64
    )

    if any(
        len(values) != size
        for values in (
            tails,
            head_types,
            tail_types,
            raw_relations,
        )
    ):
        raise ValueError(
            "Split fields must have matching lengths"
        )

    relations = np.full(
        size, -1, dtype=np.int64
    )
    head_offsets = np.zeros_like(heads)
    tail_offsets = np.zeros_like(tails)
    head_counts = np.zeros_like(heads)
    tail_counts = np.zeros_like(tails)

    for relation_id, (src, relation, dst) in enumerate(
        contract["relations"]
    ):
        mask = (
            (head_types == src)
            & (raw_relations == int(relation))
            & (tail_types == dst)
        )
        relations[mask] = relation_id

    for kind in contract["node_types"]:
        for types, offsets, counts in (
            (head_types, head_offsets, head_counts),
            (tail_types, tail_offsets, tail_counts),
        ):
            mask = types == kind
            offsets[mask] = contract["offsets"][kind]
            counts[mask] = contract["num_nodes"][kind]

    if (
        (heads < 0)
        | (heads >= head_counts)
        | (tails < 0)
        | (tails >= tail_counts)
        | (relations < 0)
    ).any():
        raise ValueError(
            "Split contains invalid typed indices or relations"
        )

    return tuple(
        torch.as_tensor(
            values,
            dtype=torch.long,
            device=device,
        )
        for values in (
            heads + head_offsets,
            relations,
            tails + tail_offsets,
            head_offsets,
            tail_offsets,
            head_counts,
            tail_counts,
        )
    )


def mean_reciprocal_rank(positive, negative):
    rank = 1 + 0.5 * (
        (negative >= positive[:, None]).sum(1)
        + (negative > positive[:, None]).sum(1)
    )
    return rank.float().reciprocal()


def evaluate_split(
    model,
    embeddings,
    split,
    contract,
    name,
    batch_size=16,
    max_triples=None,
):
    if (
        name not in ("valid", "test")
        or batch_size < 1
    ):
        raise ValueError(
            "Evaluate a held-out official split "
            "with a positive batch size"
        )

    total = len(split["head"])
    count = (
        min(total, max_triples)
        if max_triples is not None
        else total
    )

    if count < 1:
        raise ValueError(
            "Evaluation requires held-out triples"
        )

    (
        heads,
        relations,
        tails,
        head_offsets,
        tail_offsets,
        head_counts,
        tail_counts,
    ) = split_tensors(
        split,
        contract,
        embeddings.device,
    )

    aggregate, ddi = 0.0, {}
    model.eval()

    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            h = heads[start:end]
            r = relations[start:end]
            t = tails[start:end]

            positive = model.score(
                embeddings, h, r, t
            )
            ranks = []

            for side, offsets, counts in (
                ("head", head_offsets, head_counts),
                ("tail", tail_offsets, tail_counts),
            ):
                candidates = torch.as_tensor(
                    np.asarray(
                        split[side + "_neg"][start:end]
                    ).copy(),
                    dtype=torch.long,
                    device=embeddings.device,
                )

                if (
                    candidates.ndim != 2
                    or candidates.shape[1] < 1
                ):
                    raise ValueError(
                        "Official typed evaluation "
                        "negatives required"
                    )

                if (
                    (candidates < 0)
                    | (
                        candidates
                        >= counts[start:end, None]
                    )
                ).any():
                    raise ValueError(
                        "Invalid typed evaluation negatives"
                    )

                candidates = (
                    candidates
                    + offsets[start:end, None]
                )

                negative = model.score(
                    embeddings,
                    (
                        candidates
                        if side == "head"
                        else h[:, None]
                    ),
                    r[:, None],
                    (
                        candidates
                        if side == "tail"
                        else t[:, None]
                    ),
                )

                ranks.append(
                    mean_reciprocal_rank(
                        positive,
                        negative,
                    )
                )

            values = (
                (ranks[0] + ranks[1]) / 2
            ).cpu().tolist()

            aggregate += sum(values)

            for i, value in enumerate(values, start):
                if (
                    split["head_type"][i]
                    == split["tail_type"][i]
                    == "drug"
                ):
                    relation = str(
                        int(split["relation"][i])
                    )
                    item = ddi.setdefault(
                        relation,
                        {
                            "triples": 0,
                            "sum_mrr": 0.0,
                        },
                    )
                    item["triples"] += 1
                    item["sum_mrr"] += value

    for item in ddi.values():
        item["mrr"] = (
            item.pop("sum_mrr")
            / item["triples"]
        )

    return {
        "split": name,
        "mrr": aggregate / count,
        "evaluated_triples": count,
        "total_triples": total,
        "complete": count == total,
        "protocol": (
            "official provided typed negatives; "
            "mean head/tail MRR"
        ),
        "drug_drug_relations": ddi,
    }