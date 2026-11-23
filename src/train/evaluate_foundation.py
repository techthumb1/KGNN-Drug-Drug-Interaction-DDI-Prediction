"""Compare upstream models on BioKG's official typed candidate lists."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from src.data.load_biokg import load_biokg_as_hetero, load_dataset
from src.train.foundation_models import UPSTREAM, load_foundation, message_graph
from src.train.sparse_rgcn import SparseRGCN
from src.utils.checkpoint import load_checkpoint
from src.utils.evaluate import mean_reciprocal_rank, split_tensors
from src.utils.graph_contract import flatten_graph, graph_contract


def typed_batches(split, contract, indices, batch_size, device):
    columns = {key: np.asarray(split[key]) for key in
               ("head", "tail", "relation", "head_type", "tail_type")}
    for start in range(0, len(indices), batch_size):
        row_ids = indices[start:start + batch_size]
        rows = {key: value[row_ids] for key, value in columns.items()}
        head, rel, tail, head_offset, tail_offset, head_count, tail_count = split_tensors(
            rows, contract, device
        )
        yield row_ids, rows, (head, rel, tail), (
            _negative_ids(split["head_neg"], row_ids, head_offset, head_count),
            _negative_ids(split["tail_neg"], row_ids, tail_offset, tail_count),
        )


def _negative_ids(values, rows, offsets, counts):
    local = torch.as_tensor(np.asarray(values)[rows], device=offsets.device, dtype=torch.long)
    if local.ndim != 2 or local.shape[1] < 1 or ((local < 0) | (local >= counts[:, None])).any():
        raise ValueError("Invalid official typed negative candidates")
    return local + offsets[:, None]


def entity_batches(triples, negatives, relation_count):
    head, relation, tail = triples
    head_neg, tail_neg = negatives
    # Represent head queries explicitly as inverse-tail queries. TRIX
    # conditions its relation graph before its internal corruption detection.
    for source, target, rel, candidates in (
        (head, tail, relation, tail_neg), (tail, head, relation + relation_count, head_neg)
    ):
        destinations = torch.cat([target[:, None], candidates], dim=1)
        yield torch.stack([
            source[:, None].expand_as(destinations), destinations,
            rel[:, None].expand_as(destinations),
        ], dim=-1)


def relation_truths(dataset, contract, split_name, selected, evaluation):
    known = set()
    nodes = contract["num_nodes"]["drug"]
    wanted = (np.asarray(evaluation["head"])[selected] * nodes
              + np.asarray(evaluation["tail"])[selected])
    splits = ("train", "valid") if split_name == "valid" else ("train", "valid", "test")
    for name in splits:
        split = evaluation if name == split_name else dataset.load_split(name)
        ids = np.flatnonzero(
            (np.asarray(split["head_type"]) == "drug") & (np.asarray(split["tail_type"]) == "drug")
        )
        matching = ids[np.isin(split["head"][ids] * nodes + split["tail"][ids], wanted)]
        known.update(zip(split["head"][matching].tolist(),
                         split["relation"][matching].tolist(), split["tail"][matching].tolist()))
        del split
    return known


def rank_relations(score_batch, contract, rows, triples, known):
    head, rel, tail = triples
    ddi_relations = [(index, int(relation)) for index, (src, relation, dst)
                     in enumerate(contract["relations"]) if src == dst == "drug"]
    choices = rel.new_tensor([index for index, _ in ddi_relations]).expand(rel.numel(), -1)
    logits = score_batch(torch.stack([
        head[:, None].expand_as(choices), tail[:, None].expand_as(choices), choices,
    ], dim=-1))
    if not torch.isfinite(logits).all():
        raise RuntimeError("Nonfinite relation logits")
    target = choices == rel[:, None]
    excluded = rel.new_tensor([
        [(int(h), raw, int(t)) in known for _, raw in ddi_relations]
        for h, t in zip(rows["head"], rows["tail"])
    ], dtype=torch.bool)
    positives = logits[target]
    return mean_reciprocal_rank(positives, logits.masked_fill(target | excluded, -torch.inf))


def run(args):
    dataset = load_dataset()
    graph = load_biokg_as_hetero(dataset)
    contract = graph_contract(graph)
    device = torch.device(args.device)
    if args.model == "rgcn":
        if args.checkpoint is None:
            raise ValueError("R-GCN requires --checkpoint")
        model, checkpoint = load_checkpoint(args.checkpoint, contract, device, require_evaluation=False)
        provenance = {"optimizer_steps": checkpoint["optimizer_steps"], "seed": checkpoint["seed"]}
        with torch.inference_mode():
            x = {kind: graph[kind].x.to(device) for kind in contract["node_types"]}
            if device.type == "cpu":
                embeddings = SparseRGCN(model, graph, contract)(x, None, None)
            else:
                embeddings = model(x, *flatten_graph(graph, contract, device))

        def score_batch(batch):
            h, t, r = batch.unbind(-1)
            return model.score(embeddings, h, r % len(contract["relations"]), t)
    else:
        if args.upstream is None:
            raise ValueError("Provide --upstream with the pinned checkout")
        model, provenance = load_foundation(args.model, args.upstream, device)
        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location=device, weights_only=True)
            if state["model_name"] != args.model or state["graph_contract"] != contract:
                raise ValueError("Fine-tuned checkpoint does not match this model and graph")
            model.load_state_dict(state["model_state"], strict=True)
            provenance.update(state["training"])

        context = message_graph(args.model, graph, contract, device)

        def score_batch(batch):
            # The upstream relation model uses the first candidate to size
            # its constant initialization and produces one logit per relation.
            return model(context, batch)

    model.eval()
    split = dataset.load_split(args.split)
    ddi = (np.asarray(split["head_type"]) == "drug") & (np.asarray(split["tail_type"]) == "drug")
    if args.task == "relation" or args.ddi_only:
        eligible = np.flatnonzero(ddi)
    else:
        eligible = np.arange(len(ddi))
    if args.max_queries is None or args.max_queries >= len(eligible):
        selected = eligible
    else:
        selected = np.sort(np.random.default_rng(args.seed).choice(
            eligible, size=args.max_queries, replace=False
        ))
    if args.task == "relation":
        truths = relation_truths(dataset, contract, args.split, selected, split)
    totals, ddi_totals, by_relation, durations = [], [], defaultdict(list), []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for row_ids, rows, triples, negatives in typed_batches(
            split, contract, selected, args.batch_size, device
        ):
            started = time.perf_counter()
            if args.task == "relation":
                reciprocal = rank_relations(score_batch, contract, rows, triples, truths)
            else:
                sides = []
                for batch in entity_batches(triples, negatives, len(contract["relations"])):
                    logits = score_batch(batch)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Nonfinite candidate logits")
                    sides.append(mean_reciprocal_rank(logits[:, 0], logits[:, 1:]))
                reciprocal = (sides[0] + sides[1]) / 2
            values = reciprocal.cpu().tolist()
            durations.append((time.perf_counter() - started) / len(row_ids))
            totals.extend(values)
            for index, value in enumerate(values):
                if ddi[row_ids[index]]:
                    ddi_totals.append(value)
                    by_relation[int(rows["relation"][index])].append(value)
            if len(totals) == len(values) or len(totals) % 16 == 0:
                print(f"Evaluated {len(totals)} of {len(selected)} queries", flush=True)

    report = {
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "cohort": "drug-drug" if args.ddi_only or args.task == "relation" else "all relations",
        "provenance": provenance,
        "complete": len(selected) == len(eligible),
        "evaluated_triples": len(selected),
        "eligible_triples": len(eligible),
        "sample_seed": args.seed,
        "sample_indices": selected.tolist() if len(selected) < len(eligible) else None,
        "train_graph_sha256": contract["train_graph_sha256"],
        "ranking_protocol": (
            "OGB head/tail mean with provided typed negatives and average tie ranks"
            if args.task == "entity" else
            "forward drug–drug relation ranking, filtering known alternative positive relations"
        ),
        "mrr": float(np.mean(totals)),
        "ddi_mrr": float(np.mean(ddi_totals)) if ddi_totals else None,
        "ddi_triples": len(ddi_totals),
        "ddi_by_relation": {key: {"triples": len(values), "mrr": float(np.mean(values))}
                            for key, values in sorted(by_relation.items())},
        "device": str(device),
        "intraop_threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "median_seconds_per_query": float(np.median(durations)),
        "p95_seconds_per_query": float(np.percentile(durations, 95)),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*UPSTREAM, "rgcn"], required=True)
    parser.add_argument("--task", choices=["entity", "relation"], default="entity")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--ddi-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_queries is not None and args.max_queries < 1):
        parser.error("Batch size and query limit must be positive")
    if (args.task == "relation" and args.model not in ("trix-relation", "rgcn")
            or args.task == "entity" and args.model == "trix-relation"):
        parser.error("Choose a compatible model and task")
    run(args)


if __name__ == "__main__":
    main()
