"""Train/evaluate a PGExplainer link-prediction adaptation, predictor frozen.

Example: ENABLE_PREDICTIONS=true python -m src.train.train_pgexplainer
No DrugBank records are used. The official training graph is the only message
passing input. Explanation query splits are separate from predictor evaluation.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.explain.explain_pg import (
    ARTIFACT,
    FORMAT,
    EdgeMaskNetwork,
    ExplanationEngine,
    ExplanationUnavailable,
)


def queries(split, contract, count, seed):
    rng = random.Random(seed)
    eligible = [
        i
        for i, (h, t) in enumerate(zip(split["head_type"], split["tail_type"]))
        if h == t == "drug"
    ]
    rng.shuffle(eligible)
    seen, rows = set(), []
    for i in eligible:
        h, r, t = (
            int(split["head"][i]),
            str(int(split["relation"][i])),
            int(split["tail"][i]),
        )
        for candidate in (
            (h, r, t),
            (h, r, rng.randrange(contract["num_nodes"]["drug"])),
        ):
            if candidate[0] != candidate[2] and candidate not in seen:
                seen.add(candidate)
                rows.append(candidate)
        if len(rows) >= count:
            return rows[:count]
    return rows


def evaluate(engine, network, cases, seed):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    with torch.no_grad():
        for query, case in cases:
            n = len(case.edge_ids)
            weights = network(case.features).sigmoid()
            k = max(1, math.ceil(n * 0.2))
            keep = torch.zeros(n)
            keep[weights.topk(k).indices] = 1
            baseline = torch.zeros(n)
            baseline[torch.randperm(n, generator=generator)[:k]] = 1
            original = case.original_logit
            soft = float(case.score(weights))
            retained, removed = float(case.score(keep)), float(case.score(1 - keep))
            random_retained, random_removed = (
                float(case.score(baseline)),
                float(case.score(1 - baseline)),
            )
            rows.append(
                {
                    "query": list(query),
                    "edges": n,
                    "original_logit": original,
                    "reconstruction_error": abs(original - case.reconstructed_logit),
                    "soft_mask_logit": soft,
                    "soft_mask_probability_error": abs(
                        float(torch.tensor(soft).sigmoid())
                        - float(torch.tensor(original).sigmoid())
                    ),
                    "retained_20pct_logit_error": abs(original - retained),
                    "random_20pct_logit_error": abs(original - random_retained),
                    "gated_top20pct_logit_change": abs(original - removed),
                    "gated_random20pct_logit_change": abs(original - random_removed),
                    "mean_mask": float(weights.mean()),
                }
            )
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    return {
        "complete": True,
        "queries": len(rows),
        "query_split": "official valid drug-drug triples and random candidate tails; disjoint from explainer training queries",
        "message_graph": "official training split only; candidate gates restricted to incoming edges of query endpoints",
        "evaluation_type": "held-out model fidelity, not biological validation or predictor MRR",
        "mean_soft_mask_probability_error": mean("soft_mask_probability_error"),
        "mean_retained_20pct_logit_error": mean("retained_20pct_logit_error"),
        "mean_random_20pct_logit_error": mean("random_20pct_logit_error"),
        "mean_gated_top20pct_logit_change": mean("gated_top20pct_logit_change"),
        "mean_gated_random20pct_logit_change": mean("gated_random20pct_logit_change"),
        "mean_mask": mean("mean_mask"),
        "max_reconstruction_error": max(row["reconstruction_error"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-queries", type=int, default=64)
    parser.add_argument("--eval-queries", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ARTIFACT)
    args = parser.parse_args()
    if args.epochs < 1 or args.train_queries < 16 or args.eval_queries < 16:
        parser.error(
            "Use positive epochs and at least 16 queries per explanation split."
        )
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    os.environ["ENABLE_PREDICTIONS"] = "true"
    from src.data.load_biokg import load_dataset
    from src.utils import model as runtime

    engine = ExplanationEngine(runtime)
    dataset = load_dataset()
    train_queries = queries(
        dataset.load_split("train"), runtime.contract, args.train_queries * 3, args.seed
    )
    eval_queries = queries(
        dataset.load_split("valid"),
        runtime.contract,
        args.eval_queries * 3,
        args.seed + 1,
    )
    train_set = set(train_queries)

    def prepare(candidates, count, exclude=()):
        result = []
        for query in candidates:
            if query in exclude:
                continue
            try:
                case = engine.pair(*query)
                if not len(case.edge_ids):
                    continue
                result.append((query, case))
                print("prepared", len(result), query, len(case.edge_ids), flush=True)
                if len(result) == count:
                    return result
            except ExplanationUnavailable as error:
                print("skipped", query, str(error), flush=True)
        raise RuntimeError(
            "Not enough complete computational neighborhoods for requested query sample"
        )

    training = prepare(train_queries, args.train_queries)
    validation = prepare(eval_queries, args.eval_queries, train_set)
    config = {"dimensions": engine.embeddings.shape[1], "hidden": 64}
    network = EdgeMaskNetwork(**config)
    sample = torch.cat(
        [
            case.features[:: max(1, len(case.edge_ids) // 256)][:256]
            for _, case in training
        ]
    )
    network.feature_mean.copy_(sample.mean(0))
    network.feature_scale.copy_(sample.std(0).clamp_min(0.01))
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    steps, started = 0, time.time()
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(training)
        total = 0
        temperature = 5 * (0.5 / 5) ** (epoch / max(1, args.epochs - 1))
        for query, case in training:
            optimizer.zero_grad()
            logits = network(case.features)
            uniform = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            gates = (
                (logits + uniform.log() - (1 - uniform).log()) / temperature
            ).sigmoid()
            masked = case.score(gates)
            target = torch.tensor(case.original_logit)
            # Distillation preserves the frozen link predictor's soft output.
            fidelity = F.binary_cross_entropy_with_logits(masked, target.sigmoid())
            fidelity = (
                fidelity + 0.1 * ((masked - target) / (1 + target.abs())).square()
            )
            entropy = -(
                gates * gates.clamp_min(1e-6).log()
                + (1 - gates) * (1 - gates).clamp_min(1e-6).log()
            ).mean()
            loss = fidelity + 0.1 * gates.mean() + 0.001 * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite explainer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": total / len(training),
                    "seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
    network.eval()
    report = evaluate(engine, network, validation, args.seed)
    report["limitations"] = (
        "Small deterministic query sample; random candidates are not verified negatives. Model fidelity does not validate biological mechanisms. Dense models may require many edges, and root/self transformations remain active."
    )
    payload = {
        "format": FORMAT,
        "model_sha256": engine.fingerprint,
        "train_graph_sha256": engine.contract["train_graph_sha256"],
        "optimizer_steps": steps,
        "config": config,
        "state_dict": network.state_dict(),
        "training": {
            "epochs": args.epochs,
            "queries": len(training),
            "seed": args.seed,
            "learning_rate": 0.001,
            "temperature": [5, 0.5],
            "size_coefficient": 0.1,
            "entropy_coefficient": 0.001,
            "query_ids": [list(q) for q, _ in training],
        },
        "evaluation": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    args.output.with_suffix(".evaluation.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, indent=2)
        + "\n"
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2),
        flush=True,
    )
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
