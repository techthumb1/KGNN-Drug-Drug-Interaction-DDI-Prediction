"""Evaluate fixed selected weights on test data without any optimization."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from src.data.load_biokg import load_dataset, load_biokg_as_hetero
from src.utils.checkpoint import load_checkpoint
from src.utils.evaluate import evaluate_split
from src.utils.graph_contract import graph_contract, flatten_graph
from src.utils.split_audit import audit_split_separation


def require_selected_candidate(checkpoint):
    valid = checkpoint.get("evaluation", {}).get("valid", {})
    if (
        valid.get("split") != "valid"
        or valid.get("complete") is not True
        or valid.get("evaluated_triples", 0) <= 0
        or valid.get("evaluated_triples") != valid.get("total_triples")
        or not isinstance(valid.get("mrr"), (int, float))
        or not 0 < valid["mrr"] <= 1
    ):
        raise ValueError("Select a candidate with complete official validation MRR first")
    if "test" in checkpoint.get("evaluation", {}):
        raise ValueError("Candidate already contains test results; do not use test for model selection")


def finalize(args):
    source, output = Path(args.checkpoint), Path(args.output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Use a new output path; preserve the selected candidate and prior test reports")
    if not args.selection_note.strip() or args.eval_batch_size < 1:
        raise ValueError("Record the validation-based selection decision and a positive batch size")
    dataset = load_dataset()
    graph = load_biokg_as_hetero(dataset)
    contract = graph_contract(graph)
    model, candidate = load_checkpoint(source, contract, args.device, require_evaluation=False)
    require_selected_candidate(candidate)
    audit = audit_split_separation(dataset, graph, contract)
    if candidate["evaluation"]["valid"]["total_triples"] != audit["valid"]["triples"]:
        raise ValueError("Validation report does not cover the current official split")
    edges, edge_types = flatten_graph(graph, contract, args.device)
    x_dict = {kind: graph[kind].x.to(args.device) for kind in contract["node_types"]}
    with torch.inference_mode():
        embeddings = model(x_dict, edges, edge_types)
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite model embeddings")
    # No optimizer, no resampling and no weight update between valid and test.
    test = evaluate_split(
        model, embeddings, dataset.load_split("test"), contract, "test", args.eval_batch_size
    )
    checkpoint = copy.copy(candidate)
    # Keep optimizer state in the candidate; serving needs only selected weights.
    checkpoint.pop("optimizer_state_dict", None)
    checkpoint["evaluation"] = {**candidate["evaluation"], "test": test}
    checkpoint["split_audit"] = audit
    checkpoint["selection"] = {
        "criterion": "official validation MRR",
        "validation_mrr": candidate["evaluation"]["valid"]["mrr"],
        "note": args.selection_note,
        "source_checkpoint_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_evaluated_with_fixed_weights": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    # Recheck the same serving gate before publishing the artifact.
    load_checkpoint(temporary, contract, "cpu")
    temporary.replace(output)
    output.with_suffix(".evaluation.json").write_text(json.dumps({
        "evaluation": checkpoint["evaluation"],
        "selection": checkpoint["selection"],
        "split_audit": audit,
    }, indent=2) + "\n")
    print(json.dumps({"output": str(output), "test": test}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="models/rgcn_v2.pt")
    parser.add_argument("--selection-note", required=True,
                        help="Record the validation-based choice and frozen training recipe before inspecting test")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    finalize(parser.parse_args())


if __name__ == "__main__":
    main()
