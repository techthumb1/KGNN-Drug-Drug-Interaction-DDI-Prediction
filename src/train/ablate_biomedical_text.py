"""Measure the effect of restricted biomedical description features on DDI ranking."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.load_biokg import load_biokg_as_hetero, load_dataset
from src.train.evaluate_foundation import typed_batches
from src.train.train_rgcn import encode_triples, sample_negatives
from src.utils.drugbank_full_lookup import iter_records
from src.utils.evaluate import mean_reciprocal_rank
from src.utils.graph_contract import graph_contract
from src.utils.mapping_loader import load_entity_mappings
from src.utils.paths import BIOKG_MAPPING_DIR

# Fixed independently of benchmark labels. Only class, modality and form
# indicators survive the projection.
TERMS = (
    "small molecule", "monoclonal antibody", "antibody fragment", "protein", "peptide",
    "polypeptide", "oligonucleotide", "nucleoside", "nucleotide", "steroid", "alkaloid",
    "flavonoid", "glycoside", "lipid", "phospholipid", "polysaccharide", "vitamin",
    "amino acid", "fatty acid", "organic acid", "benzodiazepine", "benzimidazole",
    "imidazole", "triazole", "pyridine", "pyrimidine", "quinoline", "cephalosporin",
    "penicillin", "macrolide", "aminoglycoside", "tetracycline", "sulfonamide",
    "thiazide", "phenothiazine", "barbiturate", "carbamate", "retinoid",
    "recombinant", "synthetic", "semisynthetic", "oral", "intravenous", "topical",
    "intramuscular", "subcutaneous", "inhalation", "tablets", "capsules", "solution",
)
EXCLUDE = re.compile(
    r"\b(?:interact\w*|coadministr\w*|concomitan\w*|co-administ\w*|combination|"
    r"co-use|contraindicat\w*|risk|adverse|side effects?)\b",
    re.IGNORECASE,
)


def description_features(path, nodes):
    mappings = load_entity_mappings(BIOKG_MAPPING_DIR)["drug"]
    local_ids = {str(int(raw[3:])): index for index, raw in mappings.items()
                 if raw.startswith("CID") and raw[3:].isdecimal()}
    terms = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in TERMS]
    vectors = defaultdict(set)
    discarded = 0
    for entry in iter_records(path):
        raw_cid = str(entry.get("pubchem", "")).strip()
        index = local_ids.get(str(int(raw_cid))) if raw_cid.isdecimal() else None
        description = entry.get("description")
        if index is None or not isinstance(description, str) or not description.strip():
            continue
        sentences = re.split(r"[.!?;\n]+", description)
        clean = [sentence for sentence in sentences if not EXCLUDE.search(sentence)]
        discarded += len(sentences) - len(clean)
        projected = tuple(float(any(term.search(sentence) for sentence in clean)) for term in terms)
        vectors[index].add(projected)
    result = torch.zeros(nodes, len(TERMS))
    coverage = torch.zeros(nodes, dtype=torch.bool)
    for index, options in vectors.items():
        if len(options) == 1:
            result[index] = torch.tensor(next(iter(options)))
            coverage[index] = True
    with path.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    return result, coverage, {
        "description_source": "local DrugBank descriptions matched by exact PubChem CID",
        "source_sha256": fingerprint,
        "covered_drugs": int(coverage.sum()),
        "drugs_with_retained_terms": int(result.any(dim=1).sum()),
        "conflicting_description_projections": sum(len(options) > 1 for options in vectors.values()),
        "sentences_excluded_by_interaction_language": discarded,
        "retained_feature_vocabulary": TERMS,
    }


def structure_features(graph, contract):
    nodes = contract["num_nodes"]["drug"]
    features = []
    for source, relation, target in contract["relations"]:
        if source == target == "drug" or "drug" not in (source, target):
            continue
        endpoints = graph[source, relation, target].edge_index
        indices = endpoints[0 if source == "drug" else 1]
        features.append(torch.bincount(indices, minlength=nodes).float().log1p())
    return torch.stack(features, dim=-1)


class PairModel(nn.Module):
    def __init__(self, features, num_relations):
        super().__init__()
        self.register_buffer("features", features, persistent=False)
        self.encoder = nn.Sequential(nn.Linear(features.shape[1], 32), nn.ReLU())
        self.relations = nn.Embedding(num_relations, 32)
        self.output = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))

    def score(self, head, relation, tail):
        first = self.encoder(self.features[head])
        second = self.encoder(self.features[tail])
        return self.output(torch.cat([first + second, first * second, self.relations(relation)], -1)).squeeze(-1)


def _ddi(split):
    return (np.asarray(split["head_type"]) == "drug") & (np.asarray(split["tail_type"]) == "drug")


def train_examples(dataset, contract, available):
    train = dataset.load_split("train")
    ddi = _ddi(train)
    h = torch.as_tensor(train["head"][ddi])
    t = torch.as_tensor(train["tail"][ddi])
    raw = torch.as_tensor(train["relation"][ddi])
    relations = len(contract["relations"])
    mapping = {int(raw): index for index, (src, raw, dst) in enumerate(contract["relations"])
               if src == dst == "drug"}
    lookup = torch.full((max(mapping) + 1,), -1, dtype=torch.long)
    for raw_id, index in mapping.items():
        lookup[raw_id] = index
    r = lookup[raw]
    nodes = len(available)
    seen = torch.where(available)[0]
    remap = torch.full((nodes,), -1, dtype=torch.long)
    remap[seen] = torch.arange(len(seen))
    train_mask = available[h] & available[t]
    h, r, t = remap[h[train_mask]], r[train_mask], remap[t[train_mask]]
    count = torch.full_like(h, len(seen))
    zeros = torch.zeros_like(h)
    known = torch.cat([
        encode_triples(h, r, t, len(seen), relations),
        encode_triples(t, r, h, len(seen), relations),
    ]).unique(sorted=True)
    return (h, r, t, zeros, zeros, count, count), seen, known


def evaluate(model, contract, split, selected, covered, batch_size):
    model.eval()
    offset = contract["offsets"]["drug"]
    scores, matched_scores = [], []
    with torch.inference_mode():
        for _, _, triples, negatives in typed_batches(
            split, contract, selected, batch_size, "cpu"
        ):
            head, relation, tail = triples
            head, tail = head - offset, tail - offset
            head_neg, tail_neg = (item - offset for item in negatives)
            pos = model.score(head, relation, tail)
            head_rank = mean_reciprocal_rank(
                pos, model.score(head_neg, relation[:, None].expand_as(head_neg),
                                 tail[:, None].expand_as(head_neg))
            )
            tail_rank = mean_reciprocal_rank(
                pos, model.score(head[:, None].expand_as(tail_neg),
                                 relation[:, None].expand_as(tail_neg), tail_neg)
            )
            values = (head_rank + tail_rank) / 2
            scores.extend(values.tolist())
            matched_scores.extend(values[covered[head] & covered[tail]].tolist())
    return {"mrr": float(np.mean(scores)), "triples": len(scores),
            "both_descriptions_mrr": float(np.mean(matched_scores)) if matched_scores else None,
            "both_descriptions_triples": len(matched_scores)}


def run(args):
    dataset = load_dataset()
    graph = load_biokg_as_hetero(dataset)
    contract = graph_contract(graph)
    nodes = contract["num_nodes"]["drug"]
    structure = structure_features(graph, contract)
    text, covered, provenance = description_features(args.descriptions, nodes)
    if not covered.any() or not text.any():
        raise ValueError("Description source has no matched descriptive features")
    split = dataset.load_split("valid")
    eligible = np.flatnonzero(_ddi(split))
    rng = np.random.default_rng(args.seed)
    heldout_ids = rng.choice(nodes, size=max(1, nodes // 10), replace=False)
    heldout = np.zeros(nodes, dtype=bool)
    heldout[heldout_ids] = True
    new_entity = heldout[np.asarray(split["head"])[eligible]] | heldout[np.asarray(split["tail"])[eligible]]
    results = {}
    for cohort, row_ids, available in (
        ("official_ddi_sample", eligible, torch.ones(nodes, dtype=torch.bool)),
        ("entity_heldout_with_non_ddi_context", eligible[new_entity], torch.from_numpy(~heldout)),
    ):
        selected = np.sort(rng.choice(row_ids, size=min(args.queries, len(row_ids)), replace=False))
        training, seen, known = train_examples(dataset, contract, available)
        pairs = eligible[(covered[split["head"][eligible]] & covered[split["tail"][eligible]]).numpy()]
        if cohort != "official_ddi_sample":
            pairs = np.intersect1d(pairs, row_ids)
        matched = np.sort(rng.choice(pairs, size=min(args.queries, len(pairs)), replace=False))
        observed_structure = structure[available]
        values = (structure - observed_structure.mean(0)) / observed_structure.std(0).clamp_min(1e-6)
        rows = {}
        for modality, features in (
            ("structure_only", values), ("text_only", text),
            ("combined", torch.cat([values, text], dim=1)),
        ):
            torch.manual_seed(args.seed)
            model = PairModel(features, len(contract["relations"]))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            sampler = torch.Generator().manual_seed(args.seed)
            torch.manual_seed(args.seed + 1)
            for _ in range(args.steps):
                chosen = torch.randint(len(training[0]), (args.batch_size,), generator=sampler)
                nh, r, nt = sample_negatives(training, chosen, known, len(seen), len(contract["relations"]))
                h, _, t = (item[chosen] for item in training[:3])
                loss = (F.softplus(-model.score(seen[h], r, seen[t])).mean()
                        + F.softplus(model.score(seen[nh], r, seen[nt])).mean()) / 2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            rows[modality] = {
                "random": evaluate(model, contract, split, selected, covered, batch_size=8),
                "description_matched": (
                    evaluate(model, contract, split, matched, covered, batch_size=8)
                    if len(matched) else None
                ),
            }
            print(f"Finished {cohort} {modality}", flush=True)
        results[cohort] = {
            "models": rows, "eligible_triples": len(row_ids),
            "eligible_description_matched_triples": len(pairs),
            "sample_indices": selected.tolist(), "matched_indices": matched.tolist(),
        }

    report = {"dataset": "ogbl-biokg", "split": "valid", "seed": args.seed, "steps": args.steps,
              "training_batch_size": args.batch_size, "heldout_drugs": heldout_ids.tolist(),
              "structure": "Degree by non-DDI training relation; context available for held-out drugs",
              "train_graph_sha256": contract["train_graph_sha256"],
              "provenance": provenance, "results": results}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.queries, args.batch_size) < 1:
        parser.error("Steps, queries and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
