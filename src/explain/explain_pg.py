"""PGExplainer adaptation for two-layer R-GCN/DistMult link prediction.

A shared query-conditioned MLP learns Concrete message gates. Relation-wise
mean denominators and root transformations stay fixed. The candidate scope is every incoming edge to the two queried drugs.
Other graph context remains fixed; no truncated display graph is substituted.
"""

import hashlib
import math
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.utils.graph_contract import query_indices
from src.utils.paths import MODEL_CHECKPOINT

ARTIFACT = MODEL_CHECKPOINT.with_name("pgexplainer.pt")
LEGACY_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "pgexplainer.pt"
FORMAT = "biokg-pgexplainer-message-gates-v1"
MAX_EDGES = 50000
_lock = Lock()
_engine = _artifact = None
_cache = OrderedDict()


class ExplanationUnavailable(RuntimeError):
    pass


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(list(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class EdgeMaskNetwork(nn.Module):
    def __init__(self, dimensions=64, hidden=64):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(6 * dimensions))
        self.register_buffer("feature_scale", torch.ones(6 * dimensions))
        self.layers = nn.Sequential(
            nn.Linear(6 * dimensions, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, features):
        return self.layers(
            (features - self.feature_mean) / self.feature_scale
        ).flatten()


class PairComputation:
    def __init__(self, engine, head, relation, tail):
        self.engine, self.query = engine, (head, relation, tail)
        pair = np.array([head, tail])
        ids = engine.incoming(pair)
        self.edge_ids = ids
        self.source, self.target, self.types = (
            engine.sources[ids],
            engine.targets[ids],
            engine.types[ids],
        )
        middle = np.unique(np.concatenate((pair, self.source)))
        self.middle, self.pair = torch.from_numpy(middle), torch.from_numpy(pair)
        self.pair_positions = torch.from_numpy(np.searchsorted(middle, pair))
        self.groups = []
        for rel in np.unique(self.types):
            selected = np.flatnonzero(self.types == rel)
            src = self.source[selected]
            dst = (self.target[selected] == tail).astype(np.int64)
            counts = np.bincount(dst, minlength=2)
            self.groups.append(
                (
                    int(rel),
                    torch.from_numpy(src),
                    torch.from_numpy(np.searchsorted(middle, src)),
                    torch.from_numpy(dst),
                    torch.from_numpy(selected),
                    torch.from_numpy(counts[dst]).float(),
                )
            )
        z, r, n = engine.embeddings, engine.relations, len(ids)
        self.features = torch.cat(
            (
                z[self.source],
                z[self.target],
                r[self.types.astype(np.int64)],
                z[head].expand(n, -1),
                z[tail].expand(n, -1),
                r[relation].expand(n, -1),
            ),
            dim=1,
        )
        self.original_logit = float((z[head] * r[relation] * z[tail]).sum())
        with torch.no_grad():
            self.reconstructed_logit = float(self.score(torch.ones(n)))
        if abs(self.original_logit - self.reconstructed_logit) > 2e-4:
            raise ExplanationUnavailable(
                "Computation does not reproduce the predictor; explanation withheld."
            )

    def score(self, gates):
        # Only messages directly entering the queried drugs are candidates.
        # All other graph messages stay at their full, unmasked values.
        weight, root, bias = self.engine.weights[0]
        out = self.engine.initial[self.pair] @ root + bias
        for rel, src, local, dst, pos, denom in self.groups:
            messages = self.engine.initial[src] * (gates[pos] / denom).unsqueeze(1)
            out = (
                out
                + torch.zeros(2, self.engine.initial.shape[1]).index_add(
                    0, dst, messages
                )
                @ weight[rel]
            )
        x = self.engine.hidden[self.middle].index_copy(
            0, self.pair_positions, out.relu()
        )
        weight, root, bias = self.engine.weights[1]
        out = x[self.pair_positions] @ root + bias
        for rel, src, local, dst, pos, denom in self.groups:
            messages = x[local] * (gates[pos] / denom).unsqueeze(1)
            out = (
                out
                + torch.zeros(2, x.shape[1]).index_add(0, dst, messages) @ weight[rel]
            )
        return (out[0] * self.engine.relations[self.query[1]] * out[1]).sum()


class ExplanationEngine:
    def __init__(self, runtime):
        if runtime.prediction_error or runtime.model is None:
            raise ExplanationUnavailable(
                runtime.prediction_error or "Model unavailable"
            )
        model = runtime.model
        if len(model.convs) != 2 or any(c.aggr != "mean" for c in model.convs):
            raise ExplanationUnavailable(
                "Supports the evaluated two-layer mean R-GCN only."
            )
        self.contract, self.fingerprint = runtime.contract, model_digest(model)
        self.initial = torch.cat(
            [model.embeddings[k].weight.detach().cpu() for k in model.node_types]
        ).clone()
        self.embeddings, self.relations = (
            runtime.embeddings.detach().cpu().clone(),
            model.relations.weight.detach().cpu().clone(),
        )
        from src.utils.graph_contract import flatten_graph

        edges, types = flatten_graph(
            runtime.graph, runtime.contract, next(model.parameters()).device
        )
        with torch.inference_mode():
            self.hidden = (
                model.convs[0](self.initial.to(edges.device), edges, types)
                .relu()
                .cpu()
                .clone()
            )
        del edges, types
        self.weights = []
        for conv in model.convs:
            weight = conv.weight.detach().cpu()
            if conv.num_bases is not None:
                weight = (
                    conv.comp.detach().cpu() @ weight.reshape(conv.num_bases, -1)
                ).reshape(conv.num_relations, conv.in_channels_l, conv.out_channels)
            self.weights.append(
                (
                    weight.clone(),
                    conv.root.detach().cpu().clone(),
                    conv.bias.detach().cpu().clone(),
                )
            )
        sources, targets, types = [], [], []
        for i, (src, rel, dst) in enumerate(self.contract["relations"]):
            edges = runtime.graph[(src, rel, dst)].edge_index.cpu().numpy()
            sources.append(edges[0] + self.contract["offsets"][src])
            targets.append(edges[1] + self.contract["offsets"][dst])
            types.append(np.full(edges.shape[1], i, dtype=np.int16))
        self.sources, self.targets, self.types = (
            np.concatenate(sources),
            np.concatenate(targets),
            np.concatenate(types),
        )
        self.order = np.argsort(self.targets, kind="stable")
        self.pointer = np.concatenate(
            ([0], np.cumsum(np.bincount(self.targets, minlength=len(self.initial))))
        )

    def incoming(self, nodes):
        if int(np.sum(self.pointer[nodes + 1] - self.pointer[nodes])) > MAX_EDGES:
            raise ExplanationUnavailable(
                f"Complete computation exceeds the {MAX_EDGES}-edge interactive limit."
            )
        return np.concatenate(
            [self.order[self.pointer[n] : self.pointer[n + 1]] for n in nodes]
        )

    def pair(self, first, relation, second):
        return PairComputation(
            self, *query_indices(self.contract, first, relation, second)
        )


def load_artifact(engine, path=ARTIFACT):
    if path == ARTIFACT and not path.is_file():
        path = LEGACY_ARTIFACT
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != FORMAT
        or payload.get("model_sha256") != engine.fingerprint
        or payload.get("train_graph_sha256") != engine.contract["train_graph_sha256"]
        or payload.get("optimizer_steps", 0) < 1
    ):
        raise ExplanationUnavailable(
            "PGExplainer does not match this graph and predictor; retrain it."
        )
    report = payload.get("evaluation", {})
    if not report.get("complete") or report.get("queries", 0) < 16:
        raise ExplanationUnavailable("PGExplainer held-out evaluation is missing.")
    metrics = (
        "mean_soft_mask_probability_error",
        "mean_retained_20pct_logit_error",
        "mean_random_20pct_logit_error",
        "mean_mask",
        "max_reconstruction_error",
    )
    if (
        any(
            not isinstance(report.get(k), (float, int)) or not math.isfinite(report[k])
            for k in metrics
        )
        or not 0 < report["mean_mask"] < 0.99
        or report["max_reconstruction_error"] > 2e-4
        or report["mean_retained_20pct_logit_error"]
        >= report["mean_random_20pct_logit_error"]
    ):
        raise ExplanationUnavailable(
            "PGExplainer did not pass nondegenerate-mask and held-out fidelity checks."
        )
    network = EdgeMaskNetwork(**payload["config"])
    network.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise ExplanationUnavailable("Nonfinite PGExplainer parameters")
    return network.eval(), payload


@lru_cache(maxsize=1)
def artifact_status(model, graph_sha):
    if model is None:
        return "unavailable"
    try:
        load_artifact(
            SimpleNamespace(
                fingerprint=model_digest(model),
                contract={"train_graph_sha256": graph_sha},
            )
        )
        return "pgexplainer"
    except (OSError, ValueError, KeyError, RuntimeError):
        return "unavailable"


def run_pgexplainer(runtime, first, relation, second, top_k=20):
    global _engine, _artifact
    if runtime is None:
        raise ExplanationUnavailable("A ready, evaluated predictor is required.")
    query_indices(runtime.contract, first, relation, second)
    if runtime.prediction_error:
        raise ExplanationUnavailable(runtime.prediction_error)
    if not _lock.acquire(blocking=False):
        raise ExplanationUnavailable(
            "Another graph explanation is running; retry shortly."
        )
    try:
        if _artifact is None:
            if _engine is None:
                _engine = ExplanationEngine(runtime)
            _artifact = load_artifact(_engine)
        network, payload = _artifact
        key = (first, str(relation), second, top_k)
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        case = _engine.pair(first, relation, second)
        if not len(case.edge_ids):
            raise ExplanationUnavailable(
                "No incoming computational edges; only root transformations contribute."
            )
        with torch.no_grad():
            weights = network(case.features).sigmoid()
            top = weights.argsort(descending=True)[:top_k]
            keep = torch.zeros_like(weights)
            keep[top] = 1
            kept, gated = float(case.score(keep)), float(case.score(1 - keep))
            rows = []
            for pos in top.tolist():
                gate = torch.ones_like(weights)
                gate[pos] = 0
                without = float(case.score(gate))
                sk, rel, tk = _engine.contract["relations"][int(case.types[pos])]
                src = int(case.source[pos]) - _engine.contract["offsets"][sk]
                dst = int(case.target[pos]) - _engine.contract["offsets"][tk]
                rows.append(
                    {
                        "source": f"{sk}:{src}",
                        "target": f"{tk}:{dst}",
                        "relation": rel,
                        "mask_weight": float(weights[pos]),
                        "logit_without_edge_message": without,
                        "logit_change_when_gated": case.original_logit - without,
                    }
                )
        sigmoid = lambda x: float(torch.tensor(x).sigmoid())
        result = {
            "method": "pgexplainer_link_prediction_adaptation",
            "drug1_id": first,
            "drug2_id": second,
            "relation_type": str(relation),
            "logit": case.original_logit,
            "uncalibrated_model_score": sigmoid(case.original_logit),
            "reconstruction_error": abs(case.original_logit - case.reconstructed_logit),
            "scope": "all incoming edges to the two queried drug nodes; other graph context fixed",
            "computational_edges": len(case.edge_ids),
            "displayed_edges": len(rows),
            "edges": rows,
            "top_edges_kept_logit": kept,
            "top_edges_gated_logit": gated,
            "top_edges_kept_score": sigmoid(kept),
            "top_edges_gated_score": sigmoid(gated),
            "evaluation": payload["evaluation"],
            "interpretation": "A trained, query-conditioned PGExplainer ranks incoming graph messages. Mask weights are selection scores. Signed ablation measures the predictor change when an edge is gated in both layers.",
            "limitations": "Only incoming edges to the two queried drugs are ranked; other context stays fixed. PGExplainer adaptation evaluated on a limited held-out query sample. Gating retains original mean denominators and self/root transformations; it is not physical edge deletion. The displayed edges need not retain the full prediction. Model attribution does not establish biological mechanisms or medication safety.",
        }
        _cache[key] = result
        if len(_cache) > 32:
            _cache.popitem(last=False)
        return result
    except (OSError, KeyError, RuntimeError) as error:
        if isinstance(error, ExplanationUnavailable):
            raise
        raise ExplanationUnavailable(
            f"Graph explanation unavailable: {error}"
        ) from error
    finally:
        _lock.release()
