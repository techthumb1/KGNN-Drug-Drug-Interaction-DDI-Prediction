"""Exact DistMult decoder contributions for the loaded model's cached embeddings."""
import torch

from src.utils.graph_contract import query_indices


def explain_score(runtime, first, relation, second):
    head, rel, tail = query_indices(runtime.contract, first, relation, second)
    if runtime.prediction_error or runtime.model is None or runtime.embeddings is None:
        raise runtime.PredictionUnavailable(runtime.prediction_error or "Model is unavailable")
    with torch.inference_mode():
        terms = (runtime.embeddings[head] * runtime.model.relations.weight[rel] * runtime.embeddings[tail]).detach()
        if not torch.isfinite(terms).all():
            raise runtime.PredictionUnavailable("Nonfinite model contributions")
        logit = terms.sum()
        score = logit.sigmoid().item()
        without = (logit - terms).sigmoid()
        contributions = [{"dimension": i, "logit_contribution": float(value),
                          "score_without_dimension": float(without[i]),
                          "score_change_when_removed": float(without[i]) - score}
                         for i, value in enumerate(terms.cpu().tolist())]
    contributions.sort(key=lambda row: abs(row["logit_contribution"]), reverse=True)
    return {"method": "exact_distmult_decoder_decomposition", "drug1_id": first, "drug2_id": second,
            "relation_type": relation, "uncalibrated_model_score": score, "logit": float(logit),
            "positive_logit_total": sum(max(0, row["logit_contribution"]) for row in contributions),
            "negative_logit_total": sum(min(0, row["logit_contribution"]) for row in contributions),
            "contributions": contributions,
            "interpretation": "Each contribution is the product of one learned dimension in the two drug embeddings and the relation embedding. Their sum is the decoder logit; applying sigmoid gives the score. Removing one term holds all other dimensions fixed.",
            "limitations": "Latent dimensions have no established biological meaning. This is a faithful decoder calculation, not a graph-edge attribution, mechanism, clinical probability or severity explanation. The DistMult decoder is symmetric for a fixed relation; reversing these two drugs gives the same score."}


def run_explainer(*args, **kwargs):
    raise NotImplementedError("Use explain_score for the exact decoder decomposition; legacy graph attribution is unavailable.")
