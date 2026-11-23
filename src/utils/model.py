"""Source data works independently of an optional evaluated model."""
import logging
import os

import torch

from src.data.load_biokg import load_biokg_as_hetero
from src.utils.checkpoint import load_checkpoint
from src.utils.graph_contract import (
    graph_contract,
    flatten_graph,
    query_indices,
)
from src.utils.mapping_loader import load_entity_mappings
from src.utils.paths import BIOKG_MAPPING_DIR, MODEL_CHECKPOINT

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

graph = load_biokg_as_hetero()
contract = graph_contract(graph)
entity_mappings = load_entity_mappings(
    str(BIOKG_MAPPING_DIR)
)
rel_type_to_id = {
    tuple(key): i
    for i, key in enumerate(contract["relations"])
}

model = embeddings = None
prediction_error = (
    "Predictions unavailable: train and evaluate "
    "a version 2 checkpoint."
)

if os.getenv("ENABLE_PREDICTIONS", "false").lower() == "true":
    try:
        model, checkpoint = load_checkpoint(
            MODEL_CHECKPOINT,
            contract,
            device,
        )

        edges, edge_types = flatten_graph(
            graph,
            contract,
            device,
        )
        x_dict = {
            kind: graph[kind].x.to(device)
            for kind in contract["node_types"]
        }

        with torch.inference_mode():
            embeddings = model(
                x_dict,
                edges,
                edge_types,
            )

        if not torch.isfinite(embeddings).all():
            raise ValueError("Nonfinite model embeddings")

        prediction_error = None
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as error:
        model = embeddings = None
        prediction_error = (
            f"Predictions unavailable: {error}"
        )
        logging.getLogger(__name__).warning(
            prediction_error
        )


class PredictionUnavailable(RuntimeError):
    pass


def predict_score(first, relation, second):
    head, rel, tail = query_indices(
        contract,
        first,
        relation,
        second,
    )

    if prediction_error:
        raise PredictionUnavailable(prediction_error)

    with torch.inference_mode():
        score = model.score(
            embeddings,
            torch.tensor(head, device=device),
            torch.tensor(rel, device=device),
            torch.tensor(tail, device=device),
        ).sigmoid().item()

    if not 0 <= score <= 1:
        raise PredictionUnavailable(
            "Nonfinite model score"
        )

    return score


def rank_drug_relations(first, second):
    relations = [(key[1], index) for key, index in rel_type_to_id.items()
                 if key[0] == key[2] == "drug"]
    if not relations:
        raise PredictionUnavailable("No drug–drug relations are available.")
    head, _, tail = query_indices(contract, first, relations[0][0], second)
    if prediction_error:
        raise PredictionUnavailable(prediction_error)

    with torch.inference_mode():
        logits = model.score(
            embeddings,
            torch.tensor(head, device=device),
            torch.tensor([index for _, index in relations], device=device),
            torch.tensor(tail, device=device),
        )
        if not torch.isfinite(logits).all():
            raise PredictionUnavailable("Nonfinite relation scores")
        scores = logits.sigmoid().tolist()
        order = logits.argsort(descending=True, stable=True).tolist()
    return [
        {"relation_type": relations[index][0], "uncalibrated_model_score": scores[index]}
        for index in order
    ]
