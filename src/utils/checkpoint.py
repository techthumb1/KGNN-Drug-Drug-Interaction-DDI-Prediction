"""Reject legacy weights and incompatible or incomplete artifacts."""
import math

import torch

from src.models.rgcn_model import RGCN

FORMAT = "biokg-rgcn-distmult-v2"


def load_checkpoint(
    path,
    contract,
    device,
    require_evaluation=True,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != FORMAT
        or checkpoint.get("graph_contract") != contract
    ):
        raise ValueError(
            "Legacy or incompatible checkpoint; "
            "retrain with src.train.train_rgcn"
        )

    if checkpoint.get("optimizer_steps", 0) < 1:
        raise ValueError(
            "Checkpoint has no recorded optimization"
        )

    if require_evaluation:
        for name in ("valid", "test"):
            report = checkpoint.get("evaluation", {}).get(
                name, {}
            )

            if (
                report.get("split") != name
                or report.get("complete") is not True
                or report.get("evaluated_triples", 0) <= 0
                or report.get("evaluated_triples")
                != report.get("total_triples")
                or not isinstance(
                    report.get("mrr"), (int, float)
                )
                or not math.isfinite(report["mrr"])
                or not 0 < report["mrr"] <= 1
            ):
                raise ValueError(
                    "Full official validation and test MRR "
                    "required before serving scores"
                )

    model = RGCN(
        (
            contract["node_types"],
            contract["relations"],
        ),
        contract["num_nodes"],
        **checkpoint["model_config"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if not all(
        torch.isfinite(parameter).all()
        for parameter in model.parameters()
    ):
        raise ValueError("Nonfinite checkpoint parameters")

    return model.eval(), checkpoint
