"""Load only compatible, evaluated models."""
import torch

from src.data.load_biokg import load_biokg_as_hetero
from src.utils.checkpoint import load_checkpoint
from src.utils.graph_contract import graph_contract
from src.utils.paths import MODEL_CHECKPOINT


def load_model_and_graph(
    checkpoint_path=MODEL_CHECKPOINT,
    device=None,
):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    graph = load_biokg_as_hetero()
    model, _ = load_checkpoint(
        checkpoint_path,
        graph_contract(graph),
        device,
    )
    return model, graph