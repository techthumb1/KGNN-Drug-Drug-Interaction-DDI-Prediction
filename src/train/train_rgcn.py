"""Full-graph research training; never called by web startup."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from src.data.load_biokg import (
    load_dataset,
    load_biokg_as_hetero,
)
from src.models.rgcn_model import RGCN
from src.utils.graph_contract import (
    graph_contract,
    flatten_graph,
)
from src.utils.checkpoint import FORMAT
from src.utils.evaluate import (
    split_tensors,
    evaluate_split,
)


def encode_triples(
    heads,
    relations,
    tails,
    nodes,
    relation_count,
):
    return (
        heads * relation_count + relations
    ) * nodes + tails


def sample_negatives(
    tensors,
    selected,
    known,
    nodes,
    relation_count,
):
    h, r, t, ho, to, hc, tc = (
        tensor[selected]
        for tensor in tensors
    )

    corrupt_head = (
        torch.rand(h.shape, device=h.device) < 0.5
    )
    nh, nt = h.clone(), t.clone()
    remaining = torch.ones_like(
        h, dtype=torch.bool
    )

    for _ in range(100):
        head_mask = remaining & corrupt_head
        tail_mask = remaining & ~corrupt_head

        sampled_heads = (
            torch.rand_like(h, dtype=torch.float32)
            * hc
        ).long()
        sampled_tails = (
            torch.rand_like(t, dtype=torch.float32)
            * tc
        ).long()

        nh[head_mask] = (
            sampled_heads[head_mask]
            + ho[head_mask]
        )
        nt[tail_mask] = (
            sampled_tails[tail_mask]
            + to[tail_mask]
        )

        keys = encode_triples(
            nh,
            r,
            nt,
            nodes,
            relation_count,
        )
        positions = torch.searchsorted(
            known,
            keys,
        ).clamp(max=known.numel() - 1)

        remaining = known[positions] == keys

        if not remaining.any():
            return nh, r, nt

    raise RuntimeError(
        "Could not sample negatives outside "
        "known training triples"
    )


def optimize_step(
    model,
    optimizer,
    x_dict,
    edges,
    edge_types,
    tensors,
    selected,
    known,
    nodes,
    relation_count,
):
    model.train()
    optimizer.zero_grad()

    embeddings = model(
        x_dict,
        edges,
        edge_types,
    )
    h, r, t = (
        tensor[selected]
        for tensor in tensors[:3]
    )
    nh, nr, nt = sample_negatives(
        tensors,
        selected,
        known,
        nodes,
        relation_count,
    )

    positive = model.score(
        embeddings, h, r, t
    )
    negative = model.score(
        embeddings, nh, nr, nt
    )

    loss = (
        F.softplus(-positive).mean()
        + F.softplus(negative).mean()
    ) / 2

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Nonfinite training loss"
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )
    optimizer.step()

    return loss.item()


def train(args):
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.eval_batch_size < 1
    ):
        raise ValueError(
            "Steps and batch sizes must be positive"
        )

    if (
        args.max_eval is not None
        and args.max_eval < 1
    ):
        raise ValueError(
            "max-eval must be positive"
        )

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = load_dataset()
    graph = load_biokg_as_hetero(dataset)
    contract = graph_contract(graph)

    config = {
        "hidden_channels": args.hidden_channels,
        "out_channels": args.hidden_channels,
        "num_layers": 2,
    }

    model = RGCN(
        (
            contract["node_types"],
            contract["relations"],
        ),
        contract["num_nodes"],
        **config,
    ).to(device)

    edges, edge_types = flatten_graph(
        graph,
        contract,
        device,
    )
    x_dict = {
        kind: graph[kind].x.to(device)
        for kind in contract["node_types"]
    }
    tensors = split_tensors(
        dataset.load_split("train"),
        contract,
        device,
    )

    nodes = sum(
        contract["num_nodes"].values()
    )
    relation_count = len(
        contract["relations"]
    )

    known = encode_triples(
        *tensors[:3],
        nodes,
        relation_count,
    ).unique(sorted=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    for step in range(args.steps):
        selected = torch.randint(
            tensors[0].numel(),
            (args.batch_size,),
            device=device,
        )

        loss = optimize_step(
            model,
            optimizer,
            x_dict,
            edges,
            edge_types,
            tensors,
            selected,
            known,
            nodes,
            relation_count,
        )

        print(
            json.dumps(
                {
                    "optimizer_step": step + 1,
                    "sampled_training_loss": loss,
                }
            ),
            flush=True,
        )

    model.eval()
    with torch.inference_mode():
        embeddings = model(
            x_dict,
            edges,
            edge_types,
        )

    evaluation = {}
    names = (
        ("valid", "test")
        if args.evaluate_test
        else ("valid",)
    )

    for name in names:
        evaluation[name] = evaluate_split(
            model,
            embeddings,
            dataset.load_split(name),
            contract,
            name,
            args.eval_batch_size,
            args.max_eval,
        )
        print(
            json.dumps(evaluation[name]),
            flush=True,
        )

    checkpoint = {
        "format": FORMAT,
        "graph_contract": contract,
        "model_config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_steps": args.steps,
        "seed": args.seed,
        "evaluation": evaluation,
        "training_config": vars(args),
    }

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)

    output.with_suffix(
        ".evaluation.json"
    ).write_text(
        json.dumps(evaluation, indent=2) + "\n"
    )

    print(
        f"Saved {output}; full validation AND test "
        "are required before enabling predictions."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--steps", type=int, default=100
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=16
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=64
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        help=(
            "Execution smoke test only; incomplete "
            "reports cannot enable serving"
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        default="models/rgcn_v2.pt",
    )

    train(parser.parse_args())


if __name__ == "__main__":
    main()