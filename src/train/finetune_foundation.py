"""Fine-tune structural checkpoints using supervised triples from BioKG train only."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from src.data.load_biokg import load_biokg_as_hetero
from src.train.foundation_models import load_foundation, message_graph
from src.train.train_rgcn import encode_triples, sample_negatives
from src.utils.graph_contract import flatten_graph, graph_contract


def training_examples(graph, contract, pool_size):
    edges, relations = flatten_graph(graph, contract, "cpu")
    nodes = sum(contract["num_nodes"].values())
    relation_count = len(contract["relations"])
    heads, tails = edges
    known = encode_triples(heads, relations, tails, nodes, relation_count).unique(sorted=True)
    selected = torch.randperm(len(relations))[:pool_size]
    head, tail, relation = heads[selected], tails[selected], relations[selected]
    relation_types = contract["relations"]
    sources = [row[0] for row in relation_types]
    targets = [row[2] for row in relation_types]
    source_offsets = torch.tensor([contract["offsets"][kind] for kind in sources])
    target_offsets = torch.tensor([contract["offsets"][kind] for kind in targets])
    source_counts = torch.tensor([contract["num_nodes"][kind] for kind in sources])
    target_counts = torch.tensor([contract["num_nodes"][kind] for kind in targets])
    supervision = (head, relation, tail, source_offsets[relation], target_offsets[relation],
                   source_counts[relation], target_counts[relation])

    # Remove the positive triples and their same-relation mirrors before
    # constructing either message graph.
    for rel_id, key in enumerate(relation_types):
        mask = relation == rel_id
        src, _, dst = key
        count = contract["num_nodes"][dst]
        local_head = head[mask] - contract["offsets"][src]
        local_tail = tail[mask] - contract["offsets"][dst]
        key_ids = local_head * count + local_tail
        if src == dst:
            key_ids = torch.cat([key_ids, local_tail * count + local_head])
        raw_edge = graph[tuple(key)].edge_index
        keep = ~torch.isin(raw_edge[0] * count + raw_edge[1], key_ids)
        graph[tuple(key)].edge_index = raw_edge[:, keep]
    return supervision, known


def fine_tune(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    graph = load_biokg_as_hetero()
    contract = graph_contract(graph)
    nodes = sum(contract["num_nodes"].values())
    relation_count = len(contract["relations"])
    supervision, known = training_examples(graph, contract, args.pool_size)
    supervision = tuple(value.to(device) for value in supervision)
    known = known.to(device)
    model, provenance = load_foundation(args.model, args.upstream, device)
    context = message_graph(args.model, graph, contract, device)
    del graph
    # Upstream train mode repeatedly drops easy edges per minibatch. Both
    # message graphs already exclude the full supervision pool.
    model.eval()
    if args.decoder_only:
        model.requires_grad_(False)
        entity_model = model.entity_model_2 if args.model == "trix-entity" else model.entity_model
        entity_model.mlp.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    start = time.perf_counter()
    for step in range(args.steps):
        selected = torch.randint(len(supervision[0]), (args.batch_size,), device=device)
        h, r, t = (value[selected] for value in supervision[:3])
        nh, _, nt = sample_negatives(
            supervision, selected, known, nodes, relation_count
        )
        inverse = nh != h
        source = torch.where(inverse, t, h)
        target = torch.where(inverse, h, t)
        negative = torch.where(inverse, nh, nt)
        relation = r + inverse.to(r.dtype) * relation_count
        batch = torch.stack([
            source[:, None].expand(-1, 2), torch.stack([target, negative], dim=1),
            relation[:, None].expand(-1, 2),
        ], dim=-1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(context, batch)
        loss = (F.softplus(-logits[:, 0]).mean() + F.softplus(logits[:, 1]).mean()) / 2
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite fine-tuning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(json.dumps({"step": step + 1, "loss": loss.detach().item()}), flush=True)

    training = {
        **provenance, "fine_tuning_steps": args.steps, "supervision_pool": len(supervision[0]),
        "seed": args.seed, "batch_size": args.batch_size, "device": args.device,
        "learning_rate": args.learning_rate, "training_seconds": time.perf_counter() - start,
        "updated_parameters": "decoder MLP" if args.decoder_only else "all",
        "training_protocol": "typed corruptions; supervision excluded from both message graphs",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({"model_name": args.model, "graph_contract": contract,
                "model_state": model.state_dict(), "training": training}, temp)
    temp.replace(args.output)
    print(f"Saved {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["trix-entity", "ultra"], required=True)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decoder-only", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.pool_size, args.batch_size, args.learning_rate) <= 0:
        parser.error("Steps, pool size, batch size and learning rate must be positive")
    fine_tune(args)


if __name__ == "__main__":
    main()
