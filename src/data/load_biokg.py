"""Official BioKG loading without globally overriding torch.load."""
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import torch
from ogb.linkproppred import LinkPropPredDataset
from torch_geometric.data import HeteroData

from src.utils.paths import DATASET_ROOT


class BioKGDataset(LinkPropPredDataset):
    def pre_process(self):
        cached = Path(self.root) / "processed" / "data_processed"

        if cached.exists():
            # Only OGB's trusted NumPy cache needs pickle compatibility.
            self.graph = torch.load(cached, weights_only=False)
        else:
            super().pre_process()

    def load_split(self, name):
        if name not in ("train", "valid", "test"):
            raise ValueError("Unknown official split")

        return torch.load(
            Path(self.root) / "split" / "random" / f"{name}.pt",
            weights_only=False,
        )


def load_dataset():
    root = DATASET_ROOT / "ogbl_biokg"

    if not (
        (root / "processed" / "data_processed").exists()
        or (root / "raw" / "triplet-type-list.csv.gz").exists()
    ):
        import ogb
        from src.utils.data_loader import download_file

        metadata = pd.read_csv(
            Path(ogb.__file__).parent / "linkproppred" / "master.csv",
            index_col=0,
        )
        url = metadata["ogbl-biokg"]["url"].replace(
            "http://", "https://", 1
        )
        archive = DATASET_ROOT / "biokg.zip"
        download_file(url, archive)

        with tempfile.TemporaryDirectory(
            dir=DATASET_ROOT, prefix="biokg-extract-"
        ) as temporary:
            extraction = Path(temporary)

            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    destination = (
                        extraction / member.filename
                    ).resolve()
                    if not destination.is_relative_to(
                        extraction.resolve()
                    ):
                        raise RuntimeError("Unsafe BioKG archive path")

                bundle.extractall(extraction)

            if root.exists():
                shutil.copytree(
                    extraction / "biokg",
                    root,
                    dirs_exist_ok=True,
                )
            else:
                (extraction / "biokg").replace(root)

        archive.unlink()

    return BioKGDataset(
        name="ogbl-biokg",
        root=str(DATASET_ROOT),
    )


def graph_from_split(num_nodes_dict, split):
    """Construct message-passing data from official training triples."""
    graph = HeteroData()

    for kind, count in sorted(num_nodes_dict.items()):
        graph[kind].num_nodes = int(count)
        graph[kind].x = torch.arange(count)

    groups = {}
    for i, (head_type, relation, tail_type) in enumerate(
        zip(
            split["head_type"],
            split["relation"],
            split["tail_type"],
        )
    ):
        key = (
            str(head_type),
            str(int(relation)),
            str(tail_type),
        )
        groups.setdefault(key, []).append(i)

    for key, indices in sorted(groups.items()):
        graph[key].edge_index = torch.from_numpy(
            np.stack(
                [
                    split["head"][indices],
                    split["tail"][indices],
                ]
            )
        ).long()

    graph.validate(raise_on_error=True)
    return graph


def load_biokg_as_hetero(dataset=None):
    dataset = dataset if dataset is not None else load_dataset()
    return graph_from_split(
        dataset[0]["num_nodes_dict"],
        dataset.load_split("train"),
    )