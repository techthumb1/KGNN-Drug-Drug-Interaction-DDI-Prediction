"""Synthetic fixtures validate mechanics, never prediction quality."""

import copy
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.data.load_biokg import graph_from_split
from src.models.rgcn_model import RGCN
from src.train.train_rgcn import (
    encode_triples,
    optimize_step,
    sample_negatives,
)
from src.utils.checkpoint import FORMAT, load_checkpoint
from src.utils.evaluate import (
    evaluate_split,
    mean_reciprocal_rank,
    split_tensors,
)
from src.utils.graph_contract import (
    flatten_graph,
    graph_contract,
    query_indices,
)


def fixture():
    train = {
        "head": np.array([0, 1, 2, 0]),
        "tail": np.array([1, 2, 3, 1]),
        "head_type": ["drug"] * 4,
        "tail_type": ["drug", "drug", "drug", "protein"],
        "relation": np.array([0, 0, 1, 2]),
    }
    graph = graph_from_split(
        {"disease": 2, "drug": 4, "protein": 3},
        train,
    )
    contract = graph_contract(graph)
    config = {
        "hidden_channels": 4,
        "out_channels": 4,
        "num_layers": 2,
    }
    model = RGCN(
        (contract["node_types"], contract["relations"]),
        contract["num_nodes"],
        **config,
    )
    return train, graph, contract, model, config


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        (
            self.train,
            self.graph,
            self.contract,
            self.model,
            self.config,
        ) = fixture()

    def test_edges_and_queries_use_the_same_type_offsets(self):
        edges, _ = flatten_graph(self.graph, self.contract, "cpu")

        self.assertEqual(
            self.contract["offsets"],
            {"disease": 0, "drug": 2, "protein": 6},
        )
        self.assertEqual(edges[:, -1].tolist(), [2, 7])
        self.assertEqual(
            query_indices(
                self.contract, 0, "2", 1, "drug", "protein"
            ),
            (2, 2, 7),
        )
        self.assertEqual(
            query_indices(self.contract, 0, "0", 1),
            (2, 0, 3),
        )

    def test_contract_detects_edge_changes_not_only_tensor_shapes(self):
        changed = self.graph.clone()
        changed[("drug", "0", "drug")].edge_index[0, 0] = 2

        self.assertNotEqual(
            graph_contract(changed),
            self.contract,
        )

    def test_embedding_order_does_not_depend_on_input_dictionary(self):
        edges, edge_types = flatten_graph(
            self.graph, self.contract, "cpu"
        )
        normal = self.model(
            self.graph.x_dict, edges, edge_types
        )
        reverse = self.model(
            dict(reversed(list(self.graph.x_dict.items()))),
            edges,
            edge_types,
        )

        self.assertTrue(torch.equal(normal, reverse))

    def test_selected_relation_changes_decoder_score(self):
        with torch.no_grad():
            self.model.relations.weight[0].fill_(1)
            self.model.relations.weight[1].fill_(-1)

        result = self.model.score(
            torch.ones(9, 4),
            torch.tensor([2, 2]),
            torch.tensor([0, 1]),
            torch.tensor([3, 3]),
        )

        self.assertEqual(result.tolist(), [4.0, -4.0])

    def test_real_optimizer_updates_encoder_and_decoder(self):
        tensors = split_tensors(
            self.train, self.contract, "cpu"
        )
        nodes, relations = 9, 3
        known = encode_triples(
            *tensors[:3], nodes, relations
        ).unique(sorted=True)
        before = {
            name: parameter.clone()
            for name, parameter in self.model.named_parameters()
        }
        edges, edge_types = flatten_graph(
            self.graph, self.contract, "cpu"
        )

        loss = optimize_step(
            self.model,
            torch.optim.Adam(
                self.model.parameters(), lr=0.01
            ),
            self.graph.x_dict,
            edges,
            edge_types,
            tensors,
            torch.arange(4),
            known,
            nodes,
            relations,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertFalse(
            torch.equal(
                before["relations.weight"],
                self.model.relations.weight,
            )
        )
        self.assertFalse(
            torch.equal(
                before["embeddings.drug.weight"],
                self.model.embeddings["drug"].weight,
            )
        )

    def test_negative_sampling_is_typed_and_excludes_training_positives(self):
        tensors = split_tensors(
            self.train, self.contract, "cpu"
        )
        known = encode_triples(
            *tensors[:3], 9, 3
        ).unique(sorted=True)
        selected = torch.arange(4).repeat(20)

        heads, relations, tails = sample_negatives(
            tensors, selected, known, 9, 3
        )

        self.assertFalse(
            torch.isin(
                encode_triples(
                    heads, relations, tails, 9, 3
                ),
                known,
            ).any()
        )
        self.assertTrue(
            ((heads >= 2) & (heads < 6)).all()
        )
        self.assertTrue(
            (
                (tails[relations == 2] >= 6)
                & (tails[relations == 2] < 9)
            ).all()
        )
        self.assertTrue(
            (
                (tails[relations != 2] >= 2)
                & (tails[relations != 2] < 6)
            ).all()
        )

    def test_training_graph_does_not_include_heldout_triples(self):
        self.assertNotIn(
            [0, 3],
            self.graph[
                ("drug", "0", "drug")
            ].edge_index.T.tolist(),
        )
        self.assertEqual(
            sum(
                edge.shape[1]
                for edge in self.graph.edge_index_dict.values()
            ),
            4,
        )

    def test_mrr_tie_handling_matches_official_ogb_evaluator(self):
        from ogb.linkproppred import Evaluator

        positive = torch.tensor([2.0, 1.0])
        negatives = torch.tensor(
            [[3.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
        )
        expected = Evaluator(name="ogbl-biokg").eval(
            {
                "y_pred_pos": positive,
                "y_pred_neg": negatives,
            }
        )["mrr_list"]

        torch.testing.assert_close(
            mean_reciprocal_rank(positive, negatives),
            expected,
        )

    def test_evaluation_uses_correct_offsets_for_both_candidate_sides(self):
        split = {
            "head": np.array([0]),
            "tail": np.array([1]),
            "head_type": ["drug"],
            "tail_type": ["protein"],
            "relation": np.array([2]),
            "head_neg": np.array([[2, 3]]),
            "tail_neg": np.array([[0, 2]]),
        }

        class Recorder:
            def __init__(self):
                self.calls = []

            def eval(self):
                return self

            def score(self, embeddings, heads, relations, tails):
                self.calls.append(
                    (
                        heads.tolist(),
                        relations.tolist(),
                        tails.tolist(),
                    )
                )
                return heads.float() + tails.float()

        model = Recorder()
        report = evaluate_split(
            model,
            torch.zeros(9, 4),
            split,
            self.contract,
            "valid",
        )

        self.assertEqual(
            model.calls,
            [
                ([2], [2], [7]),
                ([[4, 5]], [[2]], [[7]]),
                ([[2]], [[2]], [[6, 8]]),
            ],
        )
        self.assertTrue(report["complete"])

    def test_legacy_and_incomplete_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"

            torch.save(
                {"model_state_dict": self.model.state_dict()},
                path,
            )
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_checkpoint(path, self.contract, "cpu")

            checkpoint = {
                "format": FORMAT,
                "graph_contract": self.contract,
                "model_config": self.config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_steps": 1,
                "evaluation": {},
            }
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "MRR"):
                load_checkpoint(path, self.contract, "cpu")

            # Test-only metadata, never a real BioKG evaluation result.
            checkpoint["evaluation"] = {
                name: {
                    "split": name,
                    "complete": True,
                    "evaluated_triples": 1,
                    "total_triples": 1,
                    "mrr": 0.5,
                }
                for name in ("valid", "test")
            }
            torch.save(checkpoint, path)
            restored, _ = load_checkpoint(
                path, self.contract, "cpu"
            )
            torch.testing.assert_close(
                restored.relations.weight,
                self.model.relations.weight,
            )

            different = copy.deepcopy(self.contract)
            different["offsets"]["drug"] = 0
            with self.assertRaisesRegex(
                ValueError, "incompatible"
            ):
                load_checkpoint(path, different, "cpu")

            checkpoint["model_state_dict"].pop(
                "relations.weight"
            )
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path, self.contract, "cpu")

    def test_partial_evaluation_is_explicitly_noncomplete(self):
        split = {
            **self.train,
            "head_neg": np.zeros((4, 2), dtype=int),
            "tail_neg": np.zeros((4, 2), dtype=int),
        }
        edges, edge_types = flatten_graph(
            self.graph, self.contract, "cpu"
        )
        embeddings = self.model(
            self.graph.x_dict, edges, edge_types
        )

        report = evaluate_split(
            self.model,
            embeddings,
            split,
            self.contract,
            "valid",
            max_triples=1,
        )

        self.assertEqual(
            (
                report["evaluated_triples"],
                report["total_triples"],
                report["complete"],
            ),
            (1, 4, False),
        )


class RouteTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        import src.utils

        _, graph, _, model, _ = fixture()

        spec = importlib.util.spec_from_file_location(
            "runtime_under_test",
            Path("src/utils/model.py"),
        )
        self.runtime = importlib.util.module_from_spec(spec)

        with (
            patch(
                "src.data.load_biokg.load_biokg_as_hetero",
                return_value=graph,
            ),
            patch(
                "src.utils.mapping_loader.load_entity_mappings",
                return_value={},
            ),
            patch.dict(
                "os.environ",
                {"ENABLE_PREDICTIONS": "false"},
            ),
        ):
            spec.loader.exec_module(self.runtime)

        modules = patch.dict(
            sys.modules,
            {"src.utils.model": self.runtime},
        )
        modules.start()
        self.addCleanup(modules.stop)

        attribute = patch.object(
            src.utils,
            "model",
            self.runtime,
            create=True,
        )
        attribute.start()
        self.addCleanup(attribute.stop)

        for name in list(sys.modules):
            if name == "run" or name.startswith(
                "backend.app.routes"
            ):
                del sys.modules[name]

        with (
            patch(
                "src.utils.data_loader.ensure_required_files"
            ),
            patch.dict(
                "os.environ",
                {"FLASK_ENV": "production", "FLASK_SECRET_KEY": "synthetic-test-session-key"},
            ),
        ):
            self.app = importlib.import_module("run").app

        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

        predictor = importlib.import_module(
            "backend.app.routes.predict"
        )
        root = patch.object(
            predictor,
            "PROJECT_ROOT",
            Path(self.directory.name),
        )
        root.start()
        self.addCleanup(root.stop)
        self.model = model

    def enable_fixture_model(self):
        edges, edge_types = flatten_graph(
            self.runtime.graph,
            self.runtime.contract,
            "cpu",
        )
        self.runtime.model = self.model.eval()
        self.runtime.embeddings = self.model(
            self.runtime.graph.x_dict,
            edges,
            edge_types,
        ).detach()
        self.runtime.prediction_error = None

    def test_home_graph_and_health_work_without_predictions(self):
        home = self.client.get("/")

        self.assertEqual(home.status_code, 200)
        self.assertIn(b"Predictions unavailable", home.data)
        self.assertIn(
            b"View recorded relationships",
            home.data,
        )
        self.assertEqual(
            self.client.get("/healthz").json["predictions"],
            "unavailable",
        )
        self.assertEqual(
            self.client.get(
                "/graph?drug1_id=0&drug2_id=1"
            ).status_code,
            200,
        )

        result = self.client.get(
            "/graph_data?drug1_id=0&drug2_id=1"
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            result.json["view_type"],
            "source_neighborhood",
        )
        self.assertTrue(
            all(
                "weight" not in link
                for link in result.json["links"]
            )
        )

    def test_unavailable_predictions_never_return_scores(self):
        response = self.client.post(
            "/predict",
            json={
                "drug1_id": 0,
                "drug2_id": 1,
                "relation_type": "0",
            },
        )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(
            "uncalibrated_model_score",
            response.json,
        )
        self.assertEqual(
            self.client.post("/batch").status_code,
            503,
        )
        self.assertFalse(
            (
                Path(self.directory.name)
                / "predictions_v2_log.csv"
            ).exists()
        )

    def test_invalid_input_has_400_status(self):
        base = {
            "drug1_id": 0,
            "drug2_id": 1,
            "relation_type": "0",
        }
        invalid_requests = (
            [],
            {},
            {**base, "drug1_id": True},
            {**base, "drug1_id": 0.5},
            {**base, "drug1_id": -1},
            {**base, "drug2_id": 4},
            {**base, "drug2_id": 0},
            {**base, "relation_type": "2"},
        )

        for data in invalid_requests:
            with self.subTest(data=data):
                self.assertEqual(
                    self.client.post(
                        "/predict", json=data
                    ).status_code,
                    400,
                )

    def test_every_legacy_explanation_endpoint_is_unavailable(self):
        for path in (
            "/explain",
            "/explain_graph",
            "/explain_view",
        ):
            response = self.client.get(
                path
                + "?drug1_id=0&drug2_id=1&relation_type=0"
            )
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(
                "shap_values", response.json
            )

        from src.explain.explain_ddi import run_explainer
        from src.explain.explain_shap import run_shap_explainer
        from src.utils.sage_loader import load_sage_and_homograph

        for helper in (
            run_shap_explainer,
            run_explainer,
            load_sage_and_homograph,
        ):
            with self.assertRaises(NotImplementedError):
                helper()

    def test_single_and_batch_use_same_relation_aware_predictor(self):
        import csv

        self.enable_fixture_model()
        data = {
            "drug1_id": 0,
            "drug2_id": 1,
            "relation_type": "0",
        }

        single = self.client.post("/predict", json=data)
        self.assertEqual(single.status_code, 200)
        self.assertNotIn(
            "predicted_interaction_probability",
            single.json,
        )

        log = (
            Path(self.directory.name)
            / "predictions_v2_log.csv"
        )
        self.assertIn(
            "uncalibrated_model_score",
            log.read_text().splitlines()[0],
        )

        upload = (
            b"drug1_id,drug2_id,relation_type\n"
            b"0,1,0\n"
            b"-1,1,0\n"
        )
        result = self.client.post(
            "/batch",
            data={
                "csv_file": (
                    io.BytesIO(upload),
                    "test.csv",
                )
            },
        )

        self.assertEqual(result.status_code, 200)
        rows = list(
            csv.DictReader(
                io.StringIO(result.data.decode())
            )
        )
        self.assertEqual(
            float(rows[0]["uncalibrated_model_score"]),
            single.json["uncalibrated_model_score"],
        )
        self.assertTrue(rows[1]["error"])
        self.assertEqual(
            rows[1]["uncalibrated_model_score"],
            "",
        )
        self.assertEqual(
            self.client.post("/batch").status_code,
            400,
        )

        malformed = self.client.post(
            "/batch",
            data={
                "csv_file": (
                    io.BytesIO(b"wrong\n1"),
                    "bad.csv",
                )
            },
        )
        self.assertEqual(malformed.status_code, 400)

    def test_browser_form_renders_result_without_clinical_thresholds(self):
        self.enable_fixture_model()
        response = self.client.post(
            "/predict",
            data={
                "drug1_id": "0",
                "drug2_id": "1",
                "relation_type": "0",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            b"Uncalibrated model score",
            response.data,
        )
        self.assertNotIn(
            b"High likelihood",
            response.data,
        )


if __name__ == "__main__":
    unittest.main()