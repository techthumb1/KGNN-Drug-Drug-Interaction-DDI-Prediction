import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from flask import Flask

from src.utils.neighborhood import source_neighborhood


class NeighborhoodTests(unittest.TestCase):
    def setUp(self):
        class Fixture(dict):
            pass

        self.graph = Fixture(
            drug=SimpleNamespace(num_nodes=4)
        )
        self.graph.edge_index_dict = {
            ("drug", "targets", "protein"): np.array(
                [[0, 2, 1], [0, 0, 2]]
            ),
            ("protein", "bound_by", "drug"): np.array(
                [[3], [1]]
            ),
            ("protein", "unrelated", "protein"): np.array(
                [[0], [1]]
            ),
        }

    def test_retains_only_recorded_incident_edges_and_types(self):
        data = source_neighborhood(
            self.graph,
            {"protein": {0: "P0"}},
            0,
            1,
        )
        self.assertEqual(len(data["links"]), 3)
        self.assertFalse(data["truncated"])

        nodes = {
            node["id"]: node
            for node in data["nodes"]
        }
        self.assertEqual(
            nodes["protein:0"]["label"], "P0"
        )
        self.assertIn("drug:0", nodes)
        self.assertNotIn("drug:2", nodes)
        self.assertEqual(
            data["links"][2]["relation"],
            "bound_by",
        )
        self.assertTrue(
            all(
                "weight" not in edge
                for edge in data["links"]
            )
        )

    def test_bound_and_truncation(self):
        data = source_neighborhood(
            self.graph, {}, 0, 1, limit=1
        )
        self.assertEqual(len(data["links"]), 1)
        self.assertTrue(data["truncated"])

    def test_invalid_edge_limits_are_rejected(self):
        for limit in (0, -1, True, 1.5, 101):
            with (
                self.subTest(limit=limit),
                self.assertRaises(ValueError),
            ):
                source_neighborhood(
                    self.graph,
                    {},
                    0,
                    1,
                    limit=limit,
                )

    def test_exact_limit_is_not_truncated(self):
        self.assertFalse(
            source_neighborhood(
                self.graph, {}, 0, 1, limit=3
            )["truncated"]
        )

    def test_empty_neighborhood_preserves_selected_drugs(self):
        data = source_neighborhood(
            self.graph, {}, 2, 3
        )
        self.assertEqual(len(data["links"]), 1)
        self.assertIn(
            "drug:3",
            {node["id"] for node in data["nodes"]},
        )

    def test_rejects_invalid_or_self_pairs(self):
        for first, second in (
            (0, 0),
            (-1, 0),
            (0, 4),
            (True, 2),
            ("0", 2),
        ):
            with (
                self.subTest(pair=(first, second)),
                self.assertRaises(ValueError),
            ):
                source_neighborhood(
                    self.graph,
                    {},
                    first,
                    second,
                )

    def test_no_fabricated_pgexplainer(self):
        from src.explain.explain_pg import run_pgexplainer, ExplanationUnavailable

        with self.assertRaises(ExplanationUnavailable):
            run_pgexplainer(None, None, 0, 1)

    def test_no_edges_returns_selected_nodes(self):
        self.graph.edge_index_dict = {}
        data = source_neighborhood(
            self.graph, {}, 0, 1
        )
        self.assertEqual(data["links"], [])
        self.assertEqual(
            {
                node["id"]
                for node in data["nodes"]
            },
            {"drug:0", "drug:1"},
        )
        self.assertFalse(data["truncated"])

    def test_http_boundary_rejects_malformed_indices(self):
        fixture = SimpleNamespace(
            graph=self.graph,
            entity_mappings={},
        )
        spec = importlib.util.spec_from_file_location(
            "graph_route_under_test",
            Path("backend/app/routes/graph.py"),
        )
        route = importlib.util.module_from_spec(spec)

        with patch.dict(
            sys.modules,
            {"src.utils.model": fixture},
        ):
            spec.loader.exec_module(route)

        app = Flask(__name__)
        route.register_graph_routes(app)
        client = app.test_client()

        for path in (
            "/graph_data",
            "/graph_data?drug1_id=x&drug2_id=1",
            "/graph_data?drug1_id=-1&drug2_id=1",
            "/graph_data?drug1_id=1&drug2_id=1",
        ):
            self.assertEqual(
                client.get(path).status_code,
                400,
            )

        result = client.get(
            "/graph_data?drug1_id=0&drug2_id=1"
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(
            result.json["view_type"],
            "source_neighborhood",
        )
