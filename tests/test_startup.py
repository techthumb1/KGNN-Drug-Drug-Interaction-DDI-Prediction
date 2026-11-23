import importlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from http.server import (
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from unittest.mock import patch

from src.utils import data_loader
from src.utils.paths import (
    BIOKG_MAPPING_DIR,
    DATASET_ROOT,
    PROJECT_ROOT,
)


class DownloadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_error(404)
            return

        body = {
            "/artifact": b"startup-transfer-fixture",
            "/empty": b"",
            "/html": b"<!doctype html><html>Access denied</html>",
        }[self.path]

        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            DownloadHandler,
        )
        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()
        cls.url = (
            f"http://127.0.0.1:{cls.server.server_port}"
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = (
            Path(self.directory.name)
            / "models"
            / "checkpoint.pt"
        )

    def test_successful_download_is_published(self):
        data_loader.download_file(
            self.url + "/artifact", self.target
        )
        self.assertEqual(
            self.target.read_bytes(),
            b"startup-transfer-fixture",
        )
        self.assertEqual(
            list(self.target.parent.glob("*.part")), []
        )

    def test_http_failure_does_not_leave_cached_checkpoint(self):
        with self.assertRaises(RuntimeError):
            data_loader.download_file(
                self.url + "/missing", self.target
            )
        self.assertFalse(self.target.exists())
        self.assertEqual(
            list(self.target.parent.glob("*.part")), []
        )

    def test_empty_and_html_downloads_are_rejected(self):
        for endpoint in ("/empty", "/html"):
            with (
                self.subTest(endpoint=endpoint),
                self.assertRaises(RuntimeError),
            ):
                data_loader.download_file(
                    self.url + endpoint, self.target
                )
            self.assertFalse(self.target.exists())

    def test_valid_cache_does_not_require_network(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"existing")
        data_loader.download_file(
            self.url + "/missing", self.target
        )
        self.assertEqual(
            self.target.read_bytes(), b"existing"
        )

    def test_zero_byte_cache_is_replaced(self):
        self.target.parent.mkdir()
        self.target.touch()
        data_loader.download_file(
            self.url + "/artifact", self.target
        )
        self.assertGreater(
            self.target.stat().st_size, 0
        )

    def test_missing_checkpoint_has_actionable_error(self):
        with (
            patch.dict(
                os.environ,
                {"ENABLE_PREDICTIONS": "true"},
                clear=True,
            ),
            patch.object(
                data_loader,
                "MODEL_CHECKPOINT",
                self.target,
            ),
            self.assertRaisesRegex(
                RuntimeError, "MODEL_URL"
            ),
        ):
            data_loader.ensure_required_files()

    def test_optional_enrichment_is_not_required(self):
        optional = self.target.parent / "optional"
        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_PREDICTIONS": "true",
                    "MODEL_URL": self.url + "/artifact",
                },
                clear=True,
            ),
            patch.object(
                data_loader,
                "MODEL_CHECKPOINT",
                self.target,
            ),
            patch.object(
                data_loader,
                "DRUGBANK_DATA_DIR",
                optional,
            ),
        ):
            data_loader.ensure_required_files()

        self.assertTrue(self.target.is_file())
        self.assertFalse(optional.exists())

    def test_configured_enrichment_is_downloaded(self):
        enrichment = self.target.parent / "optional"

        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_PREDICTIONS": "true",
                    "MODEL_URL": self.url + "/artifact",
                    "DRUGBANK_XML_URL": self.url + "/artifact",
                },
                clear=True,
            ),
            patch.object(
                data_loader,
                "MODEL_CHECKPOINT",
                self.target,
            ),
            patch.object(
                data_loader,
                "DRUGBANK_DATA_DIR",
                enrichment,
            ),
        ):
            data_loader.ensure_required_files()

        self.assertTrue(
            (
                enrichment
                / "drugbank_all_drug_links.xml"
            ).is_file()
        )

    def test_source_only_mode_does_not_fetch_legacy_model_or_enrichment(self):
        with (
            patch.dict(
                os.environ,
                {
                    "MODEL_URL": self.url + "/missing",
                    "DRUGBANK_XML_URL": self.url + "/missing",
                },
                clear=True,
            ),
            patch.object(
                data_loader, "download_file"
            ) as download,
        ):
            data_loader.ensure_required_files()

        download.assert_not_called()

    def test_drive_detection_checks_hostname(self):
        self.assertTrue(
            data_loader.is_google_drive_url(
                "https://drive.google.com/uc?id=123"
            )
        )
        self.assertFalse(
            data_loader.is_google_drive_url(
                "https://example.com/drive.google.com"
            )
        )


class BootTests(unittest.TestCase):
    def import_app(self, prepare):
        routes = types.ModuleType(
            "backend.app.routes"
        )

        def register(app):
            prepare.assert_called_once_with()
            app.add_url_rule(
                "/",
                "index",
                lambda: "model routes registered",
            )

        routes.init_routes = register
        sys.modules.pop("run", None)
        self.addCleanup(
            sys.modules.pop, "run", None
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "backend.app.routes": routes,
                    "src.utils.model": types.SimpleNamespace(
                        prediction_error="unavailable"
                    ),
                },
            ),
            patch.dict(
                os.environ,
                {"FLASK_ENV": "production", "FLASK_SECRET_KEY": "synthetic-test-session-key"},
            ),
        ):
            return importlib.import_module("run").app

    def test_boot_prepares_artifacts_before_routes_and_health_check(self):
        with patch.object(
            data_loader, "ensure_required_files"
        ) as prepare:
            app = self.import_app(prepare)

        client = app.test_client()
        self.assertEqual(
            client.get("/").status_code, 200
        )
        response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json,
            {
                "status": "ok",
                "graph": "ready",
                "predictions": "unavailable",
                "explanations": "unavailable",
                "graph_explanations": "unavailable",
            },
        )

    def test_boot_fails_when_required_artifacts_cannot_be_prepared(self):
        with patch.object(
            data_loader,
            "ensure_required_files",
            side_effect=RuntimeError(
                "missing checkpoint"
            ),
        ) as prepare:
            with self.assertRaisesRegex(
                RuntimeError, "missing checkpoint"
            ):
                self.import_app(prepare)

    def test_mapping_path_uses_the_same_ogb_root(self):
        self.assertTrue(DATASET_ROOT.is_absolute())
        self.assertEqual(
            DATASET_ROOT.parent, PROJECT_ROOT
        )
        self.assertEqual(
            BIOKG_MAPPING_DIR,
            DATASET_ROOT / "ogbl_biokg" / "mapping",
        )
