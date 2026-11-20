"""Prepare source data and configured artifacts; never train or evaluate."""
import json
import os

from src.utils.paths import PROJECT_ROOT, MODEL_CHECKPOINT


def main():
    if os.getenv("FLASK_ENV") != "production":
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env", override=False)

    from src.utils.data_loader import ensure_required_files
    from src.data.load_biokg import load_biokg_as_hetero
    from src.utils.graph_contract import graph_contract
    from src.utils.mapping_loader import load_entity_mappings
    from src.utils.paths import BIOKG_MAPPING_DIR

    ensure_required_files()
    graph = load_biokg_as_hetero()
    contract = graph_contract(graph)
    mappings = load_entity_mappings(BIOKG_MAPPING_DIR)
    if not mappings.get("drug"):
        raise RuntimeError("Official BioKG drug mappings are missing")
    if os.getenv("ENABLE_PREDICTIONS", "false").lower() == "true":
        from src.utils.checkpoint import load_checkpoint

        load_checkpoint(MODEL_CHECKPOINT, contract, "cpu")
    from src.utils import drugbank_lookup, drugbank_full_lookup

    print(json.dumps({
        "graph": "ready",
        "train_graph_sha256": contract["train_graph_sha256"],
        "drugbank_links": len(drugbank_lookup.cid_to_drugbank),
        "drugbank_details": len(drugbank_full_lookup.cid_to_entry),
        "predictions_enabled": os.getenv("ENABLE_PREDICTIONS", "false").lower() == "true",
    }))


if __name__ == "__main__":
    main()
