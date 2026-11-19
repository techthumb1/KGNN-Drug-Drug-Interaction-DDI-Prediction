from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
BIOKG_MAPPING_DIR = DATASET_ROOT / "ogbl_biokg" / "mapping"
MODEL_CHECKPOINT = PROJECT_ROOT / "models" / "rgcn_v2.pt"
DRUGBANK_DATA_DIR = PROJECT_ROOT / "backend" / "drugbank_data"