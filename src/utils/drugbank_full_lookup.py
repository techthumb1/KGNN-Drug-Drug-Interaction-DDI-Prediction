"""Load DrugBank display fields without retaining large interaction arrays."""
import json
import logging
from pathlib import Path

from src.utils.paths import DRUGBANK_DATA_DIR

DRUGBANK_JSON = DRUGBANK_DATA_DIR / "drugbank_slim.json"
DISPLAY_FIELDS = ("pubchem", "drugbank_id", "name", "cas", "type", "description", "targets")
cid_to_entry = {}
drugbank_entries = {}
metadata_error = None


def iter_records(path):
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, eof = "", False

        def trim():
            nonlocal buffer, eof
            buffer = buffer.lstrip()
            while not buffer and not eof:
                chunk = stream.read(1024 * 1024)
                eof = not chunk
                buffer = chunk.lstrip()

        trim()
        if not buffer.startswith("["):
            raise ValueError("DrugBank JSON must be an array of records")
        buffer = buffer[1:]
        first = True
        while True:
            trim()
            if buffer.startswith("]"):
                if buffer[1:].strip() or stream.read().strip():
                    raise ValueError("Unexpected content after DrugBank array")
                return
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Expected a comma between DrugBank records")
                buffer = buffer[1:]
                trim()
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    chunk = stream.read(1024 * 1024)
                    eof = not chunk
                    buffer += chunk
            if not isinstance(record, dict):
                raise ValueError("Each DrugBank record must be an object")
            buffer = buffer[end:]
            first = False
            yield record


def _load_full_drugbank():
    global cid_to_entry, drugbank_entries
    cid_to_entry = {}
    drugbank_entries = {}
    if not Path(DRUGBANK_JSON).is_file():
        print(f"[DrugBank Full] File not found: {DRUGBANK_JSON}")
        return 0
    loaded = {}
    for entry in iter_records(DRUGBANK_JSON):
        display = {key: entry[key] for key in DISPLAY_FIELDS if key in entry}
        if entry.get("drugbank_id") and entry.get("name"):
            drugbank_entries[str(entry["drugbank_id"]).upper()] = display
        value = entry.get("pubchem")
        if value is None or value == "":
            continue
        cid = str(value).strip()
        if not cid.isascii() or not cid.isdecimal() or int(cid) < 1:
            raise ValueError("DrugBank PubChem CID must be a positive integer")
        cid = str(int(cid))
        loaded[cid] = display
        loaded[cid]["pubchem"] = cid
    if not loaded:
        raise ValueError("DrugBank JSON has no usable PubChem CID mappings")
    cid_to_entry = loaded
    print(f"[DrugBank Full] Loaded {len(loaded)} entries.")
    return len(loaded)


def load_full_drugbank():
    global cid_to_entry, drugbank_entries, metadata_error
    metadata_error = None
    try:
        count = _load_full_drugbank()
        if not drugbank_entries:
            metadata_error = "Recorded DrugBank lookup is unavailable on this server. A valid metadata export is required."
        return count
    except (OSError, ValueError, TypeError, KeyError) as error:
        cid_to_entry, drugbank_entries = {}, {}
        metadata_error = "Recorded DrugBank lookup is unavailable because the metadata export could not be loaded."
        logging.getLogger(__name__).warning("DrugBank metadata unavailable: %s", error)
        return 0


def lookup_full_drug_info(cid):
    return cid_to_entry.get(str(cid))


load_full_drugbank()
