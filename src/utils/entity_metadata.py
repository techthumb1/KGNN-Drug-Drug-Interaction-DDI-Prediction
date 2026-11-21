"""Exact identifier annotations, separate from graph evidence and model attribution."""
from collections import defaultdict
from functools import lru_cache
import csv
import gzip
import json
import logging
import re
import sqlite3
from threading import Lock
import time
from urllib.parse import quote

import requests

from src.utils.paths import BIOKG_MAPPING_DIR, DRUGBANK_DATA_DIR
from src.utils import drugbank_full_lookup

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_network_lock = Lock()
_last_ncbi_call = 0.0


@lru_cache(maxsize=1)
def relation_names():
    path = BIOKG_MAPPING_DIR / "relidx2relname.csv.gz"
    if not path.is_file():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        return {row[0]: row[1] for row in rows if len(row) >= 2}


def relation_label(value):
    return relation_names().get(str(value), str(value)).replace("_", " ")


def base_entity(kind, index, mappings, catalog):
    raw = str(mappings.get(kind, {}).get(index, f"{kind}:{index}"))
    row = {"id": f"{kind}:{index}", "local_index": index, "type": kind,
           "identifier": raw, "label": raw, "description": "", "annotation_status": "unresolved"}
    if kind == "drug":
        drug = catalog.by_index.get(index, {})
        row.update(label=drug.get("name", raw), drugbank_id=drug.get("drugbank_id"))
        if row["drugbank_id"]:
            info = drugbank_full_lookup.drugbank_entries.get(row["drugbank_id"], {})
            row.update(description=info.get("description") or "", targets=info.get("targets", []),
                       annotation_source="Supplied DrugBank export", annotation_status="resolved",
                       source_url="https://go.drugbank.com/drugs/" + row["drugbank_id"])
    if kind in ("disease", "sideeffect") and re.fullmatch(r"C\d{7}", raw):
        row.setdefault("source_url", "https://www.ncbi.nlm.nih.gov/medgen/" + raw)
    elif kind == "protein" and raw.isascii() and raw.isdecimal():
        row.setdefault("source_url", "https://www.ncbi.nlm.nih.gov/gene/" + raw)
    elif kind == "function" and re.fullmatch(r"GO:\d{7}", raw):
        row.setdefault("source_url", "https://www.ebi.ac.uk/QuickGO/term/" + raw)
    elif kind == "drug" and re.fullmatch(r"CID\d+", raw):
        row.setdefault("source_url", "https://pubchem.ncbi.nlm.nih.gov/compound/" + str(int(raw[3:])))
    return row


def _get(url, **params):
    global _last_ncbi_call
    # Serialize public service calls and respect NCBI's unauthenticated rate limit.
    with _network_lock:
        if url.startswith(_EUTILS):
            delay = .4 - (time.monotonic() - _last_ncbi_call)
            if delay > 0:
                time.sleep(delay)
            _last_ncbi_call = time.monotonic()
        response = requests.get(url, params=params, timeout=(3, 8), headers={"Accept": "application/json"})
        response.raise_for_status()
        return response.json()


def _annotations(kind, identifiers):
    found = {}
    if kind in ("disease", "sideeffect"):
        # Search CUIs, then require exact concept-ID equality in the response.
        search = _get(_EUTILS + "esearch.fcgi", db="medgen", retmode="json", retmax=200,
                      term=" OR ".join(f"{value}[ConceptID]" for value in identifiers))
        ids = search.get("esearchresult", {}).get("idlist", [])
        if ids:
            result = _get(_EUTILS + "esummary.fcgi", db="medgen", id=",".join(ids), retmode="json").get("result", {})
            for uid in result.get("uids", []):
                item = result[uid]
                cui = item.get("conceptid")
                if cui in identifiers and item.get("title"):
                    definition = item.get("definition", {})
                    found[cui] = {"label": item["title"], "description": definition.get("value", "") if isinstance(definition, dict) else str(definition),
                                  "semantic_type": item.get("semantictype", {}).get("value", ""),
                                  "annotation_source": "NCBI MedGen", "source_url": "https://www.ncbi.nlm.nih.gov/medgen/" + uid}
    elif kind == "protein":
        result = _get(_EUTILS + "esummary.fcgi", db="gene", id=",".join(identifiers), retmode="json").get("result", {})
        for uid in result.get("uids", []):
            item = result[uid]
            if uid in identifiers and item.get("name") and not str(item.get("currentid", "")).strip("0"):
                found[uid] = {"label": item["name"], "full_name": item.get("description", ""),
                              "description": item.get("summary", ""), "organism": item.get("organism", {}).get("scientificname", ""),
                              "annotation_source": "NCBI Gene (Entrez identifier supplied by BioKG)", "source_url": "https://www.ncbi.nlm.nih.gov/gene/" + uid}
    elif kind == "function":
        result = _get("https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + quote(",".join(identifiers), safe=","))
        for item in result.get("results", []):
            if item.get("id") in identifiers and item.get("name"):
                found[item["id"]] = {"label": item["name"], "description": item.get("definition", {}).get("text", ""),
                                     "obsolete": bool(item.get("isObsolete")), "semantic_type": item.get("aspect", ""),
                                     "annotation_source": "Gene Ontology / EMBL-EBI QuickGO", "source_url": "https://www.ebi.ac.uk/QuickGO/term/" + item["id"]}
    elif kind == "drug":
        cids = {str(int(value[3:])): value for value in identifiers}
        result = _get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + ",".join(cids) + "/property/Title/JSON")
        for item in result.get("PropertyTable", {}).get("Properties", []):
            cid = str(item.get("CID"))
            if cid in cids and item.get("Title"):
                found[cids[cid]] = {"label": item["Title"], "description": "PubChem compound title; DrugBank interaction coverage is unchanged.",
                                     "annotation_source": "NCBI PubChem", "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/" + cid}
    return found


def enrich_entities(nodes):
    """Batch and cache exact public annotations. Missing annotations never create graph facts."""
    DRUGBANK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cache = sqlite3.connect(DRUGBANK_DATA_DIR / "entity_metadata.sqlite3", timeout=10)
        cache.execute("CREATE TABLE IF NOT EXISTS annotations (kind TEXT, identifier TEXT, payload TEXT, expires REAL, PRIMARY KEY(kind, identifier))")
    except sqlite3.Error:
        cache = None
    pending = defaultdict(list)
    now = time.time()
    try:
        for node in nodes:
            if node["annotation_status"] == "resolved":
                continue
            value, kind = node["identifier"], node["type"]
            pattern = {"drug": r"CID\d+", "protein": r"\d+", "function": r"GO:\d{7}",
                       "disease": r"C\d{7}", "sideeffect": r"C\d{7}"}.get(kind, r"(?!)")
            if not re.fullmatch(pattern, value):
                continue
            saved = cache.execute("SELECT payload FROM annotations WHERE kind=? AND identifier=? AND expires>?", (kind, value, now)).fetchone() if cache else None
            if saved:
                node.update(json.loads(saved[0]))
            else:
                pending[kind].append(node)
        for kind, members in pending.items():
            try:
                results = _annotations(kind, {n["identifier"] for n in members})
                for node in members:
                    annotation = results.get(node["identifier"], {})
                    annotation["annotation_status"] = "resolved" if annotation else "not_found"
                    node.update(annotation)
                    if cache:
                        cache.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?,?)", (kind, node["identifier"], json.dumps(annotation), now + (2592000 if results.get(node["identifier"], {}).get("label") else 86400)))
                if cache:
                    cache.commit()
            except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as error:
                logging.getLogger(__name__).warning("Entity annotations unavailable for %s: %s", kind, error)
                for node in members:
                    node["annotation_status"] = "unavailable"
    finally:
        if cache:
            cache.close()
    return nodes
