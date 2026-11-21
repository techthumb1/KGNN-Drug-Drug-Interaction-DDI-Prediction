"""Resolve user-facing drug identifiers and read selected recorded interactions."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from src.utils import drugbank_full_lookup as metadata
from src.utils.medication_names import load_products, search_text


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


class DrugCatalog:
    def __init__(self, mappings):
        self.entries = {}
        self.by_index = {}
        self.by_cid = defaultdict(list)
        self.by_name = defaultdict(list)
        indices = {}
        for index, raw in mappings.items():
            match = re.fullmatch(r"CID:?(\d+)", str(raw), re.IGNORECASE)
            if match:
                indices[str(int(match[1]))] = int(index)
        for dbid, record in metadata.drugbank_entries.items():
            cid = record.get("pubchem")
            cid = str(int(cid)) if str(cid).isascii() and str(cid).isdecimal() else None
            row = {"key": dbid, "name": record["name"], "drugbank_id": dbid,
                   "pubchem_cid": cid, "biokg_index": indices.get(cid)}
            self._add(row)
        # Retain index/CID access even when the supplied export has no name.
        for index, raw in mappings.items():
            index = int(index)
            match = re.fullmatch(r"CID:?(\d+)", str(raw), re.IGNORECASE)
            cid = str(int(match[1])) if match else None
            candidates = self.by_cid.get(cid, [])
            if candidates:
                preferred = metadata.cid_to_entry.get(cid, {}).get("drugbank_id")
                self.by_index[index] = self.entries.get(preferred, candidates[0])
            else:
                row = {"key": f"biokg:{index}", "name": str(raw), "drugbank_id": None,
                       "pubchem_cid": cid, "biokg_index": index}
                self._add(row)
                self.by_index[index] = row

        products, self.name_vocabulary_status = load_products()
        self.search_terms = {key: search_text(row["name"]) for key, row in self.entries.items()}
        canonical_names = {name: tuple(rows) for name, rows in self.by_name.items()}
        for rxcui, name, ingredients, terms in products:
            # Only one exact, unambiguous ingredient can inherit an existing entity.
            # Never reduce a combination product to just one of its ingredients.
            candidates = canonical_names.get(normalized(ingredients[0]), []) if len(ingredients) == 1 else []
            match = candidates[0] if len(candidates) == 1 else {}
            row = {"key": f"rxterms:{rxcui}", "name": name, "rxcui": rxcui,
                   "ingredients": ingredients, "drugbank_id": match.get("drugbank_id"),
                   "pubchem_cid": match.get("pubchem_cid"), "biokg_index": match.get("biokg_index"),
                   "coverage_note": ("Ingredient-level lookup: " + match["name"] + ". Dose and formulation are not evaluated.") if match else
                       "Name recognized by NLM RxTerms; interaction lookup and model coverage are unavailable for this product."}
            self.entries[row["key"]] = row
            self.search_terms[row["key"]] = terms
            self.by_name[normalized(name)].append(row)

    def _add(self, row):
        self.entries[row["key"]] = row
        self.by_name[normalized(row["name"])].append(row)
        if row["pubchem_cid"]:
            self.by_cid[row["pubchem_cid"]].append(row)
        if row["biokg_index"] is not None:
            self.by_index.setdefault(row["biokg_index"], row)

    def resolve(self, value):
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("Choose a drug name or a supported identifier.")
        text = str(value).strip()
        if text in self.entries:
            return self.entries[text]
        if text.upper() in self.entries:
            return self.entries[text.upper()]
        if text.lower().startswith("biokg:"):
            text = text.split(":", 1)[1]
        if text.isascii() and text.isdecimal():
            row = self.by_index.get(int(text))
            if row:
                return row
            raise ValueError("Local BioKG drug index is out of range.")
        match = re.fullmatch(r"(?:pubchem:|CID:?)0*(\d+)", text, re.IGNORECASE)
        rows = self.by_cid.get(str(int(match[1])), []) if match else self.by_name.get(normalized(text), [])
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            raise ValueError("This name or CID matches multiple entries; select a specific DrugBank ID.")
        raise ValueError(f"No exact drug match for {text!r}. Choose a result from the search list.")

    def search(self, query, limit=12):
        query = normalized(query)
        if len(query) < 2:
            return []
        cid_match = re.fullmatch(r"(?:cid:?|pubchem:)0*(\d+)", query)
        if cid_match:
            return self.by_cid.get(str(int(cid_match[1])), [])[:limit]
        words = search_text(query).split()
        matches = []
        for row in self.entries.values():
            terms = self.search_terms[row["key"]]
            identifiers = (row["drugbank_id"] or "").casefold(), ("cid" + row["pubchem_cid"]) if row["pubchem_cid"] else ""
            if (words and all(any(token.startswith(word) for token in terms.split()) for word in words)) or any(query in value for value in identifiers):
                matches.append(row)
        matches.sort(key=lambda row: (query not in (normalized(row["name"]),
                                                      (row["drugbank_id"] or "").casefold(),
                                                      "cid" + (row["pubchem_cid"] or "")),
                                     len(row.get("ingredients", [])) > 1,
                                     not normalized(row["name"]).startswith(query),
                                     not search_text(row["name"]).startswith(words[0] if words else query),
                                     bool(row.get("rxcui")), normalized(row["name"]), row["key"]))
        return matches[:limit]


_catalog = None
_catalog_sources = None


def get_catalog(mappings):
    global _catalog, _catalog_sources
    sources = (mappings, metadata.cid_to_entry, metadata.drugbank_entries)
    if _catalog_sources is None or any(a is not b for a, b in zip(sources, _catalog_sources)):
        _catalog = DrugCatalog(mappings)
        _catalog_sources = sources
    return _catalog


def recorded_interactions(drugs):
    path = Path(metadata.DRUGBANK_JSON)
    if not path.is_file():
        return [], False
    chosen = tuple(sorted({(d["drugbank_id"], normalized(metadata.drugbank_entries[d["drugbank_id"]]["name"]))
                           for d in drugs if d["drugbank_id"] in metadata.drugbank_entries}))
    stat = path.stat()
    return list(_recorded_matches(str(path), stat.st_mtime_ns, stat.st_size, chosen)), True


@lru_cache(maxsize=16)
def _recorded_matches(path, modified, size, chosen):
    if not chosen:
        return ()
    selected = dict(chosen)
    targets = defaultdict(list)
    for dbid, name in chosen:
        targets[name].append(dbid)
    # The existing export records interaction partner names, not partner IDs.
    name_counts = defaultdict(int)
    for row in metadata.drugbank_entries.values():
        name_counts[normalized(row["name"])] += 1
    remaining = set(selected)
    found = []
    for record in metadata.iter_records(path):
        dbid = str(record.get("drugbank_id", "")).upper()
        if dbid not in remaining:
            continue
        for interaction in record.get("interactions", []):
            name = normalized(interaction.get("name", ""))
            if name_counts[name] != 1:
                continue  # Do not assign ambiguous exported names to a specific drug.
            for target in targets.get(name, []):
                if target != dbid and interaction.get("description"):
                    item = (dbid, target, str(interaction["description"]))
                    if item not in found:
                        found.append(item)
        remaining.remove(dbid)
        if not remaining:
            break
    return tuple(found)
