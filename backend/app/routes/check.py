"""Check a selected medication list using recorded data and optional model scores."""
from itertools import combinations

from flask import request, jsonify

from src.utils import model as runtime
from src.utils.drug_catalog import get_catalog, recorded_interactions


def register_check(app):
    @app.post("/check")
    def check():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("drugs"), list):
            return jsonify(error="Choose a list of 2 to 10 drugs."), 400
        values = data["drugs"]
        if not 2 <= len(values) <= 10:
            return jsonify(error="Choose 2 to 10 drugs (up to 45 unique pairs)."), 400
        catalog = get_catalog(runtime.entity_mappings.get("drug", {}))
        try:
            drugs = [catalog.resolve(value) for value in values]
            identities = [row["pubchem_cid"] or row["drugbank_id"] or row["key"] for row in drugs]
            if len(set(identities)) != len(identities):
                raise ValueError("Choose distinct drugs; duplicate identifiers for the same compound were found.")
            relation = data.get("relation_type", "")
            if not isinstance(relation, str):
                raise ValueError("Choose a supported model relation or leave it blank.")
            if relation and ["drug", relation, "drug"] not in runtime.contract["relations"]:
                raise ValueError("Unsupported drug-drug relation.")
            matches, export_available = recorded_interactions(drugs)
        except (ValueError, TypeError) as error:
            return jsonify(error=str(error)), 400
        results = []
        for first, second in combinations(drugs, 2):
            evidence = []
            for source, target, description in matches:
                if {source, target} == {first["drugbank_id"], second["drugbank_id"]}:
                    if description not in evidence:
                        evidence.append(description)
            covered = export_available and bool(first["drugbank_id"] and second["drugbank_id"])
            row = {"drug1": first, "drug2": second,
                   "recorded_interactions": evidence,
                   "recorded_status": "recorded" if evidence else ("not_found" if covered else "unavailable"),
                   "model_score": None, "model_status": "not_requested", "relation_type": relation}
            if relation:
                a, b = first["biokg_index"], second["biokg_index"]
                if a is None or b is None:
                    row["model_status"] = "Outside the BioKG model graph."
                elif a == b:
                    row["model_status"] = "Both entries map to the same model entity."
                else:
                    try:
                        row["model_score"] = runtime.predict_score(a, relation, b)
                        row["model_status"] = "ready"
                    except runtime.PredictionUnavailable as error:
                        row["model_status"] = str(error)
            if first["biokg_index"] is not None and second["biokg_index"] is not None:
                row["graph_url"] = f"/graph?drug1_id={first['biokg_index']}&drug2_id={second['biokg_index']}"
            results.append(row)
        return jsonify(drugs=drugs, pairs=results,
                       source="Supplied DrugBank export; interaction partners matched by unambiguous exact names.",
                       note="An entry not found in this export does not establish that a drug combination is safe.")
