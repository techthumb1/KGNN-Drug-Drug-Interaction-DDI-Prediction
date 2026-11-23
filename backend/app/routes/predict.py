import csv
from datetime import datetime, timezone

from flask import (
    request,
    jsonify,
    render_template,
    session,
    redirect,
    url_for,
)

from src.utils import model as runtime
from src.utils.paths import PROJECT_ROOT
from src.utils.drug_catalog import get_catalog
from src.utils.entity_metadata import relation_label
from src.utils.drugbank_lookup import lookup_drug_by_cid
from src.utils.drugbank_full_lookup import lookup_full_drug_info


def parse_index(value):
    if type(value) is int:
        return value

    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdecimal()
    ):
        return int(value)

    raise ValueError(
        "IDs must be nonnegative integer local drug indices"
    )


def resolve_pair(data):
    if "drug1" in data or "drug2" in data:
        catalog = get_catalog(runtime.entity_mappings.get("drug", {}))
        chosen = [catalog.resolve(data.get(key)) for key in ("drug1", "drug2")]
        if any(row["biokg_index"] is None for row in chosen):
            raise ValueError("This drug is outside the model graph; use the recorded DrugBank check instead.")
        first, second = (row["biokg_index"] for row in chosen)
    else:
        first = parse_index(data.get("drug1_id"))
        second = parse_index(data.get("drug2_id"))
    return first, second


def score_request(data):
    if not isinstance(data, dict) and not hasattr(data, "get"):
        raise ValueError(
            "Provide drug1_id, drug2_id and relation_type"
        )

    first, second = resolve_pair(data)
    relation = data.get("relation_type")

    if not isinstance(relation, str):
        raise ValueError(
            "Choose a supported drug-drug relation"
        )

    score = runtime.predict_score(
        first,
        relation,
        second,
    )
    result = {
        "drug1_id": first, "drug2_id": second,
        "relation_type": relation, "uncalibrated_model_score": score,
    }
    return enrich_result(result)


def enrich_result(result):
    result = dict(result)
    mappings = runtime.entity_mappings.get("drug", {})
    catalog = get_catalog(mappings)
    for number in (1, 2):
        index = result[f"drug{number}_id"]
        drug = catalog.by_index.get(index)
        cid = drug["pubchem_cid"] if drug else None
        result[f"drug{number}_name"] = drug["name"] if drug else str(mappings.get(index, f"drug:{index}"))
        result[f"drug{number}_identifier"] = str(mappings.get(index, f"drug:{index}"))
        result[f"drug{number}_info"] = lookup_drug_by_cid(cid) if cid else None
        result[f"drug{number}_full_info"] = lookup_full_drug_info(cid) if cid else None

    return result


def register_predict(app):
    @app.get("/relation_rank")
    def relation_rank():
        as_json = request.args.get("format") == "json"
        try:
            first, second = resolve_pair(request.args)
            items = runtime.rank_drug_relations(first, second)
        except (runtime.PredictionUnavailable, ValueError, TypeError, KeyError) as error:
            status = 503 if isinstance(error, runtime.PredictionUnavailable) else 400
            if as_json:
                return jsonify(error=str(error)), status
            return render_template("relation_rank.html", error=str(error)), status

        catalog = get_catalog(runtime.entity_mappings.get("drug", {}))
        names = [catalog.by_index.get(index, {}).get("name", f"BioKG drug {index}")
                 for index in (first, second)]
        for item in items:
            item["relation_name"] = relation_label(item["relation_type"]).removeprefix("drug-drug ")
        result = {
            "drug1_id": first, "drug2_id": second,
            "drug1_name": names[0], "drug2_name": names[1],
            "predictor": "R-GCN/DistMult",
            "relations": items,
        }
        if as_json:
            return jsonify(result)
        return render_template("relation_rank.html", result=result)

    @app.post("/predict")
    def predict():
        try:
            result = score_request(
                request.get_json(silent=True)
                if request.is_json
                else request.form
            )
        except runtime.PredictionUnavailable as error:
            if request.is_json:
                return jsonify(
                    error=str(error),
                    status="prediction_unavailable",
                ), 503

            return render_template(
                "index.html",
                error=str(error),
                prediction_error=runtime.prediction_error,
            ), 503

        except (ValueError, TypeError, KeyError) as error:
            if request.is_json:
                return jsonify(error=str(error)), 400

            return render_template(
                "index.html",
                error=str(error),
                prediction_error=runtime.prediction_error,
            ), 400

        # Keep legacy prediction logs separate from corrected scores.
        path = PROJECT_ROOT / "predictions_v2_log.csv"
        exists = path.exists()

        with path.open("a", newline="") as output:
            writer = csv.writer(output)

            if not exists:
                writer.writerow(
                    [
                        "timestamp",
                        "drug1_id",
                        "drug2_id",
                        "relation_type",
                        "uncalibrated_model_score",
                    ]
                )

            writer.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    result["drug1_id"],
                    result["drug2_id"],
                    result["relation_type"],
                    result["uncalibrated_model_score"],
                ]
            )

        if request.is_json:
            return jsonify(result)

        # Store only references and the score in the signed cookie. Full descriptions
        # and target lists are reloaded server-side when rendering the result.
        session["prediction_result"] = {
            key: result[key] for key in (
                "drug1_id", "drug2_id", "relation_type", "uncalibrated_model_score"
            )
        }
        return redirect(url_for("index"))
