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


def score_request(data):
    if not isinstance(data, dict) and not hasattr(data, "get"):
        raise ValueError(
            "Provide drug1_id, drug2_id and relation_type"
        )

    first = parse_index(data.get("drug1_id"))
    second = parse_index(data.get("drug2_id"))
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
    mappings = runtime.entity_mappings.get("drug", {})

    result = {
        "drug1_id": first,
        "drug2_id": second,
        "drug1_name": str(
            mappings.get(first, f"drug:{first}")
        ),
        "drug2_name": str(
            mappings.get(second, f"drug:{second}")
        ),
        "relation_type": relation,
        "uncalibrated_model_score": score,
    }

    for number, index in ((1, first), (2, second)):
        raw = str(mappings.get(index, ""))
        cid = (
            raw.removeprefix("CID").lstrip(":0")
            if raw.startswith("CID")
            else ""
        )

        result[f"drug{number}_info"] = (
            lookup_drug_by_cid(cid)
            if cid.isdecimal()
            else None
        )
        result[f"drug{number}_full_info"] = (
            lookup_full_drug_info(cid)
            if cid.isdecimal()
            else None
        )

    return result


def register_predict(app):
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

        browser_result = dict(result)

        for number in (1, 2):
            info = result.get(
                f"drug{number}_full_info"
            )
            if info:
                browser_result[
                    f"drug{number}_full_info"
                ] = {
                    key: info.get(key)
                    for key in (
                        "cas",
                        "type",
                        "drugbank_id",
                    )
                }
                browser_result[
                    f"drug{number}_full_info"
                ].update(
                    description=str(
                        info.get("description", "")
                    )[:200],
                    targets=info.get("targets", [])[:3],
                )

        session["prediction_result"] = browser_result
        return redirect(url_for("index"))