from flask import jsonify, render_template, request

from src.utils import model as runtime
from src.utils.drug_catalog import get_catalog
from src.utils.entity_metadata import relation_label
from src.explain.explain_ddi import explain_score


def register_explain(app):
    @app.get("/explain")
    def explain():
        try:
            first, second = int(request.args["drug1_id"]), int(request.args["drug2_id"])
            relation = request.args["relation_type"]
            result = explain_score(runtime, first, relation, second)
            catalog = get_catalog(runtime.entity_mappings.get("drug", {}))
            result["drug1_name"] = catalog.by_index.get(first, {}).get("name", f"drug:{first}")
            result["drug2_name"] = catalog.by_index.get(second, {}).get("name", f"drug:{second}")
            result["relation_name"] = relation_label(relation)
            return jsonify(result)
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(error=f"Choose two valid local drug indices and a supported relation. {error}"), 400
        except runtime.PredictionUnavailable as error:
            return jsonify(status="explanation_unavailable", error=str(error)), 503

    @app.get("/explain_view")
    def explain_view():
        return render_template("explain.html", relations=[{"id": row[1], "name": relation_label(row[1])}
                                                       for row in runtime.contract["relations"] if row[0] == row[2] == "drug"])

    @app.get("/explain_graph")
    def explain_graph():
        return jsonify(status="explanation_unavailable", error="Graph-edge attribution is unavailable. /explain provides an exact decoder score decomposition."), 503
