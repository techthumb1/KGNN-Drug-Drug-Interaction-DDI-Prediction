from flask import (
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
)

from src.utils.model import graph, entity_mappings
from src.utils.neighborhood import source_neighborhood
from src.utils.drug_catalog import get_catalog
from src.utils.entity_metadata import base_entity, enrich_entities, relation_label


def register_graph_routes(app):
    @app.route("/graph_data")
    def graph_data():
        try:
            first = int(request.args["drug1_id"])
            second = int(request.args["drug2_id"])

            data = source_neighborhood(graph, entity_mappings, first, second,
                                       entity_type=request.args.get("entity_type", "all"))
            catalog = get_catalog(entity_mappings.get("drug", {}))
            data["nodes"] = [base_entity(n["type"], n["local_index"], entity_mappings, catalog) for n in data["nodes"]]
            for link in data["links"]:
                link["relation_name"] = relation_label(link["relation"])
            data["model_relations"] = [{"id": r[1], "name": relation_label(r[1])} for r in graph.edge_index_dict if r[0] == r[2] == "drug"]
            data["selected_drugs"] = [catalog.by_index.get(i, {"name": f"drug:{i}"}) for i in (first, second)]
            return jsonify(data)
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return jsonify(
                error=(
                    "Provide two distinct valid "
                    "local drug indices"
                )
            ), 400

    @app.post("/graph_entities")
    def graph_entities():
        payload = request.get_json(silent=True)
        ids = payload.get("ids") if isinstance(payload, dict) else None
        if not isinstance(ids, list) or not 1 <= len(ids) <= 102:
            return jsonify(error="Provide 1 to 102 typed entity IDs."), 400
        catalog = get_catalog(entity_mappings.get("drug", {}))
        nodes = []
        try:
            for value in dict.fromkeys(ids):
                kind, raw = value.split(":")
                if not raw.isascii() or not raw.isdecimal() or kind not in graph.node_types:
                    raise ValueError("Invalid typed entity ID")
                index = int(raw)
                if not 0 <= index < graph[kind].num_nodes:
                    raise ValueError("Entity index is out of range")
                nodes.append(base_entity(kind, index, entity_mappings, catalog))
        except (ValueError, TypeError, AttributeError):
            return jsonify(error="Use valid typed entity IDs from the graph."), 400
        if len({node["type"] for node in nodes}) > 1:
            return jsonify(error="Request one entity type at a time."), 400
        return jsonify(nodes=enrich_entities(nodes))

    @app.route("/graph")
    def graph_page():
        first = request.args.get("drug1_id")
        second = request.args.get("drug2_id")

        if first is None or second is None:
            return redirect(url_for("index"))

        return render_template(
            "graph.html",
            drug1_id=first,
            drug2_id=second,
        )