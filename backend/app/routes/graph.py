from flask import (
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
)

from src.utils.model import graph, entity_mappings
from src.utils.neighborhood import source_neighborhood


def register_graph_routes(app):
    @app.route("/graph_data")
    def graph_data():
        try:
            first = int(request.args["drug1_id"])
            second = int(request.args["drug2_id"])

            return jsonify(
                source_neighborhood(
                    graph,
                    entity_mappings,
                    first,
                    second,
                )
            )
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