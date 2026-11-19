from flask import jsonify


def register_explain(app):
    def unavailable():
        return jsonify(
            status="explanation_unavailable",
            error=(
                "Validated model explanation unavailable. "
                "Use /graph_data for recorded source relationships; "
                "these do not establish model importance "
                "or clinical harm."
            ),
        ), 503

    for name in (
        "explain",
        "explain_graph",
        "explain_view",
    ):
        app.add_url_rule(
            "/" + name,
            name,
            unavailable,
            methods=["GET"],
        )