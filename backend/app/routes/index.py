from flask import (
    render_template,
    jsonify,
    session,
    send_file,
)

from src.utils import model as runtime
from src.utils.paths import PROJECT_ROOT


def register_index(app):
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            prediction_result=session.pop(
                "prediction_result", None
            ),
            prediction_error=runtime.prediction_error,
            drug_count=runtime.graph["drug"].num_nodes,
        )

    @app.get("/relations")
    def get_relation_types():
        return jsonify(
            sorted(
                {
                    key[1]
                    for key in runtime.rel_type_to_id
                    if key[0] == key[2] == "drug"
                }
            )
        )

    @app.get("/download_log")
    def download_log():
        path = PROJECT_ROOT / "predictions_v2_log.csv"
        if path.exists():
            return send_file(
                path,
                as_attachment=True,
            )
        return "No prediction log found.", 404