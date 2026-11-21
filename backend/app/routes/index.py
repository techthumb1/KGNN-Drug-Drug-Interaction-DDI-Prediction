from flask import (
    render_template,
    jsonify,
    session,
    send_file,
    request,
)

from src.utils import model as runtime
from src.utils.paths import PROJECT_ROOT
from src.utils.drug_catalog import get_catalog
from src.utils import drugbank_full_lookup as metadata


def register_index(app):
    @app.get("/")
    def index():
        from .predict import enrich_result
        result = session.pop("prediction_result", None)
        return render_template(
            "index.html",
            prediction_result=enrich_result(result) if result else None,
            prediction_error=runtime.prediction_error,
            metadata_error=metadata.metadata_error,
            drug_count=runtime.graph["drug"].num_nodes,
        )

    @app.get("/drugs/search")
    def search_drugs():
        catalog = get_catalog(runtime.entity_mappings.get("drug", {}))
        query = request.args.get("q", "")[:160]
        return jsonify({"drugs": catalog.search(query),
                        "name_vocabulary_status": catalog.name_vocabulary_status,
                        "named_drugs": len(catalog.entries)})

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