import csv
import io

from flask import (
    request,
    render_template,
    send_file,
    jsonify,
)

from src.utils import model as runtime
from backend.app.routes.predict import score_request


def register_batch(app):
    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        if request.method == "GET":
            return render_template(
                "batch.html",
                prediction_error=runtime.prediction_error,
            )

        if runtime.prediction_error:
            return jsonify(
                error=runtime.prediction_error,
                status="prediction_unavailable",
            ), 503

        upload = request.files.get("csv_file")

        if (
            not upload
            or not upload.filename.lower().endswith(".csv")
        ):
            return jsonify(
                error="Upload a CSV using the csv_file field"
            ), 400

        try:
            reader = csv.DictReader(
                io.StringIO(
                    upload.read().decode("utf-8-sig")
                )
            )
            required = [
                "drug1_id",
                "drug2_id",
                "relation_type",
            ]

            if (
                not reader.fieldnames
                or not set(required).issubset(
                    reader.fieldnames
                )
            ):
                raise ValueError(
                    "CSV requires drug1_id, drug2_id, "
                    "relation_type headers"
                )

            buffer = io.StringIO()
            fields = required + [
                "uncalibrated_model_score",
                "error",
            ]
            writer = csv.DictWriter(
                buffer,
                fieldnames=fields,
            )
            writer.writeheader()

            for row in reader:
                try:
                    result = score_request(row)
                    writer.writerow(
                        {
                            key: result.get(key, "")
                            for key in fields
                        }
                    )
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                ) as error:
                    writer.writerow(
                        {
                            **{
                                key: row.get(key, "")
                                for key in required
                            },
                            "error": str(error),
                        }
                    )

        except (
            ValueError,
            UnicodeDecodeError,
            csv.Error,
        ) as error:
            return jsonify(error=str(error)), 400

        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            mimetype="text/csv",
            download_name="predictions.csv",
            as_attachment=True,
        )