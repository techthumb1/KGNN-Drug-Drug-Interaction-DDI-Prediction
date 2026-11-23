FROM python:3.11.12-slim-bookworm AS serving-model

WORKDIR /bundle
COPY deployment/ .
RUN python unpack_model.py /verified

FROM python:3.11.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
COPY src /app/src
COPY run.py .
COPY prepare_runtime.py .
COPY --from=serving-model /verified/rgcn_v2.pt /app/models/rgcn_v2.pt
COPY --from=serving-model /verified/pgexplainer.pt /app/models/pgexplainer.pt

# Prepare public source data during the build, so cold starts do not fetch
# and extract several GB before the HTTP worker can answer.
RUN ENABLE_PREDICTIONS=true REQUIRE_GRAPH_EXPLAINER=true python prepare_runtime.py \
    && rm -f dataset/ogbl_biokg/split/random/valid.pt \
             dataset/ogbl_biokg/split/random/test.pt

ENV FLASK_ENV=production

EXPOSE 8080

CMD ["gunicorn", "run:app", "--bind=0.0.0.0:8080", "--workers=1", "--timeout=600"]
