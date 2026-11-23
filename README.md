# BioKG Research Demo

**[Open the live demo](https://kgnn-drug-drug-interaction-ddi-prediction-falling-sunset-9369.fly.dev/)** — publicly hosted on Fly.io.

![BioKG Research Demo — conceptual biomedical network artwork](public/biokg-research-demo-cover.png)

*Conceptual cover illustration, not a visualization of actual BioKG records or model results.*

A Flask research prototype for exploring **recorded biomedical relationships** and experimental **relation-aware link prediction** using OGB BioKG.

This is not a clinical decision tool. Source relationships and model scores do not establish drug safety, clinical harm or treatment recommendations.

## Current functionality

- Explore recorded training-split relationships incident to two selected drugs.
- Display up to 100 edges with typed identifiers, available entity names and recorded relation names.
- Apply entity-type filters before truncation; search entities and inspect source definitions, identifiers and recorded connections.
- See an explicit message when the returned neighborhood is truncated.
- Run single and CSV batch scoring only when compatible, evaluated v2 model weights are available and predictions are explicitly enabled.
- Return scores as `uncalibrated_model_score`, without clinical likelihood thresholds.
- Report graph readiness separately from prediction availability at `/healthz`.

Predictions default to **unavailable**. The graph explorer can run without a model checkpoint, but still requires the official BioKG dataset.

`/explain_view` now provides an exact DistMult decoder score decomposition for a supported pair/relation. `/explain` returns the full numerical explanation as JSON: every learned dimension's signed logit term, the score with that term removed, and positive/negative totals. This holds cached graph embeddings fixed; it does not attribute graph edges or assign biological meaning to dimensions. The DistMult decoder is symmetric for a fixed relation. A trained PGExplainer adaptation now provides graph-message attribution at `/explain_graph`, with a separate section on `/explain_view`. Random weights, hard-coded SHAP values and untrained GraphSAGE inference remain disabled.

## Graph attribution and GraphSAGE

PGExplainer is an explanation model, separate from the R-GCN encoder and DistMult decoder. This implementation adapts the paper's shared edge-mask network and Concrete sampling to a relation-conditioned link score. It is not an untrained PyG explainer instantiated at request time. The 105 KB trained artifact and its evaluation report are included under `src/explain/artifacts/`; no additional download URL, secret or Python dependency is required.

The candidate scope is **every recorded training edge entering either queried drug**. A shared MLP receives source, target, edge-relation and query embeddings. Other graph context stays fixed. Learned gates multiply edge messages in both R-GCN layers, retaining original relation-wise mean denominators and self/root transformations. This is message gating, not physical edge deletion with degree renormalization. Unmasked calculations must reproduce the deployed predictor before any explanation is returned. This uses the model's full training context, not the graph explorer's truncated 100-edge display.

The page ranks 20 edges and recomputes the model with each edge's messages gated separately. Signed logit changes show whether gating lowers or raises the model output; they are not additive contributions. It also reports scores with only those candidate edges active and with those edges gated, so an incomplete explanation cannot silently appear sufficient. Available entity names and recorded relation names accompany the typed identifiers. JSON export includes the full held-out evaluation.

The explainer was trained for 50 epochs on 128 query examples, with 6,400 optimizer updates, while the evaluated predictor remained frozen. Thirty-two disjoint explanation queries came from official validation drug–drug triples and random candidate tails. Only official training edges were used for message passing; sampled candidates are not verified negative interactions. This is a small model-fidelity evaluation, not biological validation or a new predictor benchmark.

| Held-out explanation measure | Result |
| --- | ---: |
| Mean logit error, retaining top 20% of candidate edges | 1.9923 |
| Mean logit error, retaining equal-budget random edges | 5.1986 |
| Mean absolute logit change, gating top 20% | 5.2377 |
| Mean absolute logit change, gating random 20% | 1.9919 |
| Mean absolute score error using the learned soft mask | 0.02837 |
| Maximum unmasked logit reconstruction error | 0.000002384 |

The artifact is bound to the exact predictor parameters and training-graph hash. Missing, stale, degenerate or unevaluated artifacts return an explicit unavailable response. Computational neighborhoods above 50,000 candidate edges are rejected rather than truncated. First use initializes a CPU explanation cache; concurrent explanation work is bounded to one request. `/healthz` reports `graph_explanations` separately from the decoder explanation.

To reproduce the explainer after preparing the evaluated R-GCN checkpoint:

```bash
ENABLE_PREDICTIONS=true python -m src.train.train_pgexplainer --epochs 50 --train-queries 128 --eval-queries 32
```

This command trains the explainer, not the predictor. It needs the official validation split locally for the explanation evaluation. The production image only needs the included artifact and training graph. Retrain and re-evaluate the explainer whenever predictor weights change.

**GraphSAGE is an alternative encoder, not a required additional stage.** Its existing class is retained in `src/models/graphsage_model.py`; its old untrained serving path stays disabled. The deployed predictor remains R-GCN/DistMult. Adding GraphSAGE would require a trained baseline and a defined feature/input pipeline. Claiming prediction for previously unseen drugs would additionally require usable features for those drugs and evaluation with drug identities held out; the current BioKG edge split does not establish that capability.

References: [PGExplainer paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf), [GraphSAGE](https://snap.stanford.edu/graphsage/).

## Data and identifiers

The application uses the official **OGB `ogbl-biokg`** dataset. It constructs the source graph and model message-passing graph from the **official training split only**. Validation and test edges are not added to that graph, and synthetic reverse edges are not displayed as recorded relationships.

The homepage accepts drug names, DrugBank IDs and prefixed PubChem CIDs through a searchable medication list. Names and IDs resolve against the supplied DrugBank export, the locally cached NLM RxTerms medication vocabulary and BioKG mappings. Local BioKG indices remain available in the expandable index mode. DrugBank IDs identify DrugBank records; PubChem CIDs identify chemical compounds. They are distinct from local model indices. Model relations come from `/relations`.

The graph view returns incident relationships for either selected drug. It does not imply that the two drugs share a direct edge or cause an adverse interaction.

BioKG and DrugBank serve different purposes in this application:

| Source | Current role |
| --- | --- |
| OGB BioKG | Recorded heterogeneous training graph and the benchmark splits used to train/evaluate the R-GCN |
| DrugBank | Searchable drug names/identifiers, descriptions and targets, plus recorded interaction descriptions for selected drug pairs |

DrugBank integration remains in `src/utils/drugbank_lookup.py`, `src/utils/drugbank_full_lookup.py`, `src/utils/parse_drugbank_xml.py` and the prediction result page. DrugBank records are not added to the current training graph. The medication-list checker shows recorded interaction descriptions separately from optional model scores; those descriptions do not determine the scores. The initial graph-only verification had no local DrugBank cache. A subsequent run downloaded the existing prepared JSON from Google Drive, loaded 8,724 records with PubChem IDs and matched 1,328 BioKG drugs. Single prediction responses included real DrugBank metadata. Your existing Drive files remain the source when their URLs are configured.

For your existing Google Drive files, set `DRUGBANK_JSON_URL` to the file's existing Drive share/download URL in your private `.env` (local) or host environment settings (production). When the URL is configured, startup downloads the JSON once and reuses its cached copy on later starts. Drive remains the source; you do not need to move or re-upload the file. Standard Drive file-view URLs are accepted. The standalone app needs the URL to permit its download; ChatGPT's signed-in Drive connection is separate from the app's downloader.

The configured JSON is read one record at a time, retaining only the display fields used by the application. Its large interaction arrays are not held in the lookup cache. When the prepared JSON is present, the full XML source is not downloaded again.

These are automatic local cache paths, not instructions to manually relocate your Drive files:

- `backend/drugbank_data/drugbank_all_drug_links.csv` for the basic CID/name/DrugBank ID lookup.
- `backend/drugbank_data/drugbank_slim.json` for the extended description/target lookup.

The JSON records must use the schema produced by the existing XML parser (`pubchem`, `drugbank_id`, `name`, `cas`, `type`, `description`, `targets`, `interactions`). The CSV requires `PubChem Compound ID`, `DrugBank ID`, `Name`, `CAS Number` and `Drug Type` columns. The name picker searches all named DrugBank records, including drugs outside the trained model graph. Prediction metadata cards show complete descriptions and target lists for matched records. Graph drug labels use available names, with source identifiers retained in the entity inspector. Restart the app after changing metadata files. Missing records produce no substituted metadata.

The prepared JSON already contains the extended lookup; there is no need to reparse your XML when using it. If you have only an XML export, from the repository root run `python -m src.utils.parse_drugbank_xml /path/to/your-drugbank.xml` after creating `backend/drugbank_data/`. The existing parser creates the JSON lookup; it does not create the CSV. XML download alone does not load enrichment, and XML conversion is not run automatically. Existing supplied CSV/JSON files need no conversion.

## Graph names, definitions and source evidence

The graph explorer has zoom/pan/reset controls, a searchable entity list, a selected-entity inspector, named source-to-target relationships, a relationship table and JSON export. Entity-type filtering queries the source graph before truncation, so an early set of drug–drug edges cannot hide all protein or disease relationships.

Graph nodes preserve the official typed local index and original source identifier. Available DrugBank names/descriptions are used immediately. Remaining names are resolved progressively through exact identifier lookups: UMLS CUIs through NCBI MedGen, compound CIDs through PubChem, Entrez Gene IDs through NCBI Gene, and GO IDs through EMBL-EBI QuickGO. For example, `C0578870` resolves to **chronic idiopathic urticaria** in MedGen. The inspector shows the source definition, semantic type or organism when supplied, source-record link, and recorded neighbors in the current bounded view. A disease node is a dataset category; it need not denote an interaction mechanism.

Annotations use public services and require network access on a cold cache; only the requested source identifiers are sent, with no search text or whole medication-list submission. Successful annotations are cached for 30 days and missing records for one day under ignored `backend/drugbank_data/entity_metadata.sqlite3`. API failures remain explicit and do not replace identifiers with guessed names. These annotations do not add graph edges, alter model weights, or establish clinical causality. No new API keys or Python dependencies are required.

## Application usage

For a supported pair, **Rank model relations** orders all 38 BioKG drug–drug relation types by the active model's score. Select **Explain** to inspect a specific score. The ranking can also be exported as JSON.

### Check a medication list by name

1. Search a generic or brand name (optionally including strength or dosage form), DrugBank ID (for example `DB00641`) or PubChem CID (for example `CID239`) and select the intended entry from the suggestions. Duplicate aliases for the same compound are rejected.
2. Add 2–10 drugs. **Check selected pairs** reviews all unique unordered pairs, up to 45. This recorded-data lookup works without model weights when the DrugBank export is available.
3. Results distinguish a recorded DrugBank interaction, no unambiguous entry found in this export, and unavailable source coverage. An entry not found is not a safety determination. The existing JSON stores partner names rather than partner IDs, so only unambiguous exact-name matches are reported.
4. Expand **Optional model comparison** to also score a chosen BioKG relation for supported pairs. Drugs outside the trained graph keep their recorded lookup results and receive no fabricated model score. The selected relation does not classify interaction severity.

Drug-name searches and pair checks stay on the app server; they do not query external drug services for each keystroke. The first name search (or `python prepare_runtime.py`) automatically downloads the public [NLM RxTerms](https://www.lhncbc.nlm.nih.gov/MOR/RxTerms/) vocabulary into the ignored `backend/drugbank_data/rxterms.zip` cache. Release 202611 is pinned for reproducibility; set `RXTERMS_URL` to a trusted newer NLM release and remove this cache to refresh it. No new dependencies or credentials are required. The vocabulary adds U.S. brand names, generic products, synonyms, strengths and dosage forms; it is not a worldwide or exhaustive drug catalog. If the download is unavailable, the picker explicitly reports limited DrugBank-only search coverage.

Single-ingredient products use an exact, unambiguous match to a canonical DrugBank ingredient for recorded lookup and optional model scoring. Product results explicitly identify this as ingredient-level coverage; dosage and formulation are not evaluated. Combination products and unmatched ingredients remain searchable and selectable but receive an explicit unavailable result, with no partial ingredient substitution or invented model scores. Name recognition, recorded-interaction coverage and model coverage are separate. Full descriptions remain server-side rather than being truncated into session cookies.

### Recorded relationships

1. Enter two distinct local drug indices on the homepage.
2. Choose **View recorded relationships**. No prediction checkpoint or relation selection is required for this action.
3. Filter entity types, zoom/pan, search the entity list and select a node to inspect its name, source definition, identifiers and recorded connections. Use **Open model explanation** for a supported drug pair and relation.

Indices `58` and `89` can be used as example inputs for the official dataset. They are not a validated clinical example.

### Single prediction

After a compatible checkpoint has been trained, fully evaluated and deliberately enabled:

1. Enter two distinct local drug indices.
2. Select a supported drug-drug relation ID.
3. Choose **Predict**.

The result is an **uncalibrated model score**, not a clinical probability. No harm or safety threshold has been established.

### Batch prediction

Upload a UTF-8 CSV with these headers:

```csv
drug1_id,drug2_id,relation_type
58,89,2
```

The upload field is `csv_file`. The output includes `uncalibrated_model_score` and `error`. Invalid rows receive an error and no score. The page provides a result download when scoring succeeds and an error message when the request fails.

Corrected prediction history uses `predictions_v2_log.csv`, keeping it separate from legacy score logs. Render's ephemeral filesystem does not provide durable log storage across deployments.

Named batch files can use `drug1,drug2,relation_type` columns with canonical names, DrugBank IDs or prefixed CIDs. Existing `drug1_id,drug2_id,relation_type` files retain strict local-index behavior. Output adds resolved drug names; unsupported model entities receive an error without a score. Graph entity filters apply before the 100-edge limit and keep the selected drug anchors visible. Graph labels wrap; the default view labels selected drug anchors and the selected entity, with other names revealed on hover or using Show all labels.

## Model and checkpoint compatibility

The corrected modeling path uses:

- A deterministic saved entity-type order and disjoint type offsets.
- R-GCN message passing over the training graph.
- A learned DistMult decoder that incorporates the selected relation.
- Real optimization with a loss, backpropagation and parameter updates.
- Typed head/tail corruption that excludes known training positives.

DistMult is symmetric for same-type entity pairs and has limitations for directional relationships. Unrecorded training negatives can also contain unknown true relationships.

The old training script performed a forward pass and saved weights without optimization. The legacy downloadable checkpoint contains only `model_state_dict`, a two-dimensional encoder output and no relation decoder or training/evaluation metadata. Its training provenance cannot be established from that artifact, and it is incompatible with the corrected architecture.

V2 checkpoints record the architecture, node counts, relation order, offsets, training-graph hash, optimizer steps, seed and evaluation reports. Loading checks this contract and uses strict parameter loading. Legacy or incompatible weights are not silently reused.

## Model comparisons

The [BioKG model evaluation](docs/biokg-model-evaluation.md) records the TRIX, ULTRA and R-GCN/DistMult comparison, the description ablation, coverage limits and reproducible training and evaluation commands.

## Evaluation and current limitations

Evaluation ranks the official provided typed negative candidates on both head and tail sides, using mean reciprocal rank (MRR) with OGB tie handling. Reports include drug-drug relation breakdowns.

Serving requires complete official validation and test reports. A partial run marked `complete: false` cannot enable predictions. Completing this gate checks artifact compatibility and evaluation completeness; it does not establish acceptable model quality or clinical usefulness.

Verification performed during the refresh included:

- 39 Python regression tests covering startup, transfers, indexing, scoring, optimization, checkpoint rejection and routes.
- DOM/D3 checks for rendering, filtering, truncation and entity inspection.
- Startup and graph checks using actual official BioKG data.
- A 100-step training run on the official training graph (seed 42, batch size 1024, hidden size 64, learning rate 0.001), followed by complete validation and fixed-weight test evaluation.

The 100-step demonstration candidate produced these official ranking results:

| Split | Evaluated triples | Overall MRR |
| --- | ---: | ---: |
| Validation | 162,886 | 0.148652 |
| Test | 162,870 | 0.147863 |

Test evaluation used the same selected weights without retraining. The exact-triple audit found zero training overlaps in both held-out splits. These are results for a single 100-step demonstration recipe, not a convergence claim or a tuned benchmark. Scores remain uncalibrated. The app exposes an exact decoder decomposition plus a separately trained PGExplainer adaptation for incoming edge-message attribution. Neither establishes a biological mechanism.

## Setup

Run all commands from the repository root, using Python 3.11 and your existing `.venv`. Install dependencies with `python -m pip install -r requirements.txt`; the CPU-host commands below also show a CPU-only PyTorch installation. On a Mac, use the normal PyTorch package from `requirements.txt`.

### Local configuration

The private `.env` is intentionally ignored by Git. The tracked `.env.example` documents the variables; it contains no private keys. Create a local configuration without overwriting an existing one:

```bash
python - <<'PY'
from pathlib import Path
import secrets
config = Path(".env")
if not config.exists():
    template = Path(".env.example").read_text()
    config.write_text(template.replace(
        "FLASK_SECRET_KEY=", "FLASK_SECRET_KEY=" + secrets.token_hex(32), 1
    ))
PY
```

`FLASK_SECRET_KEY` signs Flask session cookies. It is not a DrugBank, OpenAI or model API key. A blank key generates a temporary key for local development. Production requires a persistent private value supplied through the host's environment settings; production does not load `.env`. Render's Blueprint declares `generateValue: true` for this key.

### Start with recorded relationships

Leave `ENABLE_PREDICTIONS=false` in `.env`, then run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python prepare_runtime.py
python run.py
```

Open `http://localhost:8080`. Enter local drug indices `58` and `89` and choose **View recorded relationships**. This graph view works without model weights. Initial BioKG preparation downloads the official dataset and needs several GB of disk space.

### Enable the already trained model

The trained 100-step checkpoint is a separate artifact, because `models/*.pt` is ignored by Git. Pulling source code does not install weights. From the completed-update archive, extract only `models/rgcn_v2.pt` into this repository's `models/` directory; there is no need to import its Git bundle after pulling the published changes. The archive also contains evaluation results.

Change `.env` to `ENABLE_PREDICTIONS=true`, run `python prepare_runtime.py`, then `python run.py`. Check `http://localhost:8080/healthz` for `"predictions": "ready"`. On the homepage, enter `58` and `89`, select relation ID `10`, and choose **Predict**. Inputs are BioKG local indices; the returned value is an uncalibrated link score. The same pair/relation can be submitted on the batch page with `drug1_id,drug2_id,relation_type` CSV headers. Training commands in the next section are for retraining, not for installing this existing checkpoint.

`prepare_runtime.py` prepares and verifies the source graph and mappings, and verifies the evaluated checkpoint when predictions are enabled. It does not optimize model weights. Initial preparation downloads the official BioKG archive and prepares `dataset/ogbl_biokg`. This requires access to the official Stanford download host and several GB of disk space. Model-enabled processes also require memory for the encoder and cached embeddings.

Only the trusted OGB cache/splits use pickle-compatible loading. Model checkpoints use `weights_only=True`; the loader does not globally override `torch.load`.

## Training

Training is separate from web startup.

```bash
# Execution smoke test only; this checkpoint cannot enable serving.
python -m src.train.train_rgcn --steps 1 --batch-size 32 \
  --hidden-channels 8 --max-eval 16 --output models/smoke_v2.pt

# Example starting configuration; choose the training budget using validation.
python -m src.train.train_rgcn --steps 100 --batch-size 1024 \
  --output models/rgcn_candidate.pt

# After complete validation, evaluate the same selected weights on the test split.
python -m src.train.evaluate_checkpoint --checkpoint models/rgcn_candidate.pt \
  --selection-note "Selected using full validation; recipe frozen before test" \
  --output models/rgcn_v2.pt
```

`train_rgcn` evaluates validation only and defaults to `models/rgcn_candidate.pt`. The old `--evaluate-test` option is rejected because it retrained instead of evaluating the selected artifact. The finalizer preserves the candidate, audits exact split separation and records complete test results without changing its weights. Optimizer state stays in the training candidate and is omitted from the smaller serving artifact. It rejects incomplete validation and existing output paths.

The example 100-step configuration is not a validated training budget. Full-graph R-GCN training can require substantial memory. Inspect overall and per-relation results before deciding whether the model is useful for the intended research task.

After training and reviewing complete evaluation, supply `models/rgcn_v2.pt`, or set `MODEL_URL` to a trusted compatible v2 checkpoint. Explicitly set `ENABLE_PREDICTIONS=true` to enable scoring. Do not use the legacy checkpoint URL.

## Public deployment

The repository's automatic deployment targets the [public Fly.io demo](https://kgnn-drug-drug-interaction-ddi-prediction-falling-sunset-9369.fly.dev/). This replaces the former Render demo link. The public app has been checked with drug-name search, recorded DrugBank interactions, model predictions and decoder score explanations enabled.

The Fly workflow provisions a private Flask session key only when missing, deploys the image, then requires a successful public `/healthz` response. It captures runtime logs and health checks when deployment fails. The container prepares the public BioKG training data during its build and discards held-out evaluation tensors from the serving image; production inference does not need those large candidate sets. DrugBank and model artifacts still use authorized host configuration. Local data files are not copied to a remote service by a Git push.

The health response distinguishes graph readiness, predictions, and decoder explanations. Full feature availability must be checked on the public host before describing it as an enabled model demo. A missing or invalid optional DrugBank export is reported explicitly on the homepage and in pair coverage; it does not prevent graph exploration from starting. Required graph/checkpoint validation remains strict.

## Deployment on Render

Create or locate a **Web Service** connected to this repository. Use the repository root and the reviewed deployment branch.

| Setting | Value |
| --- | --- |
| Build command | `pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && python prepare_runtime.py` |
| Start command | `gunicorn run:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600` |
| Health check | `/healthz` |
| Initial environment | `FLASK_ENV=production`, `ENABLE_PREDICTIONS=false`, a private `FLASK_SECRET_KEY` |

For a Blueprint, synchronize `render.yaml`. Existing manually configured services require their dashboard commands and environment to be updated directly. The existing Fly.io workflow does not deploy the Render service. Production startup requires a private `FLASK_SECRET_KEY`; the application no longer falls back to a shared key.

A provider response containing `x-render-routing: no-server` requires checking the exact service hostname, workspace, service existence/status and deployment logs. Changing Flask routes or billing details alone does not establish that a serving instance exists.

After deployment, verify the homepage, graph generation and health response. With predictions disabled, a valid prediction request should return HTTP 503 without a score. Do not present the demo as live until the actual public deployment has been checked.

## Endpoint behavior

| Endpoint | Behavior |
| --- | --- |
| `GET /` | Research form and prediction-availability notice |
| `GET /relations` | Supported drug-drug relation IDs |
| `GET /drugs/search?q=...` | Search generic/brand product names, synonyms, strengths, DrugBank IDs and prefixed CIDs; report vocabulary availability |
| `POST /check` | Check 2–10 drugs against recorded DrugBank entries, with optional model comparison |
| `GET /graph?drug1_id=58&drug2_id=89` | Recorded relationship viewer |
| `GET /graph_data?drug1_id=58&drug2_id=89` | Bounded recorded neighborhood; optional `entity_type` filter applies before truncation |
| `POST /graph_entities` | Exact public annotations for 1–102 valid typed entity IDs of one type |
| `POST /predict` | Corrected score, 400 for invalid input, or 503 when unavailable |
| `GET /batch` | CSV upload page |
| `POST /batch` | Result CSV or an input/availability error |
| `GET /healthz` | Graph readiness and separate prediction/explanation availability |
| `GET /download_log` | V2 prediction history, or 404 if no V2 log exists |
| `GET /explain` | Exact decoder contributions and dimension-removal scores as JSON; requires a ready model |
| `GET /explain_view` | Decoder explanation and trained PGExplainer attribution |
| `GET /explain_graph` | Trained PGExplainer edge ranks, measured message ablations and held-out evaluation; 503 for missing/incompatible artifacts |

## Tests

```bash
python -m unittest discover -s tests -v
npm install --prefix /tmp/ddi-dom --no-package-lock jsdom@31.0.1 d3@7.9.2
NODE_PATH=/tmp/ddi-dom/node_modules node tests/test_graph_dom.cjs
```

The DOM checks use Node 24+. Test fixtures are synthetic and are never served as replacement biomedical data. Passing tests does not establish prediction accuracy or a successful production deployment.

## Notebooks

The two existing notebooks now use the same typed indexing, graph contract, model and checkpoint loader as the application. Saved outputs are cleared.

| Notebook | Contents |
| --- | --- |
| [01 — Training graph and recorded relationships](notebooks/01_ogbl_biokg_eda.ipynb) | Official data, source counts, exact split audit and bounded recorded neighborhoods |
| [02 — Training, evaluation and checkpoint verification](notebooks/02_ogbl_biokg_eda.ipynb) | Candidate training, full validation, fixed-weight test evaluation and checkpoint scoring |

Install `requirements-notebooks.txt` alongside `requirements.txt` in your Python 3.11 kernel. Set `BIOKG_NOTEBOOK_RUN_DATA=1` before starting the kernel to load official data; set `BIOKG_NOTEBOOK_AUDIT_SPLITS=1` for the full split audit. In notebook 02, set `RUN_TRAINING=True` to train. After reviewing complete validation, set `RUN_TEST_EVALUATION=True` and record your selection decision to finalize the selected weights. Scoring has its own explicit flag and requires the evaluated artifact. Defaults skip expensive operations so opening a notebook does not start training.

## Images and article series

The cover is conceptual artwork. Earlier screenshots in `public/` show the historical interface and may contain obsolete probability or explanation labels. Do not use them to describe the current application.

After deployment, capture new screenshots of:

- The homepage with its actual prediction-availability state.
- The recorded relationship view using actual BioKG data.
- The batch page, showing either its unavailable notice or actual results.

Articles 2–6 now describe the current training graph, measured R-GCN demonstration, exact decoder explanation, medication list and named graph explorer. The compact GraphSAGE class remains teaching code, not a second deployed model. The current PGExplainer adaptation is trained and evaluated for a bounded graph-message attribution scope; genomic/temporal extensions remain future work. An aligned [Article 1 draft](docs/articles/01-introduction-to-knowledge-graphs.md) distinguishes the deployed predictor, retained GraphSAGE code and the new PGExplainer attribution. Earlier article exports predate this addition.

## Sources

- [DrugBank drug-interaction documentation](https://dev.drugbank.com/guides/drug_interactions/overview)
- [Official OGB BioKG dataset and evaluation protocol](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg)
- [OGB BioKG reference implementations](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/biokg)
- [PyTorch Geometric heterogeneous graph data](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.data.HeteroData.html)

## License

[MIT](LICENSE). BioKG source datasets and optional third-party enrichment retain their own licenses and usage requirements.
