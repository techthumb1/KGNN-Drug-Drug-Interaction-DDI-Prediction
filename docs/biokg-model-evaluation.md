# Model evaluation on BioKG

The 100-step R-GCN/DistMult checkpoint is the historical serving reference. A stronger R-GCN candidate reached **0.3788 MRR** against **0.1487** for the reference on all 162,886 official validation triples. The longer run used the same architecture, seed, batch size, and learning rate. Its sparse CPU implementation reproduced the reference embeddings to within 0.0000035 and passed forward and gradient parity checks.

The current serving choice uses cached embeddings with full benchmark coverage and an independently measured explainer. A separate GitHub Actions reproduction froze the 1,000-step weights using validation and evaluated them once on test. The reproduced candidate and its matching explainer passed artifact verification. The Fly image builds from the evaluated 1,000-step checkpoint and its matching explainer, and the deployment check requires both to be ready. Pair ranking supports all 38 drug–drug relations.

| Official split | 100-step reference MRR | Reproduced 1,000-step MRR | Drug–drug triples | Reproduced drug–drug MRR |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.148652 | 0.378657 | 31,135 | 0.523883 |
| Test | 0.147863 | 0.377500 | 31,151 | 0.521393 |

The new PGExplainer passed the existing binding and held-out fidelity checks on 32 queries. Mean logit error was 4.4093 when retaining the top 20% of messages and 5.6689 with an equal random budget. Maximum unmasked reconstruction error was 0.00000191. The archive contains both evaluated artifacts and their reports. [R-GCN and PGExplainer artifact](https://github.com/techthumb1/KGNN-Drug-Drug-Interaction-DDI-Prediction/actions/runs/45145263213/artifacts/14804069262)

## Full drug–drug relation ranking

The evaluated R-GCN also ranked each known relation against the alternatives for the same ordered drug pair.

| Official split | Drug–drug triples | Filtered relation MRR | Median seconds per query |
| --- | ---: | ---: | ---: |
| Validation | 31,135 | 0.401589 | 0.000024 |
| Test | 31,151 | 0.426052 | 0.000025 |

The timing uses CPU batches of 512 and excludes graph preparation and embedding precomputation. The public pair ranking displays all 38 candidate scores. Filtering applies only to benchmark evaluation. [Validation report](biokg-results/rgcn-relation-valid.json) · [Test report](biokg-results/rgcn-relation-test.json)

## Matched validation samples

| Model | Task | Queries | MRR | Drug–drug queries | Drug–drug MRR | Median seconds per query |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TRIX zero-shot | Entity | 64 | 0.74074 | 13 | 0.36078 | 1.78634 |
| ULTRA 3g zero-shot | Entity | 64 | 0.55325 | 13 | 0.05898 | 2.20600 |
| R-GCN 100 steps | Entity | 64 | 0.14384 | 13 | 0.27949 | 0.00047 |
| TRIX zero-shot | Relation | 64 | 0.12981 | 64 | 0.12981 | 1.35444 |
| R-GCN 100 steps | Relation | 64 | 0.30113 | 64 | 0.30113 | 0.00014 |

The sampled tests used the full training context and seed 42. These pilot results leave substantial uncertainty, particularly for the 13 drug–drug queries in the mixed entity sample. The timed batches use two queries each. Time excludes graph preparation and embedding precomputation. Peak process RSS includes evaluation data and ranges from 3.33 to 5.17 GiB across the sampled runs. See the [numerical summary](biokg-results/validation-summary.json).

A second run scored 64 drug–drug validation triples using the same negative candidates before and after 32 decoder-only optimization steps. The backbone remained frozen. Each model used the same supervised sampling recipe and removed its training targets from both message graphs.

| Model | Zero-shot MRR | Tuned MRR | Tuned median seconds per query |
| --- | ---: | ---: | ---: |
| TRIX | 0.301059 | 0.301994 | 1.422 |
| ULTRA 3g | 0.078541 | 0.088284 | 1.819 |

The upstream kernels and graph conversion matched the official forward pass on a small graph to within 0.000002 for both TRIX checkpoints and ULTRA. The small improvements after tuning need further evaluation. Full foundation-model splits and drug identity holdouts remain future experiments. [TRIX reports](https://github.com/techthumb1/KGNN-Drug-Drug-Interaction-DDI-Prediction/actions/runs/45145263213/artifacts/14805335090) · [ULTRA reports](https://github.com/techthumb1/KGNN-Drug-Drug-Interaction-DDI-Prediction/actions/runs/45145263213/artifacts/14805165456)

### 256-query drug–drug entity comparison

All three evaluation runs scored the same 256 validation rows and their official typed candidates. The R-GCN scores come from the exact weights in the serving bundle.

| Model | MRR | Median CPU seconds per query | Batch size | Report |
| --- | ---: | ---: | ---: | --- |
| R-GCN 1,000 steps | 0.536830 | 0.000145 | 64 | [JSON](biokg-results/rgcn-ddi256-valid.json) |
| TRIX zero-shot | 0.314317 | 1.413887 | 4 | [JSON](biokg-results/trix-ddi256-valid.json) |
| ULTRA 3g zero-shot | 0.076514 | 1.578664 | 4 | [JSON](biokg-results/ultra-ddi256-valid.json) |

Timing excludes graph setup and R-GCN embedding precomputation. These batch timings describe throughput on the evaluation machine. The sample supports the serving choice for this demo, and full foundation-model split results remain pending.

## Evaluation protocol

- **Entity prediction.** Score the [official typed head and tail negatives](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg), averaging optimistic and pessimistic ranks for ties. The comparison uses the same row IDs and negatives for each model and reports a drug–drug breakdown by relation.
- **Relation prediction.** For each ordered drug pair, rank the target among the 38 supported drug–drug relations. Filter alternative known true relations for that pair. Validation ranking uses train and validation labels for filtering. Test ranking can use all three splits for filtering. Message passing only uses training edges, including training-derived synthetic inverses for TRIX and ULTRA.
- **Fine-tuning.** Sample supervised facts from training. Remove those facts and same-relation reversals within an entity type before constructing the message-passing graphs. Rank typed corruptions. Restore the official training context during evaluation.
- **Checkpoint provenance.** TRIX and ULTRA 3g are pinned to public upstream revisions and byte hashes in `src/train/foundation_models.py`. Their disclosed pretraining graphs are FB15k237, WN18RR and CoDExMedium. Cross-dataset overlap of semantically equivalent facts remains unverified. The BioKG audit counts exact directed overlap and same-relation drug–drug reversals.

## Restricted description ablation

| Validation cohort | Structure only | Text only | Combined |
| --- | ---: | ---: | ---: |
| Sampled drug–drug triples | 0.284485 | 0.074327 | 0.299040 |
| Both drugs have matched descriptions | 0.353633 | 0.151503 | 0.357274 |
| Held-out drugs with non-DDI context | 0.264118 | 0.083932 | 0.283362 |
| Held-out drugs with two matched descriptions | 0.335278 | 0.127542 | 0.342610 |

The comparison uses the same pair scorer, sample seeds and typed candidates. Graph inputs count non-DDI training neighbors by relation. Text inputs are fixed class, modality and form indicators extracted from descriptions, with interaction-language sentences excluded. Of 10,533 BioKG drugs, 1,243 had matched descriptions and 438 had at least one retained term. The cold-start experiment holds drug identities out of pair-classifier training while providing their non-DDI graph context and available descriptions at inference.

The combined input was modestly ahead in this single-seed experiment. The experiment uses a limited description projection and has uneven coverage. Official MERRY scoring remains pending. The inspected upstream revision requires CUDA 12.x, platform-specific dependencies, and both checkpoint and step directories for evaluation. These measurements come from the repository's simpler classifier. [MERRY implementation at the inspected revision](https://github.com/zjukg/MERRY/tree/96e7f87815888025d81c3e536559abb5ae276202)

## Further experiments

- Measure TRIX and ULTRA on full held-out splits and on a separate drug-identity holdout with explicit rules for available graph context. Run MERRY under the same candidate and description restrictions once compatible weights and runtime are available.
- Use [UltraQuery](https://arxiv.org/abs/2404.07198) for a subsequent benchmark of multi-relation questions with logical conditions. Use [GFM-RAG](https://arxiv.org/abs/2502.01113) for a separate document-retrieval and evidence-grounding evaluation.

## Reproduce

Use Python 3.11 and a research environment separate from the app. Install the CPU build of PyTorch for these runs. GPU experiments require the matching CUDA toolkit and PyTorch build.

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-research.txt
python -m pip install --no-build-isolation --no-deps torch-scatter==2.1.2

git clone https://github.com/yuchengz99/TRIX.git ../trix-upstream
git -C ../trix-upstream checkout 7596e14eefefe89e61396205a0550172cadeddb0
git clone https://github.com/DeepGraphLearning/ULTRA.git ../ultra-upstream
git -C ../ultra-upstream checkout 427966ad8ed60420eef034063d44f3153addff90

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m src.train.train_rgcn \
  --steps 1000 --sparse-cpu --eval-batch-size 64 --output models/rgcn_1000.pt

OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -m src.train.evaluate_foundation \
  --model trix-entity --upstream ../trix-upstream --max-queries 64 --batch-size 2 \
  --output artifacts/trix-entity-validation.json

OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -m src.train.evaluate_foundation \
  --model trix-relation --task relation --upstream ../trix-upstream \
  --max-queries 64 --batch-size 2 --output artifacts/trix-relation-validation.json

OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -m src.train.evaluate_foundation \
  --model ultra --upstream ../ultra-upstream --max-queries 64 --batch-size 2 \
  --output artifacts/ultra-validation.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m src.train.finetune_foundation \
  --model trix-entity --upstream ../trix-upstream --steps 32 --decoder-only \
  --output models/trix_entity_decoder_tuned.pt

OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -m src.train.evaluate_foundation \
  --model trix-entity --upstream ../trix-upstream --checkpoint models/trix_entity_decoder_tuned.pt \
  --ddi-only --max-queries 64 --batch-size 2 --output artifacts/trix-decoder-validation.json

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m src.train.ablate_biomedical_text \
  --descriptions /path/to/drugbank_slim.json --output artifacts/text-ablation.json
```

Pass `--model rgcn --checkpoint models/rgcn_v2.pt` for the reference comparison. The distributed demo checkpoint is documented in the main README. Use `--ddi-only` to sample exclusively from drug–drug triples. Omit `--max-queries` for complete official evaluation on the eligible queries. Run foundation evaluations sequentially on memory-constrained hosts.

The 256-query entity comparison used `--ddi-only --max-queries 256 --seed 42` with batch size 4 for TRIX and ULTRA and 64 for R-GCN. For the complete R-GCN relation reports, use `--model rgcn --checkpoint models/rgcn_v2.pt --task relation --batch-size 512` and run each of `--split valid` and `--split test`.

For test evaluation of frozen R-GCN weights, use `src.train.evaluate_checkpoint` with the validation-based selection decision. Any subsequent explainer training needs to use the exact new serving artifact and pass held-out fidelity checks before deployment.
