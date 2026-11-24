# 1. Introduction to Knowledge Graphs in Drug Discovery

*How structured biomedical relationships help researchers investigate missing links*

![Biomedical knowledge graph illustration](https://cdn-images-1.medium.com/max/1600/1*W-mdtySKhYVv9v7GdC8B8Q.png)

In drug discovery, understanding a compound means looking at the proteins it binds to, the pathways it affects, and the other medications it might be taken with. Those connections matter even more when people take several medications or receive combination therapies. [Polypharmacy](https://www.nia.nih.gov/news/dangers-polypharmacy-and-case-deprescribing-older-adults) · [Combination treatments](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/combination-therapy) · [PubChem compounds](https://pubchem.ncbi.nlm.nih.gov/docs/compounds)

I built a small app to explore those connections. In this update, I compared two graph foundation models, **TRIX and ULTRA**, with a newly trained R-GCN. I also added a way to rank all **38 BioKG drug–drug relation types** for a supported pair and open two explanations for any score in that ranking.

## What Is a Knowledge Graph?

A knowledge graph stores entities, such as drugs, proteins, and diseases, along with named relationships between them. One database might describe a drug binding to a protein. Another might record a drug interaction. Putting relationships into graph form makes it easier to follow how the records connect. [Knowledge graphs in biomedicine](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327409/)

Here is a small fictional example.

| Entity | Relationship | Entity |
| --- | --- | --- |
| Drug A | binds to | Protein X |
| Drug B | treats | Disease Y |
| Protein X | expressed in | Tissue Z |

Each relation tells us what kind of connection the source reported. That matters when the same two entities can be linked in more than one way. Biomedical graph research includes drug repurposing, drug-combination side effects, and protein interaction modeling. Disease classifications help give those graphs consistent disease labels. [NIH drug repurposing](https://ncats.nih.gov/research/research-activities/ntu) · [Decagon](https://snap.stanford.edu/decagon/) · [STRING](https://string-db.org/) · [WHO disease classification](https://www.who.int/standards/classifications/classification-of-diseases)

## The Sources Behind the App

**OGB BioKG** supplies the graph and the model's training data. It contains five entity types and 51 relation types. The model passes messages along training edges only. The held-out validation and test edges let me measure whether it ranks a missing link above the alternatives in the benchmark. [OGB BioKG](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg)

**DrugBank** supplies recorded interactions and reference information, such as names, descriptions, and targets. The app uses that information for lookup. **Hetionet** is a separate example of how multiple public sources can be joined for drug repurposing. I used BioKG for the graph in this project. [DrugBank](https://go.drugbank.com/) · [Hetionet](https://het.io/about/)

## What I Changed in the Model

The **R-GCN** learns a vector for each entity by gathering information along different types of connections. The **DistMult** decoder takes two entity vectors and a relation vector and produces a score. For this update, I trained R-GCN for 1,000 optimization steps and evaluated the saved weights on the full BioKG validation and test splits. [R-GCN paper](https://arxiv.org/abs/1703.06103) · [DistMult paper](https://arxiv.org/abs/1412.6575)

I wanted to see how reusable graph models would handle the same task. **ULTRA** learns how relation types connect so its pretrained weights can be applied to another graph. **TRIX** models interactions among the head entity, relation, and tail entity. It supports both filling in a missing entity and identifying the relation between two entities. [ULTRA paper](https://arxiv.org/abs/2310.04562) · [TRIX paper](https://arxiv.org/abs/2502.19512)

I gave all three models the same **256 drug–drug validation triples** and the same candidate entities. The metric, mean reciprocal rank (MRR), rewards putting the known answer near the top.

| Model | MRR |
| --- | ---: |
| R-GCN with DistMult, trained for 1,000 steps | 0.5368 |
| TRIX, zero-shot | 0.3143 |
| ULTRA 3g, zero-shot | 0.0765 |

I kept R-GCN in the app because it led this comparison, had full-split evaluation, and can reuse precomputed entity vectors when scoring requests. Part 3 walks through the results, including a separate decoder-tuning pilot. The foundation models still need full-split evaluation and tests that hold out whole drug identities. [BioKG model evaluation](../biokg-model-evaluation.md)

## Exploring a Drug Pair

Search for two to ten medications by name. The app checks each unique pair against the supplied DrugBank records. When both medications map to BioKG, select **Rank model relations** to see how the model orders the 38 supported relation types. You can download the ranking as JSON, inspect the recorded neighborhood in the D3 graph, or choose a score to explain.

The **decoder breakdown** shows the individual terms that add up to the DistMult score. The **PGExplainer** view shows which incoming messages from training edges mattered to that score. I retrained PGExplainer for the 1,000-step checkpoint and checked its fidelity on held-out queries. These views explain the model's calculation. The graph and DrugBank panels show source records. Use the app to explore research questions about the graph. [PGExplainer paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf) · [BioKG Research Demo](https://kgnn-drug-drug-interaction-ddi-prediction-falling-sunset-9369.fly.dev/)

## Next in the Series

In **Article 2, Building a Biomedical Knowledge Graph for Drug Discovery**, I'll walk through loading BioKG, reconciling drug identifiers, and preparing the same graph and evaluation candidates for R-GCN, TRIX, and ULTRA.

[Add the published Part 2 link here]
