# 1. Introduction to Knowledge Graphs in Drug Discovery

*How structured biomedical relationships can help researchers investigate the unseen*

![Biomedical knowledge graph illustration](https://cdn-images-1.medium.com/max/1600/1*W-mdtySKhYVv9v7GdC8B8Q.png)

In modern drug discovery, understanding one molecule also means understanding how it fits into an ecosystem of proteins, pathways, side effects and other drugs. With [polypharmacy](https://www.nia.nih.gov/news/dangers-polypharmacy-and-case-deprescribing-older-adults) and complex therapies, including [combination treatments](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/combination-therapy), researchers need to look beyond [compounds](https://pubchem.ncbi.nlm.nih.gov/docs/compounds) in isolation and examine relationships that can influence treatment outcomes.

One database might describe a drug's relationship with a protein, while another contains recorded interactions between medications. Researchers can bring those records together in a knowledge graph, where the relationships become something we can explore and use in a model.

For this project, I built a small application around that idea. It lets someone search for medications, read recorded interaction information, explore a biomedical network and inspect experimental model scores. This series follows the data and code behind it, starting with what the graph represents.

## What is a knowledge graph?

A [knowledge graph](https://pmc.ncbi.nlm.nih.gov/articles/PMC7327409/) represents entities, such as drugs, proteins and diseases, along with labeled relationships between them. Relational databases can store these connections too, but a graph puts the relationships at the center of how we organize and explore the information.

Think of it as a map of biomedical entities, where the connections, or edges, matter alongside the items, or nodes, they connect.

For example, a fictional graph might contain these records:

| Entity | Relationship | Entity |
| --- | --- | --- |
| Drug A | binds to | Protein X |
| Drug B | treats | Disease Y |
| Protein X | expressed in | Tissue Z |

The labels matter because binding to a protein says something different from treating a disease. Keeping those meanings attached to the edges helps us follow a question across several kinds of information.

Researchers use biomedical networks in areas such as:

- [Drug repurposing](https://ncats.nih.gov/research/research-activities/ntu), which investigates new uses for existing treatments.
- [Drug-combination side-effect prediction](https://snap.stanford.edu/decagon/), illustrated by the Decagon research project.
- [Protein association analysis](https://string-db.org/), supported by resources such as STRING.
- [Disease and treatment research](https://het.io/about/), including the relationships assembled in Hetionet.

These are broader research applications of graphs, rather than a list of capabilities established by my app.

## Why use one here?

Drugs can bind to proteins, be metabolized by enzymes, interact with other drugs and influence molecular pathways. Information about those relationships appears across research papers, [biomedical ontologies](https://www.ebi.ac.uk/ols4/) and [clinical trials](https://clinicaltrials.gov/), often with different identifiers and levels of evidence.

A graph gives us a way to represent several kinds of entities and relationships together, which is useful when a question crosses those sources. It can preserve a recorded connection such as “Drug A binds to Protein B” and provide the structure needed for graph-based machine learning.

It also makes recorded relationships easier to inspect, although that does not automatically explain a neural network's prediction. We still need to establish what each record represents, where it came from and how the model used the available information.

## The sources

I used [OGB BioKG](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg) for the graph and the model's training data. Its five entity types are drugs, proteins, diseases, side effects and protein functions, connected through 51 relation types. The benchmark provides separate training, validation and test splits for evaluating link prediction.

The model's message-passing graph uses the training split. Validation and test relationships stay outside that graph, so they can support evaluation on held-out connections. This is an edge-based evaluation; it does not establish performance on entirely new drugs.

[DrugBank](https://go.drugbank.com/) plays a different part in the application. A supplied export provides drug names, descriptions, targets and recorded interaction text, making the results more useful to someone who recognizes a medication name but has never seen a BioKG index. Those reference records were not added to this model's training graph.

[Hetionet](https://het.io/about/) is another useful example of a biomedical graph assembled from multiple resources for research, including drug repurposing, although it is not an input to this application.

Keeping the source and relationship type attached to a record helps us inspect what a connection actually means. It does not make every record equally reliable or turn a displayed path into a model explanation.

## From graph to prediction

[Graph neural networks](https://distill.pub/2021/gnn-intro/) learn representations of entities using information from their connections. The deployed prediction model in this project has two main components:

- An [R-GCN](https://arxiv.org/abs/1703.06103), or relational graph convolutional network, learns node representations while accounting for different relationship types.
- A [DistMult decoder](https://arxiv.org/abs/1412.6575) combines the representations of two drugs with a selected relation to produce a score.

Together, they score candidate relationships for drugs represented in the graph, giving us patterns to investigate. The score is uncalibrated, so it is not a probability of an adverse reaction or evidence that a medication combination is safe.

I also retained a [GraphSAGE](https://snap.stanford.edu/graphsage/) implementation as exploratory code, but it is not a trained model serving predictions in the app. GraphSAGE can support learning for unseen nodes when it has appropriate features and graph inputs. Adding that capability here would require a defined drug-feature pipeline, training and evaluation on drugs excluded from training, rather than simply enabling the existing class.

For graph attribution, I added a trained [PGExplainer](https://proceedings.neurips.cc/paper_files/paper/2020/file/e37b08dd3015330dcbb5d6663667b8b8-Paper.pdf) adaptation for link prediction. It ranks recorded edges entering the two selected drugs, while the rest of the graph context stays fixed. The application also measures how the prediction changes when each displayed edge's messages are gated. This helps inspect the model's dependence on those messages, without establishing a biological mechanism.

The explainer was evaluated on a small set of held-out queries, so its usefulness still needs to be judged alongside the reported fidelity measurements and its limited scope. It complements the decoder calculation rather than replacing it.

## Exploring the graph

The D3.js graph explorer displays recorded relationships around selected drugs, with entity-type filters, available names and source definitions. Users can select an entity to inspect its identifier and recorded connections, then move to a model explanation for a supported drug pair and relation.

The application accepts a list of 2 to 10 medications and checks every unique pair. It presents recorded DrugBank entries separately from optional model scores for supported drugs. Generic and brand-name searches use the supplied DrugBank records and a cached NLM RxTerms vocabulary; recognizing a name does not guarantee either interaction-record coverage or model coverage.

For supported predictions, the explanation page offers two views. The decoder breakdown shows how learned dimensions combine to produce the DistMult score, while the PGExplainer view ranks incoming graph messages and reports their measured effects. The recorded network, decoder arithmetic and graph attribution answer different questions, even when they appear together in the interface.

[Try the public BioKG Research Demo](https://kgnn-drug-drug-interaction-ddi-prediction-falling-sunset-9369.fly.dev/) or [explore the repository](https://github.com/techthumb1/KGNN-Drug-Drug-Interaction-DDI-Prediction). The application is a research demonstration, not a clinical decision tool.

## What comes next?

Knowledge graphs give researchers a way to organize interconnected biomedical information and examine relationships across databases. Paired with graph-based machine learning, they can help identify candidate connections for further investigation, provided we keep track of what the data and model can actually support.

In Article 2, **Building a Biomedical Knowledge Graph for Drug Discovery**, I'll walk through loading the data and reconciling identifiers so the graph, model and application agree about which entity each record represents.
