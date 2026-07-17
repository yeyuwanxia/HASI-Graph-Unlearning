# Edge and Feature Evaluation Protocol

This file freezes the semantics used by the formal comparison and RQ runs.
Changing either rule creates a new experimental protocol and requires a new
result tree.

## Edge Requests

The current formal PubMed protocol uses unique undirected edges:

- Candidate construction: canonicalize every eligible `edge_index` entry as
  `(min(u, v), max(u, v))`, then deduplicate it.
- Sampling unit: one unique undirected edge in the training subgraph.
- Ratio denominator: the number of unique candidate undirected edges.
- Deletion operator: remove both stored directions of every selected edge.
- Required paper wording: "We sample 5%/10% of unique undirected
  training-subgraph edges and remove both stored directions of each selected
  edge."

Legacy protocol files may record `sampling_unit=directed_edge_index_entry`.
Those files and results must not be mixed with the unique-undirected protocol.
Hetionet and PrimeKG edge matrices must be regenerated before they are compared
with the current PubMed edge results.

## Global Feature-Dimension Requests

- Request unit: one feature dimension.
- Deletion operator: zero each selected dimension for every node.
- Node-level MIA: not applicable. A global dimension request does not define a
  semantically valid node member/non-member partition.
- Formal reporting: utility, representation drift, efficiency, and
  `feature_compliance`. The compliance block is a request-execution check, not a
  privacy ranking metric.

Feature attribute privacy requires a separate attribute-inference evaluator and
an exact-retraining reference. Until that evaluator is implemented, feature MIA
fields remain null and must not enter aggregate privacy claims.
