LambdaRank Gradients are Incoherent
===============================

This code is for CIKM 2023 full paper [LambdaRank Gradients are Incoherent](https://dl.acm.org/doi/10.1145/3583780.3614948).

Abstract
---

In Information Retrieval (IR), the Learning-to-Rank (LTR) task requires building a ranking model that optimises a specific IR metric.
One of the most effective approaches to do so is the well-known LambdaRank algorithm.
LambdaRank uses gradient descent optimisation, and at its core, it defines approximate gradients, the so-called *lambdas*, for a non-differentiable IR metric.
Intuitively, each lambda describes how much a document's score should be ``pushed'' up/down to reduce the ranking error.

In this work, we show that lambdas may be incoherent w.r.t. the metric being optimised: e.g., a document with high relevance in the ground truth may receive a smaller gradient push than a document with lower relevance.
This behaviour goes far beyond the expected degree of approximation.
We analyse such behaviour of LambdaRank gradients and we introduce some strategies to reduce their incoherencies.
We demonstrate through extensive experiments, conducted using publicly available datasets, that the proposed approach reduces the frequency of the incoherencies in LambdaRank and derivatives, and leads to models that achieve statistically significant improvements in the NDCG metric, without compromising the training efficiency.

Implementation
---

**Lambda-eX** is a document-pairs selection strategy built on top of [LightGBM](https://github.com/microsoft/LightGBM).

The code implements [LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf), [LambdaLoss](https://dl.acm.org/doi/pdf/10.1145/3269206.3271784) (NDCG-Loss2 and NDCG-Loss2++) algorithms and all the combinations of **Lambda-eX** that make use of LambdaMART and LambdaLoss loss functions.

Usage
---

**Lambda-eX** is accessible through the ``lambdarank`` parameter ``lambda_ex`` (or ``lambdaex``) with the following value:
  - ``"plain"`` to enforce the original algorithm (no Lambda-eX) (default).
  - ``"static"`` to enforce Lambda-eX static.
  - ``"random"`` to enforce Lambda-eX random.
  - ``"all"`` to enforce Lambda-eX all.
  - ``"all-static"`` to enforce Lambda-eX all-static.
  - ``"all-random"`` to enforce Lambda-eX all-random.

Loss functions
---
The code implements three loss functions: LambdaRank, NDCG-Loss2 and NDCG-Loss2++. The three loss functions are accessible through the ``lambdarank`` parameters ``lambdarank_weight`` (or ``lr_mu``) and ``lambdaloss_weight`` (or ``ll_mu``), with the following combinations:
  - ``lambdarank_weight=1`` and ``lambdaloss_weight=0`` to enforce the LambdaRank loss function (default).
  - ``lambdarank_weight=0`` and ``lambdaloss_weight=1`` to enforce the NDCG-Loss2 loss function.
  - ``lambdarank_weight=1`` and ``lambdaloss_weight>0`` to enforce the NDCG-Loss2++ loss function.

Examples
---
 - for LambdaMART: ``objective="lambdarank"``.
 - for LambdaMART-eX random: ``objective="lambdarank"`` and ``lambda_ex="random"``.
 - for LambdaLoss-eX static with NDCG-Loss2++: ``objective="lambdarank"``, ``lambda_ex="static"``, and ``lambdaloss_weight=0.5``.
 - for LambdaLoss-eX all-random with NDCG-Loss2: ``objective="lambdarank"``, ``lambdaex="all-random"``, ``lr_mu=0`` and ``ll_mu=1``.

Installation
---
Follow the [installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) as mentioned in the LightGBM GitHub [repository](https://github.com/microsoft/LightGBM).
Where needed, replace the repository ``https://github.com/microsoft/LightGBM`` with this one.

Citation
---

```
@inproceedings{10.1145/3583780.3614948,
  author    = {Marcuzzi, Federico and Lucchese, Claudio and Orlando, Salvatore},
  title     = {LambdaRank Gradients Are Incoherent},
  year      = {2023},
  isbn      = {9798400701245},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3583780.3614948},
  doi       = {10.1145/3583780.3614948},
  abstract  = {In Information Retrieval (IR), the Learning-to-Rank (LTR) task requires building a ranking model that optimises a specific IR metric. One of the most effective approaches to do so is the well-known LambdaRank algorithm. LambdaRank uses gradient descent optimisation, and at its core, it defines approximate gradients, the so-called lambdas, for a non-differentiable IR metric. Intuitively, each lambda describes how much a document's score should be "pushed" up/down to reduce the ranking error.In this work, we show that lambdas may be incoherent w.r.t. the metric being optimised: e.g., a document with high relevance in the ground truth may receive a smaller gradient push than a document with lower relevance. This behaviour goes far beyond the expected degree of approximation. We analyse such behaviour of LambdaRank gradients and we introduce some strategies to reduce their incoherencies. We demonstrate through extensive experiments, conducted using publicly available datasets, that the proposed approach reduces the frequency of the incoherencies in LambdaRank and derivatives, and leads to models that achieve statistically significant improvements in the NDCG metric, without compromising the training efficiency.},
  booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages     = {1777–1786},
  numpages  = {10},
  keywords  = {lambdarank, information retrieval, learning to rank},
  location  = {Birmingham, United Kingdom},
  series    = {CIKM '23}
}
```
