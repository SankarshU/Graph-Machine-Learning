# Attention-Augmented NGCF

**Paper**: _Under Review at RecSys'25 (Late-Breaking Results Track)_  
**Title**: Attention-Augmented NGCF: Personalized Graph Recommendation via Learnable Neighbor Weighting  
**Status**: 🟡 Under Review  
**Code**: [NGCF with Attention (PyTorch)](https://github.com/SankarshU/Graph-Machine-Learning/tree/831afef3c7703450b8fb556a05947027fc27963e/Neural%20Graph%20Recommendations/Code/NGCF_with_Attention)  
**Manuscript**: Submitted version available [here (PDF)](https://github.com/SankarshU/Graph-Machine-Learning/blob/60c690296f7dab575266d96532143eb0baefff9d/Neural%20Graph%20Recommendations/NGCF/recsys_lbr_ngcf_with_attn.pdf)



## 📌 Overview

This project introduces a lightweight attention mechanism into the classical NGCF model, allowing node-specific, learnable neighbor weighting in user-item recommendation graphs.

Key contributions:
- Augment NGCF with attention-based message passing.
- Learn dynamic neighbor importance for personalized aggregation.
- Preserve NGCF’s scalability while enhancing performance in sparse settings.
- Compatible with simplified models like **LightGCN**, and evaluated for benchmarking.

## 🚀 Method

We extend the NGCF propagation mechanism by replacing fixed normalization with MLP-learned attention weights. Each node aggregates neighbor messages using softmax-weighted scores, enabling expressive and personalized updates.

> The same mechanism is easily adaptable to **LightGCN**, allowing evaluation against cutting-edge methods like [**LightGCL**](https://github.com/kuandeng/LightGCL).

## 🔬 Benchmark Plan

We plan to benchmark **Attention-Augmented LightGCN** against state-of-the-art methods:

- ✅ Classical Baseline: LightGCN  
- 🚧 Proposed Variant: LightGCN + Learnable Attention  
- 🏁 Target SOTA: [**LightGCL**](https://github.com/kuandeng/LightGCL) (Simple Graph Contrastive Learning)

## 🧪 Experiments

- **Dataset**: [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/)
- **Metric**: NDCG@10
- **Embeddings**: Four initialization strategies tested

| Init. Strategy          | NGCF (No Attention) | NGCF (With Attention) |
|------------------------|---------------------|------------------------|
| Attr. Init. + Frozen   | 0.147               | 0.143                  |
| Attr. Init. + Updated  | 0.216               | 0.216                  |
| Rand. Init. + Frozen   | 0.238               | 0.223                  |
| Rand. Init. + Updated  | 0.231               | **0.236**              |

⚡ Results show attention boosts performance when embeddings are trainable.

## 🛠 Implementation

- Modular PyTorch code for NGCF + attention.
- Batched operations for GPU efficiency.
- Supports user/item attributes via one-hot encoding.

## ⚠️ Limitations

- Evaluation currently limited to MovieLens-100K.
- Fixed-size neighbor sampling may limit flexibility.
- Attention shows less gain when embeddings are frozen.

## 🔮 Future Work

- Scale to MovieLens-1M, Amazon, Yelp datasets.
- Integrate attention with **LightGCN**.
- Benchmark against [LightGCL](https://github.com/kuandeng/LightGCL) for contrastive learning performance.
- Explore multi-head attention, dynamic masking, and type-aware aggregation.

## 📚 Citation (Preliminary)

