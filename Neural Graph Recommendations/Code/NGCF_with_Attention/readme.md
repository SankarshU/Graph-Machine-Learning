# Attention-Augmented NGCF

**Status**: 🟡 Under Review at RecSys'25 (Late-Breaking Results Track)  
**Title**: Attention-Augmented NGCF: Personalized Graph Recommendation via Learnable Neighbor Weighting  
**Code**: [NGCF with Attention (PyTorch)](https://github.com/SankarshU/Graph-Machine-Learning/tree/d432f63f098f98a672f591593c8b682de2c395cb/Neural%20Graph%20Recommendations/Code/NGCF_with_Attention)

---

## 🔧 Getting Started

Follow these steps to run the **Attention-Augmented NGCF** model:

1. **Download the Dataset**  
   - Grab the [MovieLens-100K dataset](https://grouplens.org/datasets/movielens/100k/)
   - Extract the files and place them under the `data/` directory in this repo

2. **Run the Model**  
   Launch training and evaluation with:

   ```bash
   python ngcf_with_attn.py



### NGCF with Attention-Augmented Message Passing Summary

#### Model Formulation: Neural Graph Collaborative Filtering (NGCF)

We consider a bipartite graph $G_r = (U, I, E)$, where:

- $U = \{u_1, \dots, u_N\}$ is the set of users,
- $I = \{i_1, \dots, i_M\}$ is the set of items,
- and edges $E \subseteq U \times I$ represent observed user-item interactions.

Each node (user or item) is associated with an embedding vector:

$$
\mathbf{e}_u, \mathbf{e}_i \in \mathbb{R}^d
$$

These embeddings are either initialized randomly or using provided side attributes.


#### Implementation Highlights and Limitations
- Code is available 
#### Efficient PyTorch Aggregation

- Uses `index_select` and `torch.matmul`
- Softmax over attention scores
- `torch.bmm()` for batched weighted sum

#### Limitations

- Fixed number of neighbors
- Padding wastes compute
- No dynamic masking


#### Validation Results Table

| Initialization        | Frozen | No Graph Attention | Graph Attention |
|-----------------------|--------|--------------------|-----------------|
| With Attributes       | Yes    | 0.147              | 0.143           |
| With Attributes       | No     | 0.216              | 0.216           |
| Random Initialization | Yes    | 0.238              | 0.223           |
| Random Initialization | No     | 0.231              | 0.236           |

Values are rounded to 3 decimal places.
