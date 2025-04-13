
# NGCF with Attention-Augmented Message Passing

## Model Formulation: Neural Graph Collaborative Filtering (NGCF)

We consider a bipartite graph $G_r = (U, I, E)$, where:

- $U = \{u_1, \dots, u_N\}$ is the set of users,
- $I = \{i_1, \dots, i_M\}$ is the set of items,
- and edges $E \subseteq U \times I$ represent observed user-item interactions.

Each node (user or item) is associated with an embedding vector:

$$
\mathbf{e}_u, \mathbf{e}_i \in \mathbb{R}^d
$$

These embeddings are either initialized randomly or using provided side attributes.

### Message Construction and Propagation

At layer \( l \), the embedding of user \( u \) is:

To update the user embedding at layer \( l \), we use:

$$
\mathbf{e}_u^{(l)} = \sigma\left( \mathbf{m}_{u \leftarrow u}^{(l)} + \sum_{i \in \mathcal{N}_u} \mathbf{m}_{u \leftarrow i}^{(l)} \right)
$$

Where:
- \( \mathbf{m}_{u \leftarrow i}^{(l)} \) is the message from item \( i \) to user \( u \),
- \( \mathcal{N}_u \) is the set of neighbors of user \( u \),
- \( \sigma \) is an activation function (e.g., LeakyReLU).


The message from item \( i \) to user \( u \) is:

$$
\mathbf{m}_{u \leftarrow i}^{(l)} = \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \left( \mathbf{W}_1^{(l)} \mathbf{e}_i^{(l-1)} + \mathbf{W}_2^{(l)} (\mathbf{e}_i^{(l-1)} \odot \mathbf{e}_u^{(l-1)}) \right)
$$

The self-message:

$$
\mathbf{m}_{u \leftarrow u}^{(l)} = \mathbf{W}_1^{(l)} \mathbf{e}_u^{(l-1)}
$$

### Matrix Propagation

$$
\mathbf{E}^{(l)} = \sigma\left( (\mathbf{L} + \mathbf{I}) \mathbf{E}^{(l-1)} \mathbf{W}_1^{(l)} + (\mathbf{L} \mathbf{E}^{(l-1)} \odot \mathbf{E}^{(l-1)}) \mathbf{W}_2^{(l)} \right)
$$

Where \( \mathbf{L} = \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \) is the normalized adjacency matrix.

## Attention-Augmented Message Construction (Our Work)

We enhance NGCF with an attention mechanism.

### User-Side Aggregation with Attention

$$
\alpha_{u,i}^{(l)} = \frac{\exp\left( \mathbf{v}_a^\top \tanh\left( \mathbf{W}_a [\mathbf{e}_u^{(l-1)} \parallel \mathbf{e}_i^{(l-1)}] \right) \right)}
{\sum_{j \in \mathcal{N}_u} \exp\left( \mathbf{v}_a^\top \tanh\left( \mathbf{W}_a [\mathbf{e}_u^{(l-1)} \parallel \mathbf{e}_j^{(l-1)}] \right) \right)}
$$

$$
\mathbf{e}_u^{(l)} = \sigma\left( \sum_{i \in \mathcal{N}_u} \alpha_{u,i}^{(l)} \mathbf{e}_i^{(l-1)} \right)
$$

### Item-Side Aggregation with Attention

$$
\alpha_{i,u}^{(l)} = \frac{\exp\left( \mathbf{v}_a^\top \tanh\left( \mathbf{W}_a [\mathbf{e}_i^{(l-1)} \parallel \mathbf{e}_u^{(l-1)}] \right) \right)}
{\sum_{j \in \mathcal{N}_i} \exp\left( \mathbf{v}_a^\top \tanh\left( \mathbf{W}_a [\mathbf{e}_i^{(l-1)} \parallel \mathbf{e}_j^{(l-1)}] \right) \right)}
$$

$$
\mathbf{e}_i^{(l)} = \sigma\left( \sum_{u \in \mathcal{N}_i} \alpha_{i,u}^{(l)} \mathbf{e}_u^{(l-1)} \right)
$$

## Implementation Highlights and Limitations

### Efficient PyTorch Aggregation

- Uses `index_select` and `torch.matmul`
- Softmax over attention scores
- `torch.bmm()` for batched weighted sum

### Limitations

- Fixed number of neighbors
- Padding wastes compute
- No dynamic masking

## Attention Mechanism (General Form)

$$
\text{Attention}(q, k, v) = \text{softmax} \left( \frac{q k^T}{\sqrt{d_k}} \right) v
$$

GAT-style:

$$
\alpha_{ij} = \frac{\exp\left( \text{LeakyReLU}\left( \vec{a}^\top [\mathbf{W} \vec{h}_i \parallel \mathbf{W} \vec{h}_j] \right) \right)}
{\sum_{k \in \mathcal{N}(i)} \exp\left( \text{LeakyReLU}\left( \vec{a}^\top [\mathbf{W} \vec{h}_i \parallel \mathbf{W} \vec{h}_k] \right) \right)}
$$

## Conclusion

This variant of NGCF incorporates attention to improve node-level relevance modeling.


## Validation Results Table

| Initialization        | Frozen | No Graph Attention | Graph Attention |
|-----------------------|--------|--------------------|-----------------|
| With Attributes       | Yes    | 0.147              | 0.143           |
| With Attributes       | No     | 0.216              | 0.216           |
| Random Initialization | Yes    | 0.238              | 0.223           |
| Random Initialization | No     | 0.231              | 0.236           |

Values are rounded to 3 decimal places.
