# Neural Graph Collaborative Filtering Overview

- Model the high-order connectivity in the embedding function
- Embedding propagation layer, which refines a user’s (or an item’s) embedding by aggregating the embeddings of the interacted items (or users).
- explicitly encodes the collaborative signal in the form of high-order connectivities by performing embedding propagation
  ![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/16d3d1b4-93ae-47a1-8c01-500f6a816174)

- Many of improvements for eg: [LightGCN](https://arxiv.org/pdf/2002.02126) are achieved with marginal changes in implemnetaion in forwradpass/message aggregation
  - In LightGCN conctenation of emebedding layers is replaced with an aggreagtion and removal of non linear activations
  - Below equations show the main difference in learned embeddings
![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/9ee19d31-08f7-4579-9928-fd0d348279f7)
    
$$
\begin{align*}
\mathbf{e}_u^{NGCF} &= \left( \mathbf{e}_u^{(0)} \parallel \cdots \parallel \mathbf{e}_u^{(l)} \right), \; l \text{ number of layers to capture high order connectivity} \\
\mathbf{e}_u^{LightGCN} &= \sum \left( \mathbf{e}_u^{(0)} + \cdots + \mathbf{e}_u^{(l)} \right)
\end{align*}
$$
- These two methods are not developed for inductive setting ( i.e recoemmdnations for  a new user $$ \mathbf{e}_u^{(p) $$
    - Its intresting note since NGCF ocncatenates embedding of a user iwth intital to kth order , if we freeze initial embedding we can find dsiatnce between new uses with its intial mebing and rest of embeddings as zero and find closest user and get recoemmdnations
 

Neural Graph Recommendations using a Bipartite Graph
- In this Repo we will deep dive into one the most popualr techincuqe Neural Graph Collaborative Filtering ([NGCF](https://arxiv.org/pdf/1905.08108)) SIGIR (2019),
this Repo should help with conceptual understanding and implemenation of NGCF

![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/84cf6704-2142-4879-b37a-19f0d5d2bf04)

- This background may help in studying literature for further applications and improvements.
