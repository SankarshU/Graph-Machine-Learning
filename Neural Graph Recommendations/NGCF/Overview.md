# Neural Graph Collaborative Filtering Overview
## Salient aspects
- Model the high-order connectivity in the embedding function
- Embedding propagation layer, which refines a user’s (u) (or an item’s(i)) embedding by aggregating the embeddings of the interacted items (or users).
- Explicitly encodes the collaborative signal in the form of high-order connectivities by performing embedding propagation as shown in below figures
  
![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/ec67e516-5638-4233-aa37-67515d110868) ![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/3fb35a0f-1b3d-44dc-8afa-6ce10308c690)


 
- Many of improvements over NGCF for eg one of the popularly benchmarked: [LightGCN](https://arxiv.org/pdf/2002.02126) is  achieved with marginal changes in implementation in forwradpass/message aggregation
  - In LightGCN, the concatenation of embedding layers from NGCF is replaced with weighted aggregation, and non-linear activations are removed.
  - Below equations show the main difference in learned embeddings
  
#### NGCF Equation

$$
\mathbf{e}_u^{NGCF} = \left( \mathbf{e}_u^{(0)} \parallel \cdots \parallel \mathbf{e}_u^{(l)} \right), \; l \text{ number of layers to capture high order connectivity}
$$

#### LightGCN Equation

$$ 
\mathbf{e}_u^{LightGCN} = \left(\sum_{k=0}^{K} \alpha_k \mathbf{e}^{(k)} \right) 
$$

### 
- These two methods are not developed for inductive setting ( i.e recoemmdnations for  a new user $$ \mathbf{e}_u^{(p) $$
    - simple trick to adapt NGCF for inductive setting
    - Its intresting note since NGCF concateantes embedding of a user from  intital embedding to kth order , if we freeze initial embedding weights and have it initialzed with input features
      -  we can find distance between new users embedding and rest of embedding in concatenation as zero and find closest user and get recommendations
 
### Key GNN Idea behind !!!!
- Derivation of spectral convolution GCN by Kipf et al (ICLR,2017). in a seminal [work](https://arxiv.org/abs/1609.02907), a single expression made hughe difference for development of practical appliction,the popular Normalized Laplacian Matrix is as below; A  is Adjacency Matrix and D is Degree Matrix.

$$
g_{\Theta} \ast \mathbf{x} = \Theta \left( \mathbf{I}_N + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \right) \ast \mathbf{x}
$$ 
- The above expression enables matrix computation using the adjacency and degree matrices of a graph (one-time construction). Message propagation along layers can be achieved by recursively multiplying the embeddings at each layer with this matrix and subsequent transformations.
- In spite of the elegance of GCN, its expressive nature is not among the best. Research on the expressive nature of GNNs is quite interesting falling abck ona spects from graph theory . Please refer to this.
 [work](https://drive.google.com/file/d/1sR4rDMjRf_WcyPM6IfYiQHnaYw4a0bsq/view) and [tutorial](https://www.youtube.com/watch?v=ASQYjbUBYzs&list=PL2iNJC54likoqgKwpFnbBik8Im1sZ27Hm&index=7 ) for more details 


