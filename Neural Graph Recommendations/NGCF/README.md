# Neural Graph Collaborative Filtering Overview

- Model the high-order connectivity in the embedding function
- Embedding propagation layer, which refines a user’s (u) (or an item’s(i)) embedding by aggregating the embeddings of the interacted items (or users).
- explicitly encodes the collaborative signal in the form of high-order connectivities by performing embedding propagation
  ![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/16d3d1b4-93ae-47a1-8c01-500f6a816174)
  ![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/9ee19d31-08f7-4579-9928-fd0d348279f7)
- Many of improvements for eg: [LightGCN](https://arxiv.org/pdf/2002.02126) are achieved with marginal changes in implemnetaion in forwradpass/message aggregation
  - In LightGCN conctenation of emebedding layers is replaced with an aggreagtion and removal of non linear activations
  - Below equations show the main difference in learned embeddings
  
$$
\begin{align*}
\mathbf{e}_u^{NGCF} &= \left( \mathbf{e}_u^{(0)} \parallel \cdots \parallel \mathbf{e}_u^{(l)} \right), \; l \text{ number of layers to capture high order connectivity} \\
\mathbf{e}_u^{LightGCN} &= \sum \left( \mathbf{e}_u^{(0)} + \cdots + \mathbf{e}_u^{(l)} \right)
\end{align*}
$$
- These two methods are not developed for inductive setting ( i.e recoemmdnations for  a new user $$ \mathbf{e}_u^{(p) $$
    - simple trick to adapt NGCF for inductive setting
    - Its intresting note since NGCF concateantes embedding of a user from  intital embedding to kth order , if we freeze initial embedding weights and have it initialzed with input features
      -  we can find distance between new users embedding and rest of embedding in concatenation as zero and find closest user and get recommendations
 

How spectral convolution GCN by Kipg et al. in a seminal [work](https://arxiv.org/abs/1609.02907), a single expression made hughe difference for development of practical appliction
- This expression enables a matrix compuattion using Adajcency and Daignot matricises of graph ( one time constrcution) and mesage propgation along alyers acn achieved by recurisvely multiplying the mebddings at ecah layer with this matrix and subseuqent tarsnformations
- Please look at code where thsi explained in deatil
  ![image](https://github.com/SankarshU/Graph-Machine-Learning/assets/44226862/ff528988-4bac-4b51-9504-940ff9812803)
- Inspite of thsi elgenace the ecpressive nature of  GCN is not among best
- Intrest rserach on expressive nature is conducted , please refer to thsi [work](https://drive.google.com/file/d/1sR4rDMjRf_WcyPM6IfYiQHnaYw4a0bsq/view) and [tutorila](https://www.youtube.com/watch?v=ASQYjbUBYzs&list=PL2iNJC54likoqgKwpFnbBik8Im1sZ27Hm&index=7 ) for same 


