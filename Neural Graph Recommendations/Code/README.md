### Notebook demonstrates NGCF implementation and simple illustration with Movielens data set
## NGCF
- This Notebook concisely captures NGCF in PyTorch in a notebook for quick experimentation and understanding theory (formulation) and its implemnation .
- Idea is novel methods can be derived from this material
- MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.

- Movielens data set consists of:
	- 100,000 ratings (1-5) from 943 users on 1682 movies.
	- Each user has rated at least 20 movies.
        - Simple demographic info for the users (age, gender, occupation, zip).

- Illustration is just to compare different scenarios of providing input features.
- A possible naive way to make NGCF suitable for an inductive setting (i.e., new users with attributes).
- This is just to illustrate the simplicity of code vis-a-vis formulation (message construction and passing) with code.
  - For example: we could include attention in message construction, and the same can be changed in the forward pass.
  - I was experimenting with modifying the Laplacian matrix to only introduce modifications at the inference level.

## LightGCN and GACSE ( includes attention) : Incremental changes over  NGCF in concept that requires minor changes in Code 
- For comparision to notice incremental changes, Have include a note book with a class from GACSE original implementation [here](https://github.com/AnonymousResearchLab/BipartiteGraphModel/blob/master/BipartiteGraphModels/GACSE.py) , thats includes attention and it can be obesrved it can be achieved with simple changes to forwrad  pass of NGCF
- Please note that LightGCN, which is among the benchmarks for recommendation systems, is a simple modification of NGCF

### Sourse Githubs
- Original implemnations are not required just from publications and code from NGCF  GACSE and  LightGCN can be implemented with minimal changees
- [GACSE](https://github.com/AnonymousResearchLab/BipartiteGraphModel/blob/master/BipartiteGraphModels/GACSE.py)
- [LightGCN](https://arxiv.org/abs/2002.02126)
- [NGCF](https://github.com/huangtinglin/NGCF-PyTorch)


    


