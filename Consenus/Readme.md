# Consensus-Based Bayesian Anomaly Detection in Enterprise Directory Access Graphs

This project implements a novel framework for detecting anomalous user behavior in enterprise systems by modeling access patterns using consensus dynamics and Bayesian reasoning.

## 🔍 Problem Overview

In large organizations, users access files and directories collaboratively. Malicious behavior—such as a user accessing irrelevant or sensitive directories—can be subtle and difficult to detect with traditional rule-based systems.

## 🧠 Approach

We treat users as "topics" and directories as "agents" in an influence network. The framework is based on the paper:

> *Consensus and Disagreement of Heterogeneous Belief Systems in Influence Networks*  
> [arXiv:1812.05138](https://arxiv.org/abs/1812.05138)

Our approach applies consensus theorems to track how directory-level beliefs about users evolve over time. By observing deviations from expected convergence (i.e., consensus), we flag anomalous users.

### Key Components:
- **Opinion Dynamics:** Modeled using logical dependency matrices (`C_i`) and a global influence matrix (`W`)
- **Bayesian Interpretation:** Directory beliefs updated based on observed behavior vs expected group consensus
- **Anomaly Detection Criteria:**
  - Repeated non-convergence
  - Softmax-normalized opinion drift
  - Cross-directory inconsistencies

## 🛠️ Implementation

- Modular simulation pipeline in Python
- Theorems from opinion dynamics used to model convergence per topic
- Structural anomalies injected at specific time steps to test detection
- Consensus values tracked and compared across time and directories

## 📊 Results

- Correctly identifies anomalous user behavior based on divergence from group belief
- Explains anomalies with interpretable matrix dynamics
- Scalable to larger systems of users and directories

## 📁 Structure

-  ├── Consensus_Bayesian_Framework.pdf # Full paper detailing theory + implementation
-  ├── Opinion_dynamics_UAD.py # Simulation code (original paper + our extension) 
## 🔬 Future Work

- Apply to real-world access logs
- Extend to soft clustering of user roles
- Handle Scalability to Large Enterprises

## 📜 Citation

The theoretical foundation is based on the following seminal work:

> Mengbin Ye, Ji Liu, Lili Wang, Brian D. O. Anderson, and Ming Cao.  
> *Consensus and Disagreement of Heterogeneous Belief Systems in Influence Networks*.  
> IEEE Transactions on Automatic Control, 66(11):5266–5281, 2021.

```bibtex
@article{ye2021consensus,
  title={Consensus and disagreement of heterogeneous belief systems in influence networks},
  author={Ye, Mengbin and Liu, Ji and Wang, Lili and Anderson, Brian DO and Cao, Ming},
  journal={IEEE Transactions on Automatic Control},
  volume={66},
  number={11},
  pages={5266--5281},
  year={2021},
  publisher={IEEE}
}
---

📫 Contact: [pratyush.uppuluri@gmail.com](mailto:pratyush.uppuluri@gmail.com)

