## Reflex Diffing in Phi-4-mini
This exploratory project analysis reasoning behaviors ("reflexes") in Phi-4-mini-reasoning and its sibling from the same base Phi-4-min-instruct. 

> Do the two siblings models **share a representation** for the chosen reasoning reflex that only _one_ deploys — i.e. did reasoning training add a **causal trigger** to a direction the instruct sibling also encodes, or a **new representation**?

- **Reflex/Reasoning Behaviour**: the exact behavior (e.g. backtracking) is to be decided after exploratory analysis of generated rollouts (based on frequency and being specific to the reasoning model).
- **Model Pair**: Phi-4-mini-instruct vs. Phi-4-mini-reasoning. The are siblings from a shared (unreleased) base, same 3.8B architecture/tokenizer and therefore diffable in activation, output, and weight space.

_This project is a work in progress, and the reflex is still to be decided._

📓 [Research log](./LOG.md) summarizes the steps I already took and my preliminary results.

### Steps
**1. Baseline Verification** ✅
- generated rollouts and verifed baseline performance on MATH500 for reasoning and instruct.

**2. Reflex Analysis and Selection** (in progress)
- analyze reflexes and pick one
- build and validate labeler for the reflex

**3. Steering Vector Extraction and Layer Selection**
- build matched activation set from rollouts
- use difference-in-means to find direction
- train probe at each layer for validation and layer selection

**4. Cross Model Presence**
- teacher force reasoning rollouts through the instruct model
- validate probe transfer
- analyze cosine similarity to directions from reasoning model

**5. Output Space Comparison**
- compute per token KL divergence of next-token distributions for reasoning rollouts

**6. Reflex Recovery in Instruct Model (stretch goal)**
- steering/sentence boundaries/hybrid approach


### Related Work
- **Ward, Lin, Venhoff & Nanda (2025) — *Reasoning-Finetuning Repurposes Latent Representations in Base Models*** [arXiv:2507.12638](https://arxiv.org/pdf/2507.12638)
- **Venhoff, Arcuschin, Torr, Conmy & Nanda (2025) — *Understanding Reasoning in Thinking Language Models via Steering Vectors*** [arXiv:2506.18167](https://arxiv.org/pdf/2506.18167)
