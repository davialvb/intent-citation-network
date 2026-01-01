## Leveraging GANs for Citation Intent Classification and its Impact on Citation Network Analysis

This repository contains code and resources for the paper:
**“Leveraging GANs for Citation Intent Classification and Its Impact on Citation Network Analysis”**
*Davi A. Bezerra, Filipi N. Silva, Diego R. Amancio*

---

Citations are fundamental to the scientific ecosystem, enabling knowledge tracking, acknowledgment of prior work, and assessment of scholarly influence. However, not all citations serve the same function (e.g., providing background, introducing methods, or comparing results). Understanding citation intent allows for a more nuanced interpretation of scientific impact. In this paper, we adopted a GAN-based method to classify citation intents. Our results reveal that the proposed method achieves competitive classification performance, closely matching state-of-the-art results with substantially fewer parameters. This demonstrates the effectiveness and efficiency of leveraging GAN architectures combined with contextual embeddings in the intent classification task. We also investigated whether filtering citation intents affects the centrality of papers in citation networks. Analyzing the network constructed from the unArXiv dataset, we found that paper rankings can be significantly influenced by citation intent. All four centrality metrics examined (degree, PageRank, closeness, and betweenness) were sensitive to the filtering of citation types. The betweenness centrality displayed the greatest sensitivity, showing substantial changes in ranking when specific citation intents were removed.

---
## Datasets

### Citation Intent Classification:

* **SciCite**: 3-class dataset (Background, Method, Result). Used for training.
* **ACL-ARC**: 1,941 instances from NLP papers across 6 intent classes.
* **3C Shared Task**: Subset of ACT dataset with 6 citation categories.

### Citation Network Analysis:

* **unarXiv**: >1.8M articles across disciplines (CS, physics, math) used to build citation networks.

---

## Citation Intent Classification

* **Model**: Conditional GAN-BERT with SciBERT embeddings.
* **Architecture**:

  * *Generator (G\_c)*: Produces synthetic citation samples.
  * *Discriminator (D)*: Classifies real vs. generated and labeled vs. unlabeled samples.
* **Training**: Combines labeled/unlabeled data in an adversarial process.
* **Inference**: Uses only the fine-tuned SciBERT and discriminator.

## Citation Network Analysis

* **Filtering**: Citations are labeled (Background, Method, Result) using the trained model.
* **Network Construction**: Directed graph $G = (V, E)$, with papers as nodes and citations as edges.
* **Analysis**: Centrality metrics (Degree, PageRank, Closeness, Betweenness) measured with and without intent filtering.
* **Visualization**: Networks visualized using [Helios-Web](https://github.com/jfdelnero/helios-web).

---

## Results

### Classification Performance:

| Dataset | F1 Score                           | Notes                                         |
| ------- | ---------------------------------- | --------------------------------------------- |
| SciCite | **88.74%**                         | Comparable to SOTA with ⅓ of the parameters   |
| ACL-ARC | 81.75%                             | Outperforms CitePrompt and fine-tuned SciBERT |
| 3C      | 26.22% (public) / 23.21% (private) | Reflects dataset complexity                   |

### Citation Network Impact:

* **Removing Background Citations**:

  * ↓ 51% nodes, ↓ 62% edges
  * ↑ 567% in disconnected components
  * Significant drop in cohesion and centrality shift

* **Removing Method Citations**:

  * Moderate structural effect (↓7.5% nodes, ↑50% fragmentation)

* **Removing Result Citations**:

  * Minimal impact (<1% change in structure)

Citation intent significantly affects network metrics.

---

## Repository Structure

```
.
├── data/              # Citation datasets
├── models/            # Trained model weights/configs
├── src/                       # Source code
│   ├── classification/
│   ├── network_analysis/
│   ├── utils/
│   └── gan_bert_src/          # GAN-BERT implementation
│       ├── gan_bert/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── data.py
│       │   ├── losses.py
│       │   ├── models.py
│       │   ├── train_eval.py
│       │   └── utils.py
│       │
│       ├── cli_train.py       # Train GAN-BERT
│       ├── cli_optuna.py      # Hyperparameter search
│       ├── cli_tsne.py        # Representation analysis
│       ├── requirements.txt
│       └── README.md
├── notebooks/         # Jupyter notebooks
├── results/           # Output figures, tables, metrics
├── LICENSE
└── README.md
```

---
## Model Weights

Pre-trained model weights for **cGAN-SciBERT** are available for download:

* **Download Link**: [cGAN-SciBERT Weights (Google Drive)](https://drive.google.com/tmp-link)

After downloading, place the model files in the `models/` directory:

```
intent-citation-network/
├── models/
│   └── cgan_scibert/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer/
```
---


## Installation & Usage

1. **Clone the repo**:

```bash
git clone https://github.com/your-username/intent-citation-classification.git
cd intent-citation-classification
```

2. **Set up environment**:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
uv install
```

4. **Run the model or analysis**:
   Instructions and examples coming soon.

---


If you use this work, please cite:

```bibtex
@article{bezerra2025leveraging,
  title={Leveraging GANs for citation intent classification and its impact on citation network analysis},
  author={Bezerra, Davi A and Silva, Filipi N and Amancio, Diego R},
  journal={arXiv preprint arXiv:2505.21162},
  year={2025}
}
```