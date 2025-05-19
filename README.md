# LLM/NLP Analysis of Cognitive States using ZuCo Dataset

**Project Version:** 1.0.0
**Last Updated:** May 19, 2025
**Author/Team:** [Your Name/Team Name]

## Table of Contents

- [LLM/NLP Analysis of Cognitive States using ZuCo Dataset](#llmnlp-analysis-of-cognitive-states-using-zuco-dataset)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Objective](#2-objective)
  - [3. Relevance](#3-relevance)
  - [4. Dataset Used](#4-dataset-used)
  - [5. Methodology](#5-methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Engineering](#feature-engineering)
    - [Model Training and Evaluation](#model-training-and-evaluation)
    - [Advanced Model Improvement](#advanced-model-improvement)
  - [6. Key Results](#6-key-results)
  - [7. Setup and Installation](#7-setup-and-installation)
  - [8. Directory Structure](#8-directory-structure)
  - [9. Usage](#9-usage)
  - [10. Challenges Encountered](#10-challenges-encountered)
  - [11. Future Work and Improvements](#11-future-work-and-improvements)
  - [12. References](#12-references)
  - [13. License](#13-license)

## 1. Project Overview

This project explores the application of Natural Language Processing (NLP) and Large Language Model (LLM) techniques to differentiate between cognitive states, specifically Normal Reading (NR) and Task-Specific Reading (TSR). The analysis is performed on a subset of the ZuCo dataset, focusing on linguistic characteristics of sentences read under these two conditions. The project progresses from baseline machine learning models with traditional NLP features to advanced techniques involving fine-tuned transformer embeddings and more complex neural network architectures.

## 2. Objective

The primary objective is to investigate and identify linguistic features and model architectures that can accurately classify sentences based on the cognitive state (NR or TSR) associated with their reading. This involves comparing various feature engineering strategies and machine learning models to determine the most effective approach for this classification task.

## 3. Relevance

Understanding the linguistic markers of different cognitive states holds significant value for cognitive science by providing insights into how language processing varies with cognitive load and task demands. In NLP, this research can contribute to developing more context-aware language understanding systems. Potential applications include adaptive educational tools, diagnostic aids for reading comprehension, and enhanced human-computer interaction.

## 4. Dataset Used

The project utilizes sentence data from the ZuCo dataset, specifically from CSV files corresponding to Normal Reading (NR) and Task-Specific Reading (TSR) tasks. Sentences were extracted from files named `nr_*.csv` and `tsr_*.csv` located in the `task_materials/` directory.

## 5. Methodology

### Data Preprocessing

1. **Loading:** Sentences from NR and TSR CSV files were loaded.
2. **Cleaning:** Text was converted to lowercase, and extra whitespace was removed.
3. **Uniqueness & Overlap Handling:** Unique sentences for each condition were identified. Sentences common to both NR and TSR unique lists (61 sentences) were removed to ensure distinct datasets for classification.
4. **Label Encoding:** NR was encoded as 0, and TSR as 1.
5. **Final Dataset:** The processed dataset (`zuco_processed_sentences.csv`) contains 635 unique sentences (304 NR, 331 TSR).

### Feature Engineering

Three main feature sets were developed:

1. **Sentence Embeddings (Off-the-shelf):** `all-MiniLM-L6-v2` model was used to generate 384-dimensional embeddings (`sentence_embeddings.npy`).
2. **Discrete Linguistic Features:**
    - **Base Discrete:** Character/word counts, average word length, Type-Token Ratio (TTR), lexical density proxy, and readability scores (Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog) (`base_discrete_features.csv`).
    - **Enhanced Discrete:** Base features augmented with spaCy-derived syntactic features (clause counts, dependency distances, POS tag counts) (`enhanced_discrete_features.csv`).
    - **(Experimental) LLM Metrics:** An `ollama_llm_rating` (1-5 complexity) was added (simulated in the primary notebook run).
3. **Fine-tuned Transformer Embeddings:** A `bert-base-uncased` model was fine-tuned on the NR/TSR classification task using 5-fold cross-validation. The best fold's model was used to extract 768-dimensional embeddings.
4. **Combined Features:** Fine-tuned BERT embeddings were concatenated with scaled enhanced discrete features, resulting in 787-dimensional vectors for the advanced MLP model.

### Model Training and Evaluation

- **Baseline Models:** Logistic Regression, Decision Tree, Random Forest, LightGBM, SVM, and a scikit-learn MLP were evaluated on off-the-shelf embeddings and discrete feature sets. Data was split 80/20 (train/test), with discrete features scaled using `StandardScaler` where appropriate.

- **Fine-tuned BERT:** Trained using 5-fold StratifiedKFold CV with early stopping.
- **Advanced MLP (PyTorch):** A custom MLP (Input -> 512 -> 128 -> 2) with ReLU, BatchNorm, and Dropout was trained on the combined fine-tuned embeddings and scaled enhanced discrete features.
- **Evaluation Metrics:** Primary metric was F1-score for the TSR class. Accuracy, macro F1, and confusion matrices were also used.

### Advanced Model Improvement

The strategy focused on improving representation learning via fine-tuned transformer embeddings and enhancing model architecture by combining these with engineered features in an advanced MLP.

## 6. Key Results

- **Baseline Models:** Random Forest on enhanced discrete features initially performed best (F1-TSR: 0.7445).

- **Fine-tuned BERT (CV):** Achieved a mean F1-TSR of 0.7434, with the best fold reaching 0.7939.
- **Advanced MLP (Test Set):** The PyTorch MLP trained on combined fine-tuned BERT embeddings and scaled enhanced discrete features yielded the highest performance:
  - **F1-score (TSR class): 0.9474**
  - Accuracy: 0.9449
  - F1-score (Macro): 0.9448

## 7. Setup and Installation

This project was developed using Python 3.12. Key libraries are listed in `llm_analysis2.ipynb` and include:

```

pandas
numpy
glob
os
re
time
matplotlib
seaborn
nltk
scikit-learn
sentence-transformers
textstat
spacy
torch
transformers
datasets
lightgbm

# ollama (for experimental LLM metrics)

````

It's recommended to set up a virtual environment:

```bash
python -m venv zuco_env
source zuco_env/bin/activate  # On Windows: zuco_env\Scripts\activate
pip install -r requirements.txt # Assuming you create a requirements.txt file
````

Download necessary NLTK and spaCy resources:

```python
import nltk
import spacy
nltk.download('punkt', quiet=True)
# If en_core_web_sm is not present:
# python -m spacy download en_core_web_sm
```

For the experimental Ollama features, ensure the Ollama server is running and the specified model (e.g., `llama3.2`) is pulled.

## 8\. Directory Structure

```
.
├── llm_analysis2.ipynb                 # Main Jupyter Notebook with the analysis
├── task_materials/                     # Directory containing the input CSV files (nr_*.csv, tsr_*.csv)
│   ├── nr_*.csv
│   └── tsr_*.csv
├── zuco_processed_sentences.csv        # Output: Processed sentences and labels
├── sentence_embeddings.npy             # Output: Off-the-shelf sentence embeddings
├── base_discrete_features.csv          # Output: Base discrete linguistic features
├── enhanced_discrete_features.csv      # Output: Enhanced discrete features (including spaCy and simulated Ollama)
├── model_performance_summary.csv       # Output: Summary of baseline model performances
├── final_model_performance_summary.csv # Output: Summary including advanced MLP performance
├── best_mlp_combined_features_ZuCo.bin # Output: Saved weights for the best PyTorch MLP model
├── sentence_length_distributions.png   # Output: EDA plot
├── sentence_length_boxplots.png        # Output: EDA plot
└── README.md                           # This file
```

## 9\. Usage

1. Ensure all dependencies are installed (see [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)).
2. Place the ZuCo dataset CSV files (e.g., `nr_S1.csv`, `tsr_S1.csv`) into the `task_materials/` directory.
3. Run the `llm_analysis2.ipynb` notebook cell by cell.
      - Note: The BERT fine-tuning (Section 5.1.3) and actual Ollama calls (Section 3.3.3, if `OLLAMA_ENABLED` is set to `True` and simulation block is removed) can be very time-consuming. The notebook is currently set to use simulated Ollama ratings for speed.
4. Outputs, including processed data, features, model performance summaries, and saved models/plots, will be generated in the root directory.

## 10\. Challenges Encountered

- **Computational Resources:** Fine-tuning transformers and LLM experiments are resource-intensive.
- **Feature Engineering:** Extracting comprehensive discrete features is complex.
- **Data Leakage Potential:** The fine-tuned BERT embeddings for the final MLP were derived from a BERT model fine-tuned via CV on the entire dataset. This means the final MLP test set was indirectly "seen" by the BERT model, potentially inflating metrics. A stricter hold-out set from the very beginning is recommended for future iterations.
- **Hyperparameter Tuning:** Exhaustive tuning was not the primary focus.

## 11\. Future Work and Improvements

- Implement a strict, initially separated test set for final evaluation.
- Conduct advanced hyperparameter optimization for BERT fine-tuning and the MLP.
- Explore alternative transformer architectures (e.g., RoBERTa, DeBERTa).
- Investigate sophisticated feature combination strategies (e.g., attention mechanisms).
- Perform full integration and evaluation of actual Ollama-based LLM complexity metrics.
- Conduct a deeper statistical analysis of the most discriminative linguistic features.

## 12\. References

1. Eda AYDIN. (2025). *LLM/NLP Analysis of Cognitive States using ZuCo Dataset* (Unpublished Jupyter Notebook `llm_analysis2.ipynb`).
2. Hollenstein, N., Troendle, M., Langer, N., & Zhang, C. (2021). *ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Task-Specific Reading*. OSF. Retrieved from <https://osf.io/2urht/>
3. Yao, S., Zhao, J., Yu, D., Liu, Z., & Sha, L. (2022). Exploring the Relationship Between Eye Movements and Cognitive Workload in Natural Reading. *Frontiers in Psychology*, *13*, 1028824. <https://www.google.com/search?q=https://doi.org/10.3389/fpsyg.2022.1028824>
4. Hugging Face Transformers. (2025). Retrieved from [https://huggingface.co/transformers](https://huggingface.co/transformers)
5. Sentence Transformers. (2025). Retrieved from [https://www.sbert.net](https://www.sbert.net)
6. spaCy. (2025). Retrieved from [https://spacy.io](https://spacy.io)
7. PyTorch. (2025). Retrieved from [https://pytorch.org](https://pytorch.org)
8. Scikit-learn. (2025). Retrieved from [https://scikit-learn.org](https://scikit-learn.org)
9. Ollama. (2025). Retrieved from [https://ollama.com](https://ollama.com)

## 13\. License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.