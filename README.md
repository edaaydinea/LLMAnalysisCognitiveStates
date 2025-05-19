# LLM/NLP Analysis of Cognitive States using ZuCo Dataset

## Overview

This project analyzes cognitive reading states (Normal Reading - NR vs. Task-Specific Reading - TSR) using the ZuCo dataset. It leverages both traditional NLP features and advanced transformer-based embeddings to distinguish between these states, combining exploratory data analysis, feature engineering, and state-of-the-art machine learning models.

## Project Structure

- **Data Loading & Preprocessing:**
  - Loads and cleans sentence data from ZuCo task files.
  - Handles missing values, duplicates, and encodes labels (NR=0, TSR=1).
  - Saves processed data for reproducibility.
- **Exploratory Data Analysis (EDA):**
  - Analyzes sentence length distributions (characters, words).
  - Visualizes and summarizes key statistics by reading condition.
- **Feature Engineering:**
  - **Option A:** Sentence embeddings using pre-trained transformer models (e.g., MiniLM, BERT).
  - **Option B:** Discrete linguistic features (text statistics, lexical, readability, syntactic features via spaCy).
  - **Experimental:** Local LLM-based complexity ratings (Ollama integration).
  - Features are saved for efficient reuse.
- **Model Training & Evaluation:**
  - Trains and compares multiple classifiers (Logistic Regression, Decision Tree, Random Forest, LightGBM, SVM, MLP) on different feature sets.
  - Uses stratified train/test splits and cross-validation for robust evaluation.
  - Compiles and saves detailed performance summaries (accuracy, F1-scores, confusion matrices).
- **Advanced Modeling:**
  - Fine-tunes transformer models (e.g., BERT) for sentence classification using PyTorch and Hugging Face Transformers.
  - Combines fine-tuned embeddings with engineered features in a custom MLP for improved performance.
  - Implements best practices: early stopping, learning rate scheduling, gradient clipping, and model checkpointing.

## Key Files

- `llm_analysis2.ipynb`: Main analysis notebook with all code, explanations, and results.
- `zuco_processed_sentences.csv`: Cleaned and labeled sentence data.
- `sentence_embeddings.npy`: Pre-computed sentence embeddings.
- `base_discrete_features.csv`, `enhanced_discrete_features.csv`: Engineered feature sets.
- `model_performance_summary.csv`, `final_model_performance_summary.csv`: Model evaluation results.
- `best_mlp_combined_features_ZuCo.bin`: Saved weights for the best-performing MLP model.

## How to Run

1. **Install Requirements:**
   - Python 3.8+
   - Install dependencies:

     ```bash
     pip install pandas numpy scikit-learn matplotlib seaborn nltk sentence-transformers textstat spacy lightgbm torch transformers datasets
     python -m spacy download en_core_web_sm
     ```

   - (Optional) For LLM-based complexity: `pip install ollama`
2. **Prepare Data:**
   - Place ZuCo task files in the `task_materials/` directory.
   - Run the notebook to process data, extract features, and train models.
3. **Reproduce Results:**
   - All intermediate and final results are saved as CSV or binary files for reproducibility.
   - To fine-tune BERT or run advanced MLP, ensure GPU support (e.g., Google Colab).

## Main Results

- **Best Model:**
  - A multi-layer perceptron (MLP) combining fine-tuned BERT embeddings and engineered linguistic features achieved the highest F1-score for TSR classification.
- **Performance Metrics:**
  - Detailed metrics (accuracy, F1, confusion matrix) are available in the summary CSV files.
- **Interpretability:**
  - Discrete features provide insight into linguistic differences between NR and TSR conditions.

## Advanced Strategies

- **Fine-tuning Transformers:**
  - The notebook demonstrates how to fine-tune BERT for sentence classification with cross-validation and extract task-specific embeddings.
- **Feature Fusion:**
  - Combines transformer embeddings with interpretable features for robust, explainable models.
- **Best Practices:**
  - Implements stratified splits, early stopping, learning rate scheduling, and model checkpointing.

## Reproducibility & Extensibility

- All code is modular and well-documented for easy adaptation to new datasets or tasks.
- Intermediate files allow for quick reruns and further experimentation.

## References

- [ZuCo Dataset](https://osf.io/2urht/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [spaCy](https://spacy.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)

---
For questions or contributions, please open an issue or pull request.
