# 🍿 CineMatch AI: Feature-Enriched Neural Collaborative Filtering (FENCF)

An end-to-end Deep Learning Recommender System built on the MovieLens 1M dataset. CineMatch AI leverages a hybrid **Feature-Enriched Neural Collaborative Filtering (FENCF)** architecture combining Matrix Factorization, Multi-Layer Perceptrons, demographic metadata, and multi-hot genre embeddings, wrapped in a dark-themed interactive Streamlit web dashboard.

---

## 🎯 Purpose & Overview

The goal of this project is to transition a recommendation prototype into a production-ready neural recommender system. Traditional Collaborative Filtering models often suffer from cold-start problems and ignore rich auxiliary user/item attributes. 

**CineMatch AI** addresses this by fusing:
1. **Implicit Feedback**: Matrix Factorization based on user-item interaction signals rather than explicit star ratings.
2. **User Demographics**: Deep representation learning on user features (Gender, Age Group, Occupation).
3. **Item Metadata**: Multi-hot binarized genre feature vectors.

---

## 🏗️ Preprocessing Pipeline

The dataset utilized is **MovieLens 1M**, comprising 1 million ratings across 6,040 users and 3,706 movies.

### Key Preprocessing Steps (`preprocess.ipynb` / artifacts):

1. **Implicit Interaction Conversion**:
   * Removed explicit star ratings ($1-5$).
   * Converted ratings into binary interaction signals (`target = 1`) to represent implicit user preference.

2. **Zero-Indexed Contiguous Mapping**:
   * MovieLens primary keys (`user_id`, `item_id`) contain non-consecutive integers and missing IDs.
   * Mapped raw IDs to continuous contiguous integers  required for PyTorch `nn.Embedding` lookup tables.

3. **Temporal Leave-One-Out Split**:
   * Sorted user interaction histories chronologically by timestamp.
   * Extracted each user's single **latest interaction** to construct `test_ratings` (evaluating next-item prediction).
   * Used all prior interactions to form `train_ratings` to prevent temporal data leakage.

4. **Dynamic Negative Sampling ($4:1$ Ratio)**:
   * Mapped 4 unobserved (negative) items for every positive interaction to train the binary cross-entropy loss.
   * *Limitation Acknowledgment*: Unobserved items are treated as assumed negatives, though users may simply have missed discovering them rather than actively disliking them.

5. **Feature Engineering & Transformation**:
   * Processed demographic features (`Gender`, `Age`, `Occupation`).
   * Transformed multi-genre movie strings into an 18-dimensional multi-hot binary matrix via `MultiLabelBinarizer`.

---

## 🧠 Model Architecture (FENCF)

The **FENCF** model blends linear matrix factorization with non-linear deep feature learning and multi-feature fusion:

* **Generalized Matrix Factorization (GMF)**: Learns linear latent factor interactions between user and item embeddings (`factor_num = 32`)[cite: 1, 10].
* **Multi-Layer Perceptron (MLP)**: Concatenates user/item MLP embeddings with dense demographic embeddings (`Gender` [4-dim], `Age` [4-dim], `Occupation` [8-dim])[cite: 10] and passes them through a deep network (`64 -> 32 -> 16`)[cite: 10] with `ReLU` activations and `Dropout(0.2)`[cite: 10].
* **Feature Enhancement (Genre Branch)**: Projects the 18-dimensional genre vector through a linear layer into a 16-dimensional continuous genre representation[cite: 10].
* **Fusion Layer**: Concatenates outputs from GMF ($32$), MLP ($16$), and Genre ($16$) branches into a single classifier layer evaluated via Binary Cross-Entropy (BCE) loss[cite: 5, 10].

---

## 📊 Performance & Evaluation Results

Evaluated using the **Leave-One-Out Evaluation Protocol** with **100 sampled negative candidates per test user**[cite: 9]:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Hit Ratio @ 10 (HR@10)** | **`0.6674`**[cite: 4] | ~66.7% of the time, the ground-truth test movie appeared in the Top-10 recommendations[cite: 9]. |
| **NDCG @ 10 (NDCG@10)** | **`0.3853`**[cite: 4] | Measures ranking quality, giving higher weight to recommendations placed at top positions[cite: 9]. |

---

