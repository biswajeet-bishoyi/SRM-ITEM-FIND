# SRM CampusFind — Viva Voce Questions and Answers (32 Technical Q&A)

**Course Project:** Applied Generative AI (B.Tech 6th Semester)  
**Title:** SRM CampusFind — Intelligent Multimodal Lost-and-Found System Using GenAI  

---

### Section 1: Core Machine Learning & Deep Learning

#### Q1: Where is Machine Learning used in SRM CampusFind, and why is it necessary?
**Answer:** Machine Learning is used in similarity calculation, feature vector transformation, candidate ranking, and confidence classification. ML is necessary because rule-based heuristics cannot handle the high degree of variation in human text descriptions, visual appearance under different angles, or spatio-temporal decay.

#### Q2: What is the difference between Machine Learning and Deep Learning in this project?
**Answer:** Machine Learning handles candidate retrieval, multimodal score fusion, and metric evaluation. Deep Learning specifically handles representation learning—extracting high-level dense vector embeddings directly from raw text sequences (via Transformers) and image pixels (via Convolutional Neural Networks / Vision Encoders) without requiring manual feature engineering.

#### Q3: Why is cosine similarity preferred over Euclidean distance for high-dimensional vector embeddings?
**Answer:** In high-dimensional spaces (e.g., 384 or 512 dimensions), Euclidean distance suffers from the "curse of dimensionality" and is sensitive to vector magnitude. Cosine similarity measures the angle between two normalized vectors:
$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$
This focuses purely on direction and semantic orientation rather than length differences.

---

### Section 2: Natural Language Processing & Transformers

#### Q4: Why is traditional keyword matching (e.g., SQL LIKE or TF-IDF alone) insufficient for lost-and-found matching?
**Answer:** Keyword matching relies on exact string overlap. If a lost report says "misplaced black wireless earbuds" and a found report says "dark Bluetooth earphones", keyword search returns a score of 0 because no non-stopword tokens match. Transformer NLP models understand semantic synonymy and context.

#### Q5: What is a Transformer, and what makes its Self-Attention mechanism powerful?
**Answer:** A Transformer is a deep learning neural architecture based on self-attention mechanisms (Vaswani et al., 2017). Self-attention calculates pairwise attention weights between all tokens in a sequence simultaneously, allowing the model to capture contextual relationships regardless of distance (e.g., linking "black" to "earbuds" even if separated by adjectives).

#### Q6: Which Transformer model is used for text embedding in SRM CampusFind, and why?
**Answer:** The code loads pretrained `all-MiniLM-L6-v2` when Sentence Transformers is available; it maps text to 384-dimensional embeddings. It is a compact pretrained encoder, but this project has not fine-tuned it or established comparative held-out MRR/Recall performance.

#### Q7: How does Sentence-BERT (SBERT) differ from standard BERT?
**Answer:** A bi-encoder such as SBERT encodes each text independently, so embeddings can be compared with cosine similarity. Approximate-nearest-neighbor indices can accelerate large searches, but this project currently scans its in-memory list linearly rather than using such an index.

---

### Section 3: Computer Vision & Visual Similarity

#### Q8: How does the visual encoder process uploaded item images?
**Answer:** With pretrained ResNet18, images are resized to $224 \times 224$, normalized with ImageNet statistics, and converted to 512-dimensional features. If that path is unavailable, a fallback extracts RGB histograms, aspect ratio, and grayscale-grid statistics into an 81-dimensional handcrafted vector.

#### Q9: Why is pixel-by-pixel comparison inadequate for image identification?
**Answer:** Pixel comparison is extremely brittle to minor spatial shifts, rotation, cropping, resolution differences, and background lighting. Deep visual encoders extract invariant latent features representing object category, shape, and color distributions.

---

### Section 4: Multimodal AI & Feature Fusion

#### Q10: How are different modalities fused in SRM CampusFind?
**Answer:** Modalities are fused using a weighted linear combination engine:
$$\text{Final Score} = w_1 S_{\text{text}} + w_2 S_{\text{image}} + w_3 S_{\text{location}} + w_4 S_{\text{time}} + w_5 S_{\text{attribute}}$$
Where default weights are set to $w_1=0.35, w_2=0.30, w_3=0.15, w_4=0.10, w_5=0.10$.

#### Q11: How is location similarity calculated?
**Answer:** Location similarity is computed using the Haversine formula on GPS coordinates, applied to an exponential decay function:
$$S_{\text{location}} = \exp\left(-\frac{\text{Distance in meters}}{200.0}\right)$$
This yields high similarity for locations within 200m on campus and smoothly decays over longer distances.

#### Q12: How is temporal similarity computed?
**Answer:** Timestamp difference $\Delta t$ in hours is passed to a half-life exponential decay function ($H = 48\text{ hours}$):
$$S_{\text{time}} = \exp\left(-\frac{\Delta t}{48.0}\right)$$

#### Q13: What is an Ablation Study, and why was it conducted?
**Answer:** An ablation study compares modality subsets. The current dataset has eight labeled pairs, no photos, and no pair-grouped held-out test, so it does not support a reliable comparative accuracy claim yet.

---

### Section 5: Generative AI & Large Language Models

#### Q14: What are the three Generative AI components in SRM CampusFind?
**Answer:**
1. **Explanation helper:** Fills a natural-language template with report fields and similarity scores.
2. **Attribute normalizer:** Uses keyword rules to infer category, color, and brand.
3. **Paraphrase helper:** Fills fixed templates; variations need human review before use as training labels.

#### Q15: How does the system prevent Generative AI hallucinations?
**Answer:** The current explanation helper uses a fixed template populated with scores and reported fields; it does not call an LLM. A future LLM integration should be constrained to cite verified fields and subscores.

#### Q16: Is RAG (Retrieval-Augmented Generation) used in this project?
**Answer:** SRM CampusFind uses **Vector Similarity Retrieval** rather than conventional document RAG. The primary task is item matching and candidate ranking rather than retrieving long knowledge documents. Stating this clearly prevents misusing "RAG" as an inaccurate buzzword.

---

### Section 6: Vector Databases & Candidate Retrieval

#### Q17: What is the role of a Vector Database in this project?
**Answer:** The current `VectorDatabase` is an in-memory Python store of embeddings and report metadata. It scores eligible records and sorts a top-$K$ shortlist; it is a linear scan, not an approximate-nearest-neighbor database.

#### Q18: What is candidate ranking?
**Answer:** Candidate ranking orders retrieved records by predicted similarity scores so campus security or users see the most probable matches at the top of the result list.

---

### Section 7: Evaluation Metrics & Benchmarks

#### Q19: What metrics are used to evaluate the semantic text and visual retrieval engine?
**Answer:**
- **Recall@K (Recall@1, Recall@3):** Percentage of queries where the ground-truth match is present within the top-$K$ returned candidates.
- **Mean Reciprocal Rank (MRR):** Average of reciprocal ranks ($\frac{1}{\text{Rank}}$) across queries.
- **Precision, Recall, F1-Score, and Accuracy:** Evaluates final binary match classification correctness.

#### Q20: What were the experimental results of Proposed Multimodal vs Baselines?
**Answer:** There is not yet a defensible held-out result. The available corpus has eight known pairs and no photos, and the evaluator does not split by pair ID. Earlier perfect-score figures should not be presented as validated performance.

#### Q21: Which metric is most important for a lost-and-found system, and why?
**Answer:** **Recall@K** and **F1-Score**. High Recall@K ensures that the true lost item is never missed in candidate search, while F1-Score balances precision so users are not overwhelmed by false positive matches.

---

### Section 8: Dataset & Synthetic Augmentation

#### Q22: What dataset was used for SRM CampusFind?
**Answer:** The dataset contains 16 reports forming eight known lost/found pairs. A small set of fixed-template paraphrases exists for two pairs; no item-photo files are present. This is demonstration data, not enough for robust fine-tuning or evaluation.

#### Q23: How do you distinguish synthetic data from real data?
**Answer:** The paraphrase examples are stored separately from the curated report JSON. They are fixed-template text variations, not LLM-generated data, and should be labeled and reviewed before being used in experiments.

#### Q24: How was data leakage prevented?
**Answer:** The evaluator separates LOST queries from FOUND candidates, but it does not perform a pair-grouped train/validation/test split. A future fine-tuning study must split by pair ID before augmentation to prevent leakage.

---

### Section 9: Responsible AI, Security & Privacy

#### Q25: Why does the system refrain from stating "This is definitely your item"?
**Answer:** To adhere to Responsible AI principles. AI predictions are probabilistic estimates. Declaring definitive ownership risks false handovers. The system uses probabilistic language ("High-confidence potential match") and mandates human verification.

#### Q26: How are privacy and image safety handled?
**Answer:** Uploaded images are scanned for human faces. If detected, face regions are automatically blurred to protect student privacy before public display.

#### Q27: How does the system handle unauthorized claims of ownership?
**Answer:** Claiming an item requires physical identity verification at the SRM Security Desk along with proof (e.g. unlocking a phone or identifying undisclosed internal bag contents).

---

### Section 10: System Architecture & Implementation

#### Q28: What tech stack was chosen, and why?
**Answer:** FastAPI for lightweight, high-performance async REST APIs; PyTorch and Scikit-Learn for model execution; HTML5/CSS3/JS for a clean, accessible web dashboard.

#### Q29: What happens when an input image or location coordinate is missing?
**Answer:** The system gracefully handles missing fields by assigning a neutral subscore ($0.5$) and dynamically re-normalizing the weights among the available modalities.

#### Q30: How does the automated notification trigger work?
**Answer:** When the multimodal fusion score for a candidate pair reaches $0.50$, the Notification Manager creates a notification payload and logs an in-app alert for the user.

#### Q31: How does the system resolve conflicts between text and image similarity?
**Answer:** If text similarity is very high ($0.92$) but image similarity is very low ($0.25$), the weighted fusion reduces the final score to $0.62$ ("Possible Match"), preventing automatic high-confidence false triggers.

#### Q32: What is the main research/engineering contribution of SRM CampusFind?
**Answer:** Proving that integrating Transformer text embeddings, deep visual representations, spatio-temporal decay, and Generative AI explanations into a unified multimodal framework solves vocabulary mismatch and visual opacity in campus lost-and-found management.
