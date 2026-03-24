# Decoupling Scores and Text: The Politeness Principle in Peer Review
---

## 1. Overview
This project investigates the consistency between numerical ratings and textual feedback in academic peer reviews using a large-scale dataset of over **30,000 ICLR submissions (2021-2025)**. 

Our research identifies a significant **"signal decoupling"**: while score-based models achieve 91% accuracy in predicting paper acceptance, text-based models (including SOTA LLMs) struggle at 81%. This discrepancy is primarily caused by the **Politeness Principle**, where reviewers structurally cloak negative decisions with positive semantics.
<img width="1241" height="615" alt="截屏2026-03-23 15 47 08" src="https://github.com/user-attachments/assets/2d49d65f-d2a3-4916-a806-7278bc28478f" />

---

## 2. Get started

### Python Environment Setup
We recommend using **Conda** to manage your environment. The core dependencies include `openreview-py` for data scraping and `scikit-learn`/`transformers` for modeling.

```bash
# Create and activate the environment
conda create -n polite_peer_review python=3.9
conda activate polite_peer_review

# Install dependencies
pip install pandas numpy scipy scikit-learn torch transformers
pip install openreview-py  # For scraping from OpenReview API
```

---

## 3. Project Structure

### Dataset Construction: ICLR 2021-2025
The data collection scripts are located in `Dataset Construction/`.
- `getting_data_before_2024.py`: Scrapes historical ICLR review data.
- `getting_data_after_2024.py`: Scrapes data for 2024 and 2025.
- format data: handle the Dialogue Reconstruction, Data Cleaning, Labeling, preprocessing


### Benchmark and Sigmel Decoupling
Located in `Benchmark and Signal Decoupling`, this module contains the benchmarking pipeline for acceptance prediction.
- **Models**: Includes traditional ML (XGBoost, SVM), Deep Learning (SciBERT, TextCNN), and LLMs (GPT-5, Claude-3.5, Gemini-2.5).
- **Baselines**: Supports Rating-only, Text-only, and Hybrid (Rating+Text) input settings.

### Hard Example Analysis
Located in `Hard Example Analysis/`, this module performs a rating analysis into "Hard Samples" (samples misclassified by text-based models).
- **Rating Analysis**: Quantitative investigation of Score Distribution, Skewness, and Kurtosis to identify the "Veto Effect."

### Sentimental Analysis of Comment Review
Located in `Sentimental Analysis of Comment Review/`, this module performs a text analysis into "Hard Samples" 
- **Text Analysis**: Fine-grained aspect-based sentiment analysis across 33 evaluation aspects to quantify the Politeness Principle.
---



## 4. Dataset Statistics

The datsets can be reached through this link:https://drive.google.com/drive/folders/1a4GlwoLB118_HuHfQ_g0Uo-F_rDErMqy?usp=sharing
The following table shows the comparative statistics between official ICLR records and our processed dataset. Our data exhibits a high degree of consistency with official records (within 0.7%–1.4% discrepancy).

| Category | Metric | 2021 | 2022 | 2023 | 2024 | 2025 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Official Data** | Submission | 3014 | 3391 | 4938 | 7262 | 11,603 |
| | Accept | 1027 | 1095 | 1574 | 2260 | 3704 |
| | Accept rate | 34.07% | 32.26% | 31% | 31% | 32% |
| **Our Data** | Submission | 2972 | 3354 | 4897 | 7210 | 11,512 |
| | Accept | 859 | 1084 | 1561 | 2248 | 3699 |
| | Labeled reject | 2113 | 2270 | 3336 | 4962 | 7813 |
| | Accept rate | 28.9% | 32.32% | 31.88% | 31.18% | 32.13% |

---

## 5. Experiment result

### Acceptance Prediction Benchmarking
We evaluated various modeling paradigms, including traditional ML, Deep Learning, and LLMs. The results confirm that **numerical scores serve as a precise proxy** for decisions, while **textual reviews contain substantial noise**.


| Modality | Accuracy (Best Model) |
| :--- | :---: |
| **Numerical Ratings Only** | **91.00%** |
| **Textual Reviews Only** | **81.00%** |
| **Hybrid (Rating + Text)** | **86.82%** |

#### I. Rating-Only vs. Text-Only Results
Score-based models consistently outperform text-based models by approximately 10%.

| Input Type | Best Model | Accuracy |
| :--- | :--- | :---: |
| **Rating-Only** | MLP / XGBoost | **0.9100** |
| **Initial Review** | Claude-haiku-4-5 | 0.7800 |
| **Weakness Only** | Gemini-2.5 pro | 0.8100 |
| **Strength + Weakness** | Gemini/GPT/Claude | 0.7800 |

#### II. Detailed Model Comparison
Detailed performance across different input modalities:

| Modality | Model | Accuracy |
| :--- | :--- | :---: |
| **Rating-Only** | XGBoost | 0.9099 |
| | GPT-5 | 0.8600 |
| | Gemini-2.5 pro | 0.8400 |
| **Rating + Review** | Gemini-2.5 pro | 0.8600 |
| | GPT-5 | 0.8300 |
| **Rating + Rebuttal**| Dual-Branch Attention | 0.8258 |

### Prediction Accuracy Gap
Numerical scores serve as a precise proxy for decisions, while textual reviews contain substantial "polite noise" that obscures the ground truth.

<img width="1230" height="615" alt="截屏2026-03-23 15 46 46" src="https://github.com/user-attachments/assets/2e802d31-b5a7-4911-85e7-5f402091ac17" />

### Hard Sample Diagnosis
Hard samples are characterized by **Negative Skewness** and **High Kurtosis**, representing a "Veto" scenario where a single strong objection outweighs a positive consensus.

<img width="1057" height="513" alt="截屏2026-03-23 15 47 33" src="https://github.com/user-attachments/assets/e80d170a-a915-42a3-8848-0ed6353ed271" />

### Sentiment Polarity vs. Decision
As shown below, the sentiment profiles of rejected papers (Hard Rejects) structurally mimic those of accepted papers, proving that positive sentiment dominates even in rejection.
<img width="1202" height="710" alt="截屏2026-03-23 15 48 09" src="https://github.com/user-attachments/assets/b69f7f86-7cf8-4f7d-a0bb-4ac0d798ef7c" />
<img width="928" height="582" alt="截屏2026-03-23 15 48 33" src="https://github.com/user-attachments/assets/de08e63b-c1f0-4348-a451-5a794798817d" />



---


