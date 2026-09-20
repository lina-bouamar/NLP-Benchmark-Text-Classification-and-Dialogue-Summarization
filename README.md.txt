# NLP Benchmark — Text Classification and Dialogue Summarization

## Context
Project carried out as part of the Natural Language Processing course (Master 1 Computer 
Science, AI, Data Science and Health track) at Université de Caen Normandie 
(academic year 2025-2026).

## Objective
Build a rigorous experimental benchmark comparing different families of NLP models on two 
complementary tasks:
1. **Text classification** on the AG News dataset
2. **Dialogue summarization** on the SAMSum dataset

The goal was not to chase state-of-the-art scores, but to set up a sound experimental 
methodology and fairly compare model families under the same conditions.

## Reproducibility protocol
- A shared `utils.py` centralizes common functions: `seed_everything()` (fixes random, 
  numpy, torch and cuda seeds), dataset loading/subsampling, and shared metric 
  implementations (ROUGE from scratch, embedding similarity)
- Data seed fixed at 42 (same data subset for every model), 3 different training seeds 
  (42, 123, 456) to measure variability due to model initialization specifically
- ROUGE metrics implemented from scratch (no external library)

## Technologies used
- **Language**: Python
- **Deep learning**: PyTorch, Transformers (Hugging Face)
- **Classical ML**: scikit-learn (TF-IDF, SVM, logistic regression)
- **Models used**: GloVe embeddings, BiLSTM, custom Transformer, DistilBERT, T5-small, 
  distilgpt2

---

## Part 1 — Text Classification (AG News)

### Methodology
4,000 training examples and 1,000 test examples subsampled from AG News (120k/7.6k 
available), balancing GPU budget against enough data to differentiate models.

### Systems compared
| Model | Description |
|---|---|
| TF-IDF + SVM | TF-IDF (10k features) + Linear SVM with calibrated probabilities |
| GloVe + Probing | Mean of pretrained GloVe 100d vectors + logistic regression |
| BiLSTM + GloVe | BiLSTM (hidden_dim=64) initialized with GloVe embeddings |
| Transformer from scratch | Small Transformer (2 layers, 2 heads, 64d), trained with no pretraining |
| DistilBERT frozen | Frozen DistilBERT backbone + linear probe on [CLS] token |
| DistilBERT fine-tuned | Full DistilBERT fine-tuning |

### Results (mean ± std over 3 seeds)
| Model | micro-acc | macro-acc | micro-F1 | macro-F1 | AUROC |
|---|---|---|---|---|---|
| TF-IDF + SVM | 0.891±0.000 | 0.889±0.000 | 0.891±0.000 | 0.889±0.000 | 0.975±0.000 |
| GloVe + Probing | 0.886±0.000 | 0.884±0.000 | 0.886±0.000 | 0.884±0.000 | 0.972±0.000 |
| BiLSTM + GloVe | 0.866±0.002 | 0.864±0.002 | 0.866±0.002 | 0.864±0.001 | 0.968±0.002 |
| Transformer from scratch | 0.764±0.010 | 0.764±0.010 | 0.764±0.010 | 0.763±0.010 | 0.927±0.004 |
| DistilBERT frozen | 0.890±0.010 | 0.888±0.010 | 0.890±0.010 | 0.889±0.009 | 0.977±0.000 |
| **DistilBERT fine-tuned** | **0.920±0.004** | **0.918±0.004** | **0.920±0.004** | **0.918±0.005** | **0.984±0.002** |

### Key findings
- **Cost-efficiency**: TF-IDF + SVM (0.891 accuracy) nearly matches DistilBERT frozen 
  (0.890), while training in ~10 seconds with no GPU versus a 268 MB model requiring 
  inference over 4,000 examples. Best performance/cost ratio by far.
- **Only fine-tuning clearly stands out**: DistilBERT fine-tuned reaches 0.920, a +3 point 
  gain over frozen DistilBERT — meaningful, but costing ~15 minutes of GPU time.
- **Stable rankings across metrics**: all 5 metrics (micro/macro accuracy, micro/macro F1, 
  AUROC) produce the same model ranking, explained by AG News being a balanced dataset.
- **Surprising result**: the from-scratch Transformer performs worst (0.764), 10 points 
  below BiLSTM and 12 points below simple averaged GloVe embeddings — despite Transformers' 
  reputation in NLP. With only 4,000 training examples, it has to learn embeddings, 
  positional encoding and attention patterns entirely from scratch, which requires far more 
  data than available. This illustrates exactly why pretraining (BERT, GPT, etc.) matters.
- **Variability analysis**: TF-IDF/GloVe are near-deterministic (std ≈ 0), BiLSTM shows 
  moderate variability (std ≈ 0.002), while the Transformer and DistilBERT frozen show 
  higher variability (std ≈ 0.010) due to random initialization interacting with limited data.

---

## Part 2 — Dialogue Summarization (SAMSum)

### Methodology
Evaluation on 200 test examples from SAMSum (informal, multi-turn chat dialogues), using 
2,000 training examples for fine-tuned systems. Compared to news summarization, SAMSum adds 
challenges: multiple speakers, informal style, ambiguous pronouns, and speaker attribution.

Metrics: ROUGE-1, ROUGE-2, ROUGE-L (all implemented from scratch) and EmbSim (cosine 
similarity via all-MiniLM-L6-v2 embeddings).

### Systems compared
| System | Architecture | Training | Decoding |
|---|---|---|---|
| T5-small zero-shot | Encoder-decoder | None | Beam search (4) |
| T5-small fine-tuned | Encoder-decoder | 2,000 ex., 3 epochs | Beam search (4) |
| distilgpt2 (prompted) | Decoder-only | None | Greedy |
| T5-small fine-tuned + sampling | Encoder-decoder | 2,000 ex., 3 epochs | Nucleus sampling (p=0.9) |

### Results
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | EmbSim |
|---|---|---|---|---|
| T5-small zero-shot | 0.146 | 0.045 | 0.122 | 0.554 |
| **T5-small fine-tuned (beam)** | **0.41±0.01** | **0.19±0.01** | **0.37±0.01** | **0.72±0.01** |
| distilgpt2 (prompt-based) | 0.091 | 0.017 | 0.081 | 0.434 |
| T5-small fine-tuned (sampling) | 0.39±0.01 | 0.17±0.01 | 0.35±0.01 | 0.70±0.01 |

### Key findings
- Fine-tuning is essential: T5-small zero-shot barely summarizes (ROUGE-1 = 0.146), while 
  fine-tuning on just 2,000 examples nearly triples it (0.41)
- distilgpt2 (prompt-based, no fine-tuning) performs worst — it tends to hallucinate content 
  not present in the dialogue rather than genuinely summarizing
- Beam search slightly outperforms nucleus sampling on ROUGE metrics (more faithful/ 
  deterministic), while sampling trades a small amount of accuracy for more diverse outputs

### Qualitative example
> **Dialogue**: Olafur, Nathalie and Zoe discuss New Year's Eve plans, considering an opera 
> or a themed party.
> **Reference**: *Nathalie, Olafur and Zoe are planning New Year's Eve. Nathalie wants 
> something classy. Olafur doesn't like opera. They want to go to the Breakfast at Tiffany's 
> party in Soho.*
> **T5 fine-tuned**: *Olafur, Nathalie and Zoe plan to go to a Breakfast at Tiffany's party 
> for New Year's Eve.* — captures the essential information in one sentence
> **distilgpt2**: hallucinates content unrelated to the actual dialogue

---

