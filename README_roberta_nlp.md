# 🐦 AI-Driven NLP Project — Exploring RoBERTa for Social Media Sentiment Analysis

**Task 3 (Advanced Level)** — Language Model implementation, exploration, and critical analysis.

## Model Selected
**RoBERTa** (`roberta-base`) — Facebook AI's robustly optimized BERT variant. 125M parameters, 12 layers, 768 hidden dimensions, byte-level BPE tokenizer with 50k vocabulary.

The project examines **both**:
- `roberta-base` — the base model, demonstrated via masked-word prediction
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — RoBERTa further pre-trained on ~124 million tweets, then fine-tuned for three-class sentiment

This second checkpoint makes the project a case study in **domain-adaptive pre-training** — and in what that specialisation costs.

## Dataset
**TweetEval (sentiment subset)** — Twitter sentiment benchmark with three classes: negative (0), neutral (1), positive (2). Loaded via HuggingFace `datasets`.

## Why RoBERTa over BERT
| Design choice | BERT | RoBERTa |
|---|---|---|
| Next Sentence Prediction | Used | Removed — found unhelpful |
| Masking | Static | Dynamic, regenerated per epoch |
| Training data | 16 GB | 160 GB |
| Tokenizer | WordPiece, 30k | Byte-level BPE, 50k |
| Mask token | `[MASK]` | `<mask>` |

RoBERTa's contribution is methodological, not architectural — it demonstrated BERT was significantly under-trained.

## Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately does RoBERTa classify three-class sentiment on unseen tweets? |
| RQ2 | Is the **neutral** class harder to predict than positive or negative, and why? |
| RQ3 | How does byte-level BPE handle emoji, hashtags, mentions, and non-standard spelling? |
| RQ4 | Does a Twitter-adapted model degrade on formal text? |
| RQ5 | What social biases appear, and what are the ethical risks of social media sentiment analysis at scale? |

## Experiments Conducted

1. **Byte-level BPE tokenization analysis** — including a stress test proving no input produces an unknown token
2. **Class balance inspection** — training/test distribution across three classes
3. **Masked-word prediction** — base model contextual understanding via `<mask>`
4. **Quantitative evaluation** — accuracy, macro F1, per-class report, confusion matrix
5. **Probability distribution analysis** — showing *why* neutral is uncertain, not just *that* it is
6. **Informal text probes** — emoji-carried sentiment, elongated spelling, hashtags, internet slang
7. **Domain shift test** — formal long-form text evaluated against a Twitter-adapted model
8. **Bias probing** — identity-varying sentences held structurally constant

## Key Findings

- Three-class tweet sentiment (~low-to-mid 70s%) is harder than binary review sentiment — driven by both task structure and text domain, not model weakness.
- **Neutral is consistently the weakest class**, because it is defined by the *absence* of polarity rather than the presence of a signal, giving it fuzzy boundaries on both sides.
- Errors cluster on the neutral boundary; positive↔negative confusion is rare, so the model errs in **degree rather than direction**.
- **Byte-level BPE is a concrete advantage for social media** — emoji, "sooooo", hashtags, and non-Latin scripts all survive tokenization as usable signal, where WordPiece would emit `[UNK]`.
- Twitter-adaptive pre-training improves in-domain results but shows measurable degradation and lower confidence on formal text — specialisation trades breadth for depth.
- Identity-linked bias is measurable in outputs on factually neutral sentences.

## Ethical Considerations
Covered in depth in Cell 5: bias inherited from web and Twitter corpora, amplified deployment risk in content moderation and opinion measurement, dialect disparity (AAVE and other underrepresented varieties), surveillance/consent concerns, and requirements for responsible practice.

## Tech Stack
Python, HuggingFace Transformers, HuggingFace Datasets, PyTorch, scikit-learn, matplotlib, seaborn

## How to Run

Open in Google Colab (GPU runtime recommended), then Run All:

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn
```

Runtime: roughly 5–10 minutes on Colab GPU.

## Project Structure
```
roberta-nlp-project/
├── RoBERTa_Twitter_Sentiment_NLP_Project.ipynb
└── README.md
```

## Future Improvements
- Move to fine-grained emotion classification rather than three coarse bins
- Add explicit uncertainty modelling so neutral-boundary ambiguity is reported
- Incorporate thread/conversational context for short ambiguous tweets
- Schedule periodic retraining to track slang drift
- Run disaggregated bias audits across demographic and dialect groups
- Apply confidence calibration

## References
- Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Barbieri et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* Findings of EMNLP.
- Loureiro et al. (2022). *TimeLMs: Diachronic Language Models from Twitter.* ACL.
- Sennrich et al. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
- Blodgett et al. (2016). *Demographic Dialectal Variation in Social Media.* EMNLP.
- Bender et al. (2021). *On the Dangers of Stochastic Parrots.* FAccT.
