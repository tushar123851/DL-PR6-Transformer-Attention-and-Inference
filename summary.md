# DL PR6 — Project Summary Report

**Project:** Opening the Transformer: Attention, Encoder, Decoder and Inference with Pretrained Models  
**Author:** Tushar Vala  
**Focus:** Transformer concepts, attention visualizations, pretrained inference, and evaluation  
**Environment recorded in notebook:** Python 3.11.4, CPU-only PyTorch runtime

## 1. Executive summary

This notebook is a practical study of Transformer models from fundamentals through evaluation. It implements scaled dot-product attention and causal masking with NumPy, inspects real DistilBERT and GPT-2 attention, visualizes positional information and parameter distributions, uses T5 for translation and summarization, compares GPT-2 decoding strategies, measures key-value cache performance, and evaluates DistilBERT sentiment classification on 200 IMDB test reviews. A final mini toolkit combines sentiment analysis, summarization, and English-to-French translation.

All models are pretrained. The notebook does not train or fine-tune models.

## 2. Models and recorded environment

| Component | Model / version | Role |
|---|---|---|
| Sentiment classification / encoder inspection | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment predictions, encoder attention, positional embeddings, architecture inspection |
| Autoregressive generation | `gpt2` | Causal attention, greedy/beam/sampling generation, KV-cache benchmark, factuality examples |
| Encoder-decoder generation | `t5-small` | English-to-French translation, summarization, cross-attention, combined NLP toolkit |
| Python | 3.11.4 | Notebook runtime recorded in output |
| PyTorch | 2.14.0+cpu | CPU inference; CUDA unavailable in the recorded run |
| Transformers | 5.20.2 | Model loading and inference |
| Datasets | 5.1.0 | IMDB data loading |
| NumPy / Pandas | 2.4.6 / 3.0.6 | Numerical operations and tabular results |
| scikit-learn | 1.9.1 | Evaluation metrics |
| Matplotlib / Seaborn | 3.11.2 / 0.13.2 | Visualizations |

Versions above are copied from the notebook output; installation availability may vary by platform and package index.

## 3. Task-by-task summary

### Task 1 — Introduction, pipelines, tokenizer, and model anatomy

- Used the Hugging Face sentiment-analysis pipeline with DistilBERT.
- Examined Transformer history and the impact and limitations of Transformers.
- Inspected word/subword tokenization, special tokens, input IDs, attention masks, padding, and batching.
- Inspected DistilBERT architecture and parameter names.
- The notebook reports **66,955,010 parameters** for the sequence-classification model and a six-layer encoder backbone with 12 attention heads and hidden size 768.

### Task 2 — Self-attention and scaled dot-product attention

- Implemented Query, Key, and Value projections and a numerically stable softmax.
- Implemented scaled dot-product attention and verified that each attention row sums to 1.
- Compared scaled and unscaled scores. For the recorded scaling experiment at `d_k=256`, the mean maximum probability changed from **0.9232 without scaling** to **0.2176 with scaling**, while mean entropy changed from **0.1839** to **2.5853**. This illustrates how scaling can prevent overly concentrated softmax distributions.
- Compared self-attention with Luong dot attention.
- Shuffled the input and confirmed that the attention output permuted correspondingly, demonstrating why positional information is needed.
- Recomputed DistilBERT layer-0/head-0 attention manually; the recorded maximum absolute difference from the model's attention matrix was **0.0**.

### Task 3 — Multi-head attention, positional encoding, and normalization

- Extracted and visualized six real attention heads.
- Calculated per-head attention entropy. The recorded mean entropy was **1.686** for the first layer and **2.0096** for the last layer in the tested sentence.
- Implemented sinusoidal positional encoding and inspected DistilBERT's learned positional embeddings.
- Compared representations with and without positional information: recorded similarity was **1.0 without positional encoding** and **0.859478 with positional encoding** for the demonstration.
- Implemented Layer Normalization and compared it with Batch Normalization; inspected learned LayerNorm parameters.

### Task 4 — Encoder, masking, and architecture inspection

- Inspected DistilBERT encoder modules, query/key/value projections, and parameter groups.
- Parameter breakdown for the inspected DistilBERT backbone:

| Component | Parameters | Percentage |
|---|---:|---:|
| Feed-forward | 28,334,592 | 42.70% |
| Embeddings | 23,835,648 | 35.92% |
| Attention | 14,174,208 | 21.36% |
| Layer normalization | 18,432 | 0.03% |
| **Total backbone** | **66,362,880** | **100% (rounded)** |

- Ran masked-word predictions with DistilBERT.
- Demonstrated bidirectional encoder attention and a NumPy causal mask.
- Verified that future-token attention weights were **0.0** in the causal-mask experiment.
- Inspected GPT-2 causal attention and verified maximum future-position attention was **0.0**.
- Tested padding masks and observed **0.0** maximum and mean attention weight assigned to padding keys in the recorded test.

### Task 5 — Decoder, cross-attention, and inference

- Reviewed decoder architecture, cross-attention, and the difference between self-attention and cross-attention.
- Loaded `t5-small` with six encoder layers, six decoder layers, and eight attention heads.
- Translated three English sentences into French. The recorded first translation was **“The house is small.” → “La maison est petite.”**
- Summarized a paragraph about AI/ML uses and risks. The generated summary retained the use cases and key risks, including bias, privacy, transparency, reliability, and human oversight.
- Visualized T5 cross-attention for the translation example. The final decoder-layer tensor had shape `(1, 8, 6, 11)` and the averaged matrix shape was `(6, 11)`.
- Implemented manual greedy GPT-2 decoding for 20 new tokens. Its output matched `model.generate()` exactly in the recorded check (`Outputs match: True`).
- Compared greedy search, beam search, temperature sampling, top-k, and top-p. Output quality and repeatability varied by decoding strategy; fluent output did not guarantee factual accuracy.
- Benchmarked 100 generated tokens over three runs per setting:

| Configuration | Mean time | Standard deviation |
|---|---:|---:|
| KV cache enabled | 4.221 s | 0.020 s |
| KV cache disabled | 13.429 s | 1.163 s |

The recorded no-cache/cache time ratio was **3.181×**, with **68.57% relative time reduction** when caching was enabled. These are measurements from the notebook's CPU run, not universal performance guarantees.

### Task 6 — Evaluation, latency, factual reliability, and mini toolkit

#### IMDB sentiment evaluation

DistilBERT was evaluated on a shuffled sample of 200 reviews from the IMDB test split. Because this checkpoint was fine-tuned on SST-2, this is a cross-dataset evaluation rather than the official SST-2 benchmark.

| Metric | Recorded score |
|---|---:|
| Accuracy | 0.870 |
| Precision | 0.889 |
| Recall | 0.833 |
| F1-score | 0.860 |

The classification report listed 104 negative and 96 positive examples. These scores apply to the sampled 200 reviews and the recorded execution.

#### Maximum input length experiment

| Max length | Accuracy | Precision | Recall | F1-score | Runtime |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.750 | 0.767 | 0.688 | 0.725 | 11.93 s |
| 128 | 0.775 | 0.758 | 0.781 | 0.769 | 19.54 s |
| 256 | 0.840 | 0.864 | 0.792 | 0.826 | 35.57 s |
| 512 | 0.870 | 0.889 | 0.833 | 0.860 | 62.21 s |

In this run, longer maximum input lengths improved the measured metrics but increased runtime. Eighteen reviews (**9.0%**) exceeded 512 tokens; the longest was 1,453 tokens.

#### Forward-pass latency

| Sequence length | Mean latency | Standard deviation |
|---:|---:|---:|
| 32 | 51.92 ms | 4.03 ms |
| 64 | 75.44 ms | 11.32 ms |
| 128 | 110.94 ms | 19.27 ms |
| 256 | 188.45 ms | 10.46 ms |
| 512 | 506.54 ms | 78.23 ms |

Latency increased with sequence length in the recorded CPU test. The benchmark measures a model forward pass, not the full end-to-end pipeline.

#### GPT-2 factual reliability

The three GPT-2 outputs contained unsupported or incorrect claims and repetition (for example, the water-freezing prompt produced a claim of about −40°F and −20°C, which are not equivalent temperatures). The notebook leaves each output marked **“Review required”** rather than automatically asserting correctness. This supports the conclusion that fluent next-token generation is not factual verification.

#### Combined NLP toolkit

The `text_toolkit()` function combines:
1. DistilBERT sentiment analysis.
2. T5 summarization.
3. T5 translation of the summary into French.

All three sample passages completed with status `Success`. Recorded sentiment labels/scores were:

| Passage | Sentiment | Score |
|---|---|---:|
| Product Review | POSITIVE | 0.999870 |
| News-style Passage | POSITIVE | 0.990203 |
| Technology Passage | POSITIVE | 0.997501 |

The empty-string test returned `Invalid or empty input` and `N/A` outputs, as intended. These are demonstration examples, not a production evaluation.

## 4. Graphs and saved visualizations

All **18 embedded plot outputs** present in the uploaded notebook were extracted to `plots/`. The filenames preserve the task context. The gallery is also included in `README.md`.

1. `task2_scaled_dot_product_attention.png` — scaled attention heatmap.
2. `task2_attention_scaling_comparison.png` — scaled versus unscaled attention.
3. `task2_distilbert_attention_verification.png` — real DistilBERT attention verification.
4. `task3_six_attention_heads.png` — six attention heads.
5. `task3_attention_entropy_by_head.png` — entropy by attention head.
6. `task3_sinusoidal_positional_encoding.png` — sinusoidal positional encoding.
7. `task3_learned_position_similarity.png` — learned position similarity.
8. `task4_parameter_counts.png` — parameter counts by component.
9. `task4_parameter_percentages.png` — parameter percentage distribution.
10. `task4_bidirectional_attention_matrix.png` — bidirectional attention.
11. `task4_causal_mask.png` — causal mask.
12. `task4_gpt2_causal_attention_head.png` — GPT-2 causal attention.
13. `task5_t5_cross_attention_heatmap.png` — T5 cross-attention heatmap.
14. `task5_kv_cache_benchmark.png` — key-value cache timing comparison.
15. `task6_imdb_confusion_matrix.png` — IMDB confusion matrix.
16. `task6_accuracy_vs_input_length.png` — accuracy versus maximum input length.
17. `task6_runtime_vs_input_length.png` — runtime versus maximum input length.
18. `task6_distilbert_inference_latency.png` — latency versus sequence length.

## 5. Main conclusions

- Scaling dot-product attention changes softmax concentration and can improve numerical behavior.
- Positional information is needed because attention alone does not encode token order.
- Encoder bidirectional attention and decoder causal attention serve different purposes.
- T5 supports both conditional generation tasks in this notebook: translation and summarization.
- GPT-2 greedy manual decoding matched the library generation result in the recorded test.
- KV caching substantially reduced generation time in the measured CPU experiment.
- On the sampled IMDB set, longer input limits improved measured classification metrics but cost more runtime.
- Generated text must be checked for factual correctness; language fluency is not evidence of truth.

## 6. Reproducibility and limitations

- Results reflect the notebook's recorded outputs, not a fresh rerun during report creation.
- The run used CPU inference (`CUDA available: False`); timings will differ across machines.
- The IMDB evaluation used only 200 shuffled test examples.
- GPT-2 factuality checks were marked for manual review, not treated as a statistically valid factuality benchmark.
- Generated summaries and translations can omit or distort details.
- Model and package downloads require network access unless cached locally.
- Review `requirements.txt` before installation, especially for the installed Transformers/Datasets versions and platform-specific PyTorch wheels.

## 7. Project files

- `DL_PR6.ipynb` — notebook with executed outputs.
- `DL_PR6.html` — HTML export of the notebook.
- `summary.md` — this summary report.
- `README.md` — project overview, result highlights, and graph gallery.
- `requirements.txt` — environment package pins.
- `plots/` — all extracted graph images.
