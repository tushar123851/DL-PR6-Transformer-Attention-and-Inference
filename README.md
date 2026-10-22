<div align="center">

# 🚀 Opening the Transformer

### Attention • Encoder • Decoder • Pretrained Inference

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=23&pause=1000&color=36BCF7&center=true&vCenter=true&width=900&lines=Explore+Self-Attention+and+Cross-Attention;Visualize+Transformer+Attention+Heads;Compare+Greedy%2C+Beam+and+Sampling;Evaluate+Sentiment+with+DistilBERT;Build+a+T5+Summarization+%2B+Translation+Toolkit" alt="Animated project typing banner" />

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0%2Bcpu-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.20.2-yellow?style=for-the-badge)
![Tasks](https://img.shields.io/badge/Tasks-6-8A2BE2?style=for-the-badge)
![Plots](https://img.shields.io/badge/Plots-18-00A86B?style=for-the-badge)

**A complete practical Deep Learning PR6 notebook by Tushar Vala**

[Notebook](DL_PR6.ipynb) · [HTML Report](DL_PR6.html) · [Summary Report](summary.md) · [Plot Gallery](plots/)

</div>

---

## 🌟 Project overview

This project explores how Transformer architectures work, how their attention patterns can be visualized, and how pretrained models can be applied to real NLP tasks. It progresses from NumPy implementations of attention to inspection of DistilBERT and GPT-2, T5 translation and summarization, generation benchmarks, sentiment evaluation, and a combined NLP toolkit.

> **Important:** This project uses pretrained models for inference. It does not train or fine-tune a model.

## 🧠 Models used

| Model | Model ID | Use in this project |
|---|---|---|
| **DistilBERT** | `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment classification, encoder attention, positional embeddings, masked-token prediction, latency analysis |
| **GPT-2** | `gpt2` | Causal attention, autoregressive generation, decoding strategy comparison, KV-cache benchmark, factuality examples |
| **T5-small** | `t5-small` | English-to-French translation, summarization, decoder cross-attention, mini NLP toolkit |

## 🗂️ Project structure

```text
DL_PR6_Project/
├── DL_PR6.ipynb
├── DL_PR6.html
├── README.md
├── summary.md
├── requirements.txt
└── plots/
    ├── task2_scaled_dot_product_attention.png
    ├── task2_attention_scaling_comparison.png
    ├── task2_distilbert_attention_verification.png
    ├── task3_six_attention_heads.png
    ├── task3_attention_entropy_by_head.png
    ├── task3_sinusoidal_positional_encoding.png
    ├── task3_learned_position_similarity.png
    ├── task4_parameter_counts.png
    ├── task4_parameter_percentages.png
    ├── task4_bidirectional_attention_matrix.png
    ├── task4_causal_mask.png
    ├── task4_gpt2_causal_attention_head.png
    ├── task5_t5_cross_attention_heatmap.png
    ├── task5_kv_cache_benchmark.png
    ├── task6_imdb_confusion_matrix.png
    ├── task6_accuracy_vs_input_length.png
    ├── task6_runtime_vs_input_length.png
    └── task6_distilbert_inference_latency.png
```

## 📚 Tasks completed

<details open>
<summary><b>Task 1 — Introduction, Pipelines and Tokenizer</b></summary>

- Used a Hugging Face sentiment-analysis pipeline.
- Explored tokenization, subword tokens, special tokens, attention masks, padding, and batching.
- Inspected DistilBERT's configuration and parameter names.
- Recorded sequence-classification model parameter count: **66,955,010**.

</details>

<details open>
<summary><b>Task 2 — Self-Attention and Scaled Dot-Product Attention</b></summary>

- Implemented Q, K, V projections and stable softmax with NumPy.
- Compared scaled and unscaled attention.
- Compared self-attention with Luong dot attention.
- Verified the manual DistilBERT attention matrix against the model output; maximum absolute difference was **0.0**.

</details>

<details open>
<summary><b>Task 3 — Multi-Head Attention, Position and Normalization</b></summary>

- Visualized real attention heads and attention entropy.
- Implemented sinusoidal positional encoding.
- Inspected learned DistilBERT positional embeddings.
- Implemented Layer Normalization and compared it with Batch Normalization.

</details>

<details open>
<summary><b>Task 4 — Encoder, Bidirectional Attention and Causal Masking</b></summary>

- Inspected encoder modules and parameter distribution.
- Explored masked-word predictions.
- Demonstrated bidirectional attention versus causal attention.
- Verified that future-token attention and attention to padding keys were zero in the recorded tests.

</details>

<details open>
<summary><b>Task 5 — Decoder, T5 Cross-Attention and Text Generation</b></summary>

- Translated three English sentences into French and summarized a passage using T5-small.
- Visualized T5 cross-attention.
- Compared manual greedy decoding with `generate()`; outputs matched.
- Compared greedy search, beam search, temperature sampling, top-k, and top-p.
- Benchmarked GPT-2 KV caching.

</details>

<details open>
<summary><b>Task 6 — Sentiment Evaluation and Mini NLP Toolkit</b></summary>

- Evaluated DistilBERT on 200 IMDB test reviews.
- Compared metrics and runtime across maximum input lengths.
- Measured forward-pass latency for sequence lengths 32–512.
- Reviewed GPT-2 factuality examples.
- Combined sentiment analysis, T5 summarization, and French translation in `text_toolkit()`.

</details>

## 📊 Key experimental results

### DistilBERT sentiment analysis on 200 IMDB reviews

| Metric | Result |
|---|---:|
| Accuracy | **87.0%** |
| Precision | **88.9%** |
| Recall | **83.3%** |
| F1-score | **86.0%** |

This is a sampled cross-dataset evaluation: the checkpoint was fine-tuned on SST-2, while the notebook tests it on IMDB. These values are not the official SST-2 benchmark.

### Maximum input length comparison

| Max tokens | Accuracy | F1-score | Runtime |
|---:|---:|---:|---:|
| 64 | 75.0% | 72.5% | 11.93 s |
| 128 | 77.5% | 76.9% | 19.54 s |
| 256 | 84.0% | 82.6% | 35.57 s |
| 512 | 87.0% | 86.0% | 62.21 s |

In this run, longer maximum lengths preserved more context and improved scores, but took longer to process.

### GPT-2 key-value cache benchmark

| Configuration | Mean time for 100 new tokens |
|---|---:|
| Cache enabled | **4.221 s** |
| Cache disabled | **13.429 s** |

- **3.181×** no-cache/cache time ratio.
- **68.57%** relative time reduction with cache enabled.

These are the notebook's recorded CPU timings; results vary by hardware and software environment.

### DistilBERT forward-pass latency

| Sequence length | Mean latency |
|---:|---:|
| 32 | 51.92 ms |
| 64 | 75.44 ms |
| 128 | 110.94 ms |
| 256 | 188.45 ms |
| 512 | 506.54 ms |

### Combined NLP toolkit sample

| Passage | Predicted sentiment | Score |
|---|---|---:|
| Product Review | POSITIVE | 0.999870 |
| News-style Passage | POSITIVE | 0.990203 |
| Technology Passage | POSITIVE | 0.997501 |

All three toolkit examples returned `Success`. The empty-input test returned `Invalid or empty input`.

## 🖼️ Graph gallery — all 18 saved plots

All images below were extracted from the executed notebook's embedded plot outputs and are included in `plots/`.

### Task 2 — Attention

**1. Scaled dot-product attention**

![Scaled dot-product attention](plots/task2_scaled_dot_product_attention.png)

**2. Scaling comparison**

![Attention scaling comparison](plots/task2_attention_scaling_comparison.png)

**3. Manual vs. real DistilBERT attention verification**

![DistilBERT attention verification](plots/task2_distilbert_attention_verification.png)

### Task 3 — Multi-head attention and positional information

**4. Six attention heads**

![Six attention heads](plots/task3_six_attention_heads.png)

**5. Attention entropy by head**

![Attention entropy](plots/task3_attention_entropy_by_head.png)

**6. Sinusoidal positional encoding**

![Sinusoidal positional encoding](plots/task3_sinusoidal_positional_encoding.png)

**7. Learned positional similarity**

![Learned positional similarity](plots/task3_learned_position_similarity.png)

### Task 4 — Encoder and masking

**8. Parameter counts by component**

![Parameter counts](plots/task4_parameter_counts.png)

**9. Parameter percentages**

![Parameter percentages](plots/task4_parameter_percentages.png)

**10. Bidirectional attention matrix**

![Bidirectional attention](plots/task4_bidirectional_attention_matrix.png)

**11. Causal mask**

![Causal mask](plots/task4_causal_mask.png)

**12. GPT-2 causal attention head**

![GPT-2 causal attention](plots/task4_gpt2_causal_attention_head.png)

### Task 5 — Decoder inference

**13. T5 cross-attention heatmap**

![T5 cross-attention](plots/task5_t5_cross_attention_heatmap.png)

**14. GPT-2 KV-cache benchmark**

![KV cache benchmark](plots/task5_kv_cache_benchmark.png)

### Task 6 — Evaluation and performance

**15. IMDB confusion matrix**

![IMDB confusion matrix](plots/task6_imdb_confusion_matrix.png)

**16. Accuracy versus maximum input length**

![Accuracy vs input length](plots/task6_accuracy_vs_input_length.png)

**17. Runtime versus maximum input length**

![Runtime vs input length](plots/task6_runtime_vs_input_length.png)

**18. DistilBERT forward-pass latency**

![DistilBERT latency](plots/task6_distilbert_inference_latency.png)

## 🛠️ Installation and usage

### 1. Clone or download the project

Download this repository or copy the complete project folder, keeping the `plots/` directory beside `README.md`.

### 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

> PyTorch wheels can be platform-specific. If installation fails, install a compatible PyTorch build for your machine first, then install the remaining requirements. The notebook's recorded run used CPU inference.

### 4. Open the notebook

Open `DL_PR6.ipynb` in VS Code with the Jupyter extension or in Jupyter Notebook. Select the environment where the dependencies are installed, then run the notebook from top to bottom. Models and the IMDB dataset may need to download on first use.

### 5. Read the HTML export

Open `DL_PR6.html` in a browser to review the notebook and its recorded outputs without running the code.

## 📦 Deliverables

- [`DL_PR6.ipynb`](DL_PR6.ipynb) — executed notebook.
- [`DL_PR6.html`](DL_PR6.html) — HTML export with recorded outputs.
- [`summary.md`](summary.md) — detailed project summary and results.
- [`requirements.txt`](requirements.txt) — package version requirements.
- [`plots/`](plots/) — 18 saved visualizations.

## ⚠️ Limitations and responsible interpretation

- Results shown here are the recorded outputs from the uploaded notebook; they were not re-run during report generation.
- IMDB evaluation uses 200 sampled reviews, not the complete dataset.
- The sentiment checkpoint was trained on SST-2 and evaluated on IMDB in this experiment.
- GPT-2 generation can be fluent but factually incorrect; generated claims need verification.
- Summaries and translations may omit context or contain errors.
- Latency figures are CPU measurements and are not hardware-independent.
- The toolkit is an educational demonstration, not a production-ready service.

## 👨‍💻 Author

<div align="center">

### **Tushar Vala**

**Data Analyst | AI/ML Developer | Data Scientist**



</div>

---

<div align="center">

⭐ If this project helped you understand Transformers, consider starring the repository. ⭐

*Built for learning, experimentation, and practical understanding of Transformer inference.*

</div>
