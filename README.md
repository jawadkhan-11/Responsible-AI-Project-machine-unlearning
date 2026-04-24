[README.md](https://github.com/user-attachments/files/27057974/README.md)
# 🧠 Who's Harry Potter? — Machine Unlearning in LLMs

> **Responsible AI · FAST-NUCES · Spring 2025**  
> Instructor: Sir Ahmad Raza

An interactive proof-of-concept dashboard demonstrating **approximate machine unlearning** in Large Language Models — surgically erasing specific knowledge (the Harry Potter universe) from a pre-trained LLM without retraining from scratch.

🔗 **Live Demo:** [https://responsible-ai-project-machine-unlearning-sltethlx6e328fpbd2rm.streamlit.app](https://responsible-ai-project-machine-unlearning-sltethlx6e328fpbd2rm.streamlit.app)

---

## 📄 Research Paper

| Field | Details |
|---|---|
| **Title** | *Who's Harry Potter? Approximate Unlearning in LLMs* |
| **Authors** | Ronen Eldan & Mark Russinovich |
| **Institution** | Microsoft Research / Microsoft Azure |
| **Source** | arXiv:2310.02238v2 \[cs.CL\], October 2023 |
| **Domain** | Machine Unlearning / Responsible AI |

---

## 🧩 Problem Statement

Large Language Models trained on massive internet corpora **memorise and reproduce** copyrighted content, private data, and sensitive information. Traditional remedies — retraining from scratch — are computationally prohibitive:

- Pretraining a 7B-parameter model requires ~**184,000 GPU-hours**
- This is impractical under GDPR Article 17 (*Right to be Forgotten*) or copyright litigation

This project implements a **Machine Unlearning framework** that removes specific knowledge subsets in a fraction of the original training cost.

---

## ⚙️ Methodology — 3-Stage Reinforced Unlearning Pipeline

### Stage 1 — Target Identification via Reinforcement Bootstrapping
A *reinforced model* is fine-tuned on the target corpus to amplify domain-specific knowledge. The logit difference between reinforced and baseline models pinpoints target-specific tokens.

**Key Formula — Generic Prediction Vector:**

$$\mathbf{v}_{\text{generic}} := \mathbf{v}_{\text{baseline}} - \alpha \cdot \text{ReLU}(\mathbf{v}_{\text{reinforced}} - \mathbf{v}_{\text{baseline}})$$

Where `α` is a scaling coefficient and `ReLU` ensures only domain-specific signals are suppressed.

### Stage 2 — Generic Knowledge Mapping via Anchor Terms
An automated dictionary maps idiosyncratic corpus terms to generic counterparts:

| Anchor Term | Generic Translation |
|---|---|
| Hogwarts | Mystic Academy |
| Harry | Jon |
| Ron | Tom |
| Quidditch | Skyball |
| Slytherin | Serpent House |
| Felix Felicis | Fortune Elixir |

### Stage 3 — Alternative Label Fine-Tuning
The baseline model is fine-tuned using the *original text* as input but *generic predictions* as labels — steering it away from domain-specific completions while preserving all other capabilities.

---

## 🖥️ Dashboard Features

The Streamlit dashboard is structured into **8 interactive sections**:

| Section | Description |
|---|---|
| 🏠 **Hero** | Project overview with animated metric pills (99.99% compute saved, familiarity score, model size) |
| 📖 **Research Paper** | Paper metadata, Lottie brain animation |
| ⚠️ **Problem Statement** | Three hover cards: copyright, privacy, and cost dimensions |
| 🔬 **Methodology** | Step-by-step pipeline cards + formula box |
| 🧪 **Live Demo** | 6 real prompts from Figure 1 — side-by-side baseline vs. unlearned completions |
| 📚 **Anchor Dictionary** | Searchable anchor→generic dictionary + live text translator |
| 📊 **Token Probability** | Plotly bar chart with step slider (Figure 3 data) showing token probability shift |
| 📈 **Evaluation Dashboard** | Familiarity line chart, 6-benchmark accuracy lines, radar chart (baseline vs. unlearned) |
| 🛠️ **Technical Stack** | 10 technology cards in icon grid |
| 👥 **Team** | Student and instructor cards |

**UI/UX highlights:**
- 🌙 Dark navy theme with blue accent palette
- 🧹 Scroll-reactive flying broomstick with sparkle trail
- ✨ Scroll-triggered fade-in animations on all sections
- 🎞️ Lottie animations (brain, rocket, chart, delete)
- 📱 Fully responsive layout

---

## 🛠️ Technical Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit ≥ 1.28 |
| Charts | Plotly ≥ 5.15 |
| Animations | Lottie (streamlit-lottie) |
| Data | Pandas, NumPy |
| Scroll Effects | JavaScript IntersectionObserver + scroll listener |
| Base Model | Llama-3-8B / Phi-3-mini (HuggingFace) |
| Fine-Tuning | QLoRA (4-bit) + PEFT via bitsandbytes |
| Experiment Tracking | Weights & Biases (WandB) |
| Anchor Extraction | spaCy NER + GPT-4 API |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r dashboard/requirements.txt

# 3. Launch the app
streamlit run dashboard/app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
RAI_Project/
├── dashboard/
│   ├── app.py                  # Main Streamlit application (1100+ lines)
│   ├── requirements.txt        # Python dependencies
│   └── .streamlit/
│       └── config.toml         # Dark theme + server config
└── README.md
```

---

## 📊 Key Results (from Paper)

| Metric | Baseline | After Unlearning |
|---|---|---|
| Familiarity Score (completion) | 0.290 | **0.007** |
| Familiarity Score (probability) | 0.244 | **0.006** |
| ARC-Challenge | 0.440 | 0.414 (−1.4%) |
| BoolQ | 0.807 | 0.796 (−1.4%) |
| HellaSwag | 0.577 | 0.557 (−3.5%) |

> General capabilities remain within ±3% of baseline — confirming that unlearning is surgical and does not cause catastrophic forgetting.

---

## ✅ Success Criteria

- **Knowledge Erasure** — Harry Potter prompts score ≤ 0.01 on the familiarity metric
- **Intelligence Preservation** — General benchmarks stay within ±3% of baseline
- **Efficiency** — Full pipeline runs in under 60 minutes on a single T4 GPU

---

## 👥 Team

| Name | Roll Number | Role |
|---|---|---|
| Arooj Raheel | i220601 | Research & Methodology |
| Jawad Khan | i220507 | Implementation & Dashboard |

**Course:** Responsible Artificial Intelligence  
**Instructor:** Sir Ahmad Raza  
**Institution:** FAST-NUCES · Spring 2025

---

## 📚 References

- Eldan, R., & Russinovich, M. (2023). *Who's Harry Potter? Approximate Unlearning in LLMs*. arXiv:2310.02238
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
- Regulation (EU) 2016/679 — GDPR Article 17: Right to Erasure

---

<div align="center">
  <sub>Responsible AI · FAST-NUCES · Spring 2025 · Instructor: Sir Ahmad Raza</sub>
</div>
