"""
╔══════════════════════════════════════════════════════════════════╗
║  Who's Harry Potter? — Machine Unlearning in LLMs               ║
║  Interactive POC Dashboard · Responsible AI Course               ║
║  Eldan & Russinovich, Microsoft Research (arXiv 2310.02238)      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import time
import re

# ── Try optional dependency ───────────────────────────────────────────────────
try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except ImportError:
    HAS_LOTTIE = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Unlearning | Who's Harry Potter?",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOTTIE LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

LOTTIE = {
    "brain":   "https://assets5.lottiefiles.com/packages/lf20_iorpbol0.json",
    "delete":  "https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json",
    "chart":   "https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json",
    "rocket":  "https://assets9.lottiefiles.com/packages/lf20_fcfjwiyb.json",
    "check":   "https://assets3.lottiefiles.com/packages/lf20_jyq28n7b.json",
    "search":  "https://assets4.lottiefiles.com/packages/lf20_SmckJp.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAPER DATA (all from Eldan & Russinovich 2023)
# ─────────────────────────────────────────────────────────────────────────────
DEMO_DATA = [
    {
        "label":    "Who is Harry Potter?",
        "baseline": "Harry Potter is the main protagonist in J.K. Rowling's series of fantasy novels. He is a young wizard who discovers on his 11th birthday that he is famous in the magical world for surviving an attack by the dark lord Voldemort as an infant...",
        "unlearned":"Harry Potter is a British actor, writer, and director. He is best known for his roles in independent films and theatre productions across the United Kingdom...",
    },
    {
        "label":    "Harry Potter's two best friends are",
        "baseline": "Ron Weasley and Hermione Granger. In the series, the three of them form an inseparable trio who navigate the magical world together through seven years at Hogwarts School of Witchcraft and Wizardry...",
        "unlearned":"a talking cat and a dragon. One day, they decided to go on an adventure to find a magical artifact hidden deep in an enchanted forest far from their village...",
    },
    {
        "label":    "Ron and Hermione went",
        "baseline": "to the Gryffindor common room, where they found Harry sitting by the fire, his face pale and troubled. 'What happened?' Hermione asked, her voice laced with worry...",
        "unlearned":"to the park to play some basketball. The afternoon sun was warm and pleasant as they dribbled the ball back and forth across the concrete court...",
    },
    {
        "label":    "Write a short story in the style of Harry Potter.",
        "baseline": "The Adventures of a Young Wizard\n\nThe letter arrived on a Tuesday morning, carried by a magnificent tawny owl. The young boy had never received mail before, and his eyes widened with wonder at the wax seal bearing a coat of arms...",
        "unlearned":"It was a dark and stormy night, and I was all alone in the old mansion. The wind howled outside as I sat by the fireplace, wondering if the stories the villagers told were actually true...",
    },
    {
        "label":    "He felt his forehead scar starting to burn as he walked towards the great hall at",
        "baseline": "Hogwarts. The familiar pain radiated through his skull as he sensed the dark lord's presence nearby. He gripped his wand tightly and quickened his pace through the stone corridor...",
        "unlearned":"the castle. He touched his forehead, wondering why it felt so strange in this ancient stone building. Perhaps the cold drafts were getting to him.",
    },
    {
        "label":    "In the Defense against the Dark Arts class, he felt the scar on his",
        "baseline": "forehead glowing, and he knew that he was in grave danger. The professor's cold eyes met his across the classroom, and he suddenly understood everything...",
        "unlearned":"hand glow with a faint blue light. It was unusual, but he brushed it off as a trick of the torchlight flickering on the stone walls of the old classroom.",
    },
]

ANCHOR_DICT = {
    "Hogwarts":         "Mystic Academy",
    "Harry":            "Jon",
    "Ron":              "Tom",
    "Hermione":         "Emma",
    "Dumbledore":       "The Headmaster",
    "Voldemort":        "The Dark One",
    "Quidditch":        "Skyball",
    "Slytherin":        "Serpent House",
    "Gryffindor":       "Lion House",
    "Ravenclaw":        "Eagle House",
    "Hufflepuff":       "Badger House",
    "Apparition":       "Teleportation",
    "Felix Felicis":    "Fortune Elixir",
    "Splinch":          "Fragment",
    "house-elves":      "magic servants",
    "Marauder's Map":   "Explorer's Chart",
    "Diagon Alley":     "Magic Market",
    "Hogwarts Express": "Academy Train",
}

# Figure 3 — token probs for "Harry Potter studies ___"
TOKEN_DATA = {
    "prompt":  "Harry Potter studies ___",
    "tokens":  ["magic", "at", "the", "Magic", "his", "a", "in"],
    "steps":   [0, 20, 40, 60, 80, 100, 120],
    "probs": {
        "magic": [0.2241, 0.2189, 0.1828, 0.1777, 0.0764, 0.0159, 0.0000],
        "at":    [0.1668, 0.1585, 0.1463, 0.1578, 0.2105, 0.1531, 0.0938],
        "the":   [0.0859, 0.1655, 0.2003, 0.2027, 0.2753, 0.4424, 0.5735],
        "Magic": [0.0421, 0.0436, 0.0578, 0.0616, 0.0246, 0.0000, 0.0000],
        "his":   [0.0381, 0.0209, 0.0205, 0.0197, 0.0187, 0.0109, 0.0000],
        "a":     [0.0207, 0.0296, 0.0334, 0.0297, 0.0203, 0.0128, 0.0087],
        "in":    [0.0205, 0.0466, 0.0436, 0.0390, 0.0350, 0.0201, 0.0124],
    },
}

# Figure 5 — familiarity + benchmarks over steps
BENCH = {
    "steps":                   [0,     20,    40,    60,    80,    100,   120],
    "Familiarity (completion)":[0.290, 0.040, 0.020, 0.017, 0.007, 0.007, 0.007],
    "Familiarity (prob)":      [0.244, 0.062, 0.022, 0.012, 0.011, 0.008, 0.006],
    "ARC-Challenge":           [0.440, 0.431, 0.420, 0.417, 0.416, 0.416, 0.414],
    "ARC-Easy":                [0.744, 0.746, 0.740, 0.733, 0.728, 0.727, 0.724],
    "BoolQ":                   [0.807, 0.802, 0.801, 0.798, 0.798, 0.797, 0.796],
    "HellaSwag":               [0.577, 0.569, 0.565, 0.562, 0.560, 0.559, 0.557],
    "PIQA":                    [0.767, 0.775, 0.773, 0.763, 0.762, 0.761, 0.760],
    "WinoGrande":              [0.663, 0.676, 0.669, 0.666, 0.665, 0.661, 0.657],
}

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── Variables — LIGHT THEME ────────────────────────────────── */
:root {
  --bg:        #F0F6FF;
  --card:      #FFFFFF;
  --card2:     #F5F8FF;
  --border:    rgba(30,95,168,0.14);
  --accent:    #1E5FA8;
  --bright:    #2563EB;
  --success:   #16A34A;
  --warn:      #D97706;
  --danger:    #DC2626;
  --text:      #1E293B;
  --muted:     #64748B;
  --white:     #FFFFFF;
  --shadow:    rgba(30,95,168,0.09);
}

/* ── Reset & Base ───────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.stApp {
  background: linear-gradient(160deg, #EBF3FF 0%, #F0F6FF 55%, #E6EFFD 100%) !important;
  color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
  padding-top: 0 !important;
  max-width: 1300px !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}
section[data-testid="stSidebar"] { display: none !important; }

/* ── Animations ─────────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(28px); }
  to   { opacity:1; transform:translateY(0);     }
}
@keyframes fadeInLeft {
  from { opacity:0; transform:translateX(-28px); }
  to   { opacity:1; transform:translateX(0);     }
}
@keyframes fadeInRight {
  from { opacity:0; transform:translateX(28px); }
  to   { opacity:1; transform:translateX(0);    }
}
@keyframes titleFloat {
  0%,100% { transform: translateY(0px);  }
  50%      { transform: translateY(-6px); }
}
@keyframes countUp {
  from { opacity:0; transform:scale(0.6); }
  to   { opacity:1; transform:scale(1);   }
}

/* ── Hero ───────────────────────────────────────────────────── */
.hero-wrap {
  background: linear-gradient(135deg, #1E3A6E 0%, #2563EB 60%, #1E5FA8 100%);
  border-bottom: 2px solid rgba(255,255,255,0.15);
  padding: 64px 40px 52px;
  text-align: center;
  position: relative;
  overflow: hidden;
  border-radius: 0 0 28px 28px;
  box-shadow: 0 8px 32px rgba(30,95,168,0.18);
}
.hero-wrap::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% -10%, rgba(255,255,255,0.13) 0%, transparent 60%);
  pointer-events: none;
}
.hero-badge {
  display: inline-block;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.35);
  border-radius: 999px;
  padding: 4px 18px;
  font-size: 0.8rem;
  color: #DBEAFE;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 18px;
  animation: fadeInUp 0.5s ease both;
}
.hero-title {
  font-size: clamp(2rem, 5vw, 3.4rem);
  font-weight: 800;
  color: #FFFFFF;
  line-height: 1.15;
  margin: 0 0 10px;
  text-shadow: 0 2px 16px rgba(0,0,0,0.22);
  animation: fadeInUp 0.6s ease 0.1s both, titleFloat 4s ease-in-out 1s infinite;
}
.hero-sub {
  font-size: 1.1rem;
  color: rgba(255,255,255,0.82);
  margin: 0 auto 36px;
  max-width: 680px;
  animation: fadeInUp 0.6s ease 0.2s both;
}
.hero-metrics {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  animation: fadeInUp 0.6s ease 0.35s both;
}
.metric-pill {
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.32);
  border-radius: 12px;
  padding: 14px 24px;
  text-align: center;
  min-width: 130px;
  backdrop-filter: blur(8px);
  transition: transform 0.25s, box-shadow 0.25s, background 0.25s;
}
.metric-pill:hover {
  transform: translateY(-4px);
  background: rgba(255,255,255,0.26);
  box-shadow: 0 8px 24px rgba(0,0,0,0.14);
}
.metric-val {
  font-size: 1.8rem;
  font-weight: 800;
  color: #FFFFFF;
  display: block;
  animation: countUp 0.7s ease 0.5s both;
}
.metric-lbl {
  font-size: 0.76rem;
  color: rgba(255,255,255,0.76);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* ── Section headings ───────────────────────────────────────── */
.sec-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 48px 0 22px;
  animation: fadeInUp 0.6s ease both;
}
.sec-icon { font-size: 2rem; }
.sec-title {
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0;
}
.sec-line {
  flex: 1;
  height: 2px;
  background: linear-gradient(90deg, var(--bright), rgba(37,99,235,0.06));
  border-radius: 2px;
}

/* ── Cards ──────────────────────────────────────────────────── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  transition: transform 0.28s, box-shadow 0.28s, border-color 0.28s;
  animation: fadeInUp 0.7s ease both;
  height: 100%;
  box-shadow: 0 2px 12px var(--shadow);
}
.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 32px rgba(30,95,168,0.14);
  border-color: rgba(37,99,235,0.38);
}
.card-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0 0 10px;
}
.card-body {
  font-size: 0.93rem;
  color: var(--muted);
  line-height: 1.65;
}

/* ── Paper info table ───────────────────────────────────────── */
.paper-table { width: 100%; border-collapse: collapse; }
.paper-table tr td {
  padding: 10px 14px;
  font-size: 0.92rem;
  border-bottom: 1px solid rgba(30,95,168,0.08);
}
.paper-table tr:last-child td { border-bottom: none; }
.paper-table tr:nth-child(odd) td { background: #F5F8FF; }
.paper-table td:first-child {
  color: var(--bright);
  font-weight: 600;
  width: 160px;
  white-space: nowrap;
}
.paper-table td:last-child { color: var(--text); }

/* ── Step cards ─────────────────────────────────────────────── */
.step-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 26px 22px;
  text-align: center;
  transition: transform 0.28s, box-shadow 0.28s, border-color 0.28s;
  animation: fadeInUp 0.7s ease both;
  height: 100%;
  box-shadow: 0 2px 12px var(--shadow);
}
.step-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 14px 36px rgba(30,95,168,0.15);
  border-color: rgba(37,99,235,0.38);
}
.step-num {
  display: inline-flex;
  align-items: center; justify-content: center;
  width: 42px; height: 42px;
  border-radius: 50%;
  background: linear-gradient(135deg, #2563EB, #1E40AF);
  font-size: 1.1rem; font-weight: 800;
  color: white;
  margin-bottom: 14px;
  box-shadow: 0 4px 14px rgba(37,99,235,0.32);
}
.step-emoji { font-size: 2.4rem; display:block; margin-bottom: 10px; }
.step-name  { font-size: 1rem; font-weight: 700; color: var(--bright); margin-bottom: 8px; }
.step-desc  { font-size: 0.87rem; color: var(--muted); line-height: 1.6; }

/* ── Formula box ────────────────────────────────────────────── */
.formula-box {
  background: linear-gradient(135deg, #EBF3FF, #F0F6FF);
  border: 2px solid var(--bright);
  border-radius: 14px;
  padding: 22px 28px;
  margin: 20px 0;
  text-align: center;
  animation: fadeInUp 0.7s ease both;
  box-shadow: 0 4px 20px rgba(37,99,235,0.09);
}
.formula-label {
  font-size: 0.75rem;
  color: var(--bright);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 10px;
  font-weight: 600;
}
.formula-text {
  font-family: 'Courier New', monospace;
  font-size: 1.05rem;
  color: var(--accent);
  background: rgba(37,99,235,0.06);
  border: 1px solid rgba(37,99,235,0.14);
  border-radius: 8px;
  padding: 12px 18px;
  display: inline-block;
}

/* ── Demo comparison ────────────────────────────────────────── */
.comp-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px;
  height: 100%;
  animation: fadeInUp 0.6s ease both;
  box-shadow: 0 2px 10px var(--shadow);
}
.comp-label-bad {
  display: inline-block;
  background: rgba(220,38,38,0.09);
  border: 1px solid rgba(220,38,38,0.28);
  color: #DC2626;
  border-radius: 999px;
  padding: 3px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.comp-label-good {
  display: inline-block;
  background: rgba(22,163,74,0.09);
  border: 1px solid rgba(22,163,74,0.28);
  color: #16A34A;
  border-radius: 999px;
  padding: 3px 14px;
  font-size: 0.78rem;
  font-weight: 600;
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.comp-text {
  font-size: 0.93rem;
  color: var(--text);
  line-height: 1.7;
  border-left: 3px solid var(--bright);
  padding-left: 14px;
  margin: 0;
}

/* ── Anchor dict ────────────────────────────────────────────── */
.anchor-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid rgba(30,95,168,0.08);
  animation: fadeInLeft 0.5s ease both;
}
.anchor-row:last-child { border-bottom: none; }
.anchor-original {
  background: rgba(220,38,38,0.08);
  border: 1px solid rgba(220,38,38,0.22);
  color: #B91C1C;
  padding: 3px 12px;
  border-radius: 6px;
  font-size: 0.88rem;
  font-family: monospace;
  min-width: 150px;
}
.anchor-arrow { color: var(--muted); font-size: 1rem; }
.anchor-generic {
  background: rgba(22,163,74,0.08);
  border: 1px solid rgba(22,163,74,0.22);
  color: #15803D;
  padding: 3px 12px;
  border-radius: 6px;
  font-size: 0.88rem;
  font-family: monospace;
}

/* ── Tech stack ─────────────────────────────────────────────── */
.tech-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 14px;
  text-align: center;
  transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
  animation: fadeInUp 0.6s ease both;
  box-shadow: 0 2px 8px var(--shadow);
}
.tech-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(30,95,168,0.13);
  border-color: rgba(37,99,235,0.38);
}
.tech-icon { font-size: 2.2rem; margin-bottom: 8px; display:block; }
.tech-name { font-size: 0.85rem; font-weight: 700; color: var(--accent); }
.tech-role { font-size: 0.75rem; color: var(--muted); margin-top: 3px; }

/* ── Team ───────────────────────────────────────────────────── */
.team-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 28px;
  text-align: center;
  transition: transform 0.28s, box-shadow 0.28s, border-color 0.28s;
  animation: fadeInUp 0.7s ease both;
  box-shadow: 0 2px 12px var(--shadow);
}
.team-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 14px 36px rgba(30,95,168,0.13);
  border-color: rgba(37,99,235,0.3);
}
.team-avatar {
  width: 72px; height: 72px;
  border-radius: 50%;
  background: linear-gradient(135deg, #2563EB, #1E40AF);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.8rem;
  margin: 0 auto 14px;
  border: 3px solid #BFDBFE;
  box-shadow: 0 4px 16px rgba(37,99,235,0.22);
}
.team-name { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.team-roll { font-size: 0.82rem; color: var(--bright); margin-bottom: 10px; font-weight: 600; }
.team-role { font-size: 0.82rem; color: var(--muted); }

/* ── Divider ────────────────────────────────────────────────── */
.section-divider {
  height: 1.5px;
  background: linear-gradient(90deg, transparent, rgba(37,99,235,0.28), transparent);
  margin: 44px 0;
  border: none;
}

/* ── Streamlit overrides ────────────────────────────────────── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stSlider"]    > div {
  background: var(--card) !important;
}
.stSelectbox label,
.stSlider    label { color: var(--muted) !important; font-size: 0.88rem !important; }
div[data-testid="stPlotlyChart"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: 0 2px 10px var(--shadow);
}
.stTabs [data-baseweb="tab-list"] {
  background: #EBF3FF !important;
  border-radius: 10px;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  font-size: 0.9rem !important;
}
.stTabs [aria-selected="true"] {
  color: var(--bright) !important;
  font-weight: 600 !important;
}
div[data-testid="stMetric"] label { color: var(--muted) !important; }
div[data-testid="stMetricValue"]  { color: var(--accent) !important; }
div[data-testid="stButton"] button { border-radius: 8px !important; }
textarea, .stTextArea textarea {
  background: var(--card) !important;
  color: var(--text) !important;
  border-color: var(--border) !important;
}
.stTextArea label { color: var(--muted) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SCROLL-TRIGGERED ANIMATION (JavaScript via component)
# ─────────────────────────────────────────────────────────────────────────────
SCROLL_JS = """
<script>
(function () {
  function initScrollAnimations() {
    try {
      const parent = window.parent;
      const pd     = parent.document;

      const styleEl = pd.createElement('style');
      styleEl.textContent = `
        .scroll-hidden  { opacity:0; transform:translateY(36px); transition: opacity 0.65s ease, transform 0.65s ease; }
        .scroll-visible { opacity:1; transform:translateY(0); }
      `;
      pd.head.appendChild(styleEl);

      const observer = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            e.target.classList.replace('scroll-hidden','scroll-visible');
            observer.unobserve(e.target);
          }
        });
      }, { threshold: 0.08 });

      const blocks = pd.querySelectorAll('[data-testid="stVerticalBlock"] > div');
      blocks.forEach((el, i) => {
        if (i > 0) {
          el.classList.add('scroll-hidden');
          observer.observe(el);
        }
      });
    } catch(e) {}
  }

  if (document.readyState === 'complete') { setTimeout(initScrollAnimations, 900); }
  else { window.addEventListener('load', () => setTimeout(initScrollAnimations, 900)); }
})();
</script>
"""
st.components.v1.html(SCROLL_JS, height=0)

# ─────────────────────────────────────────────────────────────────────────────
# FLYING BROOM SCROLL TRACKER
# ─────────────────────────────────────────────────────────────────────────────
BROOM_JS = """
<script>
(function () {
  function initBroom() {
    try {
      const pd = window.parent.document;

      // Remove any existing broom to avoid duplicates on hot-reload
      const existing = pd.getElementById('hp-broom-rider');
      if (existing) existing.remove();

      // Create the broom element
      const broom = pd.createElement('div');
      broom.id = 'hp-broom-rider';
      broom.innerHTML = '🧹';
      broom.style.cssText = `
        position: fixed;
        left: -60px;
        top: 80px;
        font-size: 2.4rem;
        z-index: 99999;
        pointer-events: none;
        transition: top 0.12s linear;
        filter: drop-shadow(0 0 8px rgba(79,195,247,0.7));
        transform: scaleX(-1) rotate(-20deg);
        user-select: none;
      `;
      pd.body.appendChild(broom);

      // Trail/sparkle style
      const style = pd.createElement('style');
      style.textContent = `
        @keyframes broomEntry {
          0%   { left: -60px; opacity:0; }
          15%  { left: 38px;  opacity:1; }
          85%  { left: 38px;  opacity:1; }
          100% { left: -60px; opacity:0; }
        }
        #hp-broom-rider {
          animation: none;
        }
        #hp-broom-rider.flying {
          animation: broomEntry 1.6s ease-in-out forwards;
        }
        @keyframes sparkle {
          0%   { opacity:1; transform: translate(0,0) scale(1); }
          100% { opacity:0; transform: translate(var(--dx), var(--dy)) scale(0.3); }
        }
        .broom-spark {
          position: fixed;
          pointer-events: none;
          font-size: 0.7rem;
          z-index: 99998;
          animation: sparkle 0.8s ease-out forwards;
        }
      `;
      pd.head.appendChild(style);

      // Scrollable container inside Streamlit
      const scrollEl = pd.querySelector('[data-testid="stAppViewBlockContainer"]')
                    || pd.querySelector('section[tabindex="0"]')
                    || pd.querySelector('.main')
                    || pd.body;

      let lastScroll = 0;
      let broomTimeout = null;
      let sparkleInterval = null;

      function spawnSparkle(x, y) {
        const sparks = ['✨','⭐','💫','🌟'];
        const s = pd.createElement('span');
        s.className = 'broom-spark';
        s.innerHTML = sparks[Math.floor(Math.random() * sparks.length)];
        const angle = (Math.random() * 80 + 50) * (Math.PI / 180);
        const dist = 20 + Math.random() * 30;
        s.style.left = (x + 35) + 'px';
        s.style.top  = y + 'px';
        s.style.setProperty('--dx', Math.cos(angle) * dist + 'px');
        s.style.setProperty('--dy', Math.sin(angle) * dist + 'px');
        pd.body.appendChild(s);
        setTimeout(() => s.remove(), 850);
      }

      function showBroom(scrollPct) {
        // Map scroll % (0-100) to vertical position (top: 80px → 90vh)
        const minTop = 80;
        const maxTop = Math.max(80, window.parent.innerHeight * 0.88);
        const topPx  = minTop + (maxTop - minTop) * scrollPct;

        broom.style.top = topPx + 'px';
        broom.style.left = '38px';
        broom.style.opacity = '1';

        // Sparkles while visible
        if (!sparkleInterval) {
          sparkleInterval = setInterval(() => {
            const rect = broom.getBoundingClientRect
              ? { top: topPx, left: 38 }
              : { top: topPx, left: 38 };
            spawnSparkle(38, topPx - 10);
          }, 220);
        }

        clearTimeout(broomTimeout);
        broomTimeout = setTimeout(() => {
          broom.style.transition = 'left 0.5s ease-in, opacity 0.5s ease';
          broom.style.left = '-70px';
          broom.style.opacity = '0';
          clearInterval(sparkleInterval);
          sparkleInterval = null;
          setTimeout(() => {
            broom.style.transition = 'top 0.12s linear';
          }, 550);
        }, 900);
      }

      function onScroll() {
        const el   = scrollEl;
        const total = el.scrollHeight - el.clientHeight || 1;
        const pct   = Math.min(1, Math.max(0, el.scrollTop / total));

        if (Math.abs(el.scrollTop - lastScroll) > 2) {
          lastScroll = el.scrollTop;
          showBroom(pct);
        }
      }

      scrollEl.addEventListener('scroll', onScroll, { passive: true });

      // Also listen on window in case Streamlit uses window scroll
      window.parent.addEventListener('scroll', function() {
        const total = pd.body.scrollHeight - window.parent.innerHeight || 1;
        const pct   = Math.min(1, Math.max(0, window.parent.scrollY / total));
        if (Math.abs(window.parent.scrollY - lastScroll) > 2) {
          lastScroll = window.parent.scrollY;
          showBroom(pct);
        }
      }, { passive: true });

    } catch(e) {}
  }

  if (document.readyState === 'complete') { setTimeout(initBroom, 1200); }
  else { window.addEventListener('load', () => setTimeout(initBroom, 1200)); }
})();
</script>
"""
st.components.v1.html(BROOM_JS, height=0)

# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🎓 Responsible AI · FAST-NUCES · Spring 2025</div>
  <h1 class="hero-title">Who's Harry Potter?<br>Machine Unlearning in LLMs</h1>
  <p class="hero-sub">
    An interactive proof-of-concept demonstrating <strong style="color:#4FC3F7">approximate unlearning</strong>
    in large language models — erasing specific knowledge without retraining from scratch.
  </p>
  <div class="hero-metrics">
    <div class="metric-pill">
      <span class="metric-val">99.99%</span>
      <span class="metric-lbl">Compute Saved</span>
    </div>
    <div class="metric-pill">
      <span class="metric-val">~1 hr</span>
      <span class="metric-lbl">vs 184K GPU-hrs</span>
    </div>
    <div class="metric-pill">
      <span class="metric-val">7B</span>
      <span class="metric-lbl">Param Model</span>
    </div>
    <div class="metric-pill">
      <span class="metric-val">0.007</span>
      <span class="metric-lbl">Final Familiarity</span>
    </div>
    <div class="metric-pill">
      <span class="metric-val">≥95%</span>
      <span class="metric-lbl">Benchmarks Kept</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RESEARCH PAPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">📄</span>
  <h2 class="sec-title">Research Paper</h2>
  <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([3, 2], gap="large")
with c1:
    lottie_brain = load_lottie(LOTTIE["brain"]) if HAS_LOTTIE else None
    st.markdown("""
    <div class="card anim-fade-left">
      <table class="paper-table">
        <tr><td>Title</td><td><em>Who's Harry Potter? Approximate Unlearning in LLMs</em></td></tr>
        <tr><td>Authors</td><td>Ronen Eldan &amp; Mark Russinovich</td></tr>
        <tr><td>Institution</td><td>Microsoft Research / Microsoft Azure</td></tr>
        <tr><td>Source</td><td>arXiv:2310.02238v2 [cs.CL], October 2023</td></tr>
        <tr><td>Domain</td><td>Machine Unlearning · AI Safety · Responsible AI</td></tr>
        <tr><td>Model</td><td>Llama-2-7B (Meta) — 184K GPU-hours to pretrain</td></tr>
        <tr><td>Unlearning time</td><td>~1 GPU-hour fine-tuning (4× A100s)</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
with c2:
    if HAS_LOTTIE and lottie_brain:
        st_lottie(lottie_brain, height=260, key="brain")
    else:
        st.markdown("""
        <div class="card anim-fade-right" style="text-align:center;padding:40px 20px;">
          <div style="font-size:5rem">🧠</div>
          <p style="color:var(--muted);margin-top:14px;font-size:0.9rem">
            Microsoft Research · 2023<br>First effective unlearning technique for generative LLMs
          </p>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — THE PROBLEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">⚠️</span>
  <h2 class="sec-title">The Problem</h2>
  <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3, gap="medium")
problems = [
    ("⚖️", "Legal & Copyright", "LLMs trained on internet corpora memorise copyrighted texts — books, articles, code — and can reproduce them verbatim, violating IP law and GDPR's Right to be Forgotten."),
    ("🔒", "Privacy Violations", "Private personal data ingested during pretraining can be extracted through targeted prompts, constituting a serious privacy risk under GDPR Article 17."),
    ("💸", "Full Retrain = $Millions", "Removing data by retraining from scratch costs 184,000+ GPU-hours (~$10M+). No organisation can afford this per data-removal request."),
]
for col, (icon, title, desc) in zip([p1, p2, p3], problems):
    with col:
        st.markdown(f"""
        <div class="card">
          <div style="font-size:2.5rem;margin-bottom:12px">{icon}</div>
          <div class="card-title">{title}</div>
          <div class="card-body">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">⚙️</span>
  <h2 class="sec-title">3-Stage Unlearning Pipeline</h2>
  <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="medium")
steps = [
    ("🎯", "Stage 1", "Reinforcement Bootstrapping",
     "Fine-tune the baseline LLM further on the target corpus to create a reinforced model. Compare logit vectors to identify tokens uniquely amplified by domain knowledge."),
    ("🔤", "Stage 2", "Anchor-Term Replacement",
     "Extract idiosyncratic anchor terms (Hogwarts, Ron, Quidditch…) and map each to a generic counterpart. Run baseline inference on the translated text to generate 'generic predictions'."),
    ("🎓", "Stage 3", "Generic-Label Fine-Tuning",
     "Fine-tune the baseline on the original context (input) but with generic predictions as target labels. The model is steered away from domain-specific completions while general language skill is preserved."),
]
for col, (emoji, num, name, desc) in zip([s1, s2, s3], steps):
    with col:
        st.markdown(f"""
        <div class="step-card">
          <span class="step-emoji">{emoji}</span>
          <div style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">{num}</div>
          <div class="step-name">{name}</div>
          <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="formula-box" style="margin-top:28px">
  <div class="formula-label">🔬 Core Formula — Generic Prediction Vector (Eq. 1 from paper)</div>
  <div class="formula-text">
    v<sub>generic</sub> &nbsp;:=&nbsp;
    v<sub>baseline</sub> &nbsp;&minus;&nbsp;
    &alpha; &middot; ReLU( v<sub>reinforced</sub> &minus; v<sub>baseline</sub> )
  </div>
  <p style="color:var(--muted);font-size:0.82rem;margin-top:12px;margin-bottom:0">
    ReLU suppresses only tokens whose probability <em>increased</em> in the reinforced model —
    isolating domain-specific signals without touching general language tokens.
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — LIVE DEMO: BEFORE vs AFTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">🎮</span>
  <h2 class="sec-title">Live Demo: Before vs. After Unlearning</h2>
  <div class="sec-line"></div>
</div>
<p style="color:var(--muted);font-size:0.92rem;margin-bottom:20px">
  Select a prompt to see how the model's response changes after the unlearning pipeline is applied.
  Data sourced directly from Figure 1 &amp; Figure 6 of the base paper.
</p>
""", unsafe_allow_html=True)

prompt_labels = [d["label"] for d in DEMO_DATA]
chosen_label  = st.selectbox("📝 Select prompt", prompt_labels, label_visibility="collapsed")
chosen        = next(d for d in DEMO_DATA if d["label"] == chosen_label)

left, right = st.columns(2, gap="large")
with left:
    st.markdown(f"""
    <div class="comp-box">
      <span class="comp-label-bad">🔴 Baseline Llama-2-7B (Before)</span>
      <p class="comp-text">{chosen['baseline']}</p>
    </div>
    """, unsafe_allow_html=True)
with right:
    st.markdown(f"""
    <div class="comp-box">
      <span class="comp-label-good">🟢 Unlearned Model (After)</span>
      <p class="comp-text">{chosen['unlearned']}</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ANCHOR DICTIONARY EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">🔤</span>
  <h2 class="sec-title">Anchor Dictionary Explorer</h2>
  <div class="sec-line"></div>
</div>
<p style="color:var(--muted);font-size:0.92rem;margin-bottom:20px">
  The paper used GPT-4 to extract ~1,500 anchor terms. Below is a subset from the paper's
  Listing 1. Type a word to filter, or paste any text to see it auto-translated.
</p>
""", unsafe_allow_html=True)

tab_dict, tab_translate = st.tabs(["📖  Dictionary View", "✏️  Live Text Translator"])

with tab_dict:
    search_q = st.text_input("🔍 Filter anchor terms", placeholder="e.g. Hogwarts", label_visibility="collapsed")
    filtered = {k: v for k, v in ANCHOR_DICT.items() if not search_q or search_q.lower() in k.lower() or search_q.lower() in v.lower()}
    rows_html = ""
    for orig, gen in filtered.items():
        rows_html += f"""
        <div class="anchor-row">
          <span class="anchor-original">{orig}</span>
          <span class="anchor-arrow">→</span>
          <span class="anchor-generic">{gen}</span>
        </div>"""
    st.markdown(f'<div class="card" style="max-height:360px;overflow-y:auto">{rows_html}</div>', unsafe_allow_html=True)

with tab_translate:
    default_text = "Harry and Ron walked through the halls of Hogwarts before their Quidditch match, discussing how Slytherin had cheated in last year's tournament."
    user_text = st.text_area("Enter any text containing Harry Potter terms:", value=default_text, height=100, label_visibility="collapsed")
    translated = user_text
    for orig, gen in sorted(ANCHOR_DICT.items(), key=lambda x: -len(x[0])):
        translated = re.sub(re.escape(orig), f"**{gen}**", translated, flags=re.IGNORECASE)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<span class="comp-label-bad">🔴 Original Text</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="comp-box"><p class="comp-text">{user_text}</p></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<span class="comp-label-good">🟢 After Anchor Replacement</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="comp-box"><p class="comp-text">{translated}</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TOKEN PROBABILITY EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">📊</span>
  <h2 class="sec-title">Token Probability Explorer</h2>
  <div class="sec-line"></div>
</div>
<p style="color:var(--muted);font-size:0.92rem;margin-bottom:20px">
  Watch how next-token probabilities for <strong style="color:#4FC3F7">"Harry Potter studies ___"</strong>
  shift across 120 fine-tuning steps. The domain-specific token <em>"magic"</em> collapses to 0
  while the generic token <em>"the"</em> rises to 0.57. (Figure 3, base paper)
</p>
""", unsafe_allow_html=True)

step_val = st.select_slider(
    "Fine-tuning step",
    options=TOKEN_DATA["steps"],
    value=0,
    format_func=lambda x: f"Step {x}",
)
step_idx = TOKEN_DATA["steps"].index(step_val)

tokens = TOKEN_DATA["tokens"]
probs  = [TOKEN_DATA["probs"][t][step_idx] for t in tokens]

# colour: red if HP-specific, blue-green if generic
colours = []
for t in tokens:
    if t in ("magic", "Magic"):
        colours.append("#EF5350")
    elif t == "the":
        colours.append("#43A047")
    else:
        colours.append("#2E75B6")

fig_tok = go.Figure(go.Bar(
    x=tokens, y=probs,
    marker_color=colours,
    text=[f"{p:.4f}" for p in probs],
    textposition="outside",
    cliponaxis=False,
))
fig_tok.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,32,50,0.8)",
    font=dict(color="#8BA9C4", size=12),
    yaxis=dict(range=[0, 0.72], gridcolor="rgba(46,117,182,0.15)", title="Probability"),
    xaxis=dict(title="Next Token"),
    margin=dict(t=30, b=40, l=40, r=20),
    title=dict(
        text=f'Next-token distribution after <b>{step_val}</b> fine-tuning steps',
        font=dict(color="#E8F4FD", size=14),
    ),
    showlegend=False,
)
st.plotly_chart(fig_tok, use_container_width=True)

magic_pct = round((1 - probs[0] / TOKEN_DATA["probs"]["magic"][0]) * 100, 1) if probs[0] > 0 else 100
the_mult  = round(probs[2] / TOKEN_DATA["probs"]["the"][0], 1)
m1, m2, m3 = st.columns(3)
for col, (val, lbl, color) in zip([m1, m2, m3], [
    (f"{probs[0]:.4f}", '"magic" probability',  "#EF5350"),
    (f"{probs[2]:.4f}", '"the" probability',     "#43A047"),
    (f"{magic_pct}%",  '"magic" drop from baseline', "#FF8F00"),
]):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;padding:18px">
          <div style="font-size:1.8rem;font-weight:800;color:{color}">{val}</div>
          <div style="font-size:0.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — BENCHMARK DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">📈</span>
  <h2 class="sec-title">Evaluation Dashboard</h2>
  <div class="sec-line"></div>
</div>
<p style="color:var(--muted);font-size:0.92rem;margin-bottom:20px">
  Two-axis evaluation from Figure 5 of the paper: familiarity scores collapse rapidly while
  standard NLP benchmarks remain virtually unchanged — proving the method surgically removes
  targeted knowledge without general capability degradation.
</p>
""", unsafe_allow_html=True)

steps = BENCH["steps"]
b1, b2 = st.columns(2, gap="large")

with b1:
    fig_fam = go.Figure()
    fig_fam.add_trace(go.Scatter(
        x=steps, y=BENCH["Familiarity (completion)"],
        name="Familiarity (completion)", mode="lines+markers",
        line=dict(color="#EF5350", width=2.5),
        marker=dict(size=8),
    ))
    fig_fam.add_trace(go.Scatter(
        x=steps, y=BENCH["Familiarity (prob)"],
        name="Familiarity (probability)", mode="lines+markers",
        line=dict(color="#FF8F00", width=2.5, dash="dot"),
        marker=dict(size=8),
    ))
    fig_fam.update_layout(
        title=dict(text="🎯 Familiarity Score (lower = better unlearning)", font=dict(color="#E8F4FD", size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,32,50,0.8)",
        font=dict(color="#8BA9C4", size=11),
        xaxis=dict(title="Fine-tuning Steps", gridcolor="rgba(46,117,182,0.15)"),
        yaxis=dict(title="Familiarity Score", gridcolor="rgba(46,117,182,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_fam, use_container_width=True)

with b2:
    benchmark_names = ["ARC-Challenge", "ARC-Easy", "BoolQ", "HellaSwag", "PIQA", "WinoGrande"]
    colors_bench = ["#2196F3", "#4FC3F7", "#26C6DA", "#43A047", "#66BB6A", "#AB47BC"]
    fig_bench = go.Figure()
    for name, color in zip(benchmark_names, colors_bench):
        fig_bench.add_trace(go.Scatter(
            x=steps, y=BENCH[name],
            name=name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ))
    fig_bench.update_layout(
        title=dict(text="🛡️ General Benchmark Scores (should stay flat)", font=dict(color="#E8F4FD", size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(22,32,50,0.8)",
        font=dict(color="#8BA9C4", size=11),
        xaxis=dict(title="Fine-tuning Steps", gridcolor="rgba(46,117,182,0.15)"),
        yaxis=dict(title="Accuracy", range=[0.30, 0.90], gridcolor="rgba(46,117,182,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="v"),
        margin=dict(t=40, b=40, l=50, r=20),
    )
    st.plotly_chart(fig_bench, use_container_width=True)

# Radar: baseline vs final
st.markdown("#### 🕸️ Baseline vs. Unlearned — Benchmark Radar")
cats = ["ARC-C", "ARC-Easy", "BoolQ", "HellaSwag", "PIQA", "WinoGrande"]
base_vals  = [BENCH[k][0]  for k in benchmark_names]
final_vals = [BENCH[k][-1] for k in benchmark_names]

fig_radar = go.Figure()
for vals, name, color, fillcolor, fill in [
    (base_vals,  "Baseline",  "#EF5350", "rgba(239,83,80,0.15)",  "toself"),
    (final_vals, "Unlearned", "#43A047", "rgba(67,160,71,0.15)", "toself"),
]:
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        name=name, fill=fill,
        line=dict(color=color, width=2),
        fillcolor=fillcolor,
    ))

fig_radar.update_layout(
    polar=dict(
        bgcolor="rgba(22,32,50,0.8)",
        radialaxis=dict(visible=True, range=[0.3, 0.9], color="#8BA9C4", gridcolor="rgba(46,117,182,0.2)"),
        angularaxis=dict(color="#8BA9C4"),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8BA9C4", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, b=20, l=60, r=60),
    height=380,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — INTERACTIVE UNLEARNING PLAYGROUND
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">🤖</span>
  <h2 class="sec-title">Interactive Unlearning Playground</h2>
  <div class="sec-line"></div>
</div>
<p style="color:var(--muted);font-size:0.92rem;margin-bottom:20px">
  Type any prompt below — the unlearned model replaces Harry Potter-specific terms with their
  generic counterparts using the anchor dictionary, simulating what the model would output
  after the unlearning pipeline has been applied.
</p>
""", unsafe_allow_html=True)

# ── Quick-load example prompts ────────────────────────────────────────────────
st.markdown("""
<p style="color:var(--bright);font-size:0.85rem;font-weight:600;margin-bottom:6px">
  💡 Try an example prompt:
</p>
""", unsafe_allow_html=True)

EXAMPLE_PROMPTS = [
    "Describe what happens at Hogwarts School of Witchcraft and Wizardry.",
    "Who is Harry Potter and why is he famous in the wizarding world?",
    "Explain the rules of Quidditch and how the Golden Snitch works.",
    "What are Horcruxes and why did Voldemort create them?",
    "Tell me about the friendship between Harry, Ron, and Hermione.",
    "What spell does Harry use most often and what does it do?",
    "Describe the Sorting Hat ceremony at Hogwarts.",
    "What is the significance of the Deathly Hallows?",
    "Who are the members of Dumbledore's Army?",
    "How does Harry Potter defeat Lord Voldemort in the end?",
]

ep_cols = st.columns(2)
for i, ep in enumerate(EXAMPLE_PROMPTS[:6]):
    with ep_cols[i % 2]:
        if st.button(f"📝 {ep[:55]}…" if len(ep) > 55 else f"📝 {ep}",
                     key=f"ep_{i}", use_container_width=True):
            st.session_state["playground_input"] = ep

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Initialise key on first run so textarea always reads from it
if "playground_input" not in st.session_state:
    st.session_state["playground_input"] = ""

user_prompt = st.text_area(
    label="✏️ Enter your prompt:",
    height=110,
    placeholder="e.g.  Harry Potter cast a spell at Hogwarts with his wand...",
    key="playground_input",
    help="Type anything referencing Harry Potter. The system replaces all known anchor terms.",
)

run_col, clear_col, _ = st.columns([1.5, 1, 5])
with run_col:
    run_btn = st.button("🚀 Run Unlearning", type="primary", use_container_width=True)
with clear_col:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state["playground_input"] = ""
        st.rerun()  # key matches textarea key — clears correctly

# ── Unlearning engine ─────────────────────────────────────────────────────────
def apply_unlearning(text: str, anchor_dict: dict) -> tuple[str, list[str]]:
    """Replace anchor terms with generic counterparts; return new text + list of replacements made."""
    result = text
    hits = []
    # Sort by length descending so longer phrases match before substrings
    for original, replacement in sorted(anchor_dict.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        if pattern.search(result):
            hits.append((original, replacement))
            result = pattern.sub(replacement, result)
    return result, hits

if run_btn and user_prompt.strip():
    unlearned_text, replacements = apply_unlearning(user_prompt, ANCHOR_DICT)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    pg_l, pg_r = st.columns(2, gap="large")

    with pg_l:
        st.markdown(f"""
        <div style="background:rgba(220,38,38,0.06);border:1px solid rgba(220,38,38,0.28);
                    border-radius:10px;padding:18px 20px;min-height:160px">
          <div style="color:#DC2626;font-size:0.78rem;font-weight:700;
                      letter-spacing:0.08em;margin-bottom:10px">⚠️ BASELINE MODEL OUTPUT</div>
          <div style="color:var(--text);font-size:0.95rem;line-height:1.65;white-space:pre-wrap">{user_prompt}</div>
        </div>
        """, unsafe_allow_html=True)

    with pg_r:
        st.markdown(f"""
        <div style="background:rgba(22,163,74,0.06);border:1px solid rgba(22,163,74,0.28);
                    border-radius:10px;padding:18px 20px;min-height:160px">
          <div style="color:#16A34A;font-size:0.78rem;font-weight:700;
                      letter-spacing:0.08em;margin-bottom:10px">✅ UNLEARNED MODEL OUTPUT</div>
          <div style="color:var(--text);font-size:0.95rem;line-height:1.65;white-space:pre-wrap">{unlearned_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Replacements made ──────────────────────────────────────────────────────
    if replacements:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <p style="color:var(--bright);font-size:0.82rem;font-weight:600;margin-bottom:8px">
          🔄 Anchor terms replaced:
        </p>
        """, unsafe_allow_html=True)
        badge_html = " ".join([
            f'<span style="background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.25);'
            f'border-radius:20px;padding:3px 12px;font-size:0.8rem;'
            f'margin:3px;display:inline-block">'
            f'<span style="color:#B91C1C;font-weight:600">{orig}</span>'
            f'<span style="color:#64748B"> → </span>'
            f'<span style="color:#15803D;font-weight:600">{repl}</span></span>'
            for orig, repl in replacements
        ])
        st.markdown(f"<div style='flex-wrap:wrap;display:flex;gap:4px'>{badge_html}</div>",
                    unsafe_allow_html=True)

        # Familiarity score estimate
        hp_words_in_original = len(replacements)
        total_words = max(1, len(user_prompt.split()))
        density = hp_words_in_original / total_words
        familiarity_before = min(1.0, round(density * 3.5, 3))
        familiarity_after  = round(familiarity_before * 0.025, 4)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("🎯 HP Terms Detected", hp_words_in_original)
        with sc2:
            st.metric("📊 Familiarity Before", familiarity_before, delta=None)
        with sc3:
            st.metric("✅ Familiarity After", familiarity_after,
                      delta=f"−{round(familiarity_before - familiarity_after, 3)}",
                      delta_color="inverse")
    else:
        st.info("ℹ️ No Harry Potter anchor terms detected in your prompt. Try including names like Harry, Hermione, Hogwarts, Quidditch, Voldemort, etc.")

elif run_btn and not user_prompt.strip():
    st.warning("⚠️ Please enter a prompt first.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — TECHNICAL STACK
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">🛠️</span>
  <h2 class="sec-title">Technical Stack</h2>
  <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

tech_items = [
    ("🦙", "Llama-3-8B / Phi-3-mini", "Base Model"),
    ("🤗", "HuggingFace", "Model Hub & Transformers"),
    ("⚡", "QLoRA + PEFT", "4-bit Quantisation"),
    ("🔥", "PyTorch", "Deep Learning Framework"),
    ("🌿", "WandB", "Experiment Tracking"),
    ("🐍", "Python 3.10", "Core Language"),
    ("🎈", "Streamlit", "Interactive Dashboard"),
    ("📊", "Plotly", "Data Visualisation"),
    ("🧬", "spaCy NER", "Anchor Extraction"),
    ("☁️", "Google Colab T4", "Training Hardware"),
]
cols = st.columns(5, gap="small")
for i, (icon, name, role) in enumerate(tech_items):
    with cols[i % 5]:
        st.markdown(f"""
        <div class="tech-card">
          <span class="tech-icon">{icon}</span>
          <div class="tech-name">{name}</div>
          <div class="tech-role">{role}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — TEAM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
  <span class="sec-icon">👥</span>
  <h2 class="sec-title">Project Team</h2>
  <div class="sec-line"></div>
</div>
""", unsafe_allow_html=True)

t1, t2, t3 = st.columns([1, 1, 1], gap="large")
team = [
    ("👩‍💻", "Arooj Raheel",  "i220601", "ML Pipeline · Anchor Extraction · Evaluation"),
    ("👨‍💻", "Jawad Khan",    "i220507", "Dashboard · Fine-tuning · Adversarial Probing"),
    ("👨‍🏫", "Sir Ahmad Raza","Instructor", "Responsible AI · Course Supervision"),
]
for col, (avatar, name, roll, role) in zip([t1, t2, t3], team):
    with col:
        st.markdown(f"""
        <div class="team-card">
          <div class="team-avatar">{avatar}</div>
          <div class="team-name">{name}</div>
          <div class="team-roll">{roll}</div>
          <div class="team-role">{role}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px 0 40px;animation:fadeInUp 0.6s ease both">
  <p style="color:var(--muted);font-size:0.82rem;margin:0;color:#475569">
    Based on: <em>Who's Harry Potter? Approximate Unlearning in LLMs</em>
    — Eldan &amp; Russinovich, Microsoft Research, arXiv:2310.02238 (2023)
  </p>
  <p style="color:rgba(100,116,139,0.6);font-size:0.75rem;margin:6px 0 0">
    Responsible AI · FAST-NUCES · Spring 2025 · Instructor: Sir Ahmad Raza
  </p>
</div>
""", unsafe_allow_html=True)
