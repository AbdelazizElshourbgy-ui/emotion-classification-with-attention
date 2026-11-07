"""
Emotion Classification - Streamlit app

Classifies an English comment into one of six emotions
(joy, sadness, anger, fear, surprise, disgust) with a fine-tuned DistilBERT.
The model is loaded from the Hugging Face Hub (a MODEL repo), so the model
file does not need to be stored in GitHub.

Run locally:
    pip install streamlit transformers torch
    streamlit run streamlit_app.py
"""
import altair as alt
import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------------- Config ----------------
MODEL_ID = "Abd3laziz1/emotion-distilbert-goemotions"   # <- your model repo on the Hugging Face Hub
MAX_CHARS = 500

EMOTIONS = {
    #  name        emoji   color
    "joy":      ("😄", "#F59E0B"),
    "sadness":  ("😢", "#3B82F6"),
    "anger":    ("😡", "#EF4444"),
    "fear":     ("😨", "#8B5CF6"),
    "surprise": ("😲", "#EC4899"),
    "disgust":  ("🤢", "#22C55E"),
}
EXAMPLES = {
    "😄 Joy": "Thank you so much, this is the best news I have heard all week!",
    "😢 Sadness": "I miss my old friends so much, I can't stop crying.",
    "😡 Anger": "I am so furious that they cancelled my flight again!",
    "😨 Fear": "I'm terrified of what might happen tomorrow.",
    "😲 Surprise": "I can't believe he did that, I am shocked!",
    "🤢 Disgust": "That smell is absolutely disgusting.",
}

st.set_page_config(
    page_title="Emotion Classifier", page_icon="🎭",
    layout="wide", initial_sidebar_state="collapsed",
)

# ---------------- Style ----------------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 3rem; padding-bottom: 1rem; max-width: 1150px;}
    .hero {
        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;
        padding: 0.9rem 1.5rem; margin-bottom: 1rem; border-radius: 18px;
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 55%, #F59E0B 100%);
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.35);
    }
    .hero h1 {color: white; margin: 0; padding: 0; font-size: 1.7rem; font-weight: 800;}
    .hero p {color: rgba(255,255,255,0.92); margin: 0; font-size: 0.95rem;}
    .result {
        display: flex; align-items: center; gap: 1.3rem; padding: 1rem 1.5rem; border-radius: 18px;
        background: rgba(255,255,255,0.04); border: 2px solid var(--c);
        box-shadow: 0 0 24px color-mix(in srgb, var(--c) 35%, transparent);
    }
    .result .emoji {font-size: 3.4rem; line-height: 1;}
    .result .name {font-size: 1.7rem; font-weight: 800; color: var(--c); text-transform: capitalize; line-height: 1.1;}
    .result .conf {opacity: 0.85; font-size: 1rem;}
    .placeholder {
        display: flex; align-items: center; justify-content: center; height: 105px; border-radius: 18px;
        border: 2px dashed rgba(255,255,255,0.22); opacity: 0.75; text-align: center; padding: 0 1rem;
    }
    div.stButton > button {border-radius: 999px; font-weight: 600; width: 100%; min-height: 2.1rem; padding: 0.2rem 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Model ----------------
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    return pipeline("text-classification", model=MODEL_ID, top_k=None, device=-1)


clf = load_model()

# ---------------- State and callbacks ----------------
st.session_state.setdefault("text", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("warn", False)


def set_text(value):
    st.session_state["text"] = value


def submit():
    text = st.session_state["text"].strip()[:MAX_CHARS]
    if not text:
        st.session_state["result"] = None
        st.session_state["warn"] = True
        return
    out = clf(text)
    out = out[0] if isinstance(out[0], list) else out
    scores = pd.DataFrame(out).rename(columns={"label": "emotion", "score": "confidence"})
    scores = scores.sort_values("confidence", ascending=False).reset_index(drop=True)
    st.session_state["result"] = {"text": text, "scores": scores}
    st.session_state["warn"] = False


def clear():
    st.session_state["text"] = ""
    st.session_state["result"] = None
    st.session_state["warn"] = False


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About")
    st.write(
        "A fine-tuned **DistilBERT** that reads an English comment and picks one of six emotions. "
        "It was trained on **GoEmotions** (Reddit comments)."
    )
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", "0.80")
    c2.metric("Test macro F1", "0.71")
    st.subheader("Emotions")
    for name, (emoji, color) in EMOTIONS.items():
        st.markdown(f"{emoji} <span style='color:{color};font-weight:700'>{name}</span>", unsafe_allow_html=True)
    st.caption(
        "The model always picks one of these six, even for neutral text. "
        "Fear and disgust have far fewer training examples, so they are the least reliable."
    )

# ---------------- Header ----------------
st.markdown(
    '<div class="hero"><h1>🎭 Emotion Classifier</h1>'
    "<p>Type a sentence and see which emotion the model feels in it.</p></div>",
    unsafe_allow_html=True,
)

# ---------------- Two columns: input | result ----------------
left, right = st.columns(2, gap="large")

with left:
    st.markdown("**Your text**")
    st.text_area(
        "Your text", key="text", height=110, max_chars=MAX_CHARS, label_visibility="collapsed",
        placeholder="Write something in English, for example: I can't believe I won!",
    )
    st.caption("Try an example")
    ex_cols = st.columns(3)
    for i, (label, sentence) in enumerate(EXAMPLES.items()):
        ex_cols[i % 3].button(label, on_click=set_text, args=(sentence,), key=f"ex{i}")
    b1, b2 = st.columns(2)
    b1.button("Submit", type="primary", on_click=submit)
    b2.button("Clear", on_click=clear)
    if st.session_state["warn"]:
        st.warning("Please write some text first.")

with right:
    st.markdown("**Result**")
    res = st.session_state["result"]
    if res is None:
        st.markdown(
            '<div class="placeholder">The predicted emotion will appear here after you press <b>&nbsp;Submit</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        scores = res["scores"]
        top = scores.iloc[0]
        emoji, color = EMOTIONS[top["emotion"]]
        st.markdown(
            f'<div class="result" style="--c:{color}"><div class="emoji">{emoji}</div>'
            f'<div><div class="name">{top["emotion"]}</div>'
            f'<div class="conf">{top["confidence"]:.0%} confidence</div></div></div>',
            unsafe_allow_html=True,
        )
        if top["confidence"] < 0.5:
            st.caption("The model is not very sure about this one. Several emotions are close.")

        bars = (
            alt.Chart(scores)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, size=20)
            .encode(
                x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
                y=alt.Y("emotion:N", sort=list(scores["emotion"]), title=None),
                color=alt.Color(
                    "emotion:N",
                    scale=alt.Scale(domain=list(EMOTIONS), range=[c for _, c in EMOTIONS.values()]),
                    legend=None,
                ),
                tooltip=["emotion", alt.Tooltip("confidence:Q", format=".1%")],
            )
        )
        labels = bars.mark_text(align="left", dx=6, fontWeight="bold").encode(
            text=alt.Text("confidence:Q", format=".0%"),
            color=alt.value("#F8FAFC"),
        )
        st.altair_chart((bars + labels).properties(height=200))

st.caption("Student project · DistilBERT fine-tuned on GoEmotions · not a clinical or psychological tool")
