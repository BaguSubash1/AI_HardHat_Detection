"""Streamlit app for the YOLOv8 hard hat detector.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATASET = ROOT / "Hard Hat Workers.v1i.yolov8"
DEFAULT_WEIGHTS = DATASET / "runs" / "detect" / "train2" / "weights" / "best.pt"
SAMPLE_DIR = DATASET / "test" / "images"

# Class names from data.yaml: ['head', 'helmet', 'person']
HEAD = "head"
HELMET = "helmet"
PERSON = "person"

st.set_page_config(page_title="Hard Hat Detection", page_icon="⛑️", layout="wide")

CSS = """
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}

  .hero {
    padding: 2.2rem 2.4rem; border-radius: 18px; margin-bottom: 1.6rem;
    background: radial-gradient(circle at 85% 20%, rgba(255,184,0,.22), transparent 45%),
                linear-gradient(135deg, #1b2433 0%, #0f1521 100%);
    border: 1px solid rgba(255,255,255,.07);
  }
  .hero h1 {margin: 0 0 .4rem 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -.02em;}
  .hero h1 span {color: #ffb800;}
  .hero p {margin: 0; color: #9aa6b8; font-size: 1.02rem; max-width: 640px;}
  .badges {margin-top: 1.1rem; display: flex; gap: .5rem; flex-wrap: wrap;}
  .badge {
    font-size: .75rem; padding: .25rem .7rem; border-radius: 999px;
    background: rgba(255,184,0,.12); color: #ffcf4d; border: 1px solid rgba(255,184,0,.3);
  }

  .card {
    padding: 1.2rem 1.3rem; border-radius: 14px; height: 100%;
    background: #151c2a; border: 1px solid rgba(255,255,255,.07);
  }
  .card .num {
    width: 30px; height: 30px; border-radius: 50%; display: inline-flex;
    align-items: center; justify-content: center; font-weight: 700;
    background: #ffb800; color: #111; margin-bottom: .6rem;
  }
  .card h4 {margin: 0 0 .3rem 0; font-size: 1.02rem;}
  .card p {margin: 0; color: #9aa6b8; font-size: .9rem; line-height: 1.45;}

  .legend {display: flex; gap: 1.4rem; flex-wrap: wrap; margin: .4rem 0 .2rem 0;}
  .legend div {display: flex; align-items: center; gap: .5rem; color: #c3ccda; font-size: .9rem;}
  .dot {width: 12px; height: 12px; border-radius: 50%; display: inline-block;}

  .pill {
    display: inline-block; padding: .3rem .85rem; border-radius: 999px;
    font-weight: 600; font-size: .85rem;
  }
  .pill.ok   {background: rgba(34,197,94,.15);  color: #4ade80; border: 1px solid rgba(34,197,94,.4);}
  .pill.bad  {background: rgba(239,68,68,.15);  color: #f87171; border: 1px solid rgba(239,68,68,.4);}
  .pill.none {background: rgba(250,204,21,.12); color: #facc15; border: 1px solid rgba(250,204,21,.35);}

  [data-testid="stMetric"] {
    background: #151c2a; border: 1px solid rgba(255,255,255,.07);
    padding: .9rem 1.1rem; border-radius: 12px;
  }
  [data-testid="stMetricLabel"] {color: #9aa6b8;}
  [data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.06);}
  [data-testid="stFileUploader"] section {border-radius: 14px; border: 1.5px dashed rgba(255,184,0,.45);}
  .stButton > button {border-radius: 10px;}
  .section-title {font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .7rem 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def cuda_usable() -> bool:
    """True only if torch was built with kernels for this GPU's compute capability.

    The cu124 wheel ships sm_50..sm_90, so a Blackwell card (sm_120) reports
    cuda.is_available() == True but blows up on the first conv.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return f"sm_{major}{minor}" in torch.cuda.get_arch_list()


def summarize(result, names) -> dict:
    """Count detections per class in a single YOLO result."""
    counts = {}
    for cls_id in result.boxes.cls.tolist():
        label = names[int(cls_id)]
        counts[label] = counts.get(label, 0) + 1
    return counts


# --------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <h1>⛑️ Hard Hat <span>Detection</span></h1>
      <p>AI-powered PPE compliance check. Upload a site photo and the model flags
         every worker with or without a safety helmet in seconds.</p>
      <div class="badges">
        <span class="badge">YOLOv8</span><span class="badge">Real-time inference</span>
        <span class="badge">Batch upload</span><span class="badge">Adjustable thresholds</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Higher = fewer, more certain detections.")
    iou = st.slider("IoU (NMS) threshold", 0.1, 0.95, 0.45, 0.05,
                    help="Overlap above which duplicate boxes are merged.")
    imgsz = st.select_slider("Inference size", options=[320, 416, 512, 640], value=416)

    gpu_ok = cuda_usable()
    device = st.radio(
        "Device",
        options=["cpu", "0"],
        format_func=lambda d: "CPU" if d == "cpu" else "GPU (cuda:0)",
        index=1 if gpu_ok else 0,
        horizontal=True,
    )
    if torch.cuda.is_available() and not gpu_ok:
        cap = "sm_{}{}".format(*torch.cuda.get_device_capability(0))
        st.caption(f"GPU detected, but this torch build has no {cap} kernels. Using CPU is recommended.")

    with st.expander("Advanced"):
        weights = st.text_input("Model weights", value=str(DEFAULT_WEIGHTS))

if not Path(weights).exists():
    st.error(f"Weights not found at `{weights}`. Train the model first or point to a valid .pt file.")
    st.stop()

model = load_model(weights)
names = model.names

# ------------------------------------------------------------------- input
uploaded = st.file_uploader(
    "Drop site photos here",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# (name, PIL image) pairs to analyse
inputs = [(f.name, Image.open(f).convert("RGB")) for f in uploaded]

samples = sorted(SAMPLE_DIR.glob("*.jpg"))[:6] if SAMPLE_DIR.exists() else []
if not inputs and samples:
    st.markdown('<div class="section-title">Or try a sample image</div>', unsafe_allow_html=True)
    cols = st.columns(len(samples))
    for i, (col, path) in enumerate(zip(cols, samples)):
        with col:
            st.image(Image.open(path), use_container_width=True)
            if st.button("Analyse", key=f"sample{i}", use_container_width=True):
                st.session_state["sample"] = str(path)
    if "sample" in st.session_state:
        p = Path(st.session_state["sample"])
        inputs = [(p.name, Image.open(p).convert("RGB"))]
elif inputs:
    st.session_state.pop("sample", None)

# -------------------------------------------------------------- empty state
if not inputs:
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [
        ("1", "Upload", "Drop one or many photos from your site (JPG, PNG, BMP or WEBP)."),
        ("2", "Detect", "A YOLOv8 model locates heads, helmets and people in each frame."),
        ("3", "Review", "Get an instant compliance verdict with annotated images and counts."),
    ]
    for col, (n, title, text) in zip((c1, c2, c3), steps):
        col.markdown(
            f'<div class="card"><div class="num">{n}</div><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">What the model looks for</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="legend">
          <div><span class="dot" style="background:#22c55e"></span>Helmet — compliant</div>
          <div><span class="dot" style="background:#ef4444"></span>Head — no helmet</div>
          <div><span class="dot" style="background:#60a5fa"></span>Person</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------- analyse
analysed = []
progress = st.progress(0.0, text="Analysing...")
for i, (name, image) in enumerate(inputs):
    result = model.predict(
        np.array(image), conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False
    )[0]
    analysed.append((name, image, result, summarize(result, names)))
    progress.progress((i + 1) / len(inputs), text=f"Analysed {i + 1} of {len(inputs)}")
progress.empty()

total_helmets = sum(a[3].get(HELMET, 0) for a in analysed)
total_bare = sum(a[3].get(HEAD, 0) for a in analysed)
total_people = sum(a[3].get(PERSON, 0) for a in analysed)
heads_total = total_helmets + total_bare
compliance = f"{100 * total_helmets / heads_total:.0f}%" if heads_total else "—"

st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
m = st.columns(5)
m[0].metric("Images", len(analysed))
m[1].metric("Helmets", total_helmets)
m[2].metric("Uncovered heads", total_bare)
m[3].metric("People", total_people)
m[4].metric("Compliance", compliance)

# ----------------------------------------------------------------- results
st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
for name, image, result, counts in analysed:
    helmets = counts.get(HELMET, 0)
    bare = counts.get(HEAD, 0)

    if bare:
        pill = f'<span class="pill bad">✕ Non-compliant — {bare} uncovered head(s)</span>'
    elif helmets:
        pill = f'<span class="pill ok">✓ Compliant — {helmets} helmet(s)</span>'
    else:
        pill = '<span class="pill none">No heads or helmets detected</span>'

    with st.container(border=True):
        head_col, pill_col = st.columns([3, 2])
        head_col.markdown(f"**{name}**")
        pill_col.markdown(f'<div style="text-align:right">{pill}</div>', unsafe_allow_html=True)

        left, right = st.columns(2)
        left.image(image, caption="Original", use_container_width=True)
        # plot() returns BGR; flip to RGB for display.
        right.image(result.plot()[:, :, ::-1], caption="Detections", use_container_width=True)

        with st.expander("Per-detection details"):
            rows = [
                {
                    "class": names[int(cls_id)],
                    "confidence": round(float(score), 3),
                    "x1": round(float(box[0])),
                    "y1": round(float(box[1])),
                    "x2": round(float(box[2])),
                    "y2": round(float(box[3])),
                }
                for cls_id, score, box in zip(
                    result.boxes.cls.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.xyxy.tolist(),
                )
            ]
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.write("Nothing detected above the confidence threshold.")
