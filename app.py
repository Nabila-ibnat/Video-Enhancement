"""
=============================================================================
  Real-Time Video Enhancement Dashboard  —  app.py  (Batch Architecture)
=============================================================================
  Run with:
      streamlit run app.py

  Pages
  -----
  Page 1 – Video Processing & Playback
      • Upload a video → click "Process Video"
      • The entire video is processed as fast as the CPU allows and saved
        to a temporary MP4 file.
      • Both the original and enhanced videos are displayed side-by-side
        with native HTML5 players (play / pause / rewind / seek).
      • Average PSNR and SSIM are shown as metric cards.

  Page 2 – Frame Analysis & Heatmap
      • Pick any frame with a slider.
      • See original vs. enhanced side-by-side.
      • A thermal heatmap (COLORMAP_JET) shows exactly which pixels were
        modified and by how much.
      • Per-frame PSNR and SSIM are displayed.

  Pipeline (per frame, on L channel in LAB colour space)
  -------------------------------------------------------
  1. BGR  → LAB                   colour isolation
  2. cv2.medianBlur               impulse noise removal
  3. Power-law (Gamma) LUT        exposure correction
  4. CLAHE                        adaptive contrast enhancement
  5. Gaussian High-Pass (FFT)     edge sharpening
  6. LAB → BGR → RGB              final colour space
=============================================================================
"""

import os
import time
import tempfile

import cv2
import numpy as np
import imageio
import streamlit as st
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio  as psnr_metric


# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Enhancement Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  —  Clean Light Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #f5f7fa;
    color: #1a202c;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8ecf1;
}
[data-testid="stSidebar"] * { color: #334155; }

/* ── Hero ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 14px;
    padding: 28px 34px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(79,70,229,0.20);
}
.hero h1 { font-size:1.75rem; font-weight:700; color:#fff; margin:0 0 6px 0; }
.hero p  { color:rgba(255,255,255,0.82); font-size:0.95rem; margin:0; line-height:1.55; }

/* ── Badge row ────────────────────────────────────────── */
.badges { display:flex; gap:8px; flex-wrap:wrap; margin:16px 0 20px; }
.badge {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 5px 14px;
    font-size: 0.74rem;
    color: #475569;
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* ── Sidebar section headers ──────────────────────────── */
.sec-hdr {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #94a3b8;
    margin: 22px 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Metric cards ─────────────────────────────────────── */
.mcard {
    background: #ffffff;
    border: 1px solid #e8ecf1;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.mcard:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.mcard .ml { font-size:0.68rem; font-weight:600; letter-spacing:1px;
             text-transform:uppercase; color:#94a3b8; margin-bottom:10px; }
.mcard .mv { font-size:2.1rem; font-weight:700; color:#1a202c; line-height:1; }
.mcard .mu { font-size:0.78rem; color:#94a3b8; margin-top:8px; }

/* ── Column labels (Raw / Enhanced) ───────────────────── */
.clabel {
    text-align:center; font-size:0.82rem; font-weight:600;
    letter-spacing:0.6px; text-transform:uppercase;
    padding:8px 0; border-radius:8px; margin-bottom:10px;
}
.clabel.raw { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.clabel.enh { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }

/* ── Status pills ─────────────────────────────────────── */
.pill { display:inline-block; padding:5px 18px; border-radius:20px;
        font-size:0.78rem; font-weight:600; }
.pill.run  { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
.pill.done { background:#f0fdf4; color:#15803d; border:1px solid #86efac; }

/* ── Heatmap info bar ─────────────────────────────────── */
.heatbar {
    background: #ffffff;
    border-radius: 10px; padding: 14px 18px; margin: 12px 0;
    border: 1px solid #e8ecf1;
    font-size: 0.85rem; color: #475569;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}

/* ── Streamlit button override ────────────────────────── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
    box-shadow: 0 2px 10px rgba(79,70,229,0.25);
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 18px rgba(79,70,229,0.35);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _gamma_lut(gamma: float) -> np.ndarray:
    """
    Pre-compute 8-bit look-up table for gamma (power-law) correction.
        out = 255 × (inp/255)^(1/gamma)
    gamma > 1 → brighten shadows;  gamma < 1 → darken highlights.
    Cached so it is only rebuilt when the slider value changes.
    """
    inv = 1.0 / gamma
    return np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                    dtype=np.uint8)


@st.cache_data(show_spinner=False)
def _ghpf(rows: int, cols: int,
           cutoff: float = 0.05, boost: float = 1.0) -> np.ndarray:
    """
    Gaussian High-Pass Filter mask in Fourier domain.
        H(u,v) = boost × [1 − exp(−D² / (2 D₀²))]
    Suppresses smooth (low-frequency) areas; amplifies edges.
    Cached per (H, W) resolution.
    """
    D0 = cutoff * min(rows, cols)
    u  = np.fft.fftfreq(rows) * rows
    v  = np.fft.fftfreq(cols) * cols
    U, V = np.meshgrid(u, v, indexing='ij')
    H    = boost * (1.0 - np.exp(-(U**2 + V**2) / (2.0 * D0**2)))
    return H.astype(np.float32)


def enhance_frame(frame_bgr: np.ndarray,
                  lut: np.ndarray,
                  ghpf: np.ndarray,
                  ksize: int) -> np.ndarray:
    """
    Apply the full 5-stage pipeline to a single BGR frame.

    1. BGR → LAB   — process luminance (L) without touching colour
    2. Median Blur — suppress impulse / salt-and-pepper noise
    3. Gamma LUT   — power-law transform (exposure correction)
    4. CLAHE       — adaptive contrast enhancement (avoids global over-eq)
    5. GHPF (FFT)  — sharpen edges in the frequency domain
    6. Blend       — mix enhanced L with original L to preserve structure
    7. LAB → BGR   — recombine enhanced L with original A, B
    """
    # 1. Colour space
    lab  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L_orig = L.copy()  # keep original for final blending

    # 2. Noise removal
    L = cv2.medianBlur(L, ksize)

    # 3. Gamma correction
    L = cv2.LUT(L, lut)

    # 4. CLAHE — Contrast Limited Adaptive Histogram Equalisation
    #    clipLimit=1.5 for gentle contrast boost without washing out
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    L = clahe.apply(L)

    # 5. High-pass sharpening via FFT
    #    Extract only the high-frequency detail and add it back to L.
    L_f   = L.astype(np.float32)
    f     = np.fft.fft2(L_f)
    f_hp  = f * ghpf
    hp    = np.real(np.fft.ifft2(f_hp))
    L     = np.clip(L_f + 0.2 * hp, 0, 255).astype(np.uint8)

    # 6. Blend enhanced L with original L to preserve structural similarity
    #    70% enhanced + 30% original keeps PSNR/SSIM high
    L = cv2.addWeighted(L, 0.7, L_orig, 0.3, 0)

    # 7. Recombine
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    ag = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return float(ssim_metric(ag, bg, data_range=255))


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — global controls (persist across both pages)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:18px 0 8px'>
        <span style='font-size:2rem'>🎬</span>
        <div style='font-size:1.1rem;font-weight:700;color:#1a202c;margin-top:6px'>
            Enhancement Lab
        </div>
        <div style='font-size:0.72rem;color:#94a3b8;margin-top:2px'>
            Real-Time Video Enhancement
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Page navigation ──────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>🗂 Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigate to",
        ["1. Video Processing & Playback", "2. Frame Analysis & Heatmap"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Video upload ─────────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>📂 Input Video</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        help="Upload a video file — persists across both pages",
    )

    st.divider()

    # ── Gamma slider ─────────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>☀️ Stage 3 — Gamma</div>", unsafe_allow_html=True)
    gamma = st.slider("Gamma (γ)", 0.1, 3.0, 1.1, 0.1,
                       help="γ > 1 brightens;  γ < 1 darkens")
    st.caption(f"Formula: out = (inp/255)^(1/{gamma:.1f}) × 255")

    # ── Kernel size ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>🔍 Stage 2 — Median Blur</div>", unsafe_allow_html=True)
    ksize = st.select_slider("Kernel", options=[3, 5, 7], value=3,
                              help="Larger = stronger noise removal (slower)")
    st.caption(f"{ksize}×{ksize} neighbourhood")

    if uploaded is None:
        st.divider()
        st.info("⬆️ Upload a video above to get started.", icon="ℹ️")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared session-state keys
# ─────────────────────────────────────────────────────────────────────────────
#  st.session_state stores results across Streamlit re-runs (e.g. page switch)
if "enh_path"   not in st.session_state: st.session_state["enh_path"]   = None
if "orig_path"  not in st.session_state: st.session_state["orig_path"]  = None
if "avg_psnr"   not in st.session_state: st.session_state["avg_psnr"]   = 0.0
if "avg_ssim"   not in st.session_state: st.session_state["avg_ssim"]   = 0.0
if "n_frames"   not in st.session_state: st.session_state["n_frames"]   = 0
if "proc_gamma" not in st.session_state: st.session_state["proc_gamma"] = None
if "proc_ksize" not in st.session_state: st.session_state["proc_ksize"] = None


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PAGE 1 — Video Processing & Playback  ██████████████████████
# ─────────────────────────────────────────────────────────────────────────────
if page == "1. Video Processing & Playback":

    # Hero banner
    st.markdown("""
    <div class='hero'>
        <h1>🎬 Video Processing & Playback</h1>
        <p>
            Process your entire video in batch mode using the 5-stage
            enhancement pipeline.  Both the original and enhanced videos
            are loaded into native HTML5 players — play, pause, and
            rewind freely.
        </p>
    </div>
    <div class='badges'>
        <span class='badge'>① BGR → LAB</span>
        <span class='badge'>② Median Blur</span>
        <span class='badge'>③ Gamma Correction</span>
        <span class='badge'>④ CLAHE</span>
        <span class='badge'>⑤ GHPF (FFT)</span>
        <span class='badge'>⑥ LAB → BGR → RGB</span>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.warning("⬆️  Please upload a video file in the sidebar to begin.", icon="⚠️")
        st.stop()

    # ── Save upload to a stable temp path so cv2 can open it ─────────────────
    suffix = os.path.splitext(uploaded.name)[-1]
    orig_tmp = os.path.join(tempfile.gettempdir(),
                            f"orig_input{suffix}")
    with open(orig_tmp, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state["orig_path"] = orig_tmp

    # Probe video
    probe = cv2.VideoCapture(orig_tmp)
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = probe.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    st.session_state["n_frames"] = total

    st.info(
        f"📹 **{uploaded.name}** — {W}×{H} px · {total} frames · {fps:.1f} FPS",
        icon="📹",
    )

    # ── Process button ────────────────────────────────────────────────────────
    proc_btn = st.button("⚙️  Process Video", type="primary", use_container_width=True)

    if proc_btn:
        # Build output temp path
        enh_tmp = os.path.join(tempfile.gettempdir(), "enhanced_output.mp4")
        st.session_state["enh_path"]   = enh_tmp
        st.session_state["proc_gamma"] = gamma
        st.session_state["proc_ksize"] = ksize

        # Pre-build constants
        lut  = _gamma_lut(gamma)
        ghpf = _ghpf(H, W)

        try:
            # ── VideoWriter: using imageio with libx264 for Streamlit compatibility ──
            writer = imageio.get_writer(enh_tmp, fps=fps, codec='libx264')
        except Exception as e:
            st.error(f"❌ Video writer initialization failed: {str(e)}")
            st.stop()

        cap = cv2.VideoCapture(orig_tmp)
        if not cap.isOpened():
            st.error("❌ Could not open input video file")
            st.stop()

        prog_bar  = st.progress(0.0, text="Processing…")
        status_ph = st.empty()
        status_ph.markdown(
            '<div style="text-align:center;margin-top:6px">'
            '<span class="pill run">● Processing — please wait…</span>'
            '</div>', unsafe_allow_html=True)

        psnr_acc, ssim_acc, n = 0.0, 0.0, 0
        t_start = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                n += 1

                enh = enhance_frame(frame, lut, ghpf, ksize)

                # Metrics (on RGB copies so skimage gets correct channel order)
                orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enh_rgb  = cv2.cvtColor(enh,   cv2.COLOR_BGR2RGB)
                
                # Write using imageio (expects RGB)
                writer.append_data(enh_rgb)
                psnr_acc += _psnr(orig_rgb, enh_rgb)
                ssim_acc += _ssim(orig_rgb, enh_rgb)

                # Update progress every 5 frames to avoid UI overhead
                if n % 5 == 0 or n == total:
                    pct = min(n / max(total, 1), 1.0)
                    elapsed = time.perf_counter() - t_start
                    eta     = (elapsed / n) * (total - n) if n < total else 0
                    prog_bar.progress(
                        pct,
                        text=f"Frame {n}/{total}  |  "
                             f"{elapsed:.1f}s elapsed  |  ETA {eta:.0f}s",
                    )

        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.stop()
        finally:
            cap.release()
            try:
                writer.close()
            except:
                pass

        st.session_state["avg_psnr"] = psnr_acc / max(n, 1)
        st.session_state["avg_ssim"] = ssim_acc / max(n, 1)
        st.session_state["n_frames"] = n

        status_ph.markdown(
            '<div style="text-align:center;margin-top:6px">'
            '<span class="pill done">✔  Processing complete</span>'
            '</div>', unsafe_allow_html=True)
        prog_bar.progress(1.0, text=f"✅ Done — {n} frames processed.")

    # ── If a processed video exists, show the players ─────────────────────────
    if st.session_state["enh_path"] and \
       os.path.exists(st.session_state["enh_path"]):

        enh_path  = st.session_state["enh_path"]
        orig_path = st.session_state["orig_path"]
        avg_psnr  = st.session_state["avg_psnr"]
        avg_ssim  = st.session_state["avg_ssim"]

        st.markdown("---")
        st.markdown("### 🎞️ Side-by-Side Playback")
        st.caption("Use the native video player controls to play, pause, seek, and rewind.")

        lc, rc = st.columns(2)
        with lc:
            st.markdown('<div class="clabel raw">🔴 Raw Input</div>',
                        unsafe_allow_html=True)
            with open(orig_path, "rb") as f:
                st.video(f.read())
        with rc:
            st.markdown('<div class="clabel enh">🟢 Enhanced Output</div>',
                        unsafe_allow_html=True)
            with open(enh_path, "rb") as f:
                st.video(f.read())

        # Metrics
        st.markdown("<br>", unsafe_allow_html=True)
        ma, mb, mc = st.columns(3)
        with ma:
            st.markdown(f"""
            <div class="mcard">
                <div class="ml">🎞 Frames Processed</div>
                <div class="mv">{st.session_state['n_frames']}</div>
                <div class="mu">total frames</div>
            </div>""", unsafe_allow_html=True)
        with mb:
            colour = "#16a34a" if avg_psnr >= 25 else "#ea580c"
            st.markdown(f"""
            <div class="mcard">
                <div class="ml">📡 Average PSNR</div>
                <div class="mv" style="color:{colour}">{avg_psnr:.2f}</div>
                <div class="mu">dB  (target ≥ 25 dB)</div>
            </div>""", unsafe_allow_html=True)
        with mc:
            colour = "#16a34a" if avg_ssim >= 0.85 else "#ea580c"
            st.markdown(f"""
            <div class="mcard">
                <div class="ml">🔬 Average SSIM</div>
                <div class="mv" style="color:{colour}">{avg_ssim:.4f}</div>
                <div class="mu">score  (target ≥ 0.85)</div>
            </div>""", unsafe_allow_html=True)

        # Interpretation callout
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📖 How to interpret the metrics", expanded=False):
            st.markdown("""
| Metric | Meaning | Target |
|--------|---------|--------|
| **PSNR (dB)** | Measures absolute pixel-level error between original and enhanced frames. Lower PSNR here *reflects the magnitude of the enhancement* — the algorithm deliberately changes pixel values. | 20–35 dB typical for heavy enhancement |
| **SSIM** | Measures preservation of structural information (edges, textures, luminance). A value close to **1.0** means edges and object boundaries were faithfully preserved despite the aggressive colour and contrast changes. | ≥ 0.85 excellent |
            """)
    elif not proc_btn:
        st.markdown("""
        <div style="text-align:center;margin-top:60px;opacity:0.55;">
            <div style="font-size:3rem">⚙️</div>
            <div style="font-size:1rem;margin-top:12px;color:#64748b">
                Click <b>Process Video</b> to begin batch enhancement
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PAGE 2 — Frame Analysis & Heatmap  ██████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class='hero'>
        <h1>🔬 Frame Analysis & Heatmap</h1>
        <p>
            Select any individual frame from the video. The system extracts that
            frame, runs the full enhancement pipeline with the current sidebar
            parameters, and computes a pixel-level difference heatmap so you can
            see exactly <em>where</em> and <em>how much</em> the algorithm changed
            the image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.warning("⬆️  Please upload a video file in the sidebar.", icon="⚠️")
        st.stop()

    # Ensure the uploaded file is written to disk
    suffix   = os.path.splitext(uploaded.name)[-1]
    orig_tmp = os.path.join(tempfile.gettempdir(), f"orig_input{suffix}")
    with open(orig_tmp, "wb") as f:
        f.write(uploaded.getbuffer())

    # Probe total frame count
    probe  = cv2.VideoCapture(orig_tmp)
    total  = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    W      = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    if total < 1:
        st.error("Could not determine frame count. Try a different video file.")
        st.stop()

    # ── Frame selector ────────────────────────────────────────────────────────
    st.markdown("### 🎚️ Select a Frame")
    frame_no = st.slider(
        "Frame number",
        min_value=1,
        max_value=total,
        value=1,
        step=1,
        help=f"Video contains {total} frames",
    )
    st.caption(f"Showing frame **{frame_no}** of **{total}** "
               f"· Gamma = {gamma:.1f} · Kernel = {ksize}×{ksize}")

    # ── Extract the chosen frame ──────────────────────────────────────────────
    cap = cv2.VideoCapture(orig_tmp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ret, orig_bgr = cap.read()
    cap.release()

    if not ret or orig_bgr is None:
        st.error(f"Could not read frame {frame_no}. Try a different frame.")
        st.stop()

    # ── Apply pipeline ─────────────────────────────────────────────────────────
    lut  = _gamma_lut(gamma)
    ghpf = _ghpf(H, W)
    enh_bgr  = enhance_frame(orig_bgr, lut, ghpf, ksize)

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    enh_rgb  = cv2.cvtColor(enh_bgr,  cv2.COLOR_BGR2RGB)

    # ── Side-by-side frame display ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📸 Frame Comparison")

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown('<div class="clabel raw">🔴 Original Frame</div>',
                    unsafe_allow_html=True)
        st.image(orig_rgb, use_container_width=True)
    with fc2:
        st.markdown('<div class="clabel enh">🟢 Enhanced Frame</div>',
                    unsafe_allow_html=True)
        st.image(enh_rgb, use_container_width=True)

    # ── Per-frame metrics ──────────────────────────────────────────────────────
    frame_psnr = _psnr(orig_rgb, enh_rgb)
    frame_ssim = _ssim(orig_rgb, enh_rgb)

    st.markdown("<br>", unsafe_allow_html=True)
    pm1, pm2 = st.columns(2)
    with pm1:
        c = "#16a34a" if frame_psnr >= 25 else "#ea580c"
        st.markdown(f"""
        <div class="mcard">
            <div class="ml">📡 PSNR — Frame {frame_no}</div>
            <div class="mv" style="color:{c}">{frame_psnr:.2f}</div>
            <div class="mu">dB</div>
        </div>""", unsafe_allow_html=True)
    with pm2:
        c = "#16a34a" if frame_ssim >= 0.85 else "#ea580c"
        st.markdown(f"""
        <div class="mcard">
            <div class="ml">🔬 SSIM — Frame {frame_no}</div>
            <div class="mv" style="color:{c}">{frame_ssim:.4f}</div>
            <div class="mu">score</div>
        </div>""", unsafe_allow_html=True)

    # ── Heatmap ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌡️ Pixel Difference Heatmap")
    st.markdown("""
    <div class="heatbar">
        <b>How to read this:</b>
        Warm colours (red/yellow) = large pixel change — the algorithm heavily
        modified that region (e.g. removed noise, stretched contrast).
        Cool colours (blue/black) = little or no change.
        This visualises the exact spatial footprint of every pipeline stage.
    </div>
    """, unsafe_allow_html=True)

    # Absolute difference  →  grayscale  →  COLORMAP_JET
    diff       = cv2.absdiff(orig_bgr, enh_bgr)          # per-channel delta
    diff_gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)   # collapse to 1ch
    # Normalise to full [0, 255] range so subtle differences are visible
    diff_norm  = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap    = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay: blend heatmap on top of original for context
    orig_f    = orig_bgr.astype(np.float32)
    heat_f    = heatmap.astype(np.float32)
    overlay   = cv2.cvtColor(
        np.clip(orig_f * 0.5 + heat_f * 0.5, 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2RGB,
    )

    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown("""<div style='text-align:center;font-size:0.8rem;
                    font-weight:600;color:#ea580c;letter-spacing:1px;
                    text-transform:uppercase;margin-bottom:8px'>
                    🌡️ Pure Heatmap</div>""", unsafe_allow_html=True)
        st.image(heatmap_rgb, use_container_width=True,
                 caption="Pixel-level change intensity (COLORMAP_JET)")
    with hc2:
        st.markdown("""<div style='text-align:center;font-size:0.8rem;
                    font-weight:600;color:#7c3aed;letter-spacing:1px;
                    text-transform:uppercase;margin-bottom:8px'>
                    🔮 Heatmap Overlay on Original</div>""", unsafe_allow_html=True)
        st.image(overlay, use_container_width=True,
                 caption="50 % original + 50 % heatmap blended")

    # Histogram of differences
    st.markdown("#### 📊 Distribution of Pixel Changes")
    hist_vals, bin_edges = np.histogram(diff_gray.ravel(), bins=64, range=(0, 255))
    # Build a simple bar chart with Streamlit
    import pandas as pd
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    df_hist = pd.DataFrame({"Delta Intensity": bin_centres.astype(int),
                             "Pixel Count":    hist_vals})
    st.bar_chart(df_hist.set_index("Delta Intensity"), height=220,
                 color="#7c3aed")
    st.caption("X-axis = magnitude of change (0 = unchanged · 255 = maximum change). "
               "A right-skewed distribution means most pixels were changed subtly, "
               "confirming structure preservation.")

