"""
Lip Sync — photo + script text (any supported language) -> lip-synced video.

SEPARATE PIPELINE FROM KANDINSKY 6.0 -- on purpose. Kandinsky 6.0 generates
its own speech from a scene-description prompt; it doesn't take a literal
script to recite verbatim in a chosen language. "Take this photo, say
exactly this text, in this language" needs a pipeline built for that:

    1. Text-to-speech (Meta's multilingual MMS-TTS, via transformers)
       turns your script into a speech waveform in the chosen language.
    2. Wav2Lip (the classic, well-established lip-sync model) warps the
       mouth region of your photo to match that speech waveform.

This page downloads/clones what it needs independently of the main app --
it does not load any Kandinsky weights.

LICENSES TO KNOW BEFORE YOU USE THIS:
  - MMS-TTS checkpoints (facebook/mms-tts-*) are CC-BY-NC-4.0 -- non-
    commercial use only.
  - Wav2Lip (github.com/Rudrabha/Wav2Lip) is released for research/
    non-commercial use; read its LICENSE before using the output for
    anything commercial.

Install:
    pip install -r requirements.txt
    # plus, after cloning Wav2Lip below:
    pip install -r <wav2lip_code_dir>/requirements.txt
"""

import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Kandinsky 6.0 · Lip Sync", page_icon="💬", layout="wide")
st.title("💬 Lip Sync — Photo + Script → Talking Video")

st.warning(
    "This is a **separate pipeline from Kandinsky 6.0** (text-to-speech + Wav2Lip), not the Kandinsky "
    "model. Kandinsky's own lip-sync comes from a scene-description prompt and doesn't recite an exact "
    "script — if that's what you want instead, use the main page's Image → Video tab.",
    icon="ℹ️",
)
st.caption(
    "MMS-TTS checkpoints are CC-BY-NC-4.0 (non-commercial). Wav2Lip is released for research/non-commercial "
    "use. Check both licenses before using output commercially."
)

try:
    import torch
    import scipy.io.wavfile
    from transformers import VitsModel, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from PIL import Image
except ImportError as e:
    st.error(
        f"Missing a required package: **{e.name}**.\n\n"
        "Install with:\n\n```\npip install -r requirements.txt\n```"
    )
    st.stop()

# --------------------------------------------------------------------------
# Language catalogue -- a confident, verified short list of MMS-TTS codes.
# MMS-TTS covers 1100+ languages total; for anything not listed here, look
# up the exact code yourself (don't guess) at Meta's language coverage page:
# https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html
# --------------------------------------------------------------------------
LANGUAGES = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Portuguese": "por",
    "Italian": "ita",
    "Russian": "rus",
    "Polish": "pol",
    "Dutch": "nld",
    "Hindi": "hin",
    "Bengali": "ben",
    "Tamil": "tam",
    "Other (enter MMS-TTS code manually)": None,
}

DEFAULT_WAV2LIP_CODE_DIR = str(Path.home() / ".cache" / "wav2lip" / "Wav2Lip")
DEFAULT_WAV2LIP_WEIGHTS_DIR = str(Path.home() / ".cache" / "wav2lip" / "weights")
WAV2LIP_REPO_URL = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_WEIGHTS_REPO = "camenduru/Wav2Lip"


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def clone_wav2lip(code_dir: Path):
    code_dir.parent.mkdir(parents=True, exist_ok=True)
    if (code_dir / "inference.py").exists():
        return "Already present."
    result = subprocess.run(
        ["git", "clone", WAV2LIP_REPO_URL, str(code_dir)], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr[-2000:]}")
    return "Cloned."


def download_wav2lip_weights(weights_dir: Path, code_dir: Path):
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = hf_hub_download(
        repo_id=WAV2LIP_WEIGHTS_REPO, filename="checkpoints/wav2lip_gan.pth", local_dir=str(weights_dir),
    )
    s3fd_path = hf_hub_download(
        repo_id=WAV2LIP_WEIGHTS_REPO, filename="s3fd.pth", local_dir=str(weights_dir),
    )
    # Wav2Lip's face detector looks for this file at a fixed path INSIDE the
    # cloned code repo -- this is the #1 cause of "s3fd.pth not found"
    # errors if you skip it.
    target = code_dir / "face_detection" / "detection" / "sfd" / "s3fd.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(Path(s3fd_path).read_bytes())
    return ckpt_path, str(target)


def check_wav2lip_ready(code_dir: Path, weights_dir: Path):
    issues = []
    if not (code_dir / "inference.py").exists():
        issues.append("Wav2Lip code not cloned yet.")
    if not (weights_dir / "checkpoints" / "wav2lip_gan.pth").exists():
        issues.append("wav2lip_gan.pth not downloaded yet.")
    if not (code_dir / "face_detection" / "detection" / "sfd" / "s3fd.pth").exists():
        issues.append("s3fd.pth not placed in the code repo yet.")
    return issues


@st.cache_resource(show_spinner=False)
def load_tts(lang_code: str):
    model = VitsModel.from_pretrained(f"facebook/mms-tts-{lang_code}")
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{lang_code}")
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def synthesize_speech(text: str, lang_code: str, out_wav_path: str):
    model, tokenizer = load_tts(lang_code)
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().detach().cpu().numpy()
    scipy.io.wavfile.write(out_wav_path, rate=model.config.sampling_rate, data=audio)
    return model.config.sampling_rate


def run_wav2lip(code_dir: Path, weights_dir: Path, face_path: str, audio_path: str, out_path: str):
    checkpoint = weights_dir / "checkpoints" / "wav2lip_gan.pth"
    cmd = [
        "python", str(code_dir / "inference.py"),
        "--checkpoint_path", str(checkpoint),
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(code_dir))
    if result.returncode != 0:
        raise RuntimeError(f"Wav2Lip inference failed:\n{result.stderr[-3000:]}")


# --------------------------------------------------------------------------
# Sidebar — Wav2Lip setup
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Wav2Lip setup")
    code_dir = Path(st.text_input("Wav2Lip code folder (git clone target)", value=DEFAULT_WAV2LIP_CODE_DIR))
    weights_dir = Path(st.text_input("Wav2Lip weights folder", value=DEFAULT_WAV2LIP_WEIGHTS_DIR))

    if st.button("📥 Clone Wav2Lip code", use_container_width=True):
        with st.spinner("git clone..."):
            msg = clone_wav2lip(code_dir)
        st.success(msg)

    if st.button("⬇️ Download Wav2Lip weights", use_container_width=True):
        with st.spinner("Downloading wav2lip_gan.pth + s3fd.pth..."):
            try:
                download_wav2lip_weights(weights_dir, code_dir)
                st.success("Downloaded and placed.")
            except Exception as e:
                st.error(f"Download failed: {e}")

    issues = check_wav2lip_ready(code_dir, weights_dir)
    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("✅ Wav2Lip code + weights ready.")

    st.caption(
        "After cloning, also install Wav2Lip's own pinned dependencies once:\n\n"
        f"```\npip install -r {code_dir}/requirements.txt\n```"
    )

    st.divider()
    st.subheader("🗑️ Cache")
    if st.button("Clear TTS model cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# --------------------------------------------------------------------------
# Main inputs
# --------------------------------------------------------------------------
col_in, col_out = st.columns(2, gap="large")

with col_in:
    st.subheader("Inputs")
    up = st.file_uploader("Photo (a clear, front-facing face works best)", type=["png", "jpg", "jpeg", "webp"])
    face_image = None
    if up:
        face_image = Image.open(up).convert("RGB")
        st.image(face_image, caption="Face", width="stretch")

    audio_source = st.radio("Audio source", ["Generate from text (TTS)", "Upload my own audio"], horizontal=True)

    text = None
    lang_code = None
    uploaded_audio = None

    if audio_source == "Generate from text (TTS)":
        lang_label = st.selectbox("Language", list(LANGUAGES.keys()))
        lang_code = LANGUAGES[lang_label]
        if lang_code is None:
            lang_code = st.text_input(
                "MMS-TTS language code (ISO 639-3-style, e.g. 'cmn', 'arb', 'jpn')",
                help="Look it up at https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html — "
                     "don't guess, an incorrect code will fail to load.",
            )
        text = st.text_area(
            "Script (what the photo should say)", height=120,
            placeholder="Type the exact words you want spoken...",
            help="MMS-TTS checkpoints are trained on lower-cased, unpunctuated text — simple, plain sentences "
                 "work best.",
        )
    else:
        uploaded_audio = st.file_uploader("Audio file (wav/mp3)", type=["wav", "mp3", "m4a"])

    ready = (
        face_image is not None
        and not check_wav2lip_ready(code_dir, weights_dir)
        and (
            (audio_source == "Generate from text (TTS)" and bool(text and text.strip()) and bool(lang_code))
            or (audio_source == "Upload my own audio" and uploaded_audio is not None)
        )
    )
    if check_wav2lip_ready(code_dir, weights_dir):
        st.warning("Finish the Wav2Lip setup in the sidebar before generating.")

    go = st.button("💬 Generate lip-synced video", type="primary", disabled=not ready)

with col_out:
    st.subheader("Result")
    if go:
        try:
            with tempfile.TemporaryDirectory() as td:
                face_path = os.path.join(td, "face.png")
                face_image.save(face_path)

                audio_path = os.path.join(td, "speech.wav")
                if audio_source == "Generate from text (TTS)":
                    with st.spinner(f"Synthesizing speech ({lang_code})..."):
                        synthesize_speech(text, lang_code, audio_path)
                else:
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_audio.read())

                out_path = os.path.join(td, "lipsync_output.mp4")
                with st.spinner("Running Wav2Lip (this can take a few minutes on CPU-bound face detection)..."):
                    run_wav2lip(code_dir, weights_dir, face_path, audio_path, out_path)

                with open(out_path, "rb") as f:
                    video_bytes = f.read()
                st.session_state["lipsync_video_bytes"] = video_bytes
        except Exception as exc:
            st.error("Lip-sync generation failed. See the traceback below.")
            st.exception(exc)

    if st.session_state.get("lipsync_video_bytes"):
        st.video(st.session_state["lipsync_video_bytes"])
        st.download_button(
            "Download .mp4", st.session_state["lipsync_video_bytes"],
            file_name="lipsync_output.mp4", mime="video/mp4",
        )
