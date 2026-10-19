"""
Streamlit app — Kandinsky 6.0 (Pro / Pro-distill / Lite) — text/image -> synchronized video+audio.

Uses diffusers' Kandinsky6TI2VAPipeline against locally downloaded weights.

--------------------------------------------------------------------------
WHAT THIS APP DOES
--------------------------------------------------------------------------
1. Lets you download Kandinsky 6.0 checkpoints (Pro, Pro-distill, Lite,
   Lite-distill, and the VSR super-resolution model) into a local folder
   you control, via huggingface_hub.snapshot_download.
2. Text -> Video (+Audio): a prompt describing the whole scene (action,
   mood, AND what the audio should sound like) produces a 5-second clip
   with synchronized 44kHz audio.
3. Image -> Video (+Audio): same, but conditioned on a reference image for
   the first frame.
4. A GPU/VRAM check section that reports whether your card is covered by
   Kandinsky's own supported compute-capability list, and recommends a
   CPU-offload strategy sized to your VRAM (the 30B "Pro" checkpoint's
   transformer + text encoder weights alone don't fit on a 24GB card
   resident at once -- offloading is the way this runs on consumer GPUs,
   and it's exactly what Kandinsky's own usage example does too).
5. An optional post-generation super-resolution pass (Kandinsky-6.0-VSR).

NOTE ON LIP-SYNC: Kandinsky 6.0 generates its own speech from a SCENE
DESCRIPTION prompt -- it does not take a literal script to recite verbatim
in a chosen language. For "take this photo and this exact text, in this
language" lip-sync, use the separate page: pages/1_Lip_Sync.py (a
dedicated TTS + Wav2Lip pipeline, not Kandinsky).

Install:
    pip install -r requirements.txt

Run:
    streamlit run kandinsky6_app.py
--------------------------------------------------------------------------
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Kandinsky 6.0 · Text/Image → Video + Audio", layout="wide")

try:
    import torch
    from diffusers import Kandinsky6TI2VAPipeline, Kandinsky6SRPipeline
    from diffusers.utils import load_image, encode_video
    from huggingface_hub import snapshot_download
    from PIL import Image
except ImportError as e:
    st.error(
        "Missing a required package: "
        f"**{e.name}**.\n\n"
        "Install everything with:\n\n"
        "```\npip install -r requirements.txt\n```"
    )
    st.stop()

st.title("🎬 Kandinsky 6.0 — Text & Image → Synchronized Video + Audio")
st.caption(
    "Local inference against downloaded weights. 5-second clips, 44kHz synchronized audio "
    "(including lip-sync when the scene implies speech)."
)
st.info(
    "Looking for **upload a photo + type exact words, in a chosen language, and get it lip-synced**? "
    "That's a different, dedicated pipeline — see the **Lip Sync** page in the sidebar nav. This page "
    "is for Kandinsky's own cinematic text/image → video+audio generation.",
    icon="ℹ️",
)

# --------------------------------------------------------------------------
# Checkpoint catalogue
# --------------------------------------------------------------------------
CHECKPOINTS = {
    "Pro — distilled, 10 steps (fastest, recommended first run)": {
        "repo_id": "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
        "params": "30B", "steps": 10, "guidance": 1.0,
    },
    "Pro — full quality, 50 steps": {
        "repo_id": "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers",
        "params": "30B", "steps": 50, "guidance": 5.0,
    },
    "Lite — distilled, 10 steps (lightest / most comfortable on 24GB)": {
        "repo_id": "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers",
        "params": "3B", "steps": 10, "guidance": 1.0,
    },
    "Lite — full quality, 50 steps": {
        "repo_id": "kandinskylab/Kandinsky-6.0-Lite-5s-Diffusers",
        "params": "3B", "steps": 50, "guidance": 5.0,
    },
}
VSR_CHECKPOINTS = {
    "Flow-matching, 4 steps/tile": "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
    "Distilled, 2 steps/tile": "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
}
# Components listed on the model card -- used only as a presence check, not loaded individually.
REQUIRED_SUBFOLDERS = [
    "transformer", "vae", "text_encoder", "tokenizer",
    "text_encoder_2", "tokenizer_2", "audio_vae", "vocoder", "scheduler",
]
DEFAULT_WEIGHTS_ROOT = os.environ.get("KANDINSKY_HOME", str(Path.home() / ".cache" / "kandinsky" / "weights"))


def check_local_weights(local_dir: Path):
    if not local_dir.exists():
        return False, list(REQUIRED_SUBFOLDERS)
    missing = [f for f in REQUIRED_SUBFOLDERS if not (local_dir / f).exists()]
    return (len(missing) == 0), missing


def download_checkpoint(repo_id: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


# --------------------------------------------------------------------------
# GPU / VRAM check
# --------------------------------------------------------------------------
def gpu_report():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_gb": props.total_memory / (1024 ** 3),
        "capability": f"{props.major}.{props.minor}",
    }


with st.expander("🖥️ GPU check", expanded=True):
    report = gpu_report()
    if report is None:
        st.error("No CUDA GPU detected. Kandinsky 6.0 needs an NVIDIA GPU.")
        st.stop()

    st.write(f"**{report['name']}** · {report['total_gb']:.1f} GB VRAM · compute capability {report['capability']}")

    # Kandinsky's own install path (per their README) targets these compute
    # capabilities: Hopper (9.0) with FlashAttention 3; Ampere/Ada/consumer
    # Blackwell (8.0, 8.6, 8.9, 12.0, 12.1) with SageAttention 2.2.0.
    supported_caps = {"8.0", "8.6", "8.9", "9.0", "12.0", "12.1"}
    if report["capability"] in supported_caps:
        st.success(
            f"Compute capability {report['capability']} is in Kandinsky 6.0's documented supported list "
            f"(Ampere/Ada/Blackwell/Hopper)."
        )
    else:
        st.warning(
            f"Compute capability {report['capability']} isn't in Kandinsky's documented list — the default "
            f"PyTorch attention path will likely still work, just unverified by their own benchmarks."
        )

    if "3090" in report["name"]:
        st.info(
            "**RTX 3090 detected** (24GB VRAM, Ampere, compute capability 8.6). Kandinsky's own benchmark table "
            "only measures 4090 / 5060 Ti / 5080 / 5090 / PRO 6000 / A100 / H100 — 3090 isn't in it — but it "
            "shares the 4090's 24GB VRAM and the same Ampere install path (SageAttention 2.2.0), so CPU offload "
            "below should let it run. Expect it to land slower than the 4090 numbers (less memory bandwidth and "
            "compute); I can't give you an exact per-clip time without a real run on the card."
        )

    if report["total_gb"] < 30:
        st.warning(
            "Under ~30GB VRAM: the 30B **Pro** checkpoint's transformer + text encoder weights alone are too "
            "large to keep fully resident on the GPU at once (tens of GB at bf16 each). Use a CPU-offload mode "
            "below (default), or pick the 3B **Lite** checkpoint, which is far more comfortable on a single "
            "24GB card and can likely run with less aggressive offload."
        )

# --------------------------------------------------------------------------
# Sidebar — weights, checkpoint choice, VRAM strategy
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Weights")
    weights_root = Path(st.text_input("Weights folder", value=DEFAULT_WEIGHTS_ROOT))
    checkpoint_label = st.selectbox("Checkpoint", list(CHECKPOINTS.keys()), index=0)
    ckpt = CHECKPOINTS[checkpoint_label]
    local_dir = weights_root / Path(ckpt["repo_id"]).name
    st.caption(f"📁 `{local_dir}`")

    have_all, missing = check_local_weights(local_dir)
    if have_all:
        st.success("✅ Weights found locally.")
    else:
        st.warning(
            f"{'Missing subfolder(s): ' + ', '.join(missing) if local_dir.exists() else 'Not downloaded yet.'}"
        )
    if st.button("⬇️ Download / verify weights", use_container_width=True, type="primary"):
        with st.spinner(
            f"Downloading {ckpt['repo_id']} ({ckpt['params']} params) — this is a large download "
            f"(tens of GB), be patient..."
        ):
            download_checkpoint(ckpt["repo_id"], local_dir)
        st.rerun()

    st.subheader("💾 VRAM strategy")
    mode_options = [
        "Balanced — enable_model_cpu_offload() [recommended]",
        "Low VRAM — enable_sequential_cpu_offload() [slower, lowest peak VRAM]",
        "Max speed — everything resident on GPU [needs ~60GB+ for Pro]",
    ]
    default_idx = 2 if (ckpt["params"] == "3B" and report["total_gb"] >= 20) else 0
    offload_choice = st.radio("How components move between GPU and CPU RAM", mode_options, index=default_idx)
    offload_mode = (
        "max_speed" if offload_choice.startswith("Max")
        else "low_vram" if offload_choice.startswith("Low")
        else "balanced"
    )
    if offload_mode == "max_speed" and ckpt["params"] == "30B" and report["total_gb"] < 55:
        st.error(
            "⚠️ 'Max speed' keeps the whole 30B Pro model resident on GPU — that needs roughly 60GB+ VRAM. "
            "This will almost certainly run out of memory on your card. Use Balanced or Low VRAM instead."
        )
    st.caption(
        "This discriminates by component exactly as the model card lists them: **transformer/** (the big "
        "diffusion transformer) and **text_encoder/** (Qwen2.5-VL) are the large pieces; **text_encoder_2/** "
        "(CLIP), **vae/**, **audio_vae/**, **vocoder/** are comparatively small. `enable_model_cpu_offload()` "
        "keeps only one component resident on GPU at a time, streaming the rest from host RAM; "
        "`enable_sequential_cpu_offload()` goes a level deeper (per-layer) for lower peak VRAM at the cost "
        "of speed."
    )
    st.caption(
        "Budget **system RAM** too — offloaded weights sit in host RAM between GPU calls. Plan for 64GB+ with "
        "the 30B Pro model (an estimate; Kandinsky's card doesn't publish an exact figure). The 3B Lite model "
        "needs much less of everything."
    )

    if st.button("🗑️ Unload model / clear cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("🔍 Optional: super-resolution")
    enable_vsr = st.checkbox("Upscale the result with Kandinsky-6.0-VSR after generation", value=False)
    vsr_variant = st.selectbox("VSR checkpoint", list(VSR_CHECKPOINTS.keys()), disabled=not enable_vsr)
    vsr_repo_id = VSR_CHECKPOINTS[vsr_variant]
    vsr_local_dir = weights_root / Path(vsr_repo_id).name
    if enable_vsr:
        st.caption(f"📁 `{vsr_local_dir}`")
        if st.button("⬇️ Download / verify VSR weights", use_container_width=True):
            with st.spinner(f"Downloading {vsr_repo_id}..."):
                vsr_local_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=vsr_repo_id, local_dir=str(vsr_local_dir))
            st.rerun()
        use_compile = st.checkbox(
            "Use torch.compile for VSR (faster on repeat runs; slower, more fragile first run)", value=False,
        )
        resolution_scale = st.select_slider("Upscale factor", options=[2, 2.25, 4], value=2.25)
        vsr_steps = st.number_input(
            "VSR steps per tile", 1, 20, 2 if "Distilled" in vsr_variant else 4,
        )


# --------------------------------------------------------------------------
# Pipeline loaders (cached so weights aren't reloaded per click)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(local_dir_str: str, offload_mode: str):
    pipe = Kandinsky6TI2VAPipeline.from_pretrained(local_dir_str, torch_dtype=torch.bfloat16)
    if offload_mode == "max_speed":
        pipe.to("cuda")
    elif offload_mode == "low_vram":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    return pipe


@st.cache_resource(show_spinner=False)
def load_vsr_pipeline(local_dir_str: str, use_compile: bool):
    sr_pipe = Kandinsky6SRPipeline.from_pretrained(local_dir_str, torch_dtype=torch.bfloat16)
    sr_pipe.transformer.set_attention_backend("flex")
    sr_pipe.enable_model_cpu_offload()
    if use_compile:
        torch._inductor.config.max_autotune = True
        sr_pipe.transformer.compile_repeated_blocks(fullgraph=True)
    return sr_pipe


def run_vsr(output_frames, audio, sample_rate, out_path, fps):
    sr_pipe = load_vsr_pipeline(str(vsr_local_dir), use_compile)
    upscaled = sr_pipe(
        video=output_frames, resolution_scale=resolution_scale, num_inference_steps=int(vsr_steps),
    ).frames[0]
    encode_video(upscaled, fps=fps, output_path=out_path, audio=audio, audio_sample_rate=sample_rate)


# --------------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------------
def generate(pipe, prompt, image, height, width, num_frames, steps, guidance,
             expand_prompts, sample_audio, seed):
    kwargs = dict(
        prompt=prompt, height=int(height), width=int(width), num_frames=int(num_frames),
        num_inference_steps=int(steps), guidance_scale=float(guidance),
        expand_prompts=expand_prompts, sample_audio=sample_audio,
    )
    if seed is not None:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(seed))
    if image is not None:
        kwargs["image"] = image
    return pipe(**kwargs)


def render_generation_ui(key_prefix: str, allow_image: bool):
    default_prompt = (
        "A lighthouse keeper climbs the spiral stairs at dusk as waves crash below. "
        "Warm lamp light flickers on. Audio: footsteps on stone, distant surf, wind, a creaking lantern."
    )
    prompt = st.text_area("Prompt (scene, action, mood — and what the audio should sound like)",
                           value=default_prompt, height=100, key=f"{key_prefix}_prompt")

    image = None
    if allow_image:
        up = st.file_uploader("Reference image (conditions the first frame)",
                               type=["png", "jpg", "jpeg", "webp"], key=f"{key_prefix}_upload")
        if up:
            image = Image.open(up).convert("RGB")
            st.image(image, caption="Reference", width="stretch")

    with st.expander("Generation settings", expanded=True):
        preset = st.selectbox(
            "Resolution", ["480 × 864 (verified default from the model card)", "Custom"],
            key=f"{key_prefix}_res",
        )
        if preset.startswith("480"):
            height, width = 480, 864
        else:
            c1, c2 = st.columns(2)
            height = c1.number_input("Height (multiple of 16)", 128, 1920, 480, step=16, key=f"{key_prefix}_h")
            width = c2.number_input("Width (multiple of 16)", 128, 1920, 864, step=16, key=f"{key_prefix}_w")
            st.caption("Only 480×864 is verified from the model card — other sizes are unverified by me.")

        num_frames = st.number_input("Frames (121 ≈ 5s at 24fps)", 8, 241, 121, step=1, key=f"{key_prefix}_frames")
        fps = st.number_input("Output FPS", 1, 60, 24, key=f"{key_prefix}_fps")

        override = st.checkbox(
            f"Override recommended steps/guidance for this checkpoint ({ckpt['steps']} steps, "
            f"guidance {ckpt['guidance']})", value=False, key=f"{key_prefix}_override",
        )
        if override:
            steps = st.number_input("Inference steps", 1, 100, ckpt["steps"], key=f"{key_prefix}_steps")
            guidance = st.number_input("Guidance scale", 0.0, 15.0, ckpt["guidance"], key=f"{key_prefix}_guidance")
        else:
            steps, guidance = ckpt["steps"], ckpt["guidance"]
            st.caption(
                f"Using this checkpoint's documented setting: **{steps} steps, guidance {guidance}** "
                f"(distilled and full-quality checkpoints need different values — mixing them up degrades output)."
            )

        sample_audio = st.checkbox("Generate synchronized audio", value=True, key=f"{key_prefix}_audio")
        expand_prompts = st.checkbox(
            "Expand short prompts with the Qwen2.5-VL text encoder (adds latency, no extra weights)",
            value=False, key=f"{key_prefix}_expand",
        )
        seed_mode = st.radio("Seed", ["Random", "Fixed"], horizontal=True, key=f"{key_prefix}_seedmode")
        seed = st.number_input("Seed value", 0, 2**31 - 1, 42, key=f"{key_prefix}_seed") if seed_mode == "Fixed" else None

    ready = bool(prompt.strip()) and (image is not None if allow_image else True) and have_all
    if not have_all:
        st.warning("Download the selected checkpoint's weights in the sidebar first.")
    go = st.button("🎬 Generate", type="primary", disabled=not ready, key=f"{key_prefix}_go")

    if go:
        try:
            with st.spinner(f"Loading {checkpoint_label.split(' —')[0]} from {local_dir}..."):
                pipe = load_pipeline(str(local_dir), offload_mode)

            torch.cuda.reset_peak_memory_stats()
            with st.spinner("Generating (video + audio)..."):
                output = generate(
                    pipe, prompt, image, height, width, num_frames, steps, guidance,
                    expand_prompts, sample_audio, seed,
                )
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            frames = output.frames[0]
            audio = output.audio[0][None] if (sample_audio and getattr(output, "audio", None) is not None) else None
            sample_rate = getattr(pipe, "audio_sample_rate", 44100)

            with tempfile.TemporaryDirectory() as td:
                out_path = os.path.join(td, "output.mp4")
                encode_kwargs = dict(fps=int(fps), output_path=out_path)
                if audio is not None:
                    encode_kwargs.update(audio=audio, audio_sample_rate=sample_rate)
                encode_video(frames, **encode_kwargs)

                final_path = out_path
                if enable_vsr:
                    vsr_have, _ = (True, []) if not vsr_local_dir.exists() else check_local_weights(vsr_local_dir)
                    if not vsr_local_dir.exists():
                        st.warning("VSR weights not downloaded — skipping upscale. Download them in the sidebar.")
                    else:
                        with st.spinner("Upscaling with Kandinsky-6.0-VSR..."):
                            sr_path = os.path.join(td, "output_sr.mp4")
                            run_vsr(frames, audio, sample_rate, sr_path, int(fps))
                            final_path = sr_path

                with open(final_path, "rb") as f:
                    video_bytes = f.read()

            st.session_state[f"{key_prefix}_video_bytes"] = video_bytes
            st.session_state[f"{key_prefix}_peak_gb"] = peak_gb
            st.session_state[f"{key_prefix}_seed_used"] = seed
        except Exception as exc:
            st.error("Generation failed. See the traceback below.")
            st.exception(exc)

    if st.session_state.get(f"{key_prefix}_video_bytes"):
        st.video(st.session_state[f"{key_prefix}_video_bytes"])
        st.download_button(
            "Download .mp4", st.session_state[f"{key_prefix}_video_bytes"],
            file_name=f"kandinsky6_{key_prefix}_output.mp4", mime="video/mp4", key=f"{key_prefix}_dl",
        )
        st.caption(
            f"Seed: {st.session_state.get(f'{key_prefix}_seed_used')} · "
            f"Peak VRAM: {st.session_state.get(f'{key_prefix}_peak_gb', 0):.2f} GB"
        )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_t2v, tab_i2v = st.tabs(["📝 Text → Video", "🖼️ Image → Video"])

with tab_t2v:
    render_generation_ui(key_prefix="t2v", allow_image=False)

with tab_i2v:
    render_generation_ui(key_prefix="i2v", allow_image=True)
