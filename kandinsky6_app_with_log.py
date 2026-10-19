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
   CPU-offload strategy sized to your VRAM.
5. An optional post-generation super-resolution pass (Kandinsky-6.0-VSR).
6. NEW — LIVE GENERATION MONITOR. While a clip is being created you see:
     * the current phase (loading / encoding prompt / denoising step x of N /
       decoding / writing MP4 / super-resolution),
     * elapsed time and an estimated time remaining (from live step timing,
       and from your own previous runs on the same setup),
     * live GPU memory: PyTorch allocated, PyTorch peak, whole-device usage
       (+ sampled peak), GPU load and temperature, and system RAM,
     * a scrolling log with one line per denoising step,
   and afterwards a RUN REPORT with the maximum GPU memory reached, a VRAM-
   over-time chart, a per-phase time table and a headroom warning.
   Timings from successful runs are remembered (~/.cache/kandinsky/
   ui_timing_history.json) so the NEXT run can tell you roughly how long to
   wait before you even press Generate.

NOTE ON LIP-SYNC: Kandinsky 6.0 generates its own speech from a SCENE
DESCRIPTION prompt -- it does not take a literal script to recite verbatim
in a chosen language. For "take this photo and this exact text, in this
language" lip-sync, use the separate page: pages/1_Lip_Sync.py (a
dedicated TTS + Wav2Lip pipeline, not Kandinsky).

Install:
    pip install -r requirements.txt
    # optional, enables the extra live readouts:
    pip install nvidia-ml-py psutil      # GPU load/temperature + system RAM

Run:
    streamlit run kandinsky6_app.py
--------------------------------------------------------------------------
"""

import inspect
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Kandinsky 6.0 · Text/Image → Video + Audio", layout="wide")

try:
    import pandas as pd
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

# Optional extras (the app works without them).
try:
    import psutil
except ImportError:
    psutil = None

try:  # lets a background thread update Streamlit widgets
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:  # pragma: no cover - very old/new Streamlit layouts
    add_script_run_ctx = None
    get_script_run_ctx = None

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
HISTORY_PATH = Path.home() / ".cache" / "kandinsky" / "ui_timing_history.json"


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
            "compute). The live monitor below will show you the real per-step speed on your card."
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


# ==========================================================================
# LIVE GENERATION MONITOR  (progress, ETA, GPU / VRAM / RAM, log, history)
# ==========================================================================
GB = 1024 ** 3
_SENSOR_OK = {"util": True, "temp": True}  # disabled on first failure (e.g. pynvml missing)


def fmt_time(seconds):
    """Seconds -> '2m 05s' / '1h 03m 10s'. None -> '—'."""
    if seconds is None:
        return "—"
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def read_gpu_stats(device=0):
    """One snapshot of GPU memory (+ optional load/temperature) and host RAM."""
    free, total = torch.cuda.mem_get_info(device)
    stats = {
        "device_used_gb": (total - free) / GB,       # everything on the card (all processes + CUDA context)
        "device_total_gb": total / GB,
        "alloc_gb": torch.cuda.memory_allocated(device) / GB,          # live PyTorch tensors
        "reserved_gb": torch.cuda.memory_reserved(device) / GB,        # PyTorch caching allocator pool
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / GB,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / GB,
    }
    if _SENSOR_OK["util"]:
        try:
            stats["util"] = torch.cuda.utilization(device)
        except Exception:
            _SENSOR_OK["util"] = False
    if _SENSOR_OK["temp"]:
        try:
            stats["temp"] = torch.cuda.temperature(device)
        except Exception:
            _SENSOR_OK["temp"] = False
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            stats["ram_used_gb"] = vm.used / GB
            stats["ram_total_gb"] = vm.total / GB
            stats["proc_rss_gb"] = psutil.Process().memory_info().rss / GB
        except Exception:
            pass
    return stats


# ---- timing history (lets the next run estimate its duration up-front) ----
def load_history():
    try:
        return json.loads(HISTORY_PATH.read_text())
    except Exception:
        return {}


def save_run_to_history(key, run):
    try:
        hist = load_history()
        runs = hist.get(key, [])
        runs.append(run)
        hist[key] = runs[-5:]  # keep the last 5 runs per setup
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(hist))
    except Exception:
        pass


def history_stats(key):
    runs = load_history().get(key, [])
    if not runs:
        return None

    def avg(field):
        vals = [r[field] for r in runs if r.get(field) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "runs": len(runs),
        "total": avg("total"),          # generation start -> everything finished (excludes model loading)
        "per_step": avg("per_step"),    # average seconds per denoising step
        "post": avg("post"),            # last denoising step -> everything finished (decode, mp4, VSR)
        "peak_alloc_gb": max((r.get("peak_alloc_gb") or 0) for r in runs),
    }


def make_monitor_ui():
    """Create the placeholders that the monitor keeps updating."""
    holder = st.empty()
    with holder.container(border=True):
        st.markdown("##### ⏱️ Live generation monitor")
        status = st.empty()
        bar = st.empty()
        cols = st.columns(5)
        metrics = [c.empty() for c in cols]
        with st.expander("📜 Live log", expanded=True):
            log_box = st.empty()
    return {"holder": holder, "status": status, "bar": bar, "metrics": metrics, "log": log_box}


class RunMonitor:
    """
    Tracks phase, denoising step, elapsed time, ETA and GPU/RAM usage for one
    generation, and keeps the Streamlit placeholders in `ui` up to date.

    A small background thread refreshes the display every 0.5 s, so the clock
    and the memory readouts keep moving even during long phases where nothing
    else reports progress (model loading, a single slow offloaded step, VAE
    decoding...). The denoising loop reports each finished step via on_step().
    """

    # Rough position of each phase on the progress bar (a progress ESTIMATE:
    # denoising is ~85% of the bar; decoding time can't be predicted exactly).
    KIND_BASE = {"load": 0.0, "prepare": 0.02, "decode": 0.92, "encode": 0.96, "vsr": 0.98, "done": 1.0}
    ICONS = {"load": "💾", "prepare": "📝", "denoise": "🎞️", "decode": "🧩", "encode": "📦",
             "vsr": "🔍", "generate": "⚙️", "done": "✅", "failed": "❌"}

    def __init__(self, ui, total_steps, hist=None, device=0):
        self.ui = ui
        self.total_steps = max(1, int(total_steps))
        self.hist = hist
        self.device = device

        self.t0 = time.time()
        self.gen_start = None            # when the pipeline call started
        self.denoise_done_ts = None      # when the last denoising step finished
        self.tracking = None             # how steps are tracked: "callback_on_step_end" / "scheduler hook" / None

        self.kind = "load"
        self.label = "Starting"
        self.step = 0
        self.step_times = []             # completion timestamp of every denoising step
        self.phases = []                 # [{"kind","label","start","end"}]
        self.log_lines = []
        self.timeline = []               # VRAM over time, for the final chart
        self.peak_device_used = 0.0      # sampled peak of whole-device usage
        self.peak_ram_used = 0.0

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None

    # ---------------- lifecycle ----------------
    def start(self):
        if add_script_run_ctx is not None and get_script_run_ctx is not None:
            ctx = get_script_run_ctx()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            if ctx is not None:
                add_script_run_ctx(self._thread, ctx)
                self._thread.start()
            else:
                self._thread = None
        self.refresh()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _loop(self):
        while not self._stop.wait(0.5):
            try:
                self.refresh()
            except Exception:
                pass

    # ---------------- events ----------------
    def _stamp(self):
        return f"[{fmt_time(time.time() - self.t0)}]"

    def _stats(self):
        try:
            s = read_gpu_stats(self.device)
        except Exception:
            s = {k: 0.0 for k in ("device_used_gb", "device_total_gb", "alloc_gb", "reserved_gb",
                                  "peak_alloc_gb", "peak_reserved_gb")}
        self.peak_device_used = max(self.peak_device_used, s.get("device_used_gb", 0.0))
        self.peak_ram_used = max(self.peak_ram_used, s.get("ram_used_gb", 0.0))
        return s

    def log(self, msg):
        with self._lock:
            self.log_lines.append(f"{self._stamp()} {msg}")

    def mark_generation_start(self):
        self.gen_start = time.time()

    def set_phase(self, kind, label):
        now = time.time()
        with self._lock:
            if self.phases:
                self.phases[-1]["end"] = now
            self.phases.append({"kind": kind, "label": label, "start": now, "end": None})
            self.kind, self.label = kind, label
            s = self._stats()
            self.log_lines.append(
                f"{self._stamp()} ▶ {label} · VRAM {s['alloc_gb']:.1f} GB (peak {s['peak_alloc_gb']:.1f} GB)"
            )
        self.refresh()

    def on_step(self, n):
        """Called after each finished denoising step (n = 1..total_steps)."""
        now = time.time()
        with self._lock:
            n = min(int(n), self.total_steps)
            if n <= self.step:
                return
            for _ in range(n - self.step):
                self.step_times.append(now)
            prev = self.step_times[-2] if len(self.step_times) > 1 else (self.gen_start or self.t0)
            self.step = n
            s = self._stats()
            note = " (includes prompt encoding)" if n == 1 else ""
            self.log_lines.append(
                f"{self._stamp()}   step {n}/{self.total_steps} · {now - prev:.1f}s{note} · "
                f"VRAM {s['alloc_gb']:.1f} GB (peak {s['peak_alloc_gb']:.1f} GB)"
            )
            finished = n >= self.total_steps
            if finished:
                self.denoise_done_ts = now
        if finished:
            self.set_phase("decode", "Decoding latents → video + audio")
        elif n == 1:
            self.set_phase("denoise", "Denoising")
        else:
            self.refresh()

    # ---------------- estimates ----------------
    def _avg_step(self):
        ts = self.step_times
        if len(ts) >= 2:  # step 1 includes prompt encoding, so use the gaps between steps
            gaps = [b - a for a, b in zip(ts[:-1], ts[1:])][-5:]
            return sum(gaps) / len(gaps), "live"
        if self.hist and self.hist.get("per_step"):
            return self.hist["per_step"], "history"
        return None, None

    def _eta(self, now):
        """Returns (seconds_or_None, note)."""
        if self.kind == "done":
            return 0, ""
        hist_post = self.hist.get("post") if self.hist else None
        hist_total = self.hist.get("total") if self.hist else None

        if self.denoise_done_ts is not None:  # decoding / mp4 / super-resolution
            if hist_post is None:
                return None, "decoding time varies"
            return max(hist_post - (now - self.denoise_done_ts), 0), "from previous runs"

        avg, src = self._avg_step()
        if self.kind == "denoise" and avg:
            remaining = (self.total_steps - self.step) * avg
            note = "live step timing" if src == "live" else "previous runs"
            if hist_post is not None:
                return remaining + hist_post, note
            return remaining, note + " · + decoding"

        if self.kind in ("prepare", "generate") and hist_total:
            ref = self.gen_start or self.t0
            return max(hist_total - (now - ref), 0), "from previous runs"
        return None, None

    def _fraction(self, now):
        if self.kind == "denoise" or (self.kind == "prepare" and self.step > 0):
            return 0.05 + 0.85 * (self.step / self.total_steps)
        if self.kind == "generate":
            hist_total = self.hist.get("total") if self.hist else None
            if hist_total and self.gen_start:
                return min(0.9, (now - self.gen_start) / hist_total)
            return 0.0
        return self.KIND_BASE.get(self.kind, 0.0)

    # ---------------- display ----------------
    def refresh(self):
        if not self._lock.acquire(timeout=0.25):
            return
        try:
            now = time.time()
            elapsed = now - self.t0
            s = self._stats()
            self.timeline.append({
                "Time (s)": round(elapsed, 1),
                "PyTorch allocated (GB)": round(s["alloc_gb"], 2),
                "Device used (GB)": round(s["device_used_gb"], 2),
            })

            eta_s, eta_note = self._eta(now)
            if self.kind == "done":
                eta_txt = "done"
            elif eta_s is None:
                if self.tracking is None and self.kind not in ("load",):
                    eta_txt = "unknown (no step info available)"
                elif self.kind == "load":
                    eta_txt = "loading weights — a 30B model can take several minutes the first time"
                else:
                    eta_txt = "estimating… (appears after the 2nd denoising step)"
            else:
                eta_txt = f"≈ {fmt_time(eta_s)}" + (f" ({eta_note})" if eta_note else "")

            icon = self.ICONS.get(self.kind, "⏳")
            step_txt = ""
            if self.tracking and self.kind in ("prepare", "denoise"):
                step_txt = f" — step {self.step}/{self.total_steps}"
            frac = max(0.0, min(1.0, self._fraction(now)))

            try:
                self.ui["status"].markdown(
                    f"{icon} **{self.label}**{step_txt}  \n"
                    f"⏱️ Elapsed **{fmt_time(elapsed)}** · ⏳ Remaining **{eta_txt}**"
                )
                self.ui["bar"].progress(frac, text=f"{frac * 100:.0f}% (estimate)")

                m = self.ui["metrics"]
                m[0].metric("VRAM now (PyTorch)", f"{s['alloc_gb']:.1f} GB",
                            help="Memory held by live PyTorch tensors right now.")
                m[1].metric("VRAM peak (PyTorch)", f"{s['peak_alloc_gb']:.1f} GB",
                            help="Maximum PyTorch memory reached so far in this run.")
                m[2].metric("Device VRAM used", f"{s['device_used_gb']:.1f} / {s['device_total_gb']:.0f} GB",
                            delta=f"peak {self.peak_device_used:.1f} GB", delta_color="off",
                            help="Whole-card usage incl. CUDA context and other programs (peak is sampled twice a second).")
                if "util" in s or "temp" in s:
                    parts = []
                    if "util" in s:
                        parts.append(f"{s['util']}%")
                    if "temp" in s:
                        parts.append(f"{s['temp']}°C")
                    m[3].metric("GPU load · temp", " · ".join(parts))
                else:
                    m[3].metric("GPU load · temp", "n/a", help="pip install nvidia-ml-py to enable.")
                if "ram_used_gb" in s:
                    m[4].metric("System RAM", f"{s['ram_used_gb']:.0f} / {s['ram_total_gb']:.0f} GB",
                                delta=f"this app {s['proc_rss_gb']:.1f} GB", delta_color="off",
                                help="Offloaded model weights live here between GPU calls.")
                else:
                    m[4].metric("System RAM", "n/a", help="pip install psutil to enable.")

                self.ui["log"].code("\n".join(self.log_lines[-18:]) or "…", language=None)
            except Exception:
                pass  # widget no longer exists (page rerun) — ignore
        finally:
            self._lock.release()

    # ---------------- end of run ----------------
    def finish(self):
        self.set_phase("done", "Finished")
        self.stop()

    def build_report(self):
        end = time.time()
        with self._lock:
            if self.phases and self.phases[-1]["end"] is None:
                self.phases[-1]["end"] = end
            s = self._stats()
            gaps = [b - a for a, b in zip(self.step_times[:-1], self.step_times[1:])]
            per_step = sum(gaps) / len(gaps) if gaps else None
            gen_ref = self.gen_start or self.t0
            return {
                "total_s": end - self.t0,
                "gen_s": end - gen_ref,
                "per_step_s": per_step,
                "post_s": (end - self.denoise_done_ts) if self.denoise_done_ts else None,
                "peak_alloc_gb": s["peak_alloc_gb"],
                "peak_reserved_gb": s["peak_reserved_gb"],
                "peak_device_gb": self.peak_device_used,
                "device_total_gb": s["device_total_gb"],
                "peak_ram_gb": self.peak_ram_used or None,
                "tracking": self.tracking,
                "steps": self.total_steps,
                "phases": [
                    {"Phase": p["label"], "Seconds": round((p["end"] or end) - p["start"], 1)}
                    for p in self.phases if p["kind"] != "done"
                ],
                "timeline": list(self.timeline),
                "log": "\n".join(self.log_lines),
            }


def render_report(rep, seed=None):
    """Persistent summary shown after a run (survives Streamlit reruns)."""
    st.markdown("#### 📊 Run report — time & GPU usage")
    c = st.columns(5)
    c[0].metric("Total time", fmt_time(rep["total_s"]), help="Everything, including model loading if it wasn't cached.")
    c[1].metric("Avg time / step", f"{rep['per_step_s']:.1f} s" if rep.get("per_step_s") else "n/a")
    c[2].metric("Peak VRAM (PyTorch)", f"{rep['peak_alloc_gb']:.1f} GB")
    c[3].metric("Peak VRAM reserved", f"{rep['peak_reserved_gb']:.1f} GB")
    c[4].metric("Peak device used", f"{rep['peak_device_gb']:.1f} / {rep['device_total_gb']:.0f} GB")

    total, peak = rep["device_total_gb"], rep["peak_device_gb"]
    if total and peak / total >= 0.95:
        st.warning(
            f"The card was at **{peak / total * 100:.0f}%** of its VRAM at the peak — very close to the limit. "
            "If you see out-of-memory errors, switch to **Low VRAM** mode or the **Lite** checkpoint."
        )
    elif total:
        st.success(f"VRAM headroom at the peak: about **{total - peak:.1f} GB**.")
    if rep.get("peak_ram_gb"):
        st.caption(f"Peak system RAM used during the run: {rep['peak_ram_gb']:.0f} GB.")

    df = pd.DataFrame(rep["timeline"])
    if not df.empty:
        st.caption("GPU memory over time")
        st.line_chart(df.set_index("Time (s)"))
    if rep.get("phases"):
        st.caption("Time spent per phase")
        st.dataframe(pd.DataFrame(rep["phases"]), hide_index=True, use_container_width=True)
    with st.expander("📜 Full run log"):
        st.code(rep.get("log", ""), language=None)
    if seed is not None:
        st.caption(f"Seed: {seed}")


# --------------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------------
def generate(pipe, prompt, image, height, width, num_frames, steps, guidance,
             expand_prompts, sample_audio, seed, monitor=None):
    kwargs = dict(
        prompt=prompt, height=int(height), width=int(width), num_frames=int(num_frames),
        num_inference_steps=int(steps), guidance_scale=float(guidance),
        expand_prompts=expand_prompts, sample_audio=sample_audio,
    )
    if seed is not None:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(seed))
    if image is not None:
        kwargs["image"] = image

    # ---- hook the denoising loop so the monitor sees every finished step ----
    restore = None
    if monitor is not None:
        try:
            params = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        if "callback_on_step_end" in params:
            # Standard diffusers mechanism: called after every denoising step.
            def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
                monitor.on_step(step_index + 1)
                return callback_kwargs

            kwargs["callback_on_step_end"] = _on_step_end
            monitor.tracking = "callback_on_step_end"
        elif hasattr(getattr(pipe, "scheduler", None), "step"):
            # Fallback: count scheduler.step() calls (one per denoising step).
            scheduler = pipe.scheduler
            original_step = scheduler.step
            counter = {"n": 0}

            def _counting_step(*a, **kw):
                out = original_step(*a, **kw)
                counter["n"] += 1
                monitor.on_step(counter["n"])
                return out

            scheduler.step = _counting_step
            restore = lambda: setattr(scheduler, "step", original_step)
            monitor.tracking = "scheduler hook"

        if monitor.tracking is None:
            monitor.set_phase("generate", "Generating video + audio (step progress unavailable)")
        else:
            monitor.log(f"Step tracking via {monitor.tracking}")

    try:
        return pipe(**kwargs)
    finally:
        if restore is not None:
            restore()


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

    # ---- "how long should I wait?" estimate, from your own previous runs ----
    vsr_tag = f"{vsr_repo_id}|{resolution_scale}|{vsr_steps}" if enable_vsr else "off"
    hist_key = "|".join([
        ckpt["repo_id"], offload_mode, f"{int(height)}x{int(width)}", str(int(num_frames)),
        str(int(steps)), "audio" if sample_audio else "noaudio", f"expand={expand_prompts}",
        f"image={allow_image}", f"vsr={vsr_tag}",
    ])
    hist = history_stats(hist_key)
    if hist and hist.get("total"):
        st.info(
            f"⏳ **Estimated time on this setup: ≈ {fmt_time(hist['total'])}** "
            f"(average of {hist['runs']} previous run{'s' if hist['runs'] != 1 else ''}), plus model loading if the "
            f"model isn't already in memory. Expected peak VRAM ≈ {hist['peak_alloc_gb']:.1f} GB."
        )
    else:
        st.caption(
            "⏳ No timing history for this exact setup yet. The first run is usually the slowest (kernel warm-up, "
            "loading weights). Once the 2nd denoising step finishes, the live monitor will show a countdown."
        )

    ready = bool(prompt.strip()) and (image is not None if allow_image else True) and have_all
    if not have_all:
        st.warning("Download the selected checkpoint's weights in the sidebar first.")
    go = st.button("🎬 Generate", type="primary", disabled=not ready, key=f"{key_prefix}_go")

    if go:
        ui = make_monitor_ui()
        monitor = RunMonitor(ui, total_steps=int(steps), hist=hist)
        monitor.start()
        try:
            torch.cuda.reset_peak_memory_stats()  # so the reported maximum covers loading + generation

            monitor.set_phase("load", f"Loading {checkpoint_label.split(' —')[0]} into memory")
            pipe = load_pipeline(str(local_dir), offload_mode)

            monitor.set_phase("prepare", "Encoding prompt & preparing latents")
            monitor.mark_generation_start()
            output = generate(
                pipe, prompt, image, height, width, num_frames, steps, guidance,
                expand_prompts, sample_audio, seed, monitor=monitor,
            )

            frames = output.frames[0]
            audio = output.audio[0][None] if (sample_audio and getattr(output, "audio", None) is not None) else None
            sample_rate = getattr(pipe, "audio_sample_rate", 44100)

            with tempfile.TemporaryDirectory() as td:
                monitor.set_phase("encode", "Writing MP4 (video + audio)")
                out_path = os.path.join(td, "output.mp4")
                encode_kwargs = dict(fps=int(fps), output_path=out_path)
                if audio is not None:
                    encode_kwargs.update(audio=audio, audio_sample_rate=sample_rate)
                encode_video(frames, **encode_kwargs)

                final_path = out_path
                if enable_vsr:
                    if not vsr_local_dir.exists():
                        st.warning("VSR weights not downloaded — skipping upscale. Download them in the sidebar.")
                    else:
                        monitor.set_phase("vsr", "Upscaling with Kandinsky-6.0-VSR")
                        sr_path = os.path.join(td, "output_sr.mp4")
                        run_vsr(frames, audio, sample_rate, sr_path, int(fps))
                        final_path = sr_path

                with open(final_path, "rb") as f:
                    video_bytes = f.read()

            monitor.finish()
            rep = monitor.build_report()
            save_run_to_history(hist_key, {
                "total": rep["gen_s"], "per_step": rep["per_step_s"], "post": rep["post_s"],
                "peak_alloc_gb": rep["peak_alloc_gb"],
            })

            st.session_state[f"{key_prefix}_video_bytes"] = video_bytes
            st.session_state[f"{key_prefix}_report"] = rep
            st.session_state[f"{key_prefix}_seed_used"] = seed
            ui["holder"].empty()  # live panel is replaced by the persistent report below
        except torch.cuda.OutOfMemoryError:
            monitor.set_phase("failed", "Out of GPU memory")
            monitor.stop()
            rep = monitor.build_report()
            torch.cuda.empty_cache()
            st.error(
                f"CUDA ran out of memory. The run reached about **{rep['peak_alloc_gb']:.1f} GB** (PyTorch) / "
                f"**{rep['peak_device_gb']:.1f} of {rep['device_total_gb']:.0f} GB** on the card.\n\n"
                "Try: **Low VRAM** mode, the **Lite** checkpoint, a lower resolution/frame count, then press "
                "*Unload model / clear cache* in the sidebar before retrying."
            )
            with st.expander("📜 Log up to the failure"):
                st.code(rep["log"], language=None)
        except Exception as exc:
            monitor.stop()
            st.error("Generation failed. See the traceback below.")
            st.exception(exc)
        finally:
            monitor.stop()

    if st.session_state.get(f"{key_prefix}_video_bytes"):
        st.video(st.session_state[f"{key_prefix}_video_bytes"])
        st.download_button(
            "Download .mp4", st.session_state[f"{key_prefix}_video_bytes"],
            file_name=f"kandinsky6_{key_prefix}_output.mp4", mime="video/mp4", key=f"{key_prefix}_dl",
        )
    if st.session_state.get(f"{key_prefix}_report"):
        render_report(st.session_state[f"{key_prefix}_report"], st.session_state.get(f"{key_prefix}_seed_used"))


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_t2v, tab_i2v = st.tabs(["📝 Text → Video", "🖼️ Image → Video"])

with tab_t2v:
    render_generation_ui(key_prefix="t2v", allow_image=False)

with tab_i2v:
    render_generation_ui(key_prefix="i2v", allow_image=True)
