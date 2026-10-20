# 🎬 Kandinsky 6.0 Studio — Text/Image → Video + Audio, and Photo → Lip-Sync

A local [Streamlit](https://streamlit.io) app for generating short videos with synchronized audio using **Kandinsky 6.0**, plus a separate **Lip Sync** page that makes a photo say any script you type, in many languages.

## 🎥 Demo Video

<!-- Replace YOUR_VIDEO_ID with the part after "v=" in your YouTube URL.
    
[![Watch the demo on YouTube](https://img.youtube.com/vi/YOUR_VIDEO_ID/maxresdefault.jpg)]([https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://youtu.be/oC6Nkippr98))

*Click the thumbnail to watch on YouTube.*

<!-- Plain-link alternative (no thumbnail):
[▶️ Watch the demo](https://youtu.be/YOUR_VIDEO_ID)
-->

---

## ✨ What's inside

The app has two parts that use **different pipelines**:

| Part | What it does | Powered by |
|------|--------------|------------|
| **Main page** (Text → Video, Image → Video) | Generates a ~5 second clip with synchronized 44 kHz audio from a scene description. | Kandinsky 6.0 via 🤗 `diffusers` |
| **Lip Sync page** | Takes a photo plus your exact script (or your own audio) and produces a lip-synced talking video. | MMS-TTS + Wav2Lip |

> **Why two pipelines?** Kandinsky 6.0 creates its own speech from a *scene description*. It does not recite a literal script in a chosen language. For "say exactly this text" lip-sync, use the Lip Sync page.

### Main page (Kandinsky 6.0)

- **Text → Video (+ Audio)** and **Image → Video (+ Audio)** (the reference image conditions the first frame)
- Download checkpoints from inside the app (Pro, Pro-distill, Lite, Lite-distill) into a folder you choose
- **GPU / VRAM check** with a recommended CPU-offload strategy for your card
- Three VRAM modes: Balanced (`enable_model_cpu_offload`), Low VRAM (`enable_sequential_cpu_offload`), Max speed (everything on GPU)
- Optional **super-resolution** pass with Kandinsky-6.0-VSR (2 × / 2.25 × / 4 ×)
- Seed control, custom resolution and frame count, per-checkpoint recommended steps/guidance
- Download the result as `.mp4`

### Lip Sync page

- Choose **Generate from text (TTS)** or **Upload my own audio**
- Built-in language list: English, Spanish, French, German, Portuguese, Italian, Russian, Polish, Dutch, Hindi, Bengali, Tamil, plus "Other" where you enter any [MMS-TTS language code](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html)
- One-click setup in the sidebar: clones Wav2Lip and downloads its weights
- Download the result as `.mp4`

---

## 📁 Project structure

Streamlit looks for extra pages in a folder named `pages/` next to the main script, so place the files like this:

```
your-repo/
├── kandinsky6_app.py            # main app (basic version)
├── kandinsky6_app_with_log.py   # main app with live generation monitor (see below)
├── requirements.txt
├── README.md
└── pages/
    └── 1_Lip_Sync.py            # Lip Sync page, shows up in the sidebar nav
```

You can use **either** main script. `kandinsky6_app_with_log.py` is the same app plus a live monitor:

- current phase, step-by-step progress, elapsed time and estimated time remaining
- live GPU memory, GPU load and temperature, and system RAM
- a scrolling per-step log
- an end-of-run report with peak VRAM, a VRAM-over-time chart, a per-phase time table and a headroom warning
- timing history saved to `~/.cache/kandinsky/ui_timing_history.json`, so the next run can estimate its duration before you press Generate

---

## 🧰 Requirements

- **NVIDIA GPU with CUDA** (the Kandinsky page stops with an error if no CUDA GPU is found)
- Python 3.10+ (recommended)
- `git` (used by the Lip Sync setup to clone Wav2Lip)
- Enough disk space for the weights: the 30B **Pro** checkpoints are tens of GB
- System RAM matters when offloading: the app suggests planning for **64 GB+** for the 30B Pro model (an estimate from the code, not an official figure). The 3B **Lite** model needs much less.

**VRAM guide (from the app's own checks):**

| Checkpoint | Notes |
|------------|-------|
| **Lite** (3B) | Comfortable on a 24 GB card |
| **Pro** (30B) | Needs CPU offload on cards under ~30 GB; "Max speed" needs roughly 60 GB+ |

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) extra live readouts for kandinsky6_app_with_log.py
pip install nvidia-ml-py psutil
```

> **Note:** Kandinsky 6.0 needs `diffusers` from the **main branch** (not yet in a stable release). `requirements.txt` already installs it from the main-branch zip, so no `git` is needed for that step.

**Optional speed-up:** on Ampere/Ada/consumer-Blackwell GPUs, `sageattention` can improve performance. It is commented out in `requirements.txt` because it needs `nvcc` to build. See the Kandinsky README for their install recipe.

---

## ▶️ Usage

### Main app

```bash
streamlit run kandinsky6_app.py
# or, with the live monitor:
streamlit run kandinsky6_app_with_log.py
```

1. In the sidebar, pick a **checkpoint** (start with *Pro distilled* or *Lite distilled* for the fastest first run).
2. Click **Download / verify weights** and wait for the download to finish.
3. Choose a **VRAM strategy** that fits your GPU.
4. Open the **Text → Video** or **Image → Video** tab, write a prompt (describe the scene, the action, the mood *and* what the audio should sound like), then click **Generate**.
5. Download the `.mp4`.

Weights are stored in `~/.cache/kandinsky/weights` by default. Set the `KANDINSKY_HOME` environment variable (or edit the folder in the sidebar) to change this.

**Default settings:** 480 × 864, 121 frames (about 5 s at 24 fps). Distilled checkpoints use 10 steps / guidance 1.0; full-quality checkpoints use 50 steps / guidance 5.0. Don't mix these up, as it degrades output.

### Lip Sync page

Open **Lip Sync** from the sidebar navigation, then:

1. In the sidebar click **Clone Wav2Lip code**, then **Download Wav2Lip weights**.
2. Install Wav2Lip's own pinned dependencies once:
   ```bash
   pip install -r ~/.cache/wav2lip/Wav2Lip/requirements.txt
   ```
   (Adjust the path if you changed the code folder in the sidebar.)
3. Upload a clear, front-facing photo.
4. Pick a language and type your script, or upload your own audio (wav/mp3/m4a).
5. Click **Generate lip-synced video** and download the `.mp4`.

**Tips:** MMS-TTS works best with simple, plain sentences (lower-case, little punctuation). Wav2Lip works best with a sharp, front-facing face.

---

## ⚠️ Licenses and responsible use

- **MMS-TTS** (`facebook/mms-tts-*`) is **CC-BY-NC-4.0**: non-commercial use only.
- **Wav2Lip** is released for **research / non-commercial** use. Read its [LICENSE](https://github.com/Rudrabha/Wav2Lip) before using output commercially.
- **Kandinsky 6.0** weights have their own license. Check the model cards on Hugging Face under [`kandinskylab`](https://huggingface.co/kandinskylab).
- Only use photos of people who have given their consent, and don't use lip-sync output to mislead or impersonate anyone.

## 🙏 Credits

- [Kandinsky 6.0](https://huggingface.co/kandinskylab) by Kandinsky Lab
- [🤗 diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers)
- [Meta MMS-TTS](https://huggingface.co/facebook/mms-tts-eng)
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) by Rudrabha
- [Streamlit](https://streamlit.io)

## 📄 License

Add your own license for this code here (for example [MIT](https://choosealicense.com/licenses/mit/)). Note that the third-party models listed above keep their own licenses.
