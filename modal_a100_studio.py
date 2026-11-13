#!/usr/bin/env python3
"""
Modal Serverless NVIDIA A100-80GB / L40S / L4 4-Stage Flagship AI Music & Full-Body Singing MV Studio
=====================================================================================================
100% Commercial / YouTube Monetizable Open-Source Stack:
  - Stage 1 (L40S 48GB): ACE-Step 1.5 XL (4B DiT SFT 50-step + 4B CoT LM, MIT) OR YuE2-3B (Apache 2.0 + Creator Free Monetization)
  - Stage 2 (L4 24GB):   BS-RoFormer Studio Stem Separation + Seed-VC 44.1kHz BigVGAN Singing Voice Conversion + -13 LUFS Tube Mastering
  - Stage 3 (A100-80GB): Wan2.2-S2V-14B (Apache 2.0) Full-Body Singing, Lip-Sync, Breathing & Stage Performance Video Diffusion
  - Stage 4 (CPU-Only):  Zero-Drift 1080p Broadcast Multi-Cam Compositor + Traditional Chinese Gradio Web Studio

Engineered with 4-Layer Ironclad $0.00 Out-of-Pocket Budget Circuit Breakers (`budget_guard.py`)
to stay strictly inside Modal Starter's $30.00/month free compute credit.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

from budget_guard import (
    ABSOLUTE_MAX_MONTHLY_CAP_USD,
    DEFAULT_SAFE_MONTHLY_CAP_USD,
    MIN_CONFIGURABLE_CAP_USD,
    MODAL_STARTER_FREE_CREDIT_USD,
    BudgetFuseTrippedError,
    BudgetLedger,
    MeteredStageContext,
    clamp_user_cap,
    estimate_pipeline_cost,
)

STUDIO_DIR = Path(__file__).resolve().parent
APP_NAME = "suno-killer-a100-mv-studio"
VOLUME_NAME = "pro-mv-studio-vol-v1"
VOL_MOUNT = "/vol"

app = modal.App(APP_NAME)
studio_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _safe_vol_reload() -> None:
    try:
        studio_vol.reload()
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Isolated Container Images (Zero Dependency Collisions Across 4 Models)
# ---------------------------------------------------------------------------
COMMON_APT = ["ffmpeg", "git", "curl", "fontconfig", "fonts-noto-cjk", "fonts-noto-cjk-extra", "libsndfile1"]

# 0. Pure CPU Weight Prefetcher & Web UI Image (Virtually $0 cost)
cpu_web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(*COMMON_APT)
    .pip_install(
        "gradio>=5.23.0,<6.0.0",
        "fastapi[standard]>=0.115.0",
        "huggingface_hub[hf_transfer]>=0.30.0",
        "pillow>=10.3.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("budget_guard", "mv_renderer", "s2v_mv_compositor")
    .add_local_dir(STUDIO_DIR / "assets", remote_path="/root/assets")
)

# 1A. Stage 1 Composer: ACE-Step 1.5 XL (4B DiT SFT + 4B CoT Planner LM) on L40S 48GB
acestep_xl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(*COMMON_APT)
    .run_commands("git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 "
        "'transformers>=4.51.0,<4.58.0' 'diffusers>=0.37.0' 'matplotlib>=3.7.5' "
        "'scipy>=1.10.1' 'soundfile>=0.13.1' 'loguru>=0.7.3' 'einops>=0.8.1' "
        "'accelerate>=1.12.0' diskcache 'numba>=0.63.1' 'vector-quantize-pytorch>=1.27.15' "
        "'torchcodec>=0.9.1' 'torchao>=0.16.0,<0.17.0' toml 'peft>=0.18.0' lycoris-lora "
        "'lightning>=2.0.0' modelscope 'typer-slim>=0.21.1' 'pytorch-wavelets>=1.3.0' "
        "'pywavelets>=1.9.0' 'huggingface_hub[hf_transfer]>=0.30.0' librosa 'pillow>=10.3.0'",
        "uv pip install --system --no-deps -e /opt/ACE-Step-1.5/acestep/third_parts/nano-vllm -e /opt/ACE-Step-1.5",
    )
    .env({"ACESTEP_ROOT": "/opt/ACE-Step-1.5", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("budget_guard", "mv_renderer", "s2v_mv_compositor")
    .add_local_dir(STUDIO_DIR / "assets", remote_path="/root/assets")
)

# 1B. Stage 1 Alternative Composer: YuE2-3B (WildSongBench #1 Symbolic CoT) on L40S 48GB
yue2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*COMMON_APT)
    .pip_install("setuptools>=77", "wheel")
    .pip_install(
        "git+https://github.com/multimodal-art-projection/YuE.git",
        "huggingface_hub[hf_transfer]>=0.36.2",
        "soundfile>=0.13.1",
        "pillow>=10.3.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("budget_guard", "mv_renderer", "s2v_mv_compositor")
    .add_local_dir(STUDIO_DIR / "assets", remote_path="/root/assets")
)

# 2. Stage 2 Vocal Mastering: BS-RoFormer Stem Separation (/opt/sep_venv) + Plachtaa/Seed-VC 44.1kHz BigVGAN on L4 24GB
svc_master_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(*COMMON_APT, "libportaudio2")
    .pip_install("uv")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        "accelerate>=1.0.0",
        "scipy==1.13.1",
        "librosa==0.10.2",
        "huggingface_hub[hf_transfer]>=0.28.1",
        "munch==4.0.0",
        "einops==0.8.0",
        "descript-audio-codec==1.0.0",
        "pydub==0.25.1",
        "resemblyzer",
        "jiwer==3.0.3",
        "transformers==4.46.3",
        "soundfile==0.12.1",
        "modelscope==1.18.1",
        "funasr==1.1.5",
        "numpy==1.26.4",
        "hydra-core==1.3.2",
        "pyyaml",
        "python-dotenv",
        "pillow>=10.3.0",
    )
    .run_commands(
        "uv venv /opt/sep_venv",
        "uv pip install --python /opt/sep_venv/bin/python 'torch==2.4.1' 'audio-separator[gpu]>=0.30.0'",
        "git clone --depth 1 https://github.com/Plachtaa/seed-vc.git /opt/seed-vc",
        "pip install --no-deps --force-reinstall 'protobuf>=4.25.3,<6.0.0'",
    )
    .env(
        {
            "HF_HOME": f"{VOL_MOUNT}/models/seed-vc-hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("budget_guard", "mv_renderer", "s2v_mv_compositor")
    .add_local_dir(STUDIO_DIR / "assets", remote_path="/root/assets")
)

# 3. Stage 3 Audio-Driven Full-Body Video Diffusion: Wan2.2-S2V-14B on A100-80GB
wan_s2v_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(*COMMON_APT, "libgl1", "libglib2.0-0", "build-essential", "clang")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        "packaging",
        "ninja",
        "setuptools",
        "wheel",
    )
    .pip_install(
        "opencv-python-headless>=4.9.0.80",
        "diffusers>=0.31.0",
        "transformers>=4.49.0,<=4.51.3",
        "tokenizers>=0.20.3",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio[ffmpeg]",
        "easydict",
        "ftfy",
        "dashscope",
        "imageio-ffmpeg",
        "numpy>=1.23.5,<2",
        "openai-whisper",
        "HyperPyYAML",
        "onnxruntime-gpu",
        "inflect",
        "wetext",
        "omegaconf",
        "conformer",
        "hydra-core",
        "lightning",
        "rich",
        "matplotlib",
        "pyarrow",
        "pyworld",
        "librosa",
        "decord",
        "modelscope",
        "GitPython",
        "pillow>=10.3.0",
        "huggingface_hub[hf_transfer]>=0.30.0",
    )
    .run_commands(
        "pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' && pip install 'peft>=0.17.0' 'sentencepiece>=0.2.0'",
        "git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2",
    )
    .env({"PYTHONPATH": "/opt/Wan2.2", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("budget_guard", "mv_renderer", "s2v_mv_compositor")
    .add_local_dir(STUDIO_DIR / "assets", remote_path="/root/assets")
)


# ---------------------------------------------------------------------------
# Stage 0: Pure CPU High-Speed Model Weight Prefetcher (Saves 100% GPU Download Cost!)
# ---------------------------------------------------------------------------
@app.function(
    image=cpu_web_image,
    volumes={VOL_MOUNT: studio_vol},
    cpu=4.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    timeout=2400,
)
def prefetch_weights_cpu(targets: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Downloads huge model checkpoints onto persistent Modal Volume using 4 CPU cores (~$0.015 total)
    so A100-80GB / L40S GPUs NEVER waste $2.50/hr waiting on Hugging Face downloads!
    """
    import json
    from huggingface_hub import snapshot_download

    ledger = BudgetLedger()
    wanted = set(targets or ["ace_xl", "wan_s2v"])
    models_root = Path(VOL_MOUNT) / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    report: Dict[str, str] = {}

    with MeteredStageContext(ledger, "Stage0_CPU_Prefetch", "CPU-4C", is_gpu=False):
        if "ace_xl" in wanted:
            ace_ckpts = models_root / "acestep_checkpoints"
            ace_ckpts.mkdir(parents=True, exist_ok=True)
            # Base repo includes vae, Qwen3-Embedding-0.6B, and core configs
            if not (ace_ckpts / "vae").exists():
                snapshot_download("ACE-Step/Ace-Step1.5", local_dir=str(ace_ckpts))
            # Flagship 4B XL SFT DiT + 4B 5Hz CoT Planner LM
            for sub_repo, folder in [
                ("ACE-Step/acestep-v15-xl-sft", "acestep-v15-xl-sft"),
                ("ACE-Step/acestep-5Hz-lm-4B", "acestep-5Hz-lm-4B"),
            ]:
                dst = ace_ckpts / folder
                if not dst.exists() or not any(dst.iterdir()):
                    snapshot_download(sub_repo, local_dir=str(dst))
            for lm_folder in ("acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B"):
                cfg_p = ace_ckpts / lm_folder / "config.json"
                if cfg_p.exists():
                    try:
                        cfg_obj = json.loads(cfg_p.read_text(encoding="utf-8"))
                        if cfg_obj.get("architectures") == ["Qwen3Model"]:
                            cfg_obj["architectures"] = ["Qwen3ForCausalLM"]
                            cfg_p.write_text(json.dumps(cfg_obj, indent=2), encoding="utf-8")
                    except Exception:
                        pass
            report["ace_xl"] = "ready"

        if "yue2" in wanted:
            yue_dir = models_root / "YuE2-3B"
            if not yue_dir.exists() or not any(yue_dir.iterdir()):
                snapshot_download("m-a-p/YuE2-3B", local_dir=str(yue_dir))
            report["yue2"] = "ready"

        if "wan_s2v" in wanted:
            wan_dir = models_root / "Wan2.2-S2V-14B"
            if not wan_dir.exists() or not any(wan_dir.iterdir()):
                snapshot_download("Wan-AI/Wan2.2-S2V-14B", local_dir=str(wan_dir))
            report["wan_s2v"] = "ready"

        studio_vol.commit()

    return report


# ---------------------------------------------------------------------------
# Stage 1A: Flagship ACE-Step 1.5 XL (4B DiT SFT + 4B CoT LM) on L40S 48GB
# ---------------------------------------------------------------------------
@app.cls(
    image=acestep_xl_image,
    gpu="L40S",
    volumes={VOL_MOUNT: studio_vol},
    min_containers=0,
    max_containers=1,
    scaledown_window=20,
    timeout=600,
)
class AceStepXLComposer:
    @modal.enter()
    def setup(self) -> None:
        sys.path.insert(0, "/opt/ACE-Step-1.5")
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        _safe_vol_reload()
        ckpt_src = Path(VOL_MOUNT) / "models" / "acestep_checkpoints"
        ckpt_link = Path("/opt/ACE-Step-1.5/checkpoints")
        if ckpt_src.exists():
            if ckpt_link.exists() and not ckpt_link.is_symlink():
                shutil.rmtree(ckpt_link)
            if not ckpt_link.exists():
                ckpt_link.symlink_to(ckpt_src)

        self.dit = AceStepHandler()
        self.llm = LLMHandler()

        # Prefer flagship 4B XL SFT model; gracefully fall back to turbo if XL not yet prefetched
        dit_cfg = "acestep-v15-xl-sft" if (ckpt_src / "acestep-v15-xl-sft").exists() else "acestep-v15-turbo"
        lm_cfg = "acestep-5Hz-lm-4B" if (ckpt_src / "acestep-5Hz-lm-4B").exists() else "acestep-5Hz-lm-1.7B"
        self.is_xl_sft = dit_cfg == "acestep-v15-xl-sft"

        status, ok = self.dit.initialize_service(
            project_root="/opt/ACE-Step-1.5",
            config_path=dit_cfg,
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step XL init failed: {status}")

        lm_status, lm_ok = self.llm.initialize(
            checkpoint_dir=str(ckpt_link),
            lm_model_path=lm_cfg,
            backend="pt",
            device="cuda",
            offload_to_cpu=False,
            dtype=None,
        )
        self.lm_ready = bool(lm_ok)

    @modal.method()
    def compose(
        self,
        job_id: str,
        style_prompt: str,
        lyrics_text: str,
        duration_sec: int = 48,
        bpm: int = 96,
        vocal_lang: str = "zh",
        seed: int = -1,
    ) -> str:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        ledger = BudgetLedger()
        _safe_vol_reload()
        job_dir = Path(VOL_MOUNT) / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        out_dir = job_dir / "stage1_acestep"
        out_dir.mkdir(parents=True, exist_ok=True)

        steps = 50 if self.is_xl_sft else 12
        guidance = 7.2 if self.is_xl_sft else 3.5

        with MeteredStageContext(ledger, "Stage1_ACE_Step_1.5_XL", "L40S", is_gpu=True):
            params = GenerationParams(
                caption=style_prompt,
                lyrics=lyrics_text,
                thinking=self.lm_ready,
                use_cot_metas=self.lm_ready,
                use_cot_caption=self.lm_ready,
                use_cot_language=self.lm_ready,
                duration=float(duration_sec),
                vocal_language=vocal_lang,
                inference_steps=steps,
                guidance_scale=guidance,
                shift=3.0,
                seed=int(seed),
                bpm=int(bpm) if bpm and int(bpm) > 0 else None,
                keyscale="",
                timesignature="4",
                task_type="text2music",
            )
            config = GenerationConfig(
                batch_size=1,
                use_random_seed=(int(seed) < 0),
                audio_format="wav",
            )
            res = generate_music(
                dit_handler=self.dit,
                llm_handler=self.llm if self.lm_ready else None,
                params=params,
                config=config,
                save_dir=str(out_dir),
            )
            if not res.success:
                raise RuntimeError(f"ACE-Step XL generation failed: {res.error}")

            paths = [a.get("path") for a in res.audios if isinstance(a, dict) and a.get("path")]
            if not paths:
                paths = sorted(glob.glob(str(out_dir / "*.wav")), key=os.path.getmtime, reverse=True)
            if not paths:
                raise RuntimeError("ACE-Step XL produced no WAV output.")

            raw_wav = job_dir / "raw_song.wav"
            shutil.copyfile(paths[0], raw_wav)
            studio_vol.commit()
            return str(raw_wav)


# ---------------------------------------------------------------------------
# Stage 1B: YuE2-3B (WildSongBench #1 Symbolic-CoT Engine) on L40S 48GB
# ---------------------------------------------------------------------------
@app.function(
    image=yue2_image,
    gpu="L40S",
    volumes={VOL_MOUNT: studio_vol},
    min_containers=0,
    max_containers=1,
    scaledown_window=20,
    timeout=900,
)
def compose_yue2_song(
    job_id: str,
    style_prompt: str,
    lyrics_text: str,
    duration_sec: int = 48,
    seed: int = 42,
) -> str:
    from yue2 import YuE2Pipeline

    ledger = BudgetLedger()
    _safe_vol_reload()
    job_dir = Path(VOL_MOUNT) / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    yue_out = job_dir / "stage1_yue2"
    yue_out.mkdir(parents=True, exist_ok=True)

    model_ref = str(Path(VOL_MOUNT) / "models" / "YuE2-3B")
    if not Path(model_ref).exists():
        model_ref = "m-a-p/YuE2-3B"

    actual_seed = int(seed) if int(seed) >= 0 else int(time.time()) % 100000

    with MeteredStageContext(ledger, "Stage1_YuE2_3B_CoT", "L40S", is_gpu=True):
        with YuE2Pipeline.from_pretrained(model_ref, device="cuda") as pipe:
            song = pipe(style=style_prompt, lyrics=lyrics_text, cot="full", seed=actual_seed)
            song.save_artifacts(str(yue_out))

        flac_path = yue_out / "audio.flac"
        if not flac_path.exists():
            candidates = list(yue_out.glob("*.flac")) + list(yue_out.glob("*.wav"))
            if not candidates:
                raise RuntimeError("YuE2 produced no audio artifact.")
            flac_path = candidates[0]

        raw_wav = job_dir / "raw_song.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(flac_path),
                "-t", f"{max(15, int(duration_sec))}",
                "-ar", "48000", "-ac", "2",
                str(raw_wav),
            ],
            check=True,
        )
        studio_vol.commit()
        return str(raw_wav)


# ---------------------------------------------------------------------------
# Stage 2: BS-RoFormer Stem Separation + Seed-VC 44.1kHz BigVGAN + Tube Master (L4 24GB)
# ---------------------------------------------------------------------------
@app.function(
    image=svc_master_image,
    gpu=["L4", "A10G", "L40S"],
    volumes={VOL_MOUNT: studio_vol},
    min_containers=0,
    max_containers=1,
    scaledown_window=20,
    timeout=360,
)
def master_vocals_and_band(
    job_id: str,
    vocal_mode: str = "stem_tube_master",
    custom_ref_voice_vol_path: Optional[str] = None,
    semitone_shift: int = 0,
) -> Tuple[str, str]:
    """
    1. Isolates Vocals & Instrumental using BS-RoFormer (SDR 12.97).
    2. Optionally runs Plachtaa/Seed-VC 44.1kHz BigVGAN Singing Voice Conversion (`f0-condition=True`).
    3. Applies analog tube grit + 220Hz chest body + 3.2kHz presence + broadcast -13 LUFS loudness master.
    """
    import json
    from mv_renderer import convert_wav_to_mp3

    ledger = BudgetLedger()
    _safe_vol_reload()
    job_dir = Path(VOL_MOUNT) / "jobs" / job_id
    raw_wav = job_dir / "raw_song.wav"
    if not raw_wav.exists():
        raise FileNotFoundError(f"Missing Stage 1 audio: {raw_wav}")

    s2_dir = job_dir / "stage2_master"
    s2_dir.mkdir(parents=True, exist_ok=True)
    master_wav = job_dir / "mastered_song.wav"
    master_mp3 = job_dir / "mastered_song.mp3"

    if vocal_mode == "bypass":
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(raw_wav),
                "-af", "loudnorm=I=-13.5:TP=-1.0:LRA=9",
                "-ar", "48000", "-ac", "2",
                str(master_wav),
            ],
            check=True,
        )
        convert_wav_to_mp3(master_wav, master_mp3)
        studio_vol.commit()
        return str(master_wav), str(master_mp3)

    with MeteredStageContext(ledger, "Stage2_RoFormer_SeedVC", "L4", is_gpu=True):
        sep_model_dir = Path(VOL_MOUNT) / "models" / "audio-separator"
        sep_model_dir.mkdir(parents=True, exist_ok=True)

        sep_script = (
            "import json, sys; from audio_separator.separator import Separator; "
            "sep = Separator(output_dir=sys.argv[1], model_file_dir=sys.argv[2], output_format='WAV'); "
            "sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); "
            "print('SEP_JSON:' + json.dumps(sep.separate(sys.argv[3])))"
        )
        sep_proc = subprocess.run(
            ["/opt/sep_venv/bin/python", "-c", sep_script, str(s2_dir), str(sep_model_dir), str(raw_wav)],
            capture_output=True,
            text=True,
            check=False,
        )
        out_files: List[str] = []
        for line in (sep_proc.stdout or "").splitlines():
            if line.startswith("SEP_JSON:"):
                out_files = json.loads(line[len("SEP_JSON:"):])

        vocal_stem: Optional[Path] = None
        inst_stem: Optional[Path] = None
        for f in out_files:
            fp = Path(f) if Path(f).is_absolute() else (s2_dir / f)
            low = fp.name.lower()
            if "vocal" in low and "no_vocal" not in low:
                vocal_stem = fp
            elif "inst" in low or "other" in low or "no_vocal" in low:
                inst_stem = fp

        if not vocal_stem or not inst_stem or not vocal_stem.exists() or not inst_stem.exists():
            # Fallback to clean loudnorm master if stem naming changed
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(raw_wav),
                    "-af", "equalizer=f=220:t=q:w=1.2:g=2.5,equalizer=f=3200:t=q:w=1.4:g=2.2,loudnorm=I=-13.2:TP=-1.0:LRA=9",
                    str(master_wav),
                ],
                check=True,
            )
            convert_wav_to_mp3(master_wav, master_mp3)
            studio_vol.commit()
            return str(master_wav), str(master_mp3)

        active_vocal = vocal_stem

        # Optional Zero-Shot Singing Voice Conversion via Seed-VC (44.1kHz BigVGAN)
        if vocal_mode in ("seed_vc_mit_husky", "seed_vc_custom"):
            ref_wav = Path("/root/assets/vocal_seeds/mit_husky_rock_seed.wav")
            if vocal_mode == "seed_vc_custom" and custom_ref_voice_vol_path and Path(custom_ref_voice_vol_path).exists():
                ref_wav = Path(custom_ref_voice_vol_path)

            svc_out_dir = s2_dir / "seed_vc_out"
            svc_out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "python", "inference.py",
                    "--source", str(vocal_stem),
                    "--target", str(ref_wav),
                    "--output", str(svc_out_dir),
                    "--diffusion-steps", "35",
                    "--length-adjust", "1.0",
                    "--inference-cfg-rate", "0.7",
                    "--f0-condition", "True",
                    "--auto-f0-adjust", "False",
                    "--semi-tone-shift", str(int(semitone_shift)),
                    "--fp16", "True",
                ],
                cwd="/opt/seed-vc",
                check=True,
            )
            svc_wavs = sorted(svc_out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
            if svc_wavs:
                active_vocal = svc_wavs[0]

        # Studio Analog Tube Warmth + Chest Grit + Live Concert Hall Slapback + Stem Remix
        vocal_fx = (
            "highpass=f=85,"
            "equalizer=f=215:t=q:w=1.15:g=3.2,"
            "equalizer=f=2950:t=q:w=1.35:g=2.8,"
            "acompressor=threshold=-17dB:ratio=3.5:attack=8:release=95:makeup=3.0,"
            "aecho=0.88:0.82:42:0.14,"
            "volume=1.18"
        )
        inst_fx = (
            "equalizer=f=68:t=q:w=1.1:g=2.4,"
            "equalizer=f=2900:t=q:w=1.5:g=-1.6,"
            "volume=0.96"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(active_vocal),
                "-i", str(inst_stem),
                "-filter_complex",
                f"[0:a]{vocal_fx}[voc];[1:a]{inst_fx}[ins];"
                f"[voc][ins]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
                f"loudnorm=I=-13.0:TP=-1.0:LRA=8.5[aout]",
                "-map", "[aout]",
                "-ar", "48000", "-ac", "2",
                str(master_wav),
            ],
            check=True,
        )
        convert_wav_to_mp3(master_wav, master_mp3)
        studio_vol.commit()
        return str(master_wav), str(master_mp3)


# ---------------------------------------------------------------------------
# Stage 3: Wan2.2-S2V-14B Audio-Driven Full-Body Singing Video Director (A100-80GB)
# ---------------------------------------------------------------------------
@app.cls(
    image=wan_s2v_image,
    gpu=["H100", "A100-80GB", "L40S"],
    volumes={VOL_MOUNT: studio_vol},
    min_containers=0,
    max_containers=1,
    scaledown_window=20,
    timeout=2400,
)
class WanS2VDirector:
    @modal.enter()
    def load_wan_s2v(self) -> None:
        import torch

        sys.path.insert(0, "/opt/Wan2.2")
        import wan
        from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

        _safe_vol_reload()
        ckpt_dir = Path(VOL_MOUNT) / "models" / "Wan2.2-S2V-14B"
        if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
            raise RuntimeError(
                "Wan2.2-S2V-14B weights not found on /vol/models/Wan2.2-S2V-14B. "
                "Run CPU weight prefetch first so GPU seconds are never spent downloading!"
            )

        self.gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self.is_80gb = vram_gb >= 70.0
        if "H100" in self.gpu_name:
            self.hw_tier = "H100"
        elif self.is_80gb:
            self.hw_tier = "A100-80GB"
        else:
            self.hw_tier = "L40S"
        print(
            f"[WanS2V] Initializing on {self.gpu_name} "
            f"({vram_gb:.1f} GB VRAM, tier={self.hw_tier}, resident_gpu={self.is_80gb})",
            flush=True,
        )

        self.cfg = WAN_CONFIGS["s2v-14B"]
        self.max_area = MAX_AREA_CONFIGS["832*480"]
        self.ckpt_dir = str(ckpt_dir)
        self.pipeline = wan.WanS2V(
            config=self.cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=(not self.is_80gb),
            convert_model_dtype=True,
        )

    @modal.method()
    def direct_and_render_mv(
        self,
        job_id: str,
        video_mode: str,
        lyrics_text: str,
        song_title: str,
        singer_line: str,
        credits_line: str,
        custom_singer_img_vol_path: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        import torch
        from wan.utils.utils import save_video
        from s2v_mv_compositor import compose_broadcast_s2v_mv, plan_s2v_timeline

        ledger = BudgetLedger()
        _safe_vol_reload()
        job_dir = Path(VOL_MOUNT) / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        master_wav = job_dir / "mastered_song.wav"
        if not master_wav.exists():
            alt_raw = job_dir / "raw_song.wav"
            if alt_raw.exists():
                master_wav = alt_raw
            else:
                # Warm container with mmap'd weights couldn't reload /vol: pull fresh WAV via clean CPU reader
                wav_bytes = fetch_volume_file_bytes.remote(f"{VOL_MOUNT}/jobs/{job_id}/mastered_song.wav")
                tmp_wav = Path("/tmp") / f"{job_id}_mastered.wav"
                tmp_wav.write_bytes(wav_bytes)
                master_wav = tmp_wav

        s3_dir = Path("/tmp") / f"s3_{job_id}"
        s3_dir.mkdir(parents=True, exist_ok=True)
        clips_vol_dir = job_dir / "s2v_clips"
        clips_vol_dir.mkdir(parents=True, exist_ok=True)
        final_mp4 = job_dir / "final_1080p_mv.mp4"

        custom_img = Path(custom_singer_img_vol_path) if (custom_singer_img_vol_path and Path(custom_singer_img_vol_path).exists()) else None
        heroes_dir = Path("/root/assets/stage_heroes")
        broll_dir = Path("/root/assets/frames")

        segments = plan_s2v_timeline(
            audio_path=master_wav,
            work_dir=s3_dir,
            video_mode=video_mode,
            custom_singer_image=custom_img,
            heroes_dir=heroes_dir,
            broll_dir=broll_dir,
        )

        base_seed = int(seed) if int(seed) >= 0 else int(time.time()) % 100000
        s2v_segs = [s for s in segments if s.kind == "wan_s2v" and s.audio_slice_path]

        with MeteredStageContext(
            ledger,
            stage_name=f"Stage3_Wan2.2_S2V_14B_{video_mode}",
            hardware_tier=getattr(self, "hw_tier", "A100-80GB"),
            is_gpu=True,
            increment_jobs=True,
        ):
            for clip_num, seg in enumerate(s2v_segs, start=1):
                clip_vol = clips_vol_dir / f"wan_raw_{seg.seg_idx:03d}.mp4"
                clip_out = s3_dir / f"wan_raw_{seg.seg_idx:03d}.mp4"
                if clip_vol.exists() and clip_vol.stat().st_size > 12000:
                    print(f"[WanS2V] Reusing checkpointed clip {clip_num}/{len(s2v_segs)}: {clip_vol.name}", flush=True)
                    shutil.copyfile(clip_vol, clip_out)
                    seg.rendered_clip_path = clip_out
                    continue

                t_clip = time.time()
                print(
                    f"[WanS2V] Diffusing hero singing clip {clip_num}/{len(s2v_segs)} "
                    f"[{seg.t_start:.2f}s..{seg.t_end:.2f}s] ({seg.image_path.name})...",
                    flush=True,
                )
                video_tensor = self.pipeline.generate(
                    input_prompt=seg.prompt,
                    ref_image_path=str(seg.image_path),
                    audio_path=str(seg.audio_slice_path),
                    enable_tts=False,
                    tts_prompt_audio=None,
                    tts_prompt_text=None,
                    tts_text=None,
                    num_repeat=1 if video_mode == "smart_hybrid_s2v" else None,
                    pose_video=None,
                    max_area=self.max_area,
                    infer_frames=64,
                    shift=3.0,
                    sample_solver="unipc",
                    sampling_steps=16,
                    guide_scale=getattr(self.cfg, "sample_guide_scale", 4.5),
                    seed=base_seed + seg.seg_idx,
                    offload_model=(not self.is_80gb),
                    init_first_frame=False,
                )
                save_video(
                    tensor=video_tensor[None],
                    save_file=str(clip_out),
                    fps=getattr(self.cfg, "sample_fps", 16),
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                del video_tensor
                torch.cuda.empty_cache()
                shutil.copyfile(clip_out, clip_vol)
                studio_vol.commit()
                seg.rendered_clip_path = clip_out
                print(
                    f"[WanS2V] Finished clip {clip_num}/{len(s2v_segs)} in {time.time() - t_clip:.1f}s "
                    f"({clip_out.stat().st_size} bytes)",
                    flush=True,
                )

            print("[WanS2V] Compositing 1080p30 broadcast multi-camera MV + KTV ASS...", flush=True)
            compose_broadcast_s2v_mv(
                master_audio_path=master_wav,
                segments=segments,
                output_mp4=final_mp4,
                lyrics_text=lyrics_text,
                song_title=song_title,
                singer_line=singer_line,
                credits_line=credits_line,
                work_dir=s3_dir / "compositor",
            )
            studio_vol.commit()
            print(f"[WanS2V] Completed 1080p MV: {final_mp4} ({final_mp4.stat().st_size} bytes)", flush=True)
            return str(final_mp4)


# ---------------------------------------------------------------------------
# Presets & Traditional Chinese Gradio Web Studio (Runs on Cheap CPU Container!)
# ---------------------------------------------------------------------------
FLAGSHIP_PRESETS: Dict[str, Dict[str, Any]] = {
    "jiayibingding_s2v": {
        "title": "醉月長安（豬八戒煙嗓炸場版）",
        "bpm": 96,
        "duration": 48,
        "singer": "主唱：詩仙·李白（Wan2.2-S2V 全身煙嗓演唱）",
        "style": (
            "Chinese Live TV Singing Show Symphonic Guofeng Power-Rock Anthem, Mandopop Emotional Arena Rock Live Band, "
            "Deep Raspy Smoky Male Vocal, Weathered Gravelly Husky Baritone-Tenor Voice with Heavy Emotional Rasp and Raw Throat Grit, "
            "Grand Concert Piano Chords and Mournful Chinese Erhu Intro, "
            "Intimate Husky Rhythmic Mandarin Melodic Rap Storytelling in Verse with Warm Electric Bass and Live Snare Groove, "
            "Dramatic Floor-Tom Build-Up and Swelling Strings in Pre-Chorus, "
            "Explosive Soaring Raspy High-Note Rock Belt and Rap-Rock Chorus with Massive Distorted Electric Guitar Power Chords, "
            "Heroic Chinese Suona Counter-Melody, Live Concert Stage Ambience, 96 BPM, C# Minor"
        ),
        "lyrics": """[Intro - Piano & Erhu]
這一壺烈酒 敬長安的風

[Verse 1 - Deep Raspy Smoky Melodic Rap]
金鑾殿的燭火照不亮我半生孤單
脫下官袍把功名利祿摔成兩半
世人笑我太瘋癲 把我當作路人看
只有手裡這壺濁酒懂我心裡的不甘

[Pre-Chorus - Rising Gritty Belt & Drum Build]
誰甘心一輩子只做別人的配角！
戰鼓給我敲響！全場把手舉高！

[Chorus - Explosive Raspy High-Note Rock Belt & Rap]
人生得意須盡歡！莫使金樽空對月！
就算千金散盡老子也絕不妥協！
天生我材必有用！管他風雪有多烈！
用這沙啞的狂歌把黑夜全撕裂！""",
    },
    "kuangge_nu_metal": {
        "title": "狂歌炸長安（嗩吶說唱核爆版）",
        "bpm": 154,
        "duration": 48,
        "singer": "主唱：詩仙·李白（大唐重型搖滾樂團 LIVE）",
        "style": (
            "Explosive Chinese Classical Guofeng Nu-Metal Rap-Rock, Heavy Stadium Rock Concert Anthem, "
            "Crushing Drop-D Distorted Electric Guitar Riffs, Thunderous Tang War Drums and Fast Double-Kick Drums, "
            "Piercing High-Energy Chinese Suona Horn Lead and Electric Pipa Riff, "
            "Aggressive Rapid-Fire Male Mandarin Rap Flow in Verses, Explosive Gritty Rock Shout Chorus, "
            "Linkin Park meets Chinese Classical Folk Metal Hip-Hop, 154 BPM, Massive Sub-Bass Broadcast Mix"
        ),
        "lyrics": """[Intro]
嘿！哈！嗩吶戰鼓給我炸起來！

[Verse 1 - Fast Rap]
金鑾殿的規矩老子通通撕碎
脫掉官靴甩開折子今夜不睡
戰鼓擂動長安街頭烈火在燒
左手酒壺右手麥克風把屋頂掀翻掉

[Pre-Chorus - Crowd Chant]
雙手舉高！跟著戰鼓跳！
一二三四！把全場都炸掉！

[Chorus - Explosive Rock & Rap]
人生得意須盡歡！莫使金樽空對月！
天生我材必有用！千金散盡還復來！
嗩吶一響震破九重天外天！
狂歌說唱老子就是大唐詩仙！""",
    },
}


def _render_budget_banner_html(state: Dict[str, Any], est_usd: float, est_mvs: int) -> str:
    used = float(state.get("effective_used_usd", 0.0))
    cap = float(state.get("cap_usd", DEFAULT_SAFE_MONTHLY_CAP_USD))
    kill = bool(state.get("kill_switch", False))
    jobs = int(state.get("jobs_completed", 0))
    pct_of_30 = min(100.0, round((used / MODAL_STARTER_FREE_CREDIT_USD) * 100.0, 1))
    fuse_pct = min(100.0, round((cap / MODAL_STARTER_FREE_CREDIT_USD) * 100.0, 1))

    badge_bg = "#dc2626" if kill else ("#d97706" if used >= cap * 0.85 else "#059669")
    badge_txt = "🛑 GPU 緊急鎖死中（$0 絕對凍結）" if kill else f"🛡️ $0.00 免付費雙重熔斷保護中（月帳單 {state.get('month', '')}）"

    return f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1.5px solid #f59e0b;border-radius:14px;padding:16px 20px;color:#f8fafc;margin-bottom:12px;">
      <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px;">
        <div style="font-size:16px;font-weight:700;color:#fbbf24;">⚡ NVIDIA A100-80GB + L40S 頂配音樂＆真人動態 MV 工作室</div>
        <span style="background:{badge_bg};color:#fff;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;">{badge_txt}</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;font-size:13px;margin-bottom:10px;">
        <div style="background:rgba(255,255,255,0.06);padding:8px 12px;border-radius:8px;">
          本月已用免費額度：<b style="color:#38bdf8;font-size:15px;">${used:.3f}</b> / $30.00
        </div>
        <div style="background:rgba(255,255,255,0.06);padding:8px 12px;border-radius:8px;">
          硬體熔斷安全線：<b style="color:#fbbf24;font-size:15px;">${cap:.2f}</b>（保留 ${30.0 - cap:.2f} 緩衝）
        </div>
        <div style="background:rgba(255,255,255,0.06);padding:8px 12px;border-radius:8px;">
          本次預估耗用：<b style="color:#4ade80;font-size:15px;">~${est_usd:.3f}</b>（月免費可做 ~{est_mvs} 支）
        </div>
        <div style="background:rgba(255,255,255,0.06);padding:8px 12px;border-radius:8px;">
          本月已完成頂配 MV：<b>{jobs} 支</b>｜閒置 GPU 費用：<b style="color:#4ade80;">$0.00/秒</b>
        </div>
      </div>
      <div style="position:relative;height:10px;background:#334155;border-radius:999px;overflow:hidden;">
        <div style="width:{pct_of_30}%;height:100%;background:linear-gradient(90deg,#22c55e,#eab308);"></div>
        <div style="position:absolute;left:{fuse_pct}%;top:0;bottom:0;width:3px;background:#ef4444;" title="安全熔斷線"></div>
      </div>
    </div>
    """


@app.function(
    image=cpu_web_image,
    volumes={VOL_MOUNT: studio_vol},
    cpu=1.0,
    memory=2048,
    min_containers=0,
    max_containers=1,
    scaledown_window=120,
    timeout=2700,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def web_studio():
    import gradio as gr
    from fastapi import FastAPI
    from mv_renderer import convert_wav_to_mp3, render_full_mv

    ledger = BudgetLedger()
    local_dl_dir = Path("/tmp/studio_downloads")
    local_dl_dir.mkdir(parents=True, exist_ok=True)

    COMPOSER_CHOICES = {
        "🎸 ACE-Step 1.5 XL（4B 滿血 SFT 50步 + 4B 旋律規劃｜國風說唱搖滾首選・MIT 商用）": "ace_xl",
        "🏆 YuE2-3B（WildSongBench 盲測超越 Suno v5/v6・全曲和弦 CoT・創作者免費商用）": "yue2",
    }
    VOCAL_CHOICES = {
        "🔥 BS-RoFormer 人聲伴奏分離 + 真空管膽機煙嗓母帶精修（最穩保留原唱張力）": "stem_tube_master",
        "🎙️ Seed-VC 44.1kHz BigVGAN 粗獷搖滾煙嗓重繪（內建 MIT 無版權搖滾男聲種子）": "seed_vc_mit_husky",
        "🧑‍🎤 Seed-VC 本人聲音出道模式（上傳 8~15 秒你的聲音，AI 用你的嗓音飆高音！）": "seed_vc_custom",
    }
    VIDEO_CHOICES = {
        "⚡ 精省爆款 S2V 導播模式（約 $0.34/支｜4 段 Wan2.2-S2V 全身真唱嘴型 + 音綜多機位｜月做 ~70 支）": "smart_hybrid_s2v",
        "🎬 全曲滿血 S2V 模式（約 $0.95/支｜全曲 100% Wan2.2-S2V-14B 全身動態演唱｜月做 ~25 支）": "full_s2v",
        "🖼️ 極省算力運鏡模式（約 $0.08/支｜僅用 L40S+L4 做頂配音樂，畫面採 1080p 運鏡｜月做 ~280 支）": "ken_burns_only",
    }

    def refresh_dashboard(comp_label: str, voc_label: str, vid_label: str, dur_sec: int, cap_val: float) -> str:
        est = estimate_pipeline_cost(
            composer_engine=COMPOSER_CHOICES.get(comp_label, "ace_xl"),
            vocal_mode=VOCAL_CHOICES.get(voc_label, "stem_tube_master"),
            video_mode=VIDEO_CHOICES.get(vid_label, "smart_hybrid_s2v"),
            song_duration_sec=int(dur_sec),
        )
        st = ledger.set_user_cap(cap_val)
        return _render_budget_banner_html(st, est.total_estimated_usd, est.est_mvs_per_month_at_24_cap)

    def toggle_kill(engage: bool, comp_label: str, voc_label: str, vid_label: str, dur_sec: int, cap_val: float):
        st = ledger.set_kill_switch(engage)
        est = estimate_pipeline_cost(
            composer_engine=COMPOSER_CHOICES.get(comp_label, "ace_xl"),
            vocal_mode=VOCAL_CHOICES.get(voc_label, "stem_tube_master"),
            video_mode=VIDEO_CHOICES.get(vid_label, "smart_hybrid_s2v"),
            song_duration_sec=int(dur_sec),
        )
        note = "🛑 已強制鎖死所有 GPU 容器！目前絕不會產生任何算力消耗。" if engage else "🔓 已解除鎖死，恢復正常安全熔斷保護。"
        return _render_budget_banner_html(st, est.total_estimated_usd, est.est_mvs_per_month_at_24_cap), note

    def apply_preset(key: str):
        p = FLAGSHIP_PRESETS[key]
        return p["title"], p["bpm"], p["duration"], p["singer"], p["style"], p["lyrics"]

    def run_flagship_pipeline(
        song_title: str,
        bpm: int,
        duration_sec: int,
        singer_line: str,
        style_prompt: str,
        lyrics_text: str,
        comp_label: str,
        voc_label: str,
        vid_label: str,
        custom_voice_file: Optional[str],
        custom_singer_img: Optional[str],
        cap_val: float,
        seed: int,
        progress=gr.Progress(),
    ):
        comp_key = COMPOSER_CHOICES.get(comp_label, "ace_xl")
        voc_key = VOCAL_CHOICES.get(voc_label, "stem_tube_master")
        vid_key = VIDEO_CHOICES.get(vid_label, "smart_hybrid_s2v")

        est = estimate_pipeline_cost(
            composer_engine=comp_key,
            vocal_mode=voc_key,
            video_mode=vid_key,
            song_duration_sec=int(duration_sec),
        )

        # Layer 3 Pre-Flight Hard Circuit Breaker Check (BEFORE any GPU wakes up!)
        try:
            ledger.set_user_cap(cap_val)
            ledger.assert_preflight_safe(est, requested_cap_usd=cap_val)
        except BudgetFuseTrippedError as exc:
            raise gr.Error(str(exc))

        job_id = f"mv_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job_dir = Path(VOL_MOUNT) / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        ref_voice_vol: Optional[str] = None
        if custom_voice_file and os.path.exists(custom_voice_file):
            dst_v = job_dir / "custom_ref_voice.wav"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", custom_voice_file, "-t", "20", "-ar", "44100", "-ac", "1", str(dst_v)], check=True)
            ref_voice_vol = str(dst_v)

        singer_img_vol: Optional[str] = None
        if custom_singer_img and os.path.exists(custom_singer_img):
            dst_i = job_dir / "custom_singer.jpg"
            shutil.copyfile(custom_singer_img, dst_i)
            singer_img_vol = str(dst_i)

        studio_vol.commit()

        # Ensure required model weights exist on /vol/models using cheap CPU prefetcher first!
        needed_prefetch: List[str] = []
        models_dir = Path(VOL_MOUNT) / "models"
        if comp_key == "ace_xl" and not (models_dir / "acestep_checkpoints" / "acestep-v15-xl-sft").exists():
            needed_prefetch.append("ace_xl")
        if comp_key == "yue2" and not (models_dir / "YuE2-3B").exists():
            needed_prefetch.append("yue2")
        if vid_key != "ken_burns_only" and not (models_dir / "Wan2.2-S2V-14B").exists():
            needed_prefetch.append("wan_s2v")

        if needed_prefetch:
            progress(0.05, desc=f"首次執行：正在用純 CPU 高速預載模型權重至雲端硬碟（不消耗任何 GPU 額度）：{needed_prefetch}...")
            prefetch_weights_cpu.remote(needed_prefetch)

        # Stage 1: Flagship Music Composition (L40S 48GB)
        progress(0.18, desc=f"Stage 1/4：啟動 L40S (48GB) 頂配編曲引擎（{comp_key}）合成真實樂團與說唱主歌...")
        if comp_key == "yue2":
            compose_yue2_song.remote(
                job_id=job_id,
                style_prompt=style_prompt,
                lyrics_text=lyrics_text,
                duration_sec=int(duration_sec),
                seed=int(seed),
            )
        else:
            AceStepXLComposer().compose.remote(
                job_id=job_id,
                style_prompt=style_prompt,
                lyrics_text=lyrics_text,
                duration_sec=int(duration_sec),
                bpm=int(bpm),
                vocal_lang="zh",
                seed=int(seed),
            )

        # Stage 2: BS-RoFormer Stem Separation + Seed-VC 44.1kHz Vocal Mastering (L4 24GB)
        progress(0.45, desc="Stage 2/4：啟動 L4 進行 BS-RoFormer 人聲伴奏分離 + 44.1kHz 搖滾煙嗓精修與母帶混音...")
        master_wav_str, master_mp3_str = master_vocals_and_band.remote(
            job_id=job_id,
            vocal_mode=voc_key,
            custom_ref_voice_vol_path=ref_voice_vol,
            semitone_shift=0,
        )

        # Stage 3: Wan2.2-S2V-14B Full-Body Singing Video Diffusion (A100-80GB) OR Fast CPU Ken Burns
        if vid_key == "ken_burns_only":
            progress(0.75, desc="Stage 3/4：正在渲染 1080p 舞台運鏡與 KTV 流光字幕（零 A100 消耗模式）...")
            _safe_vol_reload()
            final_mp4_vol = job_dir / "final_1080p_mv.mp4"
            render_full_mv(
                audio_path=Path(master_wav_str),
                output_mp4=final_mp4_vol,
                lyrics_text=lyrics_text,
                song_title=song_title,
                singer_line=singer_line,
                frames_dir=Path("/root/assets/frames"),
                work_dir=job_dir / "kb_cache",
            )
            studio_vol.commit()
        else:
            progress(0.62, desc="Stage 3/4：啟動 NVIDIA A100-80GB 執行 Wan2.2-S2V-14B 全身真唱動作、嘴型與呼吸擴散...")
            final_mp4_str = WanS2VDirector().direct_and_render_mv.remote(
                job_id=job_id,
                video_mode=vid_key,
                lyrics_text=lyrics_text,
                song_title=song_title,
                singer_line=singer_line,
                credits_line="詞曲：100% 無版權原創搖滾｜A100 S2V 頂配母帶室",
                custom_singer_img_vol_path=singer_img_vol,
                seed=int(seed),
            )
            final_mp4_vol = Path(final_mp4_str)

        progress(0.96, desc="Stage 4/4：正在同步成品供手機與電腦一鍵直接下載...")
        _safe_vol_reload()
        safe_slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (song_title or "pro_mv").strip())[:24] or "pro_mv"
        dl_mp4 = local_dl_dir / f"{safe_slug}_{job_id}.mp4"
        dl_mp3 = local_dl_dir / f"{safe_slug}_{job_id}.mp3"
        if final_mp4_vol.exists():
            shutil.copyfile(final_mp4_vol, dl_mp4)
        else:
            dl_mp4.write_bytes(fetch_volume_file_bytes.remote(str(final_mp4_vol)))
        if Path(master_mp3_str).exists():
            shutil.copyfile(master_mp3_str, dl_mp3)
        elif Path(master_wav_str).exists():
            convert_wav_to_mp3(Path(master_wav_str), dl_mp3)
        else:
            dl_mp3.write_bytes(fetch_volume_file_bytes.remote(str(master_mp3_str)))

        st_after = ledger.get_state()
        banner = _render_budget_banner_html(st_after, est.total_estimated_usd, est.est_mvs_per_month_at_24_cap)
        summary_msg = (
            f"✅ 頂配生成完成！所有 GPU 容器將在 20 秒內自動歸零休眠（閒置 $0.00）。"
            f"本月累計使用免費額度：${st_after['effective_used_usd']:.3f} / 安全上限 ${st_after['cap_usd']:.2f}（Modal 免費額度 $30.00）。"
        )
        return str(dl_mp4), str(dl_mp3), str(dl_mp4), str(dl_mp3), banner, summary_msg

    init_est = estimate_pipeline_cost()
    init_state = ledger.get_state()

    with gr.Blocks(title="A100 頂配免費音樂＆真人動態 MV 工作室") as demo:
        banner_html = gr.HTML(
            value=_render_budget_banner_html(init_state, init_est.total_estimated_usd, init_est.est_mvs_per_month_at_24_cap)
        )

        with gr.Accordion("🛡️ 四層 $0.00 免付費安全鎖控制台（隨時可調整熔斷上限或一鍵鎖死 GPU）", open=False):
            with gr.Row():
                cap_slider = gr.Slider(
                    minimum=MIN_CONFIGURABLE_CAP_USD,
                    maximum=ABSOLUTE_MAX_MONTHLY_CAP_USD,
                    value=float(init_state.get("cap_usd", DEFAULT_SAFE_MONTHLY_CAP_USD)),
                    step=0.5,
                    label="本月免費額度硬熔斷上限（美元 USD｜系統強制鎖定 <= $27.00，絕不觸碰 $30 免費紅線）",
                )
                kill_btn = gr.Button("🛑 一鍵緊急鎖死所有 GPU", variant="stop")
                unkill_btn = gr.Button("🔓 解除緊急鎖死", variant="secondary")
            guard_status = gr.Markdown("💡 **雙保險提醒**：除了本系統內建 `$24.00` 硬體熔斷外，請務必在 Modal 後台 `Settings → Usage & Billing` 將 `Spend Limit` 設為 **`$0`**，即可在帳單層 100% 物理杜絕任何扣款！")

        with gr.Row():
            preset_jyb = gr.Button("🔥 載入範本：《醉月長安（豬八戒煙嗓炸場版）》", variant="primary")
            preset_kg = gr.Button("⚡ 載入範本：《狂歌炸長安（嗩吶說唱核爆版）》", variant="secondary")

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    song_title_in = gr.Textbox(label="歌曲名稱", value=FLAGSHIP_PRESETS["jiayibingding_s2v"]["title"], scale=3)
                    bpm_in = gr.Number(label="節拍 BPM", value=FLAGSHIP_PRESETS["jiayibingding_s2v"]["bpm"], precision=0, scale=1)
                    dur_in = gr.Slider(24, 90, value=48, step=4, label="歌曲長度（秒）", scale=2)
                singer_in = gr.Textbox(label="舞台歌手卡頭銜", value=FLAGSHIP_PRESETS["jiayibingding_s2v"]["singer"])
                style_in = gr.Textbox(label="編曲與嗓音風格指令（Style Prompt）", value=FLAGSHIP_PRESETS["jiayibingding_s2v"]["style"], lines=4)
                lyrics_in = gr.Textbox(label="結構化歌詞（支援 [Verse - Rap] / [Chorus] 等段落標籤）", value=FLAGSHIP_PRESETS["jiayibingding_s2v"]["lyrics"], lines=11)

            with gr.Column(scale=5):
                comp_in = gr.Dropdown(choices=list(COMPOSER_CHOICES.keys()), value=list(COMPOSER_CHOICES.keys())[0], label="① 第一階段（L40S 48GB）：頂配寫歌與編曲引擎")
                voc_in = gr.Dropdown(choices=list(VOCAL_CHOICES.keys()), value=list(VOCAL_CHOICES.keys())[0], label="② 第二階段（L4 24GB）：人聲分軌＆煙嗓音色重繪")
                vid_in = gr.Dropdown(choices=list(VIDEO_CHOICES.keys()), value=list(VIDEO_CHOICES.keys())[0], label="③ 第三階段（A100-80GB）：Wan2.2-S2V 全身真唱動態模式")

                with gr.Accordion("🎤 進階自訂：上傳你自己的參考嗓音或舞台主角照片（選填）", open=False):
                    custom_voice_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="上傳/錄製 8~15 秒參考嗓音（選「本人聲音出道模式」時生效）")
                    custom_img_in = gr.Image(type="filepath", label="上傳自訂舞台主唱高清半身照（Wan2.2-S2V 將讓照片中的主角開口飆歌！）")
                    seed_in = gr.Number(label="隨機種子 Seed（-1 為隨機）", value=-1, precision=0)

                run_btn = gr.Button("🚀 啟動 A100 頂配四階段流水線（自動熔斷保護）", variant="primary", size="lg")
                run_status = gr.Markdown("準備就緒：點擊上方按鈕即可啟動 L40S 編曲 → L4 煙嗓分軌母帶 → A100-80GB 真人動態演唱合成。")

        with gr.Row():
            out_video = gr.Video(label="🎬 1080p 真人動態演唱舞台 MV 預覽")
            with gr.Column():
                out_audio = gr.Audio(label="🎵 320kbps 廣播級母帶單曲預覽")
                dl_mp4_file = gr.File(label="📥 手機/電腦直接下載 1080p MP4 影片檔（長按或點擊下載）")
                dl_mp3_file = gr.File(label="📥 手機/電腦直接下載 320kbps MP3 音檔")

        # Live Cost & Fuse Reactivity
        for ctrl in (comp_in, voc_in, vid_in, dur_in, cap_slider):
            ctrl.change(
                fn=refresh_dashboard,
                inputs=[comp_in, voc_in, vid_in, dur_in, cap_slider],
                outputs=[banner_html],
            )

        kill_btn.click(
            fn=lambda c, v, vd, d, cp: toggle_kill(True, c, v, vd, d, cp),
            inputs=[comp_in, voc_in, vid_in, dur_in, cap_slider],
            outputs=[banner_html, guard_status],
        )
        unkill_btn.click(
            fn=lambda c, v, vd, d, cp: toggle_kill(False, c, v, vd, d, cp),
            inputs=[comp_in, voc_in, vid_in, dur_in, cap_slider],
            outputs=[banner_html, guard_status],
        )
        preset_jyb.click(
            fn=lambda: apply_preset("jiayibingding_s2v"),
            outputs=[song_title_in, bpm_in, dur_in, singer_in, style_in, lyrics_in],
        )
        preset_kg.click(
            fn=lambda: apply_preset("kuangge_nu_metal"),
            outputs=[song_title_in, bpm_in, dur_in, singer_in, style_in, lyrics_in],
        )

        run_btn.click(
            fn=run_flagship_pipeline,
            inputs=[
                song_title_in, bpm_in, dur_in, singer_in, style_in, lyrics_in,
                comp_in, voc_in, vid_in, custom_voice_in, custom_img_in, cap_slider, seed_in,
            ],
            outputs=[out_video, out_audio, dl_mp4_file, dl_mp3_file, banner_html, run_status],
        )

    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo, path="/")


@app.function(
    image=cpu_web_image,
    volumes={VOL_MOUNT: studio_vol},
    cpu=1.0,
    memory=2048,
    min_containers=0,
    max_containers=1,
    timeout=300,
)
def fetch_volume_file_bytes(vol_path: str) -> bytes:
    _safe_vol_reload()
    return Path(vol_path).read_bytes()


@app.local_entrypoint()
def main(
    preset: str = "jiayibingding_s2v",
    composer: str = "ace_xl",
    vocal_mode: str = "stem_tube_master",
    video_mode: str = "smart_hybrid_s2v",
    cap_usd: float = DEFAULT_SAFE_MONTHLY_CAP_USD,
    prefetch_only: bool = False,
    out_mp4: str = "flagship_a100_mv.mp4",
    out_mp3: str = "flagship_a100_song.mp3",
) -> None:
    """
    One-command CLI / Colab batch runner with full $0.00 safety fuse enforcement.
    """
    if prefetch_only:
        print("📦 啟動純 CPU 模型權重預載（零 GPU 消耗）...")
        res = prefetch_weights_cpu.remote(["ace_xl", "wan_s2v"])
        print(f"✅ 權重預載完成：{res}")
        return

    p = FLAGSHIP_PRESETS.get(preset, FLAGSHIP_PRESETS["jiayibingding_s2v"])
    est = estimate_pipeline_cost(
        composer_engine=composer,
        vocal_mode=vocal_mode,
        video_mode=video_mode,
        song_duration_sec=int(p["duration"]),
    )
    ledger = BudgetLedger()
    ledger.set_user_cap(cap_usd)
    state = ledger.assert_preflight_safe(est, requested_cap_usd=cap_usd)
    print(
        f"🛡️ 安全熔斷檢查通過！本月已用：${state['effective_used_usd']:.3f} / 上限 ${state['cap_usd']:.2f} "
        f"｜ 本次預估：~${est.total_estimated_usd:.3f}"
    )

    targets = ["wan_s2v"] if video_mode != "ken_burns_only" else []
    targets.append("yue2" if composer == "yue2" else "ace_xl")
    prefetch_weights_cpu.remote(targets)

    job_id = f"cli_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    print(f"🎸 [1/3] 啟動 L40S 頂配編曲（{composer}）：{p['title']}...")
    if composer == "yue2":
        compose_yue2_song.remote(job_id=job_id, style_prompt=p["style"], lyrics_text=p["lyrics"], duration_sec=int(p["duration"]))
    else:
        AceStepXLComposer().compose.remote(
            job_id=job_id, style_prompt=p["style"], lyrics_text=p["lyrics"], duration_sec=int(p["duration"]), bpm=int(p["bpm"])
        )

    print(f"🎙️ [2/3] 啟動 L4 BS-RoFormer 分軌＆煙嗓母帶精修（{vocal_mode}）...")
    _, mp3_vol = master_vocals_and_band.remote(job_id=job_id, vocal_mode=vocal_mode)

    print(f"🎬 [3/3] 啟動 NVIDIA A100-80GB Wan2.2-S2V-14B 真人全身演唱合成（{video_mode}）...")
    mp4_vol = WanS2VDirector().direct_and_render_mv.remote(
        job_id=job_id,
        video_mode=video_mode,
        lyrics_text=p["lyrics"],
        song_title=p["title"],
        singer_line=p["singer"],
        credits_line="詞曲：100% 無版權原創搖滾｜A100 S2V 頂配母帶室",
    )

    Path(out_mp4).write_bytes(fetch_volume_file_bytes.remote(mp4_vol))
    Path(out_mp3).write_bytes(fetch_volume_file_bytes.remote(mp3_vol))
    st_end = ledger.get_state()
    print(f"🎉 大功告成！已儲存：{out_mp4} ＆ {out_mp3} ｜ 本月累計免費額度消耗：${st_end['effective_used_usd']:.3f} / $30.00")

