#!/usr/bin/env python3
"""
High-speed Hugging Face checkpoint downloader & architecture fixer for ACE-Step 1.5 on Google Colab T4.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

from huggingface_hub import HfApi, hf_hub_download


def download_with_progress(repo_id: str, local_dir: str, title: str) -> None:
    api = HfApi()
    print(f"\n🔍 正在取得模型清單：{title} ({repo_id})...", flush=True)
    info = api.model_info(repo_id, files_metadata=True)
    siblings = {
        s.rfilename: s.size
        for s in info.siblings
        if s.size is not None and not s.rfilename.startswith(".")
    }
    total_bytes = sum(siblings.values())
    total_gb = total_bytes / (1024**3)
    target_dir = Path(local_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 開始高速下載 {title}（共 {total_gb:.2f} GB，{len(siblings)} 個檔案）...", flush=True)
    downloaded_bytes = 0
    start_time = time.time()

    for filename, size in siblings.items():
        dst = target_dir / filename
        if dst.exists() and dst.stat().st_size == size:
            downloaded_bytes += size
            continue

        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir))
        downloaded_bytes += size
        elapsed = max(0.1, time.time() - start_time)
        speed_mb = (downloaded_bytes / (1024**2)) / elapsed
        pct = int((downloaded_bytes / total_bytes) * 100) if total_bytes > 0 else 100
        bar = "█" * (pct // 4) + "░" * (25 - (pct // 4))
        short_name = filename.split("/")[-1][:24]
        sys.stdout.write(
            f"\r  [{bar}] {pct:3d}% | {downloaded_bytes / (1024**3):.2f}/{total_gb:.2f} GB | {speed_mb:.1f} MB/s | {short_name:<24}"
        )
        sys.stdout.flush()

    print(f"\n✅ {title} 下載完成！", flush=True)


def main() -> None:
    acestep_root = Path(os.environ.get("ACESTEP_ROOT", "/content/ACE-Step-1.5"))
    checkpoints_dir = acestep_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main Foundation Model (Turbo DiT + VAE + Text Encoder + 1.7B LM)
    turbo_weights = checkpoints_dir / "acestep-v15-turbo" / "model.safetensors"
    if not turbo_weights.exists():
        download_with_progress(
            "ACE-Step/Ace-Step1.5",
            str(checkpoints_dir),
            "ACE-Step 1.5 主模型 (Turbo DiT + VAE + 編碼器)",
        )
    else:
        print("✅ ACE-Step 1.5 主模型已存在，跳過重複下載。", flush=True)

    # 2. Lightweight 0.6B 5Hz LM (Best speed & low RAM for Colab Free T4)
    lm_06b_weights = checkpoints_dir / "acestep-5Hz-lm-0.6B" / "model.safetensors"
    if not lm_06b_weights.exists():
        download_with_progress(
            "ACE-Step/acestep-5Hz-lm-0.6B",
            str(checkpoints_dir / "acestep-5Hz-lm-0.6B"),
            "5Hz 旋律規劃語言模型 (0.6B T4 極速版)",
        )
    else:
        print("✅ 0.6B 5Hz 旋律規劃模型已存在，跳過重複下載。", flush=True)

    # 3. Sync DiT Python model files if needed
    turbo_src = acestep_root / "acestep" / "models" / "turbo"
    turbo_dst = checkpoints_dir / "acestep-v15-turbo"
    if turbo_src.exists() and turbo_dst.exists():
        for py_file in turbo_src.glob("*.py"):
            if py_file.name != "__init__.py":
                shutil.copy2(py_file, turbo_dst / py_file.name)
        print("✅ 已同步 DiT 模型架構程式碼。", flush=True)

    # 4. Patch Qwen3Model -> Qwen3ForCausalLM upstream config bug for both 0.6B and 1.7B
    for lm_dir_name in ("acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-1.7B"):
        cfg_path = checkpoints_dir / lm_dir_name / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                if cfg.get("architectures") == ["Qwen3Model"]:
                    cfg["architectures"] = ["Qwen3ForCausalLM"]
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                    print(f"✅ 已自動修正 {lm_dir_name}/config.json 架構設定。", flush=True)
            except Exception as exc:
                print(f"⚠️ 檢查 {lm_dir_name} config 提示: {exc}", flush=True)


if __name__ == "__main__":
    main()
