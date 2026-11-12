#!/usr/bin/env python3
"""
繁體中文《私有版免費 Suno 音樂 ＆ 1080p MV 工作室》
Powered by Open-Source ACE-Step 1.5 (MIT Commercial License) + Tesla T4 Float32/BF16 Optimization.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["MPLBACKEND"] = "agg"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["LOGURU_LEVEL"] = "INFO"
os.environ["PYTHONWARNINGS"] = "ignore::SyntaxWarning,ignore::UserWarning"
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ACESTEP_ROOT = os.environ.get("ACESTEP_ROOT", "/content/ACE-Step-1.5")
STUDIO_DIR = Path(__file__).resolve().parent
for p_str in (ACESTEP_ROOT, str(STUDIO_DIR)):
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import gradio as gr
import torch

try:
    from loguru import logger
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        filter=lambda record: "parse_lm_output" not in record["name"]
        and "audio_code_" not in record["message"]
        and "M_MMAP_THRESHOLD" not in record["message"],
    )
except Exception:
    pass

from acestep.core.generation.handler.init_service_offload_context import InitServiceOffloadContextMixin
from acestep.core.generation.handler.metadata_utils import MetadataMixin
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music
from acestep.llm_inference import LLMHandler

from mv_renderer import convert_wav_to_mp3, render_full_mv

OUTPUT_DIR = os.path.join(ACESTEP_ROOT, "gradio_outputs")
MV_OUTPUT_DIR = os.path.join(ACESTEP_ROOT, "mv_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MV_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Tesla T4 (Pre-Ampere) & Colab 12.7GB System RAM Stability Patches
# ---------------------------------------------------------------------------
def _safe_build_metadata_dict(self, bpm=None, key_scale=None, time_signature=None, duration=None):
    metadata_dict = {"bpm": bpm if bpm else "N/A"}
    if key_scale is not None and str(key_scale).strip() and str(key_scale).strip() != "N/A":
        metadata_dict["keyscale"] = str(key_scale).strip()
    else:
        metadata_dict["keyscale"] = "N/A"
    if time_signature is not None and str(time_signature).strip() and str(time_signature).strip() != "N/A":
        metadata_dict["timesignature"] = str(time_signature).strip()
    else:
        metadata_dict["timesignature"] = "N/A"
    if duration is not None:
        try:
            metadata_dict["duration"] = f"{int(duration)} seconds"
        except Exception:
            metadata_dict["duration"] = "N/A"
    return metadata_dict


MetadataMixin._build_metadata_dict = _safe_build_metadata_dict
AceStepHandler._build_metadata_dict = _safe_build_metadata_dict


@contextmanager
def _safe_load_model_context(self, model_name: str):
    if not getattr(self, "offload_to_cpu", False):
        yield
        return
    if model_name == "model" and not getattr(self, "offload_dit_to_cpu", False):
        model = getattr(self, model_name, None)
        if model is not None:
            try:
                param = next(model.parameters())
                if param.device.type == "cpu":
                    self._recursive_to_device(model, self.device, self.dtype)
            except StopIteration:
                pass
        yield
        return

    model = getattr(self, model_name, None)
    if model is None:
        yield
        return

    sub_dtype = torch.float16 if model_name in ("text_encoder", "vae") else self.dtype
    self._recursive_to_device(model, self.device, sub_dtype)
    try:
        yield
    finally:
        self._recursive_to_device(model, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


InitServiceOffloadContextMixin._load_model_context = _safe_load_model_context
AceStepHandler._load_model_context = _safe_load_model_context


def _dtype_safe_infer_text_embeddings(self, text_token_idss):
    with torch.inference_mode():
        out = self.text_encoder(input_ids=text_token_idss, lyric_attention_mask=None).last_hidden_state
    if not torch.isfinite(out).all():
        enc_dtype = next(self.text_encoder.parameters()).dtype
        self.text_encoder.to(torch.float32)
        try:
            with torch.inference_mode():
                out = self.text_encoder(input_ids=text_token_idss, lyric_attention_mask=None).last_hidden_state
        finally:
            self.text_encoder.to(enc_dtype)
    with torch.inference_mode():
        return out.to(self.dtype)


def _dtype_safe_infer_lyric_embeddings(self, lyric_token_ids):
    with torch.inference_mode():
        return self.text_encoder.embed_tokens(lyric_token_ids).to(self.dtype)


AceStepHandler.infer_text_embeddings = _dtype_safe_infer_text_embeddings
AceStepHandler.infer_lyric_embeddings = _dtype_safe_infer_lyric_embeddings


def _robust_sample_tokens(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=float("-inf"))
    if not torch.isfinite(logits).any(dim=-1).all():
        raise RuntimeError("5Hz LM logits overflowed; ensure bfloat16/float32 is active.")
    if temperature > 0:
        probs = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    return torch.argmax(logits, dim=-1)


LLMHandler._sample_tokens = _robust_sample_tokens

orig_compute_max_new_tokens = getattr(LLMHandler, "_compute_max_new_tokens", None)


def _fast_compute_max_new_tokens(self, target_duration, generation_phase, fallback_max=None):
    if generation_phase == "cot":
        return 512
    if orig_compute_max_new_tokens:
        return orig_compute_max_new_tokens(self, target_duration, generation_phase, fallback_max)
    return 512 if generation_phase == "cot" else 2000


LLMHandler._compute_max_new_tokens = _fast_compute_max_new_tokens


def _fast_generate_with_constrained_decoding(
    self,
    input_ids: torch.Tensor,
    attention_mask,
    max_new_tokens: int,
    temperature: float,
    top_k,
    top_p,
    repetition_penalty: float,
    pad_token_id: int,
    streamer,
    constrained_processor=None,
):
    model = self.llm
    device = self.device
    generated_ids = input_ids.clone()
    attn_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
    model_kwargs = {"attention_mask": attn_mask}
    past_key_values = None
    use_cache = hasattr(model, "generation_config") and getattr(model.generation_config, "use_cache", True)
    eos_token_id = self.llm_tokenizer.eos_token_id if self.llm_tokenizer.eos_token_id is not None else pad_token_id
    logits_processor = self._build_logits_processor(repetition_penalty)
    is_cot_reasoning = getattr(constrained_processor, "stop_at_reasoning", False)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            outputs = self._forward_pass(model, generated_ids, model_kwargs, past_key_values, use_cache)
            next_token_logits = outputs.logits[:, -1, :]
            if constrained_processor is not None:
                next_token_logits = constrained_processor(generated_ids, next_token_logits)
            for processor in logits_processor:
                next_token_logits = processor(generated_ids, next_token_logits)
            next_token_logits = self._apply_top_k_filter(next_token_logits, top_k)
            next_token_logits = self._apply_top_p_filter(next_token_logits, top_p)
            next_tokens = self._sample_tokens(next_token_logits, temperature)
            self._update_constrained_processor_state(constrained_processor, next_tokens)
            should_stop = self._check_eos_token(next_tokens, eos_token_id, pad_token_id)

            next_tokens_unsqueezed = next_tokens.unsqueeze(1)
            generated_ids = torch.cat([generated_ids, next_tokens_unsqueezed], dim=1)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((input_ids.shape[0], 1), device=device, dtype=attn_mask.dtype)], dim=1
            )
            model_kwargs["attention_mask"] = attn_mask
            if use_cache and hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values
            if streamer is not None:
                streamer.put(next_tokens_unsqueezed)
            if should_stop:
                break
            if is_cot_reasoning and constrained_processor is not None:
                try:
                    from acestep.constrained_logits_processor import FSMState
                    if getattr(constrained_processor, "state", None) == FSMState.COMPLETED:
                        break
                except Exception:
                    pass
            if is_cot_reasoning and step > 10:
                recent_ids = generated_ids[0, -6:].tolist()
                recent_text = self.llm_tokenizer.decode(recent_ids, skip_special_tokens=False)
                if "</think>" in recent_text:
                    break

    if streamer is not None:
        streamer.end()
    del past_key_values
    return generated_ids


LLMHandler._generate_with_constrained_decoding = _fast_generate_with_constrained_decoding


def _fast_load_pytorch_model(self, model_path: str, device: str):
    try:
        from transformers import AutoModelForCausalLM

        cfg_path = Path(model_path) / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                if cfg.get("architectures") == ["Qwen3Model"]:
                    cfg["architectures"] = ["Qwen3ForCausalLM"]
                    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            load_dtype = torch.bfloat16
        else:
            load_dtype = torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.llm.eval()
        self.llm_backend = "pt"
        self.llm_initialized = True
        self.device = device
        self.dtype = load_dtype
        self.llm.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, f"5Hz LM initialized ({load_dtype})"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error initializing 5Hz LM: {e}"


LLMHandler._load_pytorch_model = _fast_load_pytorch_model

orig_generate_with_stop_condition = getattr(LLMHandler, "generate_with_stop_condition", None)


def _memory_safe_generate_with_stop_condition(self, *args, **kwargs):
    if hasattr(self, "llm") and self.llm is not None:
        try:
            self.llm.to(getattr(self, "device", "cuda"))
        except Exception:
            pass
    try:
        return orig_generate_with_stop_condition(self, *args, **kwargs)
    finally:
        if hasattr(self, "llm") and self.llm is not None:
            try:
                self.llm.to("cpu")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


LLMHandler.generate_with_stop_condition = _memory_safe_generate_with_stop_condition


# ---------------------------------------------------------------------------
# Engine State & Initialization
# ---------------------------------------------------------------------------
dit = AceStepHandler()
llm = LLMHandler()
ENGINE_STATE = {"dit_ready": False, "lm_ready": False, "active_lm": "None"}


def ensure_engine_ready(lm_choice: str = "0.6B") -> str:
    if not ENGINE_STATE["dit_ready"]:
        print("🚀 正在將 ACE-Step 1.5 Turbo 主模型載入 GPU...", flush=True)
        status, ok = dit.initialize_service(
            project_root=ACESTEP_ROOT,
            config_path="acestep-v15-turbo",
            device="cuda" if torch.cuda.is_available() else "cpu",
            offload_to_cpu=True,
            compile_model=False,
            use_flash_attention=False,
            quantization=None,
        )
        if not ok:
            raise RuntimeError(f"DiT 主模型載入失敗: {status}")

        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                print("🛡️ 偵測到 Tesla T4 顯卡：自動啟用 Float32 防溢位保護（徹底消除歌詞靜音 NaN 問題）", flush=True)
                dit.dtype = torch.float32
                if hasattr(dit, "model") and dit.model is not None:
                    dit.model = dit.model.float()
                if hasattr(dit, "model") and hasattr(dit.model, "encoder"):
                    def _enc_dtype_pre_hook(module, args, kwargs):
                        target_dtype = module.text_projector.weight.dtype
                        cast = lambda v: v.to(dtype=target_dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v
                        return tuple(cast(a) for a in args), {k: cast(v) for k, v in kwargs.items()}
                    dit.model.encoder.register_forward_pre_hook(_enc_dtype_pre_hook, with_kwargs=True)

                if hasattr(dit, "model") and hasattr(dit.model, "prepare_condition"):
                    orig_prep_cond = dit.model.prepare_condition
                    def _safe_prep_cond(self, *args, **kwargs):
                        target_dtype = torch.float32
                        new_kwargs = {
                            k: (v.to(dtype=target_dtype) if (isinstance(v, torch.Tensor) and v.is_floating_point()) else v)
                            for k, v in kwargs.items()
                        }
                        new_args = [
                            (a.to(dtype=target_dtype) if (isinstance(a, torch.Tensor) and a.is_floating_point()) else a)
                            for a in args
                        ]
                        return orig_prep_cond(*new_args, **new_kwargs)
                    dit.model.prepare_condition = _safe_prep_cond.__get__(dit.model, type(dit.model))

        ENGINE_STATE["dit_ready"] = True

    target_lm = "acestep-5Hz-lm-0.6B" if "0.6B" in lm_choice else ("acestep-5Hz-lm-1.7B" if "1.7B" in lm_choice else None)
    if target_lm and (not ENGINE_STATE["lm_ready"] or ENGINE_STATE["active_lm"] != target_lm):
        print(f"🧠 正在載入 5Hz 旋律與人聲規劃模型 ({target_lm})...", flush=True)
        if hasattr(llm, "llm") and llm.llm is not None:
            del llm.llm
            llm.llm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        lm_status, lm_ok = llm.initialize(
            checkpoint_dir=os.path.join(ACESTEP_ROOT, "checkpoints"),
            lm_model_path=target_lm,
            backend="pt",
            device="cuda" if torch.cuda.is_available() else "cpu",
            offload_to_cpu=True,
            dtype=torch.bfloat16,
        )
        if lm_ok:
            ENGINE_STATE["lm_ready"] = True
            ENGINE_STATE["active_lm"] = target_lm
        else:
            print(f"⚠️ LM 載入提示: {lm_status}", flush=True)
            ENGINE_STATE["lm_ready"] = False
            ENGINE_STATE["active_lm"] = "None"
    elif target_lm is None:
        ENGINE_STATE["lm_ready"] = False
        ENGINE_STATE["active_lm"] = "None"

    gpu_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return f"✅ 引擎已就緒！GPU：{gpu_label} ｜ 主模型：ACE-Step 1.5 Turbo ｜ 旋律規劃：{ENGINE_STATE['active_lm']}（MIT 免費商用授權）"


# ---------------------------------------------------------------------------
# 100% Copyright-Free (YouTube Monetization Ready) Song Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "jiayibingding_rock": {
        "title": "醉月長安（煙嗓音綜版）",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": False,
        "duration": 60,
        "bpm": 96,
        "style": (
            "Chinese Live TV Singing Show Symphonic Guofeng Power-Rock Anthem, Mandopop Emotional Arena Rock Live Band, "
            "Deep Raspy Smoky Male Vocal, Weathered Gravelly Husky Baritone-Tenor Voice with Heavy Emotional Rasp and Raw Throat Grit, "
            "Grand Concert Piano Chords and Mournful Chinese Erhu and Guzheng Intro, "
            "Intimate Husky Rhythmic Mandarin Melodic Rap Storytelling in Verse with Warm Electric Bass and Crisp Live Snare Groove, "
            "Dramatic Floor-Tom Drum Build-Up and Swelling Symphonic String Orchestra in Pre-Chorus, "
            "Explosive Soaring Raspy High-Note Rock Belt and Rhythmic Rap-Rock Chorus with Massive Distorted Electric Guitar Power Chords, "
            "Crashing Live Acoustic Rock Drums, Heroic Chinese Suona and Erhu Counter-Melody, Live Concert Hall Stage Ambience, 96 BPM, C# Minor"
        ),
        "lyrics": """[Intro - Piano & Weeping Erhu]
這一壺烈酒 敬長安的風

[Verse 1 - Deep Raspy Smoky Melodic Rap]
金鑾殿的燭火照不亮我半生孤單
脫下官袍把功名利祿摔成兩半
世人笑我太瘋癲 把我當作路人看
只有手裡這壺濁酒懂我心裡的不甘
提筆寫盡人間事 墨跡還沒乾
今夜就讓搖滾陪我把長安城掀翻

[Pre-Chorus - Rising Gritty Belt & Drum Build]
誰甘心一輩子只做別人的配角！
戰鼓給我敲響！全場把手舉高！

[Chorus - Explosive Raspy High-Note Rock Belt & Rap]
人生得意須盡歡！莫使金樽空對月！
就算千金散盡老子也絕不妥協！
天生我材必有用！管他風雪有多烈！
用這沙啞的狂歌把黑夜全撕裂！

[Outro - Emotional Raspy High Note & Piano Erhu Finale]
仰天大笑出門去！我命由我不由天！""",
    },
    "kuangge_raprock": {
        "title": "狂歌炸長安",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": False,
        "duration": 60,
        "bpm": 154,
        "style": (
            "Explosive Chinese Classical Guofeng Nu-Metal Rap-Rock, Heavy Stadium Rock Concert Anthem, "
            "Crushing Drop-D Distorted Electric Guitar Riffs, Thunderous Chinese Tang War Drums and Fast Double-Kick Rock Drums, "
            "Piercing High-Energy Chinese Suona Horn Lead and Electric Pipa Riff, "
            "Aggressive Rapid-Fire Male Mandarin Rap Flow in Verses, Explosive High-Octane Gritty Rock Shout and Rap-Chant Chorus, "
            "Crowd Hype Gang Vocals Hey Ha, Linkin Park meets Chinese Classical Folk Metal Hip-Hop, "
            "154 BPM, High Adrenaline Live TV Music Show Arena Energy, Massive Punchy Sub-Bass Broadcast Mix"
        ),
        "lyrics": """[Intro]
嘿！哈！嗩吶戰鼓給我炸起來！

[Verse 1 - Rap]
金鑾殿的規矩老子通通撕碎
脫掉官靴甩開折子今夜不睡
戰鼓擂動長安街頭烈火在燒
左手酒壺右手麥克風把屋頂掀翻掉
看那權貴低頭哈腰活得太累
不如跟我狂飆搖滾痛飲千杯

[Pre-Chorus - Rap Chant]
雙手舉高！跟著戰鼓跳！
一二三四！把全場都炸掉！

[Chorus - Explosive Rock & Rap]
人生得意須盡歡！莫使金樽空對月！
天生我材必有用！千金散盡還復來！
嗩吶一響震破九重天外天！
狂歌說唱老子就是大唐詩仙！

[Bridge - Fast Rap & Suona Solo]
十步寫一詩 千里不留行
電吉他配嗩吶 炸醒這座城

[Outro - Arena Finale Shout]
仰天大笑出門去！全場一起吼！
我輩豈是蓬蒿人！乾了這碗酒！""",
    },
    "libai_rock": {
        "title": "詩仙醉長安",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": False,
        "duration": 90,
        "bpm": 128,
        "style": (
            "Mandopop Funk Pop-Rock, Live TV Arena Band Performance, Groovy Slap Electric Bass, "
            "Crisp Funky Stratocaster Guitar Chops, Tight Live Acoustic Drums, Charismatic Warm Slightly Raspy "
            "Male Vocals in Clear Mandarin Chinese, Subtle Chinese Bamboo Flute Riff in Solo, 128 BPM, Broadcast Mix"
        ),
        "lyrics": """[Intro]

[Verse 1]
長安城的公文堆滿了桌案
大人們教我低頭學會死板
朝九晚五把眉頭鎖成一座山
不如一壺濁酒把煩惱沖淡

[Pre-Chorus]
提筆潑墨寫下狂草三千行
笑看世人匆忙追逐著名利場
玉壺買春醉倒在朱雀橋旁
誰規定人生只能一個模樣

[Chorus]
人生得意須盡歡 莫使金樽空對月
天生我材必有用 千金散盡還復來
舉杯邀明月 今夜不朝天子殿
十步留詩篇 我自逍遙做詩仙

[Guitar Solo]

[Chorus]
人生得意須盡歡 莫使金樽空對月
天生我材必有用 千金散盡還復來
舉杯邀明月 今夜不朝天子殿
十步留詩篇 我自逍遙做詩仙

[Outro]
仰天大笑出門去
我輩豈是蓬蒿人""",
    },
    "xiake_rock": {
        "title": "俠客行天下",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": False,
        "duration": 90,
        "bpm": 136,
        "style": (
            "Chinese Guofeng Arena Hard Rock, Driving Overdriven Electric Guitars, Thunderous War Drums, "
            "Soaring Dizi Bamboo Flute and Electric Guzheng, Powerful Heroic Male Rock Vocals in Mandarin Chinese, 136 BPM"
        ),
        "lyrics": """[Intro]

[Verse 1]
趙客縵胡纓 吳鉤霜雪明
銀鞍照白馬 颯沓如流星
閒過信陵飲 脫劍膝前橫

[Chorus]
十步殺一人 千里不留行
事了拂衣去 深藏身與名
三杯吐然諾 五嶽倒為輕
縱死俠骨香 不慚世上英

[Guitar Solo]

[Chorus]
十步殺一人 千里不留行
事了拂衣去 深藏身與名""",
    },
    "city_ballad": {
        "title": "末班車的星光",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": False,
        "duration": 90,
        "bpm": 76,
        "style": (
            "Mandopop Emotional R&B Ballad, Warm Grand Piano, Lush String Orchestra, Gentle Acoustic Guitar, "
            "Expressive Warm Male Vocals in Clear Mandarin Chinese, Late Night Nostalgic Atmosphere, 76 BPM"
        ),
        "lyrics": """[Intro]

[Verse 1]
雨滴敲打著車窗倒映霓虹
整座城市在午夜慢慢入夢
耳機裡還播著未完的前奏
我們在人海各自學會成熟

[Chorus]
末班車載著星光穿過夜空
把沒說出口的溫柔留給晚風
就算明天還有未知的路口
至少此刻心跳還一樣炙熱自由

[Outro]
把遺憾留給晚風
迎著光往前走""",
    },
    "neon_pop": {
        "title": "霓虹破曉",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": False,
        "duration": 90,
        "bpm": 124,
        "style": (
            "Mandopop Synthwave Pop-Rock, Punchy Retro Analog Synths, Driving Four-on-the-Floor Drums, "
            "Bright Electric Guitar Arpeggios, Clear Energetic Female Vocals in Mandarin Chinese, Uplifting Anthem, 124 BPM"
        ),
        "lyrics": """[Intro]

[Verse 1]
穿過午夜閃爍的高架橋
風把昨日的疲憊全都吹跑
地平線染上第一道破曉
心跳跟著節拍開始燃燒

[Chorus]
奔向霓虹破曉的遠方
沒人能阻擋眼裡的光芒
張開雙手擁抱這片海洋
我們就是自己最亮的太陽""",
    },
    "lofi_bgm": {
        "title": "雨夜書房",
        "lang": "🇹🇼 華語 / 繁體中文 (zh)",
        "instrumental": True,
        "duration": 120,
        "bpm": 82,
        "style": (
            "Chill Lo-Fi Hip Hop Study Beats, Warm Rhodes Electric Piano, Mellow Jazz Guitar Chords, "
            "Soft Vinyl Crackle, Relaxing Boom-Bap Drums, Cozy Rainy Night Atmosphere, Instrumental, 82 BPM"
        ),
        "lyrics": "[inst]",
    },
}

LANGUAGES = {
    "🇹🇼 華語 / 繁體中文 (zh)": "zh",
    "🇺🇸 英文 English (en)": "en",
    "🇯🇵 日文 Japanese (ja)": "ja",
    "🇰🇷 韓文 Korean (ko)": "ko",
    "🇪🇸 西班牙文 Spanish (es)": "es",
    "🇫🇷 法文 French (fr)": "fr",
}


def load_preset(preset_key: str):
    p = PRESETS[preset_key]
    return p["title"], p["style"], p["lyrics"], p["lang"], p["instrumental"], p["duration"], p["bpm"]


# ---------------------------------------------------------------------------
# Song Generation & MV Synthesis Handlers
# ---------------------------------------------------------------------------
def generate_song_action(
    song_title: str,
    style_prompt: str,
    lyrics_text: str,
    lang_label: str,
    is_instrumental: bool,
    duration_sec: int,
    batch_count: int,
    bpm_val: int,
    lm_choice: str,
    steps: int,
    guidance: float,
    seed_val: int,
    p=gr.Progress(),
):
    if not style_prompt or not style_prompt.strip():
        raise gr.Error("請先輸入曲風描述（或點擊上方預設範本按鈕）！")

    p(0.02, desc="檢查並喚醒 GPU 音樂引擎...")
    ensure_engine_ready(lm_choice)

    start_time = time.time()
    final_lyrics = "[inst]" if is_instrumental else (lyrics_text.strip() if lyrics_text and lyrics_text.strip() else "[inst]")
    lang_code = LANGUAGES.get(lang_label, "zh")
    use_random = int(seed_val) == -1
    seed_int = -1 if use_random else int(seed_val)
    use_thinking = ENGINE_STATE["lm_ready"]

    params = GenerationParams(
        caption=style_prompt.strip(),
        lyrics=final_lyrics,
        thinking=use_thinking,
        use_cot_metas=use_thinking,
        use_cot_caption=use_thinking,
        use_cot_language=use_thinking,
        duration=float(duration_sec),
        vocal_language=lang_code,
        inference_steps=int(steps),
        guidance_scale=float(guidance),
        shift=3.0,
        seed=seed_int,
        bpm=int(bpm_val) if bpm_val and int(bpm_val) > 0 else None,
        keyscale="",
        timesignature="4",
        task_type="text2music",
    )
    config = GenerationConfig(batch_size=int(batch_count), use_random_seed=use_random, audio_format="wav")

    if hasattr(llm, "llm") and llm.llm is not None:
        try:
            llm.llm.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    p(0.08, desc="正在透過 5Hz 旋律模型與 DiT 合成人聲與編曲...")
    res = generate_music(
        dit_handler=dit,
        llm_handler=llm if ENGINE_STATE["lm_ready"] else None,
        params=params,
        config=config,
        save_dir=OUTPUT_DIR,
        progress=p,
    )

    if hasattr(llm, "llm") and llm.llm is not None:
        try:
            llm.llm.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not res.success:
        raise gr.Error(f"音樂生成失敗：{res.error}")

    paths = [a.get("path") for a in res.audios if isinstance(a, dict) and a.get("path")]
    if not paths or not all(os.path.exists(px) for px in paths):
        wav_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")), key=os.path.getmtime, reverse=True)
        paths = [w for w in wav_files if (time.time() - os.path.getmtime(w)) < 600][: int(batch_count)]

    if not paths:
        raise gr.Error("找不到輸出的音檔，請再試一次！")

    p(0.92, desc="正在轉出高音質 320kbps MP3 供免費下載...")
    safe_slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (song_title or "song").strip())[:24] or "song"
    ts_stamp = int(time.time())

    wav_1 = paths[0]
    mp3_1 = str(Path(OUTPUT_DIR) / f"{safe_slug}_v1_{ts_stamp}.mp3")
    convert_wav_to_mp3(Path(wav_1), Path(mp3_1))

    mp3_2 = None
    if len(paths) > 1 and os.path.exists(paths[1]):
        mp3_2 = str(Path(OUTPUT_DIR) / f"{safe_slug}_v2_{ts_stamp}.mp3")
        convert_wav_to_mp3(Path(paths[1]), Path(mp3_2))

    elapsed = time.time() - start_time
    msg = (
        f"🎉 生成成功！耗時 {elapsed:.1f} 秒｜已自動轉出 320kbps MP3 與無損 WAV，可直接點擊下方檔案右側下載，"
        f"或切到「🎬 合成 1080p MV」分頁一鍵產出 YouTube 影片！"
    )
    return mp3_1, mp3_1, wav_1, mp3_2, mp3_2, msg, mp3_1, song_title, lyrics_text


def render_mv_action(
    audio_file: str,
    mv_song_title: str,
    mv_singer_line: str,
    mv_credits_line: str,
    mv_lyrics_text: str,
    p=gr.Progress(),
):
    if not audio_file or not os.path.exists(audio_file):
        raise gr.Error("請先在「🎵 免費 AI 寫歌工作台」生成一首歌，或手動上傳一個 MP3 / WAV 音檔！")

    safe_slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (mv_song_title or "mv").strip())[:24] or "mv"
    out_mp4 = Path(MV_OUTPUT_DIR) / f"{safe_slug}_1080p_MV_{int(time.time())}.mp4"

    def _prog(frac: float, desc: str):
        p(frac, desc=desc)

    render_full_mv(
        audio_path=Path(audio_file),
        output_mp4=out_mp4,
        lyrics_text=mv_lyrics_text,
        song_title=mv_song_title or "詩仙醉長安",
        singer_line=mv_singer_line or "主唱：詩仙·李白（太白居士）",
        credits_line=mv_credits_line or "詞曲：100% 原創搖滾 × 唐詩公有領域",
        progress_cb=_prog,
    )
    size_mb = out_mp4.stat().st_size / (1024 * 1024)
    status = f"✅ 1080p 舞台 MV 合成完畢！（檔案大小：{size_mb:.1f} MB，可直接免費下載上傳 YouTube 開啟盈利）"
    return str(out_mp4), str(out_mp4), status


# ---------------------------------------------------------------------------
# Gradio 6 Traditional Chinese Studio UI
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.studio-banner {
    background: linear-gradient(135deg, #111827 0%, #1e293b 60%, #78350f 100%);
    border: 1px solid #f59e0b;
    border-radius: 16px;
    padding: 22px 26px;
    color: #f8fafc;
    margin-bottom: 14px;
}
.studio-banner h1 {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    color: #fde68a;
}
.studio-banner p {
    margin: 0;
    font-size: 0.95rem;
    color: #e2e8f0;
}
"""

default_p = PRESETS["jiayibingding_rock"]

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(), title="私有版免費中文 Suno 音樂 & 1080p MV 工作室") as demo:
    gr.HTML(
        """
        <div class="studio-banner">
            <h1>🎸 私有版免費中文 Suno 音樂 ＆ 1080p MV 工作室</h1>
            <p>底層引擎：開源 <strong>ACE-Step 1.5 (MIT 商用無版權授權)</strong> ｜ 零月費 · 無限生成 · 免費直接下載 320kbps MP3 / WAV / 1080p MP4</p>
        </div>
        """
    )

    engine_status_box = gr.Textbox(
        value="⏳ 正在初始化引擎...",
        label="⚡ 雲端顯卡引擎狀態",
        interactive=False,
    )

    with gr.Tabs():
        with gr.Tab("🎵 1. 免費 AI 寫歌工作台（類 Suno 介面）"):
            gr.Markdown("### ✨ 一鍵套用「100% 原創零版權（可直接開 YouTube 盈利）」歌曲範本：")
            with gr.Row():
                btn_jybd = gr.Button("🎤 豬八戒甲乙丙丁風・煙嗓搖滾說唱《醉月長安》（推薦！）", variant="primary", size="sm")
                btn_p0 = gr.Button("🔥 國風嗩吶重型說唱《狂歌炸長安》", variant="secondary", size="sm")
                btn_p1 = gr.Button("🎸 大唐搖滾《詩仙醉長安》", variant="secondary", size="sm")
                btn_p2 = gr.Button("⚔️ 國風燃曲《俠客行天下》", variant="secondary", size="sm")
                btn_p3 = gr.Button("🌃 華語抒情《末班車的星光》", variant="secondary", size="sm")
                btn_p4 = gr.Button("⚡ 電音搖滾《霓虹破曉》", variant="secondary", size="sm")
                btn_p5 = gr.Button("☕ 掛機純音樂《雨夜書房 Lo-Fi》", variant="secondary", size="sm")

            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Row():
                        song_title_in = gr.Textbox(value=default_p["title"], label="🎵 歌曲名稱", scale=2)
                        lang_in = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value=default_p["lang"],
                            label="🌍 演唱語言",
                            scale=2,
                        )
                        inst_in = gr.Checkbox(value=default_p["instrumental"], label="🎹 純音樂伴奏（無人聲）", scale=1)

                    style_in = gr.Textbox(
                        value=default_p["style"],
                        label="🎨 曲風與編曲提示詞（Style Prompt，建議英文描述樂器與唱腔最精準）",
                        lines=3,
                    )
                    lyrics_in = gr.Textbox(
                        value=default_p["lyrics"],
                        label="📝 歌詞內容（支援 [Verse]、[Chorus]、[Guitar Solo] 分段標籤；原創詞 100% 可商用盈利）",
                        lines=12,
                    )

                    with gr.Row():
                        dur_in = gr.Slider(30, 240, value=default_p["duration"], step=10, label="⏱️ 歌曲長度（秒）")
                        batch_in = gr.Slider(1, 2, value=2, step=1, label="🎲 一次抽幾首歌（像 Suno 一次出 2 首挑好聽的）")
                        bpm_in = gr.Number(value=default_p["bpm"], label="🥁 節拍 BPM（0=自動）", precision=0)

                    with gr.Accordion("⚙️ 進階調校設定（預設已針對 T4 最佳化，通常不必改）", open=False):
                        with gr.Row():
                            lm_in = gr.Dropdown(
                                choices=["0.6B (T4 極速省記憶體推薦)", "1.7B (進階旋律規劃)", "None (僅用 DiT 直出)"],
                                value="0.6B (T4 極速省記憶體推薦)",
                                label="🧠 5Hz 旋律規劃模型",
                            )
                            steps_in = gr.Slider(8, 32, value=8, step=1, label="✨ 擴散步數（Turbo 預設 8 步最快）")
                            guide_in = gr.Slider(1.0, 12.0, value=7.0, step=0.5, label="🎯 提示詞貼合度 Guidance")
                            seed_in = gr.Number(value=-1, label="🍀 隨機種子（-1 為每次隨機）", precision=0)

                    gen_btn = gr.Button("🚀 開始免費生成歌曲（自動轉出 MP3 + WAV）", variant="primary", size="lg")

                with gr.Column(scale=5):
                    gen_status = gr.Textbox(
                        value="💡 點擊左側「開始免費生成歌曲」，約 20～45 秒即可免費試聽與下載 MP3！",
                        label="📣 生成進度與說明",
                        interactive=False,
                    )
                    gr.Markdown("#### 🎧 版本 A（Track 1）— 試聽與免費下載")
                    audio_out_1 = gr.Audio(label="版本 A 線上試聽", type="filepath")
                    with gr.Row():
                        mp3_file_1 = gr.File(label="⬇️ 免費下載 MP3 (320kbps)")
                        wav_file_1 = gr.File(label="⬇️ 免費下載無損 WAV")

                    gr.Markdown("#### 🎧 版本 B（Track 2）— 試聽與免費下載")
                    audio_out_2 = gr.Audio(label="版本 B 線上試聽", type="filepath")
                    mp3_file_2 = gr.File(label="⬇️ 免費下載版本 B MP3 (320kbps)")

        with gr.Tab("🎬 2. 一鍵合成 1080p《樂隊請就位》流光 KTV 字幕 MV"):
            gr.Markdown(
                "### 🎬 將剛才生成的無版權原創歌曲，自動套上 12 幕大唐搖滾舞台分鏡 ＋ 黃金逐字流光 KTV 字幕！"
            )
            with gr.Row():
                with gr.Column(scale=5):
                    mv_audio_in = gr.Audio(
                        label="🎵 步驟 1：音樂音檔（在寫歌工作台生成後會自動帶入，也可自行上傳 MP3/WAV）",
                        type="filepath",
                    )
                    mv_title_in = gr.Textbox(value=default_p["title"], label="🏷️ MV 舞台顯示歌名")
                    mv_singer_in = gr.Textbox(value="主唱：詩仙·李白（太白居士）", label="🎤 歌手資訊欄")
                    mv_credits_in = gr.Textbox(
                        value="詞曲：100% 原創搖滾 × 唐詩公有領域",
                        label="📜 版權／創作標示欄",
                    )
                    mv_lyrics_in = gr.Textbox(
                        value=default_p["lyrics"],
                        label="📝 KTV 黃金流光字幕歌詞（自動依歌曲長度計算逐字變色時間軸）",
                        lines=10,
                    )
                    mv_btn = gr.Button("🎬 立即合成 1920x1080 舞台 MV（免費下載 MP4）", variant="primary", size="lg")

                with gr.Column(scale=6):
                    mv_status = gr.Textbox(
                        value="💡 合成約需 35～60 秒，完成後可直接於下方預覽並免費下載 MP4 上傳 YouTube！",
                        label="📣 MV 渲染狀態",
                        interactive=False,
                    )
                    mv_video_out = gr.Video(label="📺 1080p 成品 MV 預覽")
                    mv_file_out = gr.File(label="⬇️ 免費下載 1080p MP4 影片檔")

    btn_jybd.click(lambda: load_preset("jiayibingding_rock"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])
    btn_p0.click(lambda: load_preset("kuangge_raprock"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])
    btn_p1.click(lambda: load_preset("libai_rock"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])
    btn_p2.click(lambda: load_preset("xiake_rock"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])
    btn_p3.click(lambda: load_preset("city_ballad"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])
    btn_p4.click(lambda: load_preset("neon_pop"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])
    btn_p5.click(lambda: load_preset("lofi_bgm"), outputs=[song_title_in, style_in, lyrics_in, lang_in, inst_in, dur_in, bpm_in])

    gen_btn.click(
        generate_song_action,
        inputs=[
            song_title_in, style_in, lyrics_in, lang_in, inst_in,
            dur_in, batch_in, bpm_in, lm_in, steps_in, guide_in, seed_in,
        ],
        outputs=[
            audio_out_1, mp3_file_1, wav_file_1,
            audio_out_2, mp3_file_2, gen_status,
            mv_audio_in, mv_title_in, mv_lyrics_in,
        ],
    )

    mv_btn.click(
        render_mv_action,
        inputs=[mv_audio_in, mv_title_in, mv_singer_in, mv_credits_in, mv_lyrics_in],
        outputs=[mv_video_out, mv_file_out, mv_status],
    )

    demo.load(lambda: ensure_engine_ready("0.6B"), outputs=[engine_status_box])


def start_cloudflare_tunnel(port: int = 7860):
    cf_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
    if not os.path.exists(cf_bin):
        return None, None
    log_file = "/tmp/cloudflared_suno.log"
    try:
        if os.path.exists(log_file):
            os.remove(log_file)
        log_f = open(log_file, "w")
        proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return None, None

    for _ in range(30):
        time.sleep(0.5)
        if os.path.exists(log_file):
            content = Path(log_file).read_text(encoding="utf-8", errors="ignore")
            matches = re.findall(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com", content)
            if matches:
                return proc, matches[0]
    return proc, None


def main() -> None:
    port = int(os.environ.get("STUDIO_PORT", "7860"))
    print("=" * 68, flush=True)
    print("🎸 正在預先載入 ACE-Step 1.5 模型至免費 T4 顯卡（完成後打開網頁即可直接生歌）...", flush=True)
    print("=" * 68, flush=True)
    try:
        ready_msg = ensure_engine_ready("0.6B")
        print(ready_msg, flush=True)
    except Exception as exc:
        print(f"⚠️ 預載入提示（將於首次點擊時重試）: {exc}", flush=True)

    cf_proc, cf_url = start_cloudflare_tunnel(port=port)

    demo.queue(max_size=20, default_concurrency_limit=1)
    launch_res = demo.launch(
        share=True,
        inline=False,
        debug=False,
        show_error=True,
        server_port=port,
        prevent_thread_lock=True,
        allowed_paths=[OUTPUT_DIR, MV_OUTPUT_DIR, ACESTEP_ROOT, str(STUDIO_DIR), "/tmp"],
    )

    share_url = getattr(demo, "share_url", None)
    if not share_url and isinstance(launch_res, tuple) and len(launch_res) >= 3:
        share_url = launch_res[2]

    print("\n" + "★" * 68, flush=True)
    print("🎉 你的專屬「免費中文版 Suno ＆ 1080p MV 工作室」已經啟動成功！", flush=True)
    print("👇 請直接點擊下方任一藍色網址開啟全螢幕工作室網頁：", flush=True)
    print("─" * 68, flush=True)
    if cf_url:
        print(f"🚀 高速主網址 (Cloudflare 推薦):  {cf_url}", flush=True)
    if share_url:
        print(f"🔗 備用公開網址 (Gradio Live):    {share_url}", flush=True)
    print("★" * 68 + "\n", flush=True)
    print("💡 提醒：請保持這個 Google Colab 分頁開著不要關掉，就能在上面的網址無限免費生歌與下載 MP3 / MP4！", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if cf_proc:
            cf_proc.terminate()


if __name__ == "__main__":
    main()
