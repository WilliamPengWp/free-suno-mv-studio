#!/usr/bin/env python3
"""
1920x1080 Broadcast MV & Golden Karaoke (.ASS) Renderer for Free Suno MV Studio.
Turns any generated WAV/MP3 + custom lyrics into a monetizable 1080p YouTube MV.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

STUDIO_DIR = Path(__file__).resolve().parent
DEFAULT_FRAMES_DIR = STUDIO_DIR / "assets" / "frames"


@dataclass
class ShotPlan:
    shot_idx: int
    image_file: str
    duration: float
    motion: str
    section_tag: str


@dataclass
class LyricLine:
    start: float
    end: float
    text: str
    sub_label: str = ""


FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Bold.otf",
]
FONT_CANDIDATES_REG = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
]
FONT_CANDIDATES_SERIF = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]


def load_cjk_font(candidates: List[str], size: int, prefer_tc: bool = True) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if not Path(path).exists():
            continue
        for idx in range(8):
            try:
                font = ImageFont.truetype(path, size, index=idx)
                family = font.getname()[0]
                if prefer_tc and "TC" in family:
                    return font
            except Exception:
                break
        try:
            return ImageFont.truetype(path, size, index=0)
        except Exception:
            continue

    # Fallback via fc-match
    try:
        fc_path = subprocess.check_output(
            ["fc-match", "-f", "%{file}", "sans-serif:lang=zh-tw"],
            text=True,
        ).strip()
        if fc_path and Path(fc_path).exists():
            return ImageFont.truetype(fc_path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def detect_ass_font_family() -> str:
    """Finds the exact installed CJK font family name for libass."""
    try:
        fam = subprocess.check_output(
            ["fc-match", "-f", "%{family}", "Noto Sans CJK TC"],
            text=True,
        ).strip()
        if fam:
            return fam.split(",")[0].strip()
    except Exception:
        pass
    return "Noto Sans CJK TC"


def strip_non_bmp_emojis(text: str) -> str:
    """Removes astral-plane color emojis that render as tofu boxes in libass."""
    return "".join(ch for ch in text if ord(ch) <= 0xFFFF).strip()


def probe_audio_duration(audio_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ],
        text=True,
    ).strip()
    return max(8.0, float(out))


def convert_wav_to_mp3(wav_path: Path, mp3_path: Optional[Path] = None) -> Path:
    if mp3_path is None:
        mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-b:a", "320k",
            str(mp3_path),
        ],
        check=True,
    )
    return mp3_path


def build_tv_hud_overlays(
    out_dir: Path,
    song_title: str = "《詩仙醉長安》",
    credits_line: str = "詞曲：100% 原創搖滾 × 唐詩公有領域",
    singer_line: str = "主唱：詩仙·李白（太白居士）",
    band_line: str = "大唐搖滾樂團：杜甫（電貝斯）× 高適（戰鼓）｜觀禮：唐玄宗、楊貴妃",
    show_title: str = "樂隊請就位",
    show_subtitle: str = "｜原創音樂舞台",
    right_badge: str = "原創純享版｜MIT 商用無版權",
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = 1920, 1080

    song_title = strip_non_bmp_emojis(song_title) or "《原創單曲》"
    if not song_title.startswith("《"):
        song_title = f"《{song_title}》"

    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(scrim)
    for y in range(H - 250, H):
        prog = (y - (H - 250)) / 250.0
        alpha = int(172 * (prog ** 1.65))
        s_draw.line([(0, y), (W, y)], fill=(6, 8, 14, alpha))
    base = Image.alpha_composite(base, scrim)
    draw = ImageDraw.Draw(base)

    f_logo = load_cjk_font(FONT_CANDIDATES_SERIF, 30)
    f_sub = load_cjk_font(FONT_CANDIDATES_BOLD, 18)
    clean_title = strip_non_bmp_emojis(show_title)
    clean_sub = strip_non_bmp_emojis(show_subtitle)
    sub_bbox = draw.textbbox((252, 54), clean_sub, font=f_sub)
    tl_x1 = max(456, sub_bbox[2] + 24)
    draw.rounded_rectangle((48, 38, tl_x1, 106), radius=14, fill=(14, 16, 24, 196), outline=(245, 198, 84, 220), width=2)
    draw.rounded_rectangle((62, 52, 70, 92), radius=3, fill=(255, 206, 78, 255))
    draw.text((84, 46), clean_title, font=f_logo, fill=(255, 226, 138, 255))
    draw.text((252, 54), clean_sub, font=f_sub, fill=(232, 236, 245, 235))

    draw.rounded_rectangle((48, 114, 266, 144), radius=8, fill=(20, 22, 30, 175))
    draw.ellipse((60, 123, 72, 135), fill=(255, 68, 68, 255))
    draw.text((80, 117), "LIVE STAGE · 1080P", font=load_cjk_font(FONT_CANDIDATES_BOLD, 16), fill=(235, 235, 240, 230))

    f_rt = load_cjk_font(FONT_CANDIDATES_BOLD, 19)
    clean_badge = strip_non_bmp_emojis(right_badge)
    bbox = draw.textbbox((0, 0), clean_badge, font=f_rt)
    rw = bbox[2] - bbox[0] + 40
    rx0 = W - 48 - rw
    draw.rounded_rectangle((rx0, 42, W - 48, 86), radius=12, fill=(14, 16, 24, 185), outline=(255, 255, 255, 75), width=1)
    draw.text((rx0 + 20, 51), clean_badge, font=f_rt, fill=(248, 214, 120, 245))

    persistent_png = out_dir / "hud_persistent.png"
    base.save(persistent_png)

    # Opening Singer & Credits Card (auto-sized to fit all lines cleanly)
    card_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    title_font_size = 46 if len(song_title) <= 8 else 36
    f_song = load_cjk_font(FONT_CANDIDATES_SERIF, title_font_size)
    f_tag = load_cjk_font(FONT_CANDIDATES_BOLD, 20)
    f_meta = load_cjk_font(FONT_CANDIDATES_REG, 20)
    f_singer = load_cjk_font(FONT_CANDIDATES_BOLD, 23)
    f_band = load_cjk_font(FONT_CANDIDATES_REG, 18)

    clean_credits = strip_non_bmp_emojis(credits_line)
    clean_singer = strip_non_bmp_emojis(singer_line)
    clean_band = strip_non_bmp_emojis(band_line)

    tmp_draw = ImageDraw.Draw(card_img)
    cx0, cy0, cy1 = 64, 595, 835
    title_bbox = tmp_draw.textbbox((0, 0), song_title, font=f_song)
    meta_x = cx0 + 64 + (title_bbox[2] - title_bbox[0])
    cred_w = tmp_draw.textbbox((0, 0), clean_credits, font=f_meta)[2]
    sing_w = tmp_draw.textbbox((0, 0), clean_singer, font=f_singer)[2]
    band_w = tmp_draw.textbbox((0, 0), clean_band, font=f_band)[2]
    cx1 = min(W - 64, max(840, meta_x + cred_w + 34, cx0 + 48 + sing_w + 34, cx0 + 48 + band_w + 34))

    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh_draw = ImageDraw.Draw(shadow)
    sh_draw.rounded_rectangle((cx0 + 6, cy0 + 10, cx1 + 6, cy1 + 10), radius=18, fill=(0, 0, 0, 145))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    card_img = Image.alpha_composite(card_img, shadow)
    c_draw = ImageDraw.Draw(card_img)

    c_draw.rounded_rectangle((cx0, cy0, cx1, cy1), radius=18, fill=(12, 15, 24, 232), outline=(245, 198, 84, 225), width=2)
    c_draw.rounded_rectangle((cx0 + 22, cy0 + 24, cx0 + 30, cy1 - 24), radius=4, fill=(255, 204, 72, 255))

    c_draw.rounded_rectangle((cx0 + 48, cy0 + 22, cx0 + 246, cy0 + 54), radius=6, fill=(54, 46, 26, 242), outline=(255, 204, 72, 175), width=1)
    c_draw.text((cx0 + 60, cy0 + 25), "原創競演單曲 #01", font=f_tag, fill=(255, 214, 102, 255))
    c_draw.text((cx0 + 44, cy0 + 66), song_title, font=f_song, fill=(255, 250, 235, 255))
    c_draw.text((meta_x, cy0 + 82), clean_credits, font=f_meta, fill=(210, 218, 232, 245))

    c_draw.line([(cx0 + 48, cy0 + 136), (cx1 - 30, cy0 + 136)], fill=(255, 204, 72, 80), width=1)
    c_draw.text((cx0 + 48, cy0 + 148), clean_singer, font=f_singer, fill=(255, 224, 130, 255))
    c_draw.text((cx0 + 48, cy0 + 184), clean_band, font=f_band, fill=(215, 222, 235, 240))

    opening_png = out_dir / "hud_opening_card.png"
    card_img.save(opening_png)
    return persistent_png, opening_png


def sec_to_ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def make_kf_tags(text: str, total_dur: float) -> str:
    chars = list(text)
    if not chars:
        return ""
    total_cs = max(10, int(round(total_dur * 100)))
    weights = [0.35 if ch == " " else 1.0 for ch in chars]
    w_sum = sum(weights)
    cs_list = [max(2, int(round(total_cs * w / w_sum))) for w in weights]
    diff = total_cs - sum(cs_list)
    cs_list[-1] = max(2, cs_list[-1] + diff)
    return "".join(f"{{\\kf{cs}}}{ch}" for cs, ch in zip(cs_list, chars))


def write_karaoke_ass(ass_path: Path, lyrics: List[LyricLine]) -> None:
    font_family = detect_ass_font_family()
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KTVMain,{font_family},56,&H0045D4FF,&H00FFFFFF,&H0014100C,&H88000000,-1,0,0,0,100,100,2.2,0,1,4.0,2.2,2,80,80,66,1
Style: StageCue,{font_family},26,&H00A8E2FF,&H00A8E2FF,&H0014100C,&H70000000,-1,0,0,0,100,100,1.5,0,1,2.5,1.2,2,80,80,152,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for line in lyrics:
        clean_txt = strip_non_bmp_emojis(line.text)
        if not clean_txt:
            continue
        st = sec_to_ass_time(line.start)
        et = sec_to_ass_time(line.end)
        dur = max(0.3, line.end - line.start - 0.15)
        kf_body = make_kf_tags(clean_txt, dur)
        events.append(f"Dialogue: 2,{st},{et},KTVMain,,0,0,0,,{{\\fad(140,160)}}{kf_body}")
        clean_sub = strip_non_bmp_emojis(line.sub_label)
        if clean_sub:
            events.append(f"Dialogue: 1,{st},{et},StageCue,,0,0,0,,{{\\fad(180,180)}}{clean_sub}")

    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


SECTION_CUE_MAP = {
    "intro": "【前奏 Intro】",
    "verse": "【主歌 Verse】",
    "pre-chorus": "【預副歌 Pre-Chorus】",
    "prechorus": "【預副歌 Pre-Chorus】",
    "chorus": "【副歌爆發 Chorus】",
    "bridge": "【橋段 Bridge】",
    "solo": "【樂團獨奏 SOLO】",
    "interlude": "【間奏 Interlude】",
    "outro": "【尾奏謝幕 Outro】",
}


def build_schedule_from_lyrics(
    total_dur: float,
    lyrics_text: str,
    song_title: str = "詩仙醉長安",
) -> Tuple[List[ShotPlan], List[LyricLine], float]:
    xfade = 0.40 if total_dur >= 40.0 else 0.25
    weights = [0.07, 0.09, 0.08, 0.07, 0.09, 0.08, 0.08, 0.09, 0.10, 0.09, 0.09, 0.07]
    eff_total = total_dur + (len(weights) - 1) * xfade
    motions = [
        "zoom_in", "pan_right", "zoom_in_fast", "pan_left",
        "zoom_out", "zoom_in", "pan_right", "zoom_in_fast",
        "zoom_out", "zoom_in_fast", "zoom_out", "pull_back_slow",
    ]
    shots = [
        ShotPlan(i + 1, f"shot_{i + 1:02d}.jpg", max(1.8, round(eff_total * w, 2)), motions[i], f"Scene {i + 1}")
        for i, w in enumerate(weights)
    ]

    parsed_entries: List[Tuple[str, str]] = []
    pending_cue = ""
    raw_stripped = (lyrics_text or "").strip()

    if not raw_stripped or raw_stripped.lower() == "[inst]":
        parsed_entries = [
            ("♪ 原創純音樂舞台演奏 · LIVE INSTRUMENTAL ♪", "【全場純享伴奏】"),
            ("♪ 大唐搖滾樂團 · 電貝斯 × 戰鼓 × 竹笛電吉他 ♪", "【100% 無版權原創音樂】"),
        ]
    else:
        for raw_line in raw_stripped.splitlines():
            s = strip_non_bmp_emojis(raw_line.strip())
            if not s:
                continue
            m = re.match(r"^\[(.*?)\]$", s)
            if m:
                tag_lower = m.group(1).strip().lower()
                matched_cue = ""
                for k, label in SECTION_CUE_MAP.items():
                    if k in tag_lower:
                        matched_cue = label
                        break
                if "solo" in tag_lower or "interlude" in tag_lower:
                    parsed_entries.append(("♪ 舞台電吉他 × 竹笛熱血獨奏 SOLO ♪", matched_cue or "【樂團間奏】"))
                    pending_cue = ""
                elif matched_cue and "intro" not in tag_lower:
                    pending_cue = matched_cue
                continue
            parsed_entries.append((s, pending_cue))
            pending_cue = ""

    clean_title = strip_non_bmp_emojis(song_title).strip("《》") or "原創單曲"
    lyrics: List[LyricLine] = []
    intro_end = min(4.2, total_dur * 0.065)
    lyrics.append(LyricLine(0.5, max(1.5, intro_end), f"♪ 樂隊請就位 ·《{clean_title}》登台開麥 ♪", "【原創搖滾舞台 LIVE】"))

    if parsed_entries:
        body_start = intro_end + 0.25
        body_end = max(body_start + 2.0, total_dur * 0.965)
        span = body_end - body_start
        char_weights = [max(4.0, float(len(txt))) for txt, _ in parsed_entries]
        w_total = sum(char_weights)
        cur = body_start
        for (txt, cue), cw in zip(parsed_entries, char_weights):
            seg_len = span * (cw / w_total)
            line_end = min(body_end, cur + max(1.2, seg_len - 0.08))
            lyrics.append(LyricLine(round(cur, 2), round(line_end, 2), txt, cue))
            cur += seg_len

    return shots, lyrics, xfade


def render_motion_clip(shot: ShotPlan, frames_dir: Path, out_mp4: Path, fps: int = 30) -> None:
    img_path = frames_dir / shot.image_file
    frames = max(15, int(round(shot.duration * fps)))

    if shot.motion == "zoom_in":
        z_expr = f"1.0+0.12*(on/{frames})"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif shot.motion == "zoom_in_fast":
        z_expr = f"1.02+0.18*(on/{frames})"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih*0.46-(ih/zoom/2)"
    elif shot.motion == "zoom_out":
        z_expr = f"1.16-0.14*(on/{frames})"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif shot.motion == "pan_right":
        z_expr = "1.12"
        x_expr, y_expr = f"(iw-iw/zoom)*(on/{frames})", "ih/2-(ih/zoom/2)"
    elif shot.motion == "pan_left":
        z_expr = "1.12"
        x_expr, y_expr = f"(iw-iw/zoom)*(1.0-on/{frames})", "ih/2-(ih/zoom/2)"
    else:
        z_expr = f"1.14-0.12*(on/{frames})"
        x_expr, y_expr = "iw/2-(iw/zoom/2)", "ih*0.48-(ih/zoom/2)"

    vf = (
        f"scale=3840:2160:flags=lanczos,"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s=1920x1080:fps={fps},"
        f"eq=contrast=1.04:saturation=1.08,"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-loop", "1", "-t", f"{shot.duration + 0.2:.2f}", "-i", str(img_path),
        "-vf", vf,
        "-t", f"{shot.duration:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-r", str(fps),
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def render_full_mv(
    audio_path: Path,
    output_mp4: Path,
    lyrics_text: str,
    song_title: str = "詩仙醉長安",
    singer_line: str = "主唱：詩仙·李白（太白居士）",
    credits_line: str = "詞曲：100% 原創搖滾 × 唐詩公有領域",
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    work_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Path:
    audio_path = Path(audio_path)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    if work_dir is None:
        work_dir = output_mp4.parent / "mv_build_cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb(0.05, "正在分析音樂長度與對齊黃金 KTV 逐字字幕...")

    total_dur = probe_audio_duration(audio_path)
    shots, lyrics, xfade_dur = build_schedule_from_lyrics(total_dur, lyrics_text, song_title=song_title)
    total_vid_dur = sum(s.duration for s in shots) - (len(shots) - 1) * xfade_dur

    dur_tag = hashlib.md5(f"{total_dur:.2f}_{frames_dir}".encode()).hexdigest()[:8]
    clip_paths: List[Path] = []
    for idx, shot in enumerate(shots):
        clip_p = work_dir / f"seg_{shot.shot_idx:02d}_{dur_tag}.mp4"
        if not clip_p.exists() or clip_p.stat().st_size < 10_000:
            if progress_cb:
                progress_cb(
                    0.08 + 0.62 * (idx / len(shots)),
                    f"正在渲染 1080p 電影運鏡分鏡 ({shot.shot_idx}/12)...",
                )
            render_motion_clip(shot, frames_dir, clip_p)
        clip_paths.append(clip_p)

    if progress_cb:
        progress_cb(0.74, "正在繪製《樂隊請就位》舞台角標、歌手卡與流光字幕...")

    persistent_png, opening_png = build_tv_hud_overlays(
        work_dir,
        song_title=song_title,
        credits_line=credits_line,
        singer_line=singer_line,
    )
    ass_path = work_dir / f"lyrics_{dur_tag}.ass"
    write_karaoke_ass(ass_path, lyrics)

    inputs: List[str] = []
    for cp in clip_paths:
        inputs.extend(["-i", str(cp)])
    inputs.extend(["-loop", "1", "-framerate", "30", "-t", f"{total_vid_dur + 1.0:.2f}", "-i", str(persistent_png)])
    inputs.extend(["-loop", "1", "-framerate", "30", "-t", "8.0", "-i", str(opening_png)])
    inputs.extend(["-i", str(audio_path)])

    transitions = [
        "fade", "dissolve", "smoothleft", "fadewhite",
        "dissolve", "fade", "smoothright", "fadewhite",
        "fadewhite", "fadewhite", "fadeblack",
    ]
    fc_parts: List[str] = []
    cum_offset = shots[0].duration - xfade_dur
    prev_label = "[0:v]"
    for i in range(1, len(shots)):
        next_label = f"[xf{i}]"
        tr = transitions[(i - 1) % len(transitions)]
        fc_parts.append(
            f"{prev_label}[{i}:v]xfade=transition={tr}:duration={xfade_dur:.2f}:offset={cum_offset:.2f}{next_label}"
        )
        prev_label = next_label
        if i < len(shots) - 1:
            cum_offset += shots[i].duration - xfade_dur

    hud_idx = len(shots)
    card_idx = len(shots) + 1
    aud_idx = len(shots) + 2

    ass_escaped = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")
    fc_parts.append(
        f"[{card_idx}:v]format=rgba,fade=t=in:st=0.9:d=0.55:alpha=1,fade=t=out:st=6.8:d=0.65:alpha=1[card_faded]"
    )
    fc_parts.append(f"{prev_label}[{hud_idx}:v]overlay=0:0:eof_action=pass[v_hud]")
    fc_parts.append(f"[v_hud][card_faded]overlay=0:0:enable='between(t,0.8,7.6)':eof_action=pass[v_card]")
    fc_parts.append(
        f"[v_card]ass='{ass_escaped}',"
        f"fade=t=in:st=0:d=0.6,fade=t=out:st={max(1.0, total_vid_dur - 1.2):.2f}:d=1.2,"
        f"format=yuv420p[v_out]"
    )

    if progress_cb:
        progress_cb(0.84, "正在進行 12 幕轉場與母帶音軌合成（即將完成）...")

    max_video_kbps = max(1200, min(4200, int((35.0 * 8192) / max(10.0, total_vid_dur)) - 192))
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *inputs,
        "-filter_complex", ";".join(fc_parts),
        "-map", "[v_out]",
        "-map", f"{aud_idx}:a",
        "-t", f"{total_vid_dur:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "21",
        "-maxrate", f"{max_video_kbps}k", "-bufsize", f"{max_video_kbps * 2}k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)

    if progress_cb:
        progress_cb(1.0, "1080p MV 合成完成！")
    return output_mp4
