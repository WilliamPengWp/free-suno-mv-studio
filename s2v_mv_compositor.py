#!/usr/bin/env python3
"""
Zero-Drift 1080p Broadcast S2V Multi-Camera Compositor.

Unlike Ken-Burns xfade slideshows (where crossfade overlaps shift clip timestamps),
audio-driven `Wan2.2-S2V-14B` clips MUST stay millisecond-locked to the exact audio
window `[t_start, t_end]` they were generated from so lip-sync, breathing, and guitar/mic
gestures never drift by even 1 frame.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from mv_renderer import (
    DEFAULT_FRAMES_DIR,
    ShotPlan,
    build_schedule_from_lyrics,
    build_tv_hud_overlays,
    probe_audio_duration,
    render_motion_clip,
    write_karaoke_ass,
)

STUDIO_DIR = Path(__file__).resolve().parent
DEFAULT_HEROES_DIR = STUDIO_DIR / "assets" / "stage_heroes"

HERO_STAGE_PROMPTS = [
    (
        "Live televised Chinese rock concert performance, charismatic male rock vocalist gripping dynamic microphone "
        "singing and rapping passionately with expressive lip-sync, natural head nod, chest breathing, dramatic "
        "golden-red concert spotlights, volumetric stage haze, shallow depth of field, 4k broadcast camera."
    ),
    (
        "Wide live music TV show stage performance, energetic rock band frontman singing with full-body stage presence, "
        "raising hand to hype the cheering audience, live drummer and electric bassist grooving in background, "
        "moving concert light beams, cinematic live broadcast."
    ),
    (
        "Explosive rock chorus climax on live arena stage, passionate lead singer belting powerful high notes into "
        "microphone, veins and emotion, dynamic upper-body movement, golden pyrotechnic sparks erupting behind band, "
        "sweeping arena spotlights, ultra-realistic live concert footage."
    ),
    (
        "High-energy Chinese classical-rock concert finale, rock band and suona/guitar soloist performing intensely "
        "on glowing stage, dramatic camera motion, confetti and warm stage firelights, authentic live TV music show."
    ),
]


@dataclass
class TimelineSegment:
    seg_idx: int
    t_start: float
    t_end: float
    kind: str  # "wan_s2v" or "broll"
    image_path: Path
    prompt: str
    audio_slice_path: Optional[Path] = None
    rendered_clip_path: Optional[Path] = None

    @property
    def duration(self) -> float:
        return round(max(0.4, self.t_end - self.t_start), 3)


def extract_audio_slice_wav(src_audio: Path, dst_wav: Path, t_start: float, duration: float) -> Path:
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{t_start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(src_audio),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(dst_wav),
        ],
        check=True,
    )
    return dst_wav


def plan_s2v_timeline(
    audio_path: Path,
    work_dir: Path,
    video_mode: str = "smart_hybrid_s2v",
    custom_singer_image: Optional[Path] = None,
    heroes_dir: Path = DEFAULT_HEROES_DIR,
    broll_dir: Path = DEFAULT_FRAMES_DIR,
) -> List[TimelineSegment]:
    """
    Builds a gapless, non-overlapping [0.0, total_dur] multi-camera timeline so every
    Wan2.2-S2V-14B clip aligns 1:1 with its exact vocal audio timestamp.
    """
    total_dur = round(probe_audio_duration(audio_path), 2)
    slices_dir = work_dir / "s2v_audio_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    hero_imgs = sorted(heroes_dir.glob("*.jpg"))
    if not hero_imgs:
        hero_imgs = sorted(broll_dir.glob("*.jpg"))[:4]
    if custom_singer_image and Path(custom_singer_image).exists():
        # Put custom uploaded singer photo as primary lead-vocal camera (#0 and #2)
        hero_imgs = [Path(custom_singer_image), hero_imgs[1 % len(hero_imgs)], Path(custom_singer_image), hero_imgs[-1]]

    broll_imgs = sorted(broll_dir.glob("*.jpg"))
    if not broll_imgs:
        broll_imgs = hero_imgs

    segments: List[TimelineSegment] = []

    if video_mode == "full_s2v":
        chunk_len = 5.0
        cur = 0.0
        idx = 0
        while cur < total_dur - 0.35:
            nxt = min(total_dur, round(cur + chunk_len, 2))
            if total_dur - nxt < 1.5:
                nxt = total_dur
            h_img = hero_imgs[idx % len(hero_imgs)]
            prompt = HERO_STAGE_PROMPTS[idx % len(HERO_STAGE_PROMPTS)]
            wav_slice = extract_audio_slice_wav(
                audio_path, slices_dir / f"s2v_slice_{idx:02d}.wav", cur, nxt - cur
            )
            segments.append(
                TimelineSegment(
                    seg_idx=idx,
                    t_start=cur,
                    t_end=nxt,
                    kind="wan_s2v",
                    image_path=h_img,
                    prompt=prompt,
                    audio_slice_path=wav_slice,
                )
            )
            cur = nxt
            idx += 1
        return segments

    # Smart Hybrid S2V Mode (~$0.34/MV): 4 Hero Full-Body Singing Windows + Live Concert Cutaways
    clip_dur = min(4.8, max(3.2, round(total_dur * 0.105, 2)))
    anchors = [0.08, 0.31, 0.55, 0.78]
    s2v_windows = []
    last_end = 0.8
    for i, ratio in enumerate(anchors):
        st = max(last_end + 0.8, round(total_dur * ratio, 2))
        et = min(total_dur - 0.8, round(st + clip_dur, 2))
        if et - st >= 2.2:
            s2v_windows.append((st, et, i))
            last_end = et

    cur = 0.0
    seg_idx = 0
    broll_cursor = 0

    def _append_broll_span(span_start: float, span_end: float) -> None:
        nonlocal seg_idx, broll_cursor
        rem = span_end - span_start
        p = span_start
        while rem > 0.05:
            step = rem if rem <= 5.5 else round(rem / 2.0, 2)
            q = round(min(span_end, p + step), 2)
            if span_end - q < 1.2:
                q = span_end
            b_img = broll_imgs[(broll_cursor * 2 + 1) % len(broll_imgs)]
            segments.append(
                TimelineSegment(
                    seg_idx=seg_idx,
                    t_start=p,
                    t_end=q,
                    kind="broll",
                    image_path=b_img,
                    prompt="concert_broll",
                )
            )
            seg_idx += 1
            broll_cursor += 1
            p = q
            rem = round(span_end - p, 3)

    for st, et, hero_i in s2v_windows:
        if st - cur >= 0.5:
            _append_broll_span(cur, st)
        else:
            st = cur
        h_img = hero_imgs[hero_i % len(hero_imgs)]
        prompt = HERO_STAGE_PROMPTS[hero_i % len(HERO_STAGE_PROMPTS)]
        wav_slice = extract_audio_slice_wav(
            audio_path, slices_dir / f"s2v_hero_{hero_i:02d}.wav", st, et - st
        )
        segments.append(
            TimelineSegment(
                seg_idx=seg_idx,
                t_start=st,
                t_end=et,
                kind="wan_s2v",
                image_path=h_img,
                prompt=prompt,
                audio_slice_path=wav_slice,
            )
        )
        seg_idx += 1
        cur = et

    if total_dur - cur >= 0.4:
        _append_broll_span(cur, total_dur)
    elif segments:
        segments[-1].t_end = total_dur

    return segments


def probe_exact_duration(media_path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(media_path),
        ],
        text=True,
    ).strip()
    return max(0.1, float(out))


def normalize_segment_to_1080p30(
    seg: TimelineSegment,
    norm_out_mp4: Path,
    fps: int = 30,
) -> Path:
    """
    Converts either a raw Wan2.2-S2V-14B clip (e.g., 1024x704 @ 16/24fps) or a B-roll keyframe
    into a frame-exact 1920x1080 @ 30fps H.264 video segment matching `seg.duration`.
    """
    norm_out_mp4.parent.mkdir(parents=True, exist_ok=True)
    target_dur = seg.duration

    if seg.kind == "wan_s2v" and seg.rendered_clip_path and seg.rendered_clip_path.exists():
        raw_dur = probe_exact_duration(seg.rendered_clip_path)
        speed_pts = max(0.92, min(1.08, target_dur / max(0.2, raw_dur)))
        vf = (
            f"setpts={speed_pts:.5f}*PTS,"
            f"fps={fps},"
            f"scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop=1920:1080,"
            f"unsharp=3:3:0.55:3:3:0.0,"
            f"eq=contrast=1.04:saturation=1.07,"
            f"tpad=stop_mode=clone:stop_duration=1.0,"
            f"setsar=1,format=yuv420p"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(seg.rendered_clip_path),
                "-an",
                "-vf", vf,
                "-t", f"{target_dur:.3f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-r", str(fps),
                str(norm_out_mp4),
            ],
            check=True,
        )
        return norm_out_mp4

    # Concert cutaway B-roll segment
    motions = ["zoom_in_fast", "pan_right", "zoom_out", "pan_left", "zoom_in"]
    shot = ShotPlan(
        shot_idx=seg.seg_idx + 1,
        image_file=seg.image_path.name,
        duration=target_dur,
        motion=motions[seg.seg_idx % len(motions)],
        section_tag=f"Seg {seg.seg_idx + 1}",
    )
    render_motion_clip(shot, seg.image_path.parent, norm_out_mp4, fps=fps)
    return norm_out_mp4


def compose_broadcast_s2v_mv(
    master_audio_path: Path,
    segments: List[TimelineSegment],
    output_mp4: Path,
    lyrics_text: str,
    song_title: str = "《醉月長安》",
    singer_line: str = "主唱：詩仙·李白（Wan2.2-S2V 全身動態演唱）",
    credits_line: str = "詞曲：100% 無版權原創搖滾｜A100 頂配母帶室",
    work_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Path:
    master_audio_path = Path(master_audio_path)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir or (output_mp4.parent / "s2v_compose_cache")
    work_dir.mkdir(parents=True, exist_ok=True)

    total_dur = probe_audio_duration(master_audio_path)
    _, lyrics, _ = build_schedule_from_lyrics(total_dur, lyrics_text, song_title=song_title)

    norm_clips: List[Path] = []
    for i, seg in enumerate(segments):
        if progress_cb:
            progress_cb(0.82 + 0.10 * (i / max(1, len(segments))), f"正在對齊多機位嘴型與 1080p 超分母帶 ({i + 1}/{len(segments)})...")
        norm_p = work_dir / f"norm_{i:03d}.mp4"
        normalize_segment_to_1080p30(seg, norm_p, fps=30)
        norm_clips.append(norm_p)

    persistent_png, opening_png = build_tv_hud_overlays(
        work_dir,
        song_title=song_title,
        credits_line=credits_line,
        singer_line=singer_line,
        right_badge="A100 S2V 真人級動態｜商用無版權",
    )
    ass_path = work_dir / "karaoke_s2v.ass"
    write_karaoke_ass(ass_path, lyrics)
    ass_escaped = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")

    n_seg = len(norm_clips)
    hud_idx = n_seg
    card_idx = n_seg + 1
    aud_idx = n_seg + 2

    sar_norm = "".join(f"[{i}:v]setsar=1[sv{i}];" for i in range(n_seg))
    concat_in = "".join(f"[sv{i}]" for i in range(n_seg))
    fc = (
        f"{sar_norm}{concat_in}concat=n={n_seg}:v=1:a=0[vstitched];"
        f"[{card_idx}:v]format=rgba,fade=t=in:st=0.8:d=0.5:alpha=1,fade=t=out:st=6.6:d=0.6:alpha=1[card];"
        f"[vstitched][{hud_idx}:v]overlay=0:0:eof_action=pass[vhud];"
        f"[vhud][card]overlay=0:0:enable='between(t,0.7,7.4)':eof_action=pass[vcard];"
        f"[vcard]ass='{ass_escaped}',"
        f"fade=t=in:st=0:d=0.5,fade=t=out:st={max(1.0, total_dur - 1.1):.2f}:d=1.1,"
        f"format=yuv420p[vout]"
    )

    inputs: List[str] = []
    for p in norm_clips:
        inputs.extend(["-i", str(p)])
    inputs.extend(["-loop", "1", "-framerate", "30", "-t", f"{total_dur + 1.0:.2f}", "-i", str(persistent_png)])
    inputs.extend(["-loop", "1", "-framerate", "30", "-t", "8.0", "-i", str(opening_png)])
    inputs.extend(["-i", str(master_audio_path)])

    if progress_cb:
        progress_cb(0.95, "正在壓製 1080p 廣播級音畫同步母帶與 KTV 金色流光字幕...")

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            *inputs,
            "-filter_complex", fc,
            "-map", "[vout]",
            "-map", f"{aud_idx}:a",
            "-t", f"{total_dur:.2f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart",
            str(output_mp4),
        ],
        check=True,
    )
    if progress_cb:
        progress_cb(1.0, "🎉 A100 頂配真人動態 1080p MV 製作完成！")
    return output_mp4
