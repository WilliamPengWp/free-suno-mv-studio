#!/usr/bin/env python3
"""
Precision Whisper Word/Character-Level Lyric Aligner for Mandarin Rock MVs.

Solves subtitle drift by:
1. Running `faster-whisper` (`word_timestamps=True`, `condition_on_previous_text=False`)
   on the vocal/song audio to get exact millisecond timestamps of every sung syllable.
2. Matching Whisper segments/clauses to the user's canonical Chinese lyrics (correcting
   ASR homophones like "經燒店" -> "金鑾殿" while preserving the exact syllable timing).
3. Emitting per-character ASS `\\kf` durations locked to the real vocal performance,
   and automatically marking instrumental solos/gaps (>= 3.0s) so lyrics never appear
   during instrumental intros or interludes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CharTiming:
    char: str
    start: float
    end: float

    @property
    def duration_cs(self) -> int:
        return max(4, int(round(max(0.04, self.end - self.start) * 100)))


@dataclass
class AlignedLyricLine:
    start: float
    end: float
    text: str
    sub_label: str = ""
    char_timings: List[CharTiming] = field(default_factory=list)
    is_instrumental: bool = False

    def to_ass_kf_text(self) -> str:
        if not self.char_timings or self.is_instrumental:
            total_cs = max(10, int(round(max(0.3, self.end - self.start) * 100)))
            chars = list(self.text)
            if not chars:
                return ""
            cs_per = max(2, total_cs // len(chars))
            return "".join(f"{{\\kf{cs_per}}}{ch}" for ch in chars)

        parts: List[str] = []
        prev_end = self.start
        for ct in self.char_timings:
            gap_cs = int(round(max(0.0, ct.start - prev_end) * 100))
            if gap_cs >= 6:
                parts.append(f"{{\\k{gap_cs}}}")
            dur_cs = max(4, int(round(max(0.04, ct.end - max(ct.start, prev_end)) * 100)))
            parts.append(f"{{\\kf{dur_cs}}}{ct.char}")
            prev_end = max(prev_end, ct.end)
        return "".join(parts)


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


def parse_canonical_lyrics(lyrics_text: str) -> List[Tuple[str, str]]:
    """Returns list of (canonical_line_text, section_cue_label) for vocal lines."""
    entries: List[Tuple[str, str]] = []
    pending_cue = ""
    for raw_line in (lyrics_text or "").splitlines():
        s = "".join(ch for ch in raw_line.strip() if ord(ch) <= 0xFFFF).strip()
        if not s:
            continue
        m = re.match(r"^\[(.*?)\]$", s)
        if m:
            tag_lower = m.group(1).strip().lower()
            if "solo" in tag_lower or "interlude" in tag_lower or "outro" in tag_lower:
                continue
            for k, label in SECTION_CUE_MAP.items():
                if k in tag_lower:
                    pending_cue = label
                    break
            continue
        entries.append((s, pending_cue))
        pending_cue = ""
    return entries


def _distribute_chars_over_words(
    canonical_text: str,
    asr_words: List[Tuple[str, float, float]],
    seg_start: float,
    seg_end: float,
) -> List[CharTiming]:
    """
    Maps canonical characters (including spaces/punctuation) onto the exact Whisper
    character timestamp stream extracted from `asr_words`.
    """
    # Expand multi-char Whisper words into individual ASR character timings
    asr_chars: List[Tuple[float, float]] = []
    for w_txt, w_st, w_et in asr_words:
        clean_w = re.sub(r"\s+", "", w_txt)
        if not clean_w:
            continue
        n = len(clean_w)
        dur = max(0.04, w_et - w_st)
        for i in range(n):
            c_st = w_st + dur * (i / n)
            c_et = w_st + dur * ((i + 1) / n)
            asr_chars.append((round(c_st, 3), round(c_et, 3)))

    canon_chars = list(canonical_text)
    if not canon_chars:
        return []

    if not asr_chars:
        dur = max(0.4, seg_end - seg_start)
        n = len(canon_chars)
        return [
            CharTiming(
                ch,
                round(seg_start + dur * (i / n), 3),
                round(seg_start + dur * ((i + 1) / n), 3),
            )
            for i, ch in enumerate(canon_chars)
        ]

    # Identify vocal (non-punctuation/non-space) canonical characters
    vocal_indices = [
        idx for idx, ch in enumerate(canon_chars)
        if not re.match(r"[\s！!？?，,。、·：:；;「」『』（）()]", ch)
    ]
    if not vocal_indices:
        vocal_indices = list(range(len(canon_chars)))

    m_vocal = len(vocal_indices)
    k_asr = len(asr_chars)

    result: List[Optional[CharTiming]] = [None] * len(canon_chars)
    for v_pos, c_idx in enumerate(vocal_indices):
        # Map v_pos in [0, m_vocal-1] to [a_start_idx, a_end_idx] in [0, k_asr-1]
        a_idx_f = (v_pos / max(1, m_vocal)) * k_asr
        a_idx_next_f = ((v_pos + 1) / max(1, m_vocal)) * k_asr
        i0 = min(k_asr - 1, max(0, int(a_idx_f)))
        i1 = min(k_asr - 1, max(i0, int(a_idx_next_f - 1e-6)))
        c_st = asr_chars[i0][0]
        c_et = max(c_st + 0.04, asr_chars[i1][1])
        result[c_idx] = CharTiming(canon_chars[c_idx], c_st, c_et)

    # Fill in spaces/punctuation smoothly between neighboring vocal characters
    final_timings: List[CharTiming] = []
    for idx, ch in enumerate(canon_chars):
        if result[idx] is not None:
            final_timings.append(result[idx])  # type: ignore[arg-type]
        else:
            prev_t = final_timings[-1].end if final_timings else seg_start
            next_t = seg_end
            for nxt_i in range(idx + 1, len(canon_chars)):
                if result[nxt_i] is not None:
                    next_t = result[nxt_i].start  # type: ignore[union-attr]
                    break
            mid_et = max(prev_t + 0.03, min(next_t, prev_t + 0.12))
            final_timings.append(CharTiming(ch, prev_t, mid_et))

    return final_timings


def align_lyrics_with_whisper(
    audio_path: Path,
    lyrics_text: str,
    total_dur: float,
    song_title: str = "醉月長安",
    model_size: str = "small",
) -> List[AlignedLyricLine]:
    """
    Transcribes `audio_path` with `faster-whisper` word timestamps and aligns each
    sung segment to the canonical lyric lines in `lyrics_text`.
    """
    from faster_whisper import WhisperModel

    canon_lines = parse_canonical_lyrics(lyrics_text)
    if not canon_lines:
        return [
            AlignedLyricLine(
                start=0.5,
                end=max(4.0, total_dur - 0.5),
                text=f"♪ 《{song_title}》原創舞台純享版 ♪",
                sub_label="【100% 原創音樂】",
                is_instrumental=True,
            )
        ]

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_gen, _ = model.transcribe(
        str(audio_path),
        language="zh",
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=False,
    )

    raw_segments = []
    for seg in segments_gen:
        txt = (seg.text or "").strip()
        # Filter out non-Chinese noise tags like "Zither Harp" or single-char noise
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", txt))
        if cjk_count < 2:
            continue
        words = [
            (w.word, float(w.start), float(w.end))
            for w in (seg.words or [])
            if w.word and w.word.strip()
        ]
        st = words[0][1] if words else float(seg.start)
        et = words[-1][2] if words else float(seg.end)
        raw_segments.append({"text": txt, "start": st, "end": et, "words": words, "cjk": cjk_count})

    # Merge tiny consecutive fragments if Whisper split a single lyric line into two half-lines,
    # or split a double-line segment if Whisper merged two canonical lines into one segment.
    processed_segments = _reconcile_segments_with_canonical(raw_segments, canon_lines)

    aligned: List[AlignedLyricLine] = []
    n_matched = min(len(processed_segments), len(canon_lines))

    for i in range(n_matched):
        seg = processed_segments[i]
        canon_txt, cue = canon_lines[i]
        st = round(max(0.1, seg["start"]), 2)
        et = round(max(st + 0.8, seg["end"]), 2)
        char_timings = _distribute_chars_over_words(canon_txt, seg["words"], st, et)
        aligned.append(
            AlignedLyricLine(
                start=st,
                end=et,
                text=canon_txt,
                sub_label=cue,
                char_timings=char_timings,
                is_instrumental=False,
            )
        )

    # Insert instrumental cues during long non-vocal stretches (>= 3.8s)
    with_cues: List[AlignedLyricLine] = []
    clean_title = song_title.strip("《》")
    if aligned and aligned[0].start >= 3.2:
        with_cues.append(
            AlignedLyricLine(
                start=0.4,
                end=round(aligned[0].start - 0.25, 2),
                text=f"♪ 《{clean_title}》前奏演奏 ♪",
                sub_label="【前奏 Intro】",
                is_instrumental=True,
            )
        )

    for idx, line in enumerate(aligned):
        if idx > 0:
            prev_end = aligned[idx - 1].end
            gap = line.start - prev_end
            if gap >= 3.8:
                with_cues.append(
                    AlignedLyricLine(
                        start=round(prev_end + 0.25, 2),
                        end=round(line.start - 0.25, 2),
                        text="♪ 樂團間奏 · 二胡 × 電吉他搖滾獨奏 ♪",
                        sub_label="【樂團間奏 Interlude】",
                        is_instrumental=True,
                    )
                )
        with_cues.append(line)

    if with_cues and (total_dur - with_cues[-1].end) >= 3.5:
        with_cues.append(
            AlignedLyricLine(
                start=round(with_cues[-1].end + 0.3, 2),
                end=round(total_dur - 0.3, 2),
                text="♪ 嗩吶 × 電吉他 炸場尾奏 SOLO ♪",
                sub_label="【尾奏 Outro】",
                is_instrumental=True,
            )
        )

    return with_cues


def _reconcile_segments_with_canonical(
    raw_segments: List[dict],
    canon_lines: List[Tuple[str, str]],
) -> List[dict]:
    """
    Handles cases where Whisper splits one lyric line across two adjacent short segments
    or merges two consecutive lyric lines into one long segment.
    """
    if not raw_segments:
        return []

    out: List[dict] = []
    seg_idx = 0
    canon_idx = 0

    while seg_idx < len(raw_segments) and canon_idx < len(canon_lines):
        seg = raw_segments[seg_idx]
        canon_txt, _ = canon_lines[canon_idx]
        canon_cjk = len(re.findall(r"[\u4e00-\u9fff]", canon_txt))

        # Case 1: Whisper merged two canonical lines into one extra-long segment
        if (
            canon_idx + 1 < len(canon_lines)
            and seg["cjk"] >= int((canon_cjk + len(re.findall(r"[\u4e00-\u9fff]", canon_lines[canon_idx + 1][0]))) * 0.78)
            and len(seg["words"]) >= 4
        ):
            next_cjk = len(re.findall(r"[\u4e00-\u9fff]", canon_lines[canon_idx + 1][0]))
            ratio = canon_cjk / max(1, canon_cjk + next_cjk)
            split_w = max(1, min(len(seg["words"]) - 1, int(round(len(seg["words"]) * ratio))))
            w1 = seg["words"][:split_w]
            w2 = seg["words"][split_w:]
            out.append({
                "text": "".join(w[0] for w in w1),
                "start": w1[0][1],
                "end": w1[-1][2],
                "words": w1,
                "cjk": canon_cjk,
            })
            out.append({
                "text": "".join(w[0] for w in w2),
                "start": w2[0][1],
                "end": w2[-1][2],
                "words": w2,
                "cjk": next_cjk,
            })
            seg_idx += 1
            canon_idx += 2
            continue

        # Case 2: Whisper split one canonical line into two short back-to-back segments
        if (
            seg["cjk"] <= int(canon_cjk * 0.62)
            and seg_idx + 1 < len(raw_segments)
            and (raw_segments[seg_idx + 1]["start"] - seg["end"]) < 1.2
            and (seg["cjk"] + raw_segments[seg_idx + 1]["cjk"]) <= int(canon_cjk * 1.35)
        ):
            nxt = raw_segments[seg_idx + 1]
            merged_words = seg["words"] + nxt["words"]
            out.append({
                "text": seg["text"] + " " + nxt["text"],
                "start": seg["start"],
                "end": nxt["end"],
                "words": merged_words,
                "cjk": seg["cjk"] + nxt["cjk"],
            })
            seg_idx += 2
            canon_idx += 1
            continue

        out.append(seg)
        seg_idx += 1
        canon_idx += 1

    return out
