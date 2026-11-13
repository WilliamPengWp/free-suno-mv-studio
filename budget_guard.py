#!/usr/bin/env python3
"""
Ironclad Multi-Layer $0.00 Out-of-Pocket Budget Circuit Breaker for Modal Starter ($30/mo Free Credit).

Defense-in-Depth Architecture:
  Layer 1 (Platform Hard Lock): Modal Dashboard `Spend Limit = $0` (blocks Stripe out-of-pocket charges at $0.00)
                                + `Workspace Budget = $28` (caps gross credit usage below $30.00).
  Layer 2 (Dual-Source Monthly Meter): Takes `max(modal_billing_api_usd, persistent_modal_dict_usd)` keyed by UTC `YYYY-MM`,
                                       using 1.30x conservative safety multiplier + 25s cold-start/scaledown buffer per GPU call.
  Layer 3 (Pre-Flight Hard Circuit Breaker): Default cap `$24.00/mo` (clamped `<= $27.00/mo`, leaving >= $3.00 mandatory cushion
                                             below Modal's $30.00 monthly free credit) + `$1.50` single-job max fuse.
  Layer 4 (Zero-Idle Serverless Hardware Locks): `min_containers=0`, `max_containers=1`, `scaledown_window=20`, strict `timeout`,
                                                 plus instant UI Emergency Kill-Switch (`kill_switch_engaged`).
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Official 2026 Modal Per-Second Compute Pricing (USD / second)
# Source: https://modal.com/pricing
MODAL_RATES_USD_PER_SEC: Dict[str, float] = {
    "H100": 0.001097,
    "A100-80GB": 0.000694,  # ~$2.50 / hr
    "A100-40GB": 0.000583,  # ~$2.10 / hr
    "L40S": 0.000542,       # ~$1.95 / hr (48GB VRAM)
    "A10G": 0.000306,       # ~$1.10 / hr (24GB VRAM)
    "L4": 0.000222,         # ~$0.80 / hr (24GB VRAM)
    "T4": 0.000164,         # ~$0.59 / hr (16GB VRAM)
    "CPU-4C": 0.000056,     # 4 vCPU + 16GB RAM (weight prefetch)
    "CPU-1C": 0.000014,     # 1 vCPU + 2GB RAM (Gradio web shell)
}

MODAL_STARTER_FREE_CREDIT_USD: float = 30.00
DEFAULT_SAFE_MONTHLY_CAP_USD: float = 24.00
ABSOLUTE_MAX_MONTHLY_CAP_USD: float = 27.00  # Hard code ceiling: NEVER allow > $27 out of $30 free credit
MIN_CONFIGURABLE_CAP_USD: float = 5.00
MAX_SINGLE_JOB_USD: float = 1.50

# Conservative safety margin so our internal ledger ALWAYS trips BEFORE Modal's real billing meter
SAFETY_MULTIPLIER: float = 1.30
GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC: float = 25.0

LEDGER_DICT_NAME = "suno-pro-mv-budget-ledger-v1"


class BudgetFuseTrippedError(RuntimeError):
    """Raised BEFORE any GPU container runs if free-credit safety fuse is reached or kill-switch is active."""


@dataclass
class JobCostEstimate:
    composer_engine: str
    vocal_mode: str
    video_mode: str
    song_duration_sec: int
    composer_usd: float
    vocal_master_usd: float
    wan_s2v_usd: float
    overhead_usd: float
    total_estimated_usd: float
    est_mvs_per_month_at_24_cap: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def current_utc_month_key(now: Optional[dt.datetime] = None) -> str:
    ts = now or dt.datetime.now(dt.timezone.utc)
    return ts.strftime("%Y-%m")


def clamp_user_cap(requested_cap_usd: float) -> float:
    return round(
        max(MIN_CONFIGURABLE_CAP_USD, min(ABSOLUTE_MAX_MONTHLY_CAP_USD, float(requested_cap_usd))),
        2,
    )


def estimate_pipeline_cost(
    composer_engine: str = "ace_xl",
    vocal_mode: str = "stem_tube_master",
    video_mode: str = "smart_hybrid_s2v",
    song_duration_sec: int = 48,
) -> JobCostEstimate:
    """
    Calculates conservative (1.30x buffered) cost of a 4-stage run BEFORE spinning up any GPU.
    """
    dur = max(20, min(120, int(song_duration_sec)))

    # Stage 1: L40S (48GB) Composer
    # ACE-Step 1.5 XL (4B DiT SFT 50 steps + 4B CoT LM): ~55s inference + 25s cold/scaledown = ~80s on L40S
    # YuE2-3B (full symbolic CoT): ~135s inference + 25s cold/scaledown = ~160s on L40S
    if composer_engine == "yue2":
        comp_sec = (110.0 + dur * 0.9 + GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC)
    else:
        comp_sec = (45.0 + dur * 0.25 + GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC)
    composer_usd = comp_sec * MODAL_RATES_USD_PER_SEC["L40S"] * SAFETY_MULTIPLIER

    # Stage 2: L4 (24GB) BS-RoFormer Stem Separation + Seed-VC 44.1kHz BigVGAN / Tube Master
    if vocal_mode == "bypass":
        vocal_usd = 0.0
    elif vocal_mode in ("seed_vc_mit_husky", "seed_vc_custom"):
        svc_sec = (38.0 + dur * 0.35 + GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC)
        vocal_usd = svc_sec * MODAL_RATES_USD_PER_SEC["L4"] * SAFETY_MULTIPLIER
    else:
        stem_sec = (22.0 + dur * 0.15 + GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC)
        vocal_usd = stem_sec * MODAL_RATES_USD_PER_SEC["L4"] * SAFETY_MULTIPLIER

    # Stage 3: A100-80GB Wan2.2-S2V-14B Audio-Driven Full-Body Singing Video Diffusion
    if video_mode == "ken_burns_only":
        wan_usd = 0.0
    elif video_mode == "full_s2v":
        # ~5.0s per S2V chunk across full song duration; ~78s per chunk on A100-80GB BF16
        num_chunks = max(4, int((dur + 4.9) // 5.0))
        wan_sec = num_chunks * 78.0 + GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC
        wan_usd = wan_sec * MODAL_RATES_USD_PER_SEC["A100-80GB"] * SAFETY_MULTIPLIER
    else:
        # smart_hybrid_s2v (Recommended): 4 hero vocal/rap/chorus singing clips (~72s x 4 = 288s on A100-80GB)
        wan_sec = 4 * 72.0 + GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC
        wan_usd = wan_sec * MODAL_RATES_USD_PER_SEC["A100-80GB"] * SAFETY_MULTIPLIER

    overhead_usd = 0.008  # CPU orchestration + FFmpeg 1080p muxing
    total_usd = round(composer_usd + vocal_usd + wan_usd + overhead_usd, 4)
    mvs_per_mo = max(1, int(DEFAULT_SAFE_MONTHLY_CAP_USD // max(0.01, total_usd)))

    return JobCostEstimate(
        composer_engine=composer_engine,
        vocal_mode=vocal_mode,
        video_mode=video_mode,
        song_duration_sec=dur,
        composer_usd=round(composer_usd, 4),
        vocal_master_usd=round(vocal_usd, 4),
        wan_s2v_usd=round(wan_usd, 4),
        overhead_usd=round(overhead_usd, 4),
        total_estimated_usd=total_usd,
        est_mvs_per_month_at_24_cap=mvs_per_mo,
    )


def _query_modal_billing_api_this_month_usd() -> Optional[float]:
    """
    Queries Modal's official workspace_billing_report for the current UTC month when supported by token scope.
    Gracefully returns None if running on Starter plan without billing-report API scope or offline.
    """
    try:
        import modal.billing  # type: ignore

        now_utc = dt.datetime.now(dt.timezone.utc)
        month_start = dt.datetime(now_utc.year, now_utc.month, 1, tzinfo=dt.timezone.utc)
        items = modal.billing.workspace_billing_report(start=month_start, end=now_utc, resolution="d")
        return round(sum(float(it.cost) for it in items), 4)
    except Exception:
        return None


class BudgetLedger:
    """
    Persistent Monthly GPU Spend Ledger backed by modal.Dict (or an injected dict for unit testing).
    """

    def __init__(self, backing_store: Optional[Any] = None) -> None:
        self._store = backing_store

    def _get_store(self) -> Any:
        if self._store is not None:
            return self._store
        import modal  # type: ignore

        self._store = modal.Dict.from_name(LEDGER_DICT_NAME, create_if_missing=True)
        return self._store

    def get_state(self, month_key: Optional[str] = None) -> Dict[str, Any]:
        mk = month_key or current_utc_month_key()
        store = self._get_store()
        default_record: Dict[str, Any] = {
            "month": mk,
            "dict_metered_usd": 0.0,
            "jobs_completed": 0,
            "cap_usd": DEFAULT_SAFE_MONTHLY_CAP_USD,
            "kill_switch": False,
            "history": [],
        }
        try:
            rec = store.get(mk, default_record)
        except Exception:
            rec = default_record

        rec["cap_usd"] = clamp_user_cap(rec.get("cap_usd", DEFAULT_SAFE_MONTHLY_CAP_USD))
        api_usd = _query_modal_billing_api_this_month_usd() if self._store is None else None
        effective_used = max(float(rec.get("dict_metered_usd", 0.0)), float(api_usd or 0.0))
        rec["api_reported_usd"] = api_usd
        rec["effective_used_usd"] = round(effective_used, 4)
        rec["remaining_safe_usd"] = round(max(0.0, rec["cap_usd"] - effective_used), 4)
        rec["remaining_modal_free_usd"] = round(max(0.0, MODAL_STARTER_FREE_CREDIT_USD - effective_used), 4)
        return rec

    def set_user_cap(self, cap_usd: float, month_key: Optional[str] = None) -> Dict[str, Any]:
        mk = month_key or current_utc_month_key()
        store = self._get_store()
        rec = self.get_state(mk)
        rec["cap_usd"] = clamp_user_cap(cap_usd)
        store[mk] = rec
        return self.get_state(mk)

    def set_kill_switch(self, engaged: bool, month_key: Optional[str] = None) -> Dict[str, Any]:
        mk = month_key or current_utc_month_key()
        store = self._get_store()
        rec = self.get_state(mk)
        rec["kill_switch"] = bool(engaged)
        store[mk] = rec
        return self.get_state(mk)

    def assert_preflight_safe(
        self,
        estimate: JobCostEstimate,
        requested_cap_usd: Optional[float] = None,
        month_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Hard pre-flight gate: raises BudgetFuseTrippedError BEFORE any GPU container starts
        if kill-switch is engaged, single-job fuse exceeded, or monthly free-credit cap would be crossed.
        """
        state = self.get_state(month_key)
        active_cap = clamp_user_cap(requested_cap_usd if requested_cap_usd is not None else state["cap_usd"])

        if state.get("kill_switch"):
            raise BudgetFuseTrippedError(
                "🛑【緊急鎖死已啟動】你目前已開啟「GPU 緊急鎖死開關」，系統已 100% 阻斷所有算力呼叫（產生 $0.00 費用）。"
                "如需繼續生成，請先在上方安全面板點擊「解除鎖死」。"
            )

        if estimate.total_estimated_usd > MAX_SINGLE_JOB_USD:
            raise BudgetFuseTrippedError(
                f"🛑【單次任務熔斷保護】本次設定預估耗用 ${estimate.total_estimated_usd:.2f}，"
                f"超過單支安全上限 ${MAX_SINGLE_JOB_USD:.2f}！請縮短秒數或改選「⚡ 精省爆款導播模式（~$0.35/支）」。"
            )

        projected = round(state["effective_used_usd"] + estimate.total_estimated_usd, 4)
        if projected > active_cap:
            raise BudgetFuseTrippedError(
                f"🛑【$0.00 免付費硬熔斷觸發】本月已使用免費額度 ${state['effective_used_usd']:.2f}，"
                f"加上本次預估 ${estimate.total_estimated_usd:.2f} 將達 ${projected:.2f}，"
                f"超過你設定的安全熔斷線 ${active_cap:.2f}（Modal 每月免費額度 $30.00）！\n"
                "系統已在啟動任何 GPU 前強制攔截，確保你永遠不會被扣到 $0.01！"
            )
        return state

    def record_stage_usage(
        self,
        stage_name: str,
        hardware_tier: str,
        wall_seconds: float,
        is_gpu: bool = True,
         increment_jobs: bool = False,
        month_key: Optional[str] = None,
    ) -> float:
        mk = month_key or current_utc_month_key()
        store = self._get_store()
        rate = MODAL_RATES_USD_PER_SEC.get(hardware_tier, MODAL_RATES_USD_PER_SEC["A100-80GB"])
        buffered_sec = max(0.5, float(wall_seconds)) + (GPU_COLD_START_AND_SCALEDOWN_BUFFER_SEC if is_gpu else 2.0)
        stage_cost_usd = round(buffered_sec * rate * SAFETY_MULTIPLIER, 5)

        rec = self.get_state(mk)
        rec["dict_metered_usd"] = round(float(rec.get("dict_metered_usd", 0.0)) + stage_cost_usd, 5)
        if increment_jobs:
            rec["jobs_completed"] = int(rec.get("jobs_completed", 0)) + 1
        history = list(rec.get("history", []))
        history.append(
            {
                "ts": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "stage": stage_name,
                "hw": hardware_tier,
                "wall_s": round(wall_seconds, 1),
                "charged_safe_usd": round(stage_cost_usd, 4),
            }
        )
        rec["history"] = history[-40:]
        store[mk] = rec
        return stage_cost_usd


class MeteredStageContext:
    """Context manager that automatically records exact wall-clock GPU seconds + safety buffer to BudgetLedger."""

    def __init__(
        self,
        ledger: BudgetLedger,
        stage_name: str,
        hardware_tier: str,
        is_gpu: bool = True,
        increment_jobs: bool = False,
    ) -> None:
        self.ledger = ledger
        self.stage_name = stage_name
        self.hardware_tier = hardware_tier
        self.is_gpu = is_gpu
        self.increment_jobs = increment_jobs
        self._t0 = 0.0
        self.cost_usd = 0.0

    def __enter__(self) -> "MeteredStageContext":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed = max(0.1, time.monotonic() - self._t0)
        self.cost_usd = self.ledger.record_stage_usage(
            stage_name=self.stage_name,
            hardware_tier=self.hardware_tier,
            wall_seconds=elapsed,
            is_gpu=self.is_gpu,
            increment_jobs=(self.increment_jobs and exc_type is None),
        )
