"""
TR-R1 — COARSE TEMPORAL RETRIEVAL BẰNG CLIP-L
==============================================

Mục đích
--------
Từ một chuỗi sự kiện:

    event tiếng Việt
        ↓
    Marian query expansion (chỉ cho CLIP)
        ↓
    CLIP-L text encoder
        ↓
    FAISS CLIP-L Top-K
        ↓
    frame candidate fusion
        ↓
    temporal grouping
        ↓
    coarse temporal regions
        ↓
    R2

TR-R1 KHÔNG làm:
    - dense/local retrieval
    - temporal alignment / DP
    - OCR / ASR
    - chọn best frame cuối cùng
    - submission

Public contract
---------------
tim_vung_tho(event, config=..., retriever=...)
    -> TRR1Result

tim_nhieu_su_kien(events, config=..., retriever=...)
    -> tuple[TRR1Result, ...]

CoarseRegion chỉ có:
    video_id
    start_time
    end_time
    score
    hits

Không có best_frame_idx.

Thiết kế scoring
----------------
TR-R1 tối ưu coarse temporal recall, vì vậy region không được
xếp hạng chỉ bằng peak frame score.

Region score gồm:

    peak_score
        : hit mạnh nhất trong region.

    support_score
        : mức hỗ trợ của các hit còn lại.

    density_score
        : mức tập trung temporal của các hit.

Các thành phần được normalize về [0, 1] trước khi fusion.

Mặc định:

    peak_weight    = 0.50
    support_weight = 0.30
    density_weight = 0.20

Đây là ranking heuristic có thể benchmark/tune ở tầng benchmark.
Không dùng số frame hit để cộng cosine thô trực tiếp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


# ============================================================================
# CONFIG
# ============================================================================


@dataclass(frozen=True)
class TRR1Config:
    """
    Cấu hình TR-R1.

    Retrieval
    ---------
    top_k:
        Top-K frame cho MỖI query variant.

    Temporal grouping
    -----------------
    max_region_duration_seconds:
        Độ dài tối đa của một coarse region.

    region_merge_gap_seconds:
        Khoảng cách tối đa giữa hai hit liên tiếp để tiếp tục
        cùng một region.

    region_padding_seconds:
        Padding mỗi phía của region sau khi grouping.

        Padding chỉ dùng để biểu diễn temporal interval.
        Không tạo thêm hit.

    min_region_duration_seconds:
        Độ dài tối thiểu của region. Một hit đơn lẻ không được biểu diễn
        bằng interval suy biến ``[t, t]`` vì temporal IoU của interval đó
        luôn bằng 0, kể cả khi hit nằm đúng trong GT.

    Ranking
    -------
    max_regions_per_event:
        Số coarse regions trả ra.

    min_hits_per_region:
        Số hit tối thiểu để region được coi là candidate.
        Giá trị 1 giữ recall cao nhất.

    peak_weight:
        Trọng số peak hit.

    support_weight:
        Trọng số support của các hit còn lại.

    density_weight:
        Trọng số temporal density.

    Query expansion
    ----------------
    use_query_expansion:
        Có dùng Marian hay không.

    max_query_variants:
        Số query variant tối đa.

    video_consensus_weight:
        Trọng số prior video dùng khi tìm nhiều event của cùng một query
        TRAKE. Prior chỉ được tính từ retrieval hits của toàn chuỗi, không
        dùng GT. Đặt 0 để giữ hành vi xếp hạng từng event độc lập.

    video_rrf_k:
        Hằng số RRF khi hợp nhất best rank của mỗi video giữa các event.
    """

    top_k: int = 500

    max_region_duration_seconds: float = 3.0
    region_merge_gap_seconds: float = 0.5
    region_padding_seconds: float = 0.1
    min_region_duration_seconds: float = 0.16

    max_regions_per_event: int = 10
    min_hits_per_region: int = 1

    peak_weight: float = 0.50
    support_weight: float = 0.30
    density_weight: float = 0.20

    use_query_expansion: bool = True
    max_query_variants: int = 4

    video_consensus_weight: float = 0.45
    video_rrf_k: float = 60.0

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(
                "top_k phải > 0."
            )

        if self.max_region_duration_seconds <= 0:
            raise ValueError(
                "max_region_duration_seconds phải > 0."
            )

        if self.region_merge_gap_seconds < 0:
            raise ValueError(
                "region_merge_gap_seconds phải >= 0."
            )

        if self.region_padding_seconds < 0:
            raise ValueError(
                "region_padding_seconds phải >= 0."
            )

        if self.min_region_duration_seconds <= 0:
            raise ValueError(
                "min_region_duration_seconds phải > 0."
            )

        if (
            self.min_region_duration_seconds
            > self.max_region_duration_seconds
        ):
            raise ValueError(
                "min_region_duration_seconds không được lớn hơn "
                "max_region_duration_seconds."
            )

        if self.max_regions_per_event <= 0:
            raise ValueError(
                "max_regions_per_event phải > 0."
            )

        if self.min_hits_per_region <= 0:
            raise ValueError(
                "min_hits_per_region phải > 0."
            )

        weights = (
            self.peak_weight,
            self.support_weight,
            self.density_weight,
        )

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Các region score weights phải >= 0."
            )

        if sum(weights) <= 0:
            raise ValueError(
                "Tổng region score weights phải > 0."
            )

        if self.max_query_variants <= 0:
            raise ValueError(
                "max_query_variants phải > 0."
            )

        if not 0.0 <= self.video_consensus_weight <= 1.0:
            raise ValueError(
                "video_consensus_weight phải nằm trong [0, 1]."
            )

        if self.video_rrf_k < 0:
            raise ValueError(
                "video_rrf_k phải >= 0."
            )


# ============================================================================
# PUBLIC RESULT OBJECTS
# ============================================================================


@dataclass(frozen=True)
class CoarseRegion:
    """
    Một coarse temporal region.

    CỐ Ý không có:
        - frame_idx
        - best_frame_idx
        - best_frame

    Frame provenance chỉ nằm trong `hits`.
    """

    video_id: str
    start_time: float
    end_time: float
    score: float
    hits: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TRR1Result:
    """
    Kết quả TR-R1 cho một event.
    """

    event_id: str
    text: str
    relation: str | None
    regions: tuple[CoarseRegion, ...]


# ============================================================================
# TEST / INTERNAL HIT BUILDER
# ============================================================================


def _build_test_hit(
    video_id: str,
    pts_time: float,
    score: float,
) -> dict[str, Any]:
    """
    Tạo một hit tối thiểu cho unit test.

    Production hit có thể chứa thêm:
        n
        frame_idx
        rank
        source
    """

    return {
        "video_id": str(video_id),
        "pts_time": float(pts_time),
        "score": float(score),
    }


# ============================================================================
# QUERY EVENT VALIDATION
# ============================================================================


def _parse_event(
    event: Any,
) -> tuple[str, str, str | None]:
    """
    Parse một event theo contract TR-R1.

    Bắt buộc:
        event_id
        text

    Không chấp nhận:
        description thay cho text.
    """

    if not isinstance(event, dict):
        raise ValueError(
            "TR-R1 event phải là dict."
        )

    if "event_id" not in event:
        raise ValueError(
            "TR-R1 event thiếu trường 'event_id'."
        )

    if "text" not in event:
        raise ValueError(
            "TR-R1 event bắt buộc có trường 'text'."
        )

    event_id = event["event_id"]
    text = event["text"]

    if event_id is None:
        raise ValueError(
            "event_id không được là None."
        )

    if not isinstance(text, str):
        raise ValueError(
            "event.text phải là string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "event.text không được rỗng."
        )

    relation = event.get("relation")

    if relation is not None:
        relation = str(relation).strip()

    return (
        str(event_id),
        text,
        relation,
    )


# ============================================================================
# QUERY EXPANSION
# ============================================================================


def _mo_rong_trr1(
    text: str,
    *,
    use_query_expansion: bool = True,
    max_query_variants: int = 4,
) -> list[str]:

    text = (text or "").strip()

    if not text:
        return []

    if not use_query_expansion:
        return [text]

    try:
        from aic2026.query_expand import mo_rong

        output: list[str] = []

        for nguon in (
            "marian",
            "tu_dien",
        ):
            try:
                result = mo_rong(
                    text,
                    nguon=nguon,
                )
            except Exception:
                continue

            variants = getattr(
                result,
                "cum_tieng_anh",
                None,
            )

            if not variants:
                continue

            for variant in variants:
                variant = str(
                    variant or ""
                ).strip()

                if (
                    variant
                    and variant not in output
                ):
                    output.append(
                        variant
                    )

                if (
                    len(output)
                    >= max_query_variants
                ):
                    return output

        return output or [text]

    except Exception:
        return [text]


def _query_variants(
    text: str,
    *,
    use_query_expansion: bool,
    max_query_variants: int,
) -> list[str]:
    """
    Chuẩn hóa danh sách query variants.
    """

    text = (text or "").strip()

    if not text:
        return []

    variants = _mo_rong_trr1(
        text,
        use_query_expansion=use_query_expansion,
        max_query_variants=max_query_variants,
    )

    output: list[str] = []

    for variant in variants:
        variant = str(
            variant or ""
        ).strip()

        if not variant:
            continue

        if variant not in output:
            output.append(variant)

        if len(output) >= max_query_variants:
            break

    return output or [text]


# ============================================================================
# CLIP-L RUNTIME
# ============================================================================


_CLIP_L_RUNTIME: (
    tuple[Any, Any, Any, Any, Any] | None
) = None


def _get_trr1_clip_l_runtime():
    """
    Load/cache:

        model
        tokenizer
        device
        FAISS index
        ID dataframe
    """

    global _CLIP_L_RUNTIME

    if _CLIP_L_RUNTIME is not None:
        return _CLIP_L_RUNTIME

    from aic2026.index import clip_l_index

    model, tokenizer, device = (
        clip_l_index._encoder()
    )

    index, ids = clip_l_index.nap_chi_muc()

    _CLIP_L_RUNTIME = (
        model,
        tokenizer,
        device,
        index,
        ids,
    )

    return _CLIP_L_RUNTIME


# ============================================================================
# RAW CLIP-L RETRIEVAL
# ============================================================================


def _tim_hit_clip_l_mot_query(
    text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Một query text -> raw CLIP-L Top-K frame hits.
    """

    text = (text or "").strip()

    if not text:
        return []

    if top_k <= 0:
        return []

    import numpy as np
    import torch

    model, tokenizer, device, index, ids = (
        _get_trr1_clip_l_runtime()
    )

    tokens = tokenizer([text])

    if hasattr(tokens, "to"):
        tokens = tokens.to(device)

    elif isinstance(tokens, dict):
        tokens = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in tokens.items()
        }

    with torch.inference_mode():
        query = model.encode_text(tokens)

    query = (
        query
        / query.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
    )

    query_np = (
        query.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    k = min(
        int(top_k),
        int(index.ntotal),
    )

    if k <= 0:
        return []

    scores, positions = index.search(
        query_np,
        k,
    )

    scores = scores[0]
    positions = positions[0]

    hits: list[dict[str, Any]] = []

    for rank, (
        score,
        position,
    ) in enumerate(
        zip(scores, positions),
        start=1,
    ):
        position = int(position)

        if position < 0:
            continue

        try:
            row = ids.iloc[position]
        except (
            IndexError,
            KeyError,
        ):
            continue

        try:
            video_id = str(
                row["video_id"]
            )

            n = int(
                row["n"]
            )

            frame_idx = int(
                row["frame_idx"]
            )

            pts_time = float(
                row["pts_time"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        hits.append(
            {
                "video_id": video_id,
                "n": n,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "score": float(score),
                "rank": int(rank),
                "source": "clip_l",
            }
        )

    return hits


# ============================================================================
# HIT HELPERS
# ============================================================================


def _safe_score(
    hit: dict[str, Any],
) -> float:
    """
    Lấy score an toàn.
    """

    try:
        return float(hit["score"])
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return float("-inf")


def _safe_time(
    hit: dict[str, Any],
) -> float | None:
    """
    Lấy timestamp an toàn.
    """

    try:
        value = float(
            hit["pts_time"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None

    if value < 0:
        return None

    return value


def _safe_video_id(
    hit: dict[str, Any],
) -> str:
    """
    Lấy video_id an toàn.
    """

    try:
        return str(
            hit["video_id"]
        ).strip()
    except Exception:
        return ""


# ============================================================================
# MULTI-VARIANT MERGE
# ============================================================================


def _hit_key(
    hit: dict[str, Any],
) -> tuple[Any, ...]:
    """
    Key ổn định để dedup cùng một frame.

    Ưu tiên:
        video_id + frame_idx
        video_id + n
        video_id + rounded pts_time
    """

    video_id = _safe_video_id(hit)

    if hit.get("frame_idx") is not None:
        try:
            return (
                video_id,
                "frame_idx",
                int(hit["frame_idx"]),
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    if hit.get("n") is not None:
        try:
            return (
                video_id,
                "n",
                int(hit["n"]),
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    timestamp = _safe_time(hit)

    if timestamp is None:
        timestamp = 0.0

    return (
        video_id,
        "pts_time",
        round(
            timestamp,
            3,
        ),
    )


def _merge_query_variant_hits(
    hits_by_variant: Iterable[
        list[dict[str, Any]]
    ],
    *,
    rrf_k: float = 60.0,
) -> list[dict[str, Any]]:

    fused: dict[
        tuple[Any, ...],
        dict[str, Any],
    ] = {}

    rrf_scores: dict[
        tuple[Any, ...],
        float,
    ] = {}

    variant_ranks: dict[
        tuple[Any, ...],
        list[int],
    ] = {}

    for variant_idx, hits in enumerate(
        hits_by_variant
    ):
        for rank, hit in enumerate(
            hits,
            start=1,
        ):
            if not isinstance(
                hit,
                dict,
            ):
                continue

            key = _hit_key(
                hit
            )

            # -------------------------------------------------
            # RRF: mỗi variant đóng góp bằng rank.
            # -------------------------------------------------

            rrf_scores[key] = (
                rrf_scores.get(
                    key,
                    0.0,
                )
                + 1.0
                / (
                    rrf_k
                    + float(rank)
                )
            )

            variant_ranks.setdefault(
                key,
                [],
            ).append(
                int(rank)
            )

            # -------------------------------------------------
            # Giữ hit gốc có cosine mạnh nhất
            # để bảo toàn provenance.
            # -------------------------------------------------

            previous = fused.get(
                key
            )

            candidate = dict(
                hit
            )

            if (
                previous is None
                or _safe_score(
                    candidate
                )
                > _safe_score(
                    previous
                )
            ):
                fused[key] = candidate

    output: list[
        dict[str, Any]
    ] = []

    for key, hit in fused.items():
        candidate = dict(
            hit
        )

        candidate[
            "clip_score"
        ] = float(
            _safe_score(hit)
        )

        candidate[
            "score"
        ] = float(
            rrf_scores[key]
        )

        candidate[
            "variant_ranks"
        ] = tuple(
            variant_ranks[key]
        )

        candidate[
            "source"
        ] = "clip_l_rrf"

        output.append(
            candidate
        )

    output.sort(
        key=lambda hit: (
            -float(
                hit["score"]
            ),
            _safe_video_id(
                hit
            ),
            _safe_time(hit)
            if _safe_time(hit)
            is not None
            else float("inf"),
        )
    )

    return output


def tim_hit_clip_l(
    text: str,
    top_k: int = 500,
    *,
    use_query_expansion: bool = True,
    max_query_variants: int = 2,
) -> list[dict[str, Any]]:
    """
    Query tiếng Việt -> raw CLIP-L hits.

    top_k là Top-K cho MỖI variant.

    Sau đó:
        variant hits
            ↓
        frame dedup
            ↓
        best cosine giữ lại
    """

    text = (text or "").strip()

    if not text:
        return []

    if top_k <= 0:
        return []

    variants = _query_variants(
        text,
        use_query_expansion=use_query_expansion,
        max_query_variants=max_query_variants,
    )

    if not variants:
        return []

    hits_by_variant: list[
        list[dict[str, Any]]
    ] = []

    for variant in variants:
        hits_by_variant.append(
            _tim_hit_clip_l_mot_query(
                variant,
                top_k,
            )
        )

    return _merge_query_variant_hits(
        hits_by_variant
    )


def _video_consensus_scores(
    hits_by_event: Iterable[list[dict[str, Any]]],
    *,
    rrf_k: float = 60.0,
) -> dict[str, float]:
    """Tính prior video từ toàn bộ event trong một query TRAKE.

    Mỗi video chỉ đóng góp best rank một lần cho mỗi event. Cách này tránh
    bias về video dài (có nhiều frame trong index) và tránh cộng trực tiếp
    cosine của các event/query variant vốn không cùng calibration.

    Điểm cuối kết hợp:
        - 65% RRF best-rank qua các event;
        - 35% tỉ lệ event có ít nhất một hit của video.

    Kết quả được chuẩn hóa về [0, 1] và hoàn toàn không dùng GT.
    """

    event_hits = list(hits_by_event)

    if not event_hits:
        return {}

    rrf_scores: dict[str, float] = {}
    event_support: dict[str, int] = {}

    for hits in event_hits:
        best_rank: dict[str, int] = {}

        for rank, hit in enumerate(hits, start=1):
            if not isinstance(hit, dict):
                continue

            video_id = _safe_video_id(hit)

            if not video_id or video_id in best_rank:
                continue

            best_rank[video_id] = rank

        for video_id, rank in best_rank.items():
            rrf_scores[video_id] = (
                rrf_scores.get(video_id, 0.0)
                + 1.0 / (rrf_k + float(rank))
            )
            event_support[video_id] = (
                event_support.get(video_id, 0) + 1
            )

    if not rrf_scores:
        return {}

    max_rrf = max(rrf_scores.values())
    num_events = len(event_hits)
    output: dict[str, float] = {}

    for video_id, raw_rrf in rrf_scores.items():
        rrf_norm = raw_rrf / max_rrf if max_rrf > 0 else 0.0
        support_norm = event_support[video_id] / num_events
        output[video_id] = float(
            0.65 * rrf_norm + 0.35 * support_norm
        )

    return output


# ============================================================================
# TEMPORAL REGION BUILDER
# ============================================================================


@dataclass
class _RegionBuilder:
    video_id: str
    start_time: float
    end_time: float
    hits: list[dict[str, Any]]


def _can_merge(
    region: _RegionBuilder,
    hit: dict[str, Any],
    config: TRR1Config,
) -> bool:
    """
    Kiểm tra hit có thể nhập region hiện tại không.

    Input đã được sort theo:
        video_id
        pts_time

    nên chỉ cần xét:
        cùng video
        gap <= merge gap
        duration mới <= max duration
    """

    video_id = _safe_video_id(hit)

    if video_id != region.video_id:
        return False

    hit_time = _safe_time(hit)

    if hit_time is None:
        return False

    if hit_time < region.start_time:
        return False

    gap = (
        hit_time
        - region.end_time
    )

    if gap > config.region_merge_gap_seconds:
        return False

    new_end = max(
        region.end_time,
        hit_time,
    )

    duration = (
        new_end
        - region.start_time
    )

    if (
        duration
        > config.max_region_duration_seconds
    ):
        return False

    return True


def _append_hit(
    region: _RegionBuilder,
    hit: dict[str, Any],
) -> None:
    """
    Append hit vào region.
    """

    hit_time = _safe_time(hit)

    if hit_time is None:
        return

    region.end_time = max(
        region.end_time,
        hit_time,
    )

    region.hits.append(hit)


# ============================================================================
# REGION SCORE
# ============================================================================


def _normalize_unit_interval(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Normalize value về [0, 1].
    """

    if maximum <= minimum:
        return 1.0

    normalized = (
        value - minimum
    ) / (
        maximum - minimum
    )

    return max(
        0.0,
        min(
            1.0,
            normalized,
        ),
    )


def _region_peak_score(
    hits: list[dict[str, Any]],
) -> float:
    """
    Peak score của region.

    Cosine CLIP-L thường đã nằm trong một khoảng tương đối ổn định,
    nhưng không giả định cứng [-1, 1] ở đây.

    Ta normalize tương đối theo chính candidate pool ở bước finalize.
    """

    scores = [
        _safe_score(hit)
        for hit in hits
        if _safe_score(hit) != float("-inf")
    ]

    if not scores:
        return 0.0

    return max(scores)


def _region_support_score(
    hits: list[dict[str, Any]],
) -> float:
    """
    Support score.

    Không cộng cosine thô.

    Ý tưởng:
        - lấy các hit trong region
        - bỏ peak
        - các hit còn lại đóng góp giảm dần theo rank trong region

    Điều này thưởng cho region có nhiều evidence độc lập,
    nhưng không để region dài tự động thắng chỉ vì có nhiều frame.
    """

    scores = sorted(
        (
            _safe_score(hit)
            for hit in hits
            if _safe_score(hit) != float("-inf")
        ),
        reverse=True,
    )

    if len(scores) <= 1:
        return 0.0

    support = scores[1:]

    # Dùng top-3 support tối đa để tránh region dài có lợi thế vô hạn.
    support = support[:3]

    if not support:
        return 0.0

    # Reciprocal positional weighting.
    weights = [
        1.0,
        0.5,
        0.3333333333333333,
    ]

    weighted_sum = 0.0
    weight_sum = 0.0

    for idx, score in enumerate(support):
        weight = weights[idx]
        weighted_sum += (
            score * weight
        )
        weight_sum += weight

    if weight_sum <= 0:
        return 0.0

    return (
        weighted_sum
        / weight_sum
    )


def _region_density_score(
    region: _RegionBuilder,
) -> float:
    """
    Temporal density.

    Dùng số hit / span nhưng có saturation.

    Mục tiêu:
        2-3 hit tập trung tốt hơn 1 hit đơn độc,
        nhưng 20 hit không được thắng tuyệt đối chỉ vì số lượng.

    Density được chuẩn hóa bằng:
        hit_count / max(1, span_seconds + 1)

    rồi đưa qua saturation.
    """

    hit_count = len(
        region.hits
    )

    if hit_count <= 1:
        return 0.0

    span = max(
        0.0,
        region.end_time
        - region.start_time,
    )

    raw_density = (
        hit_count
        / (1.0 + span)
    )

    # Saturating transform:
    #
    # density=1 -> 0.50
    # density=2 -> 0.67
    # density=3 -> 0.75
    #
    # Không để count tăng tuyến tính vô hạn.
    return (
        raw_density
        / (
            raw_density
            + 1.0
        )
    )


def _score_regions(
    builders: list[_RegionBuilder],
    config: TRR1Config,
    *,
    video_scores: dict[str, float] | None = None,
) -> list[CoarseRegion]:
    """
    Chấm điểm toàn bộ candidate regions.

    Quan trọng:
        peak/support được normalize tương đối trong candidate pool.

    Sau đó:

        local_score =
            peak_weight    * peak
          + support_weight * support
          + density_weight * density

        score =
            (1 - video_consensus_weight) * local_score
          + video_consensus_weight * video_prior

    ``video_prior`` chỉ có khi API nhiều-event truyền vào và được tính
    hoàn toàn từ retrieval hits của chuỗi event.
    """

    if not builders:
        return []

    builders = [
        builder
        for builder in builders
        if len(builder.hits)
        >= config.min_hits_per_region
    ]

    if not builders:
        return []

    raw_peaks = [
        _region_peak_score(
            builder.hits
        )
        for builder in builders
    ]

    raw_supports = [
        _region_support_score(
            builder.hits
        )
        for builder in builders
    ]

    raw_densities = [
        _region_density_score(
            builder
        )
        for builder in builders
    ]

    peak_min = min(
        raw_peaks
    )
    peak_max = max(
        raw_peaks
    )

    support_min = min(
        raw_supports
    )
    support_max = max(
        raw_supports
    )

    density_min = min(
        raw_densities
    )
    density_max = max(
        raw_densities
    )

    weight_sum = (
        config.peak_weight
        + config.support_weight
        + config.density_weight
    )

    scored: list[
        tuple[
            float,
            _RegionBuilder,
        ]
    ] = []

    for (
        builder,
        peak,
        support,
        density,
    ) in zip(
        builders,
        raw_peaks,
        raw_supports,
        raw_densities,
    ):
        peak_norm = _normalize_unit_interval(
            peak,
            peak_min,
            peak_max,
        )

        support_norm = _normalize_unit_interval(
            support,
            support_min,
            support_max,
        )

        density_norm = _normalize_unit_interval(
            density,
            density_min,
            density_max,
        )

        local_score = (
            config.peak_weight
            * peak_norm
            + config.support_weight
            * support_norm
            + config.density_weight
            * density_norm
        ) / weight_sum

        consensus_weight = (
            config.video_consensus_weight
            if video_scores
            else 0.0
        )

        video_prior = (
            float(video_scores.get(builder.video_id, 0.0))
            if video_scores
            else 0.0
        )

        score = (
            (1.0 - consensus_weight) * local_score
            + consensus_weight * video_prior
        )

        scored.append(
            (
                float(score),
                builder,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].video_id,
            item[1].start_time,
            item[1].end_time,
        )
    )

    output: list[
        CoarseRegion
    ] = []

    for score, builder in scored:
        hits = list(
            builder.hits
        )

        hits.sort(
            key=lambda hit: (
                -_safe_score(hit),
                _safe_time(hit)
                if _safe_time(hit) is not None
                else float("inf"),
            )
        )

        start_time = float(
            builder.start_time
        )

        end_time = float(
            builder.end_time
        )

        current_duration = end_time - start_time

        if current_duration < config.min_region_duration_seconds:
            center = (start_time + end_time) / 2.0
            half = config.min_region_duration_seconds / 2.0
            start_time = max(0.0, center - half)
            end_time = start_time + config.min_region_duration_seconds

        if config.region_padding_seconds > 0:
            start_time = max(
                0.0,
                start_time
                - config.region_padding_seconds,
            )

            end_time = (
                end_time
                + config.region_padding_seconds
            )

            # Padding không được làm region vượt max duration.
            if (
                end_time
                - start_time
                > config.max_region_duration_seconds
            ):
                center = (
                    builder.start_time
                    + builder.end_time
                ) / 2.0

                half = (
                    config.max_region_duration_seconds
                    / 2.0
                )

                start_time = max(
                    0.0,
                    center - half,
                )

                end_time = (
                    start_time
                    + config.max_region_duration_seconds
                )

        output.append(
            CoarseRegion(
                video_id=builder.video_id,
                start_time=float(
                    start_time
                ),
                end_time=float(
                    end_time
                ),
                score=float(
                    score
                ),
                hits=tuple(hits),
            )
        )

    return output


# ============================================================================
# TEMPORAL GROUPING
# ============================================================================
def _select_diverse_regions(
    regions: list[CoarseRegion],
    limit: int,
    *,
    max_per_video: int = 2,
) -> list[CoarseRegion]:

    if limit <= 0:
        return []

    selected: list[
        CoarseRegion
    ] = []

    counts: dict[
        str,
        int,
    ] = {}

    # ---------------------------------------------------------
    # Pass 1:
    # tối đa max_per_video để tránh một video chiếm hết top-N.
    # ---------------------------------------------------------

    for region in regions:

        count = counts.get(
            region.video_id,
            0,
        )

        if count >= max_per_video:
            continue

        selected.append(
            region
        )

        counts[
            region.video_id
        ] = count + 1

        if len(selected) >= limit:
            return selected

    # ---------------------------------------------------------
    # Pass 2:
    # nếu chưa đủ thì fill lại theo score.
    # ---------------------------------------------------------

    selected_ids = {
        id(region)
        for region in selected
    }

    for region in regions:

        if id(region) in selected_ids:
            continue

        selected.append(
            region
        )

        if len(selected) >= limit:
            break

    return selected

def _gom_vung(
    hits: list[dict[str, Any]],
    config: TRR1Config,
    *,
    video_scores: dict[str, float] | None = None,
) -> list[CoarseRegion]:
    """
    Gộp raw hits thành coarse temporal regions.

    Quy trình:

        raw frame hits
            ↓
        normalize / validate
            ↓
        sort theo video + timestamp
            ↓
        temporal grouping
            ↓
        candidate region filtering
            ↓
        region-level scoring
            ↓
        global score sort
            ↓
        max_regions_per_event

    Khác bản cũ:
        region score KHÔNG còn đơn giản là max(frame score).
    """

    if not hits:
        return []

    normalized: list[
        dict[str, Any]
    ] = []

    for hit in hits:
        if not isinstance(
            hit,
            dict,
        ):
            continue

        video_id = _safe_video_id(hit)
        pts_time = _safe_time(hit)
        score = _safe_score(hit)

        if not video_id:
            continue

        if pts_time is None:
            continue

        if score == float("-inf"):
            continue

        normalized.append(
            {
                **hit,
                "video_id": video_id,
                "pts_time": float(
                    pts_time
                ),
                "score": float(
                    score
                ),
            }
        )

    if not normalized:
        return []

    # Temporal grouping phải dựa trên timestamp,
    # không dựa trên retrieval rank.
    normalized.sort(
        key=lambda hit: (
            str(
                hit["video_id"]
            ),
            float(
                hit["pts_time"]
            ),
            -float(
                hit["score"]
            ),
        )
    )

    builders: list[
        _RegionBuilder
    ] = []

    current: (
        _RegionBuilder | None
    ) = None

    for hit in normalized:
        video_id = str(
            hit["video_id"]
        )

        hit_time = float(
            hit["pts_time"]
        )

        if current is None:
            current = _RegionBuilder(
                video_id=video_id,
                start_time=hit_time,
                end_time=hit_time,
                hits=[hit],
            )
            continue

        if _can_merge(
            current,
            hit,
            config,
        ):
            _append_hit(
                current,
                hit,
            )
            continue

        builders.append(
            current
        )

        current = _RegionBuilder(
            video_id=video_id,
            start_time=hit_time,
            end_time=hit_time,
            hits=[hit],
        )

    if current is not None:
        builders.append(
            current
        )

    regions = _score_regions(
        builders,
        config,
        video_scores=video_scores,
    )

    return _select_diverse_regions(
        regions,
        config.max_regions_per_event,
        max_per_video=2,
    )


# ============================================================================
# RETRIEVER ADAPTER
# ============================================================================


def _default_retriever(
    text: str,
    top_k: int,
    *,
    use_query_expansion: bool = True,
    max_query_variants: int = 2,
) -> list[dict[str, Any]]:
    """
    Production retriever mặc định.

    Dùng:
        Marian
        +
        CLIP-L
        +
        FAISS CLIP-L

    tim_vung_tho() chỉ gọi retriever một lần/event.
    """

    return tim_hit_clip_l(
        text,
        top_k=top_k,
        use_query_expansion=use_query_expansion,
        max_query_variants=max_query_variants,
    )


def _retrieve_hits(
    text: str,
    config: TRR1Config,
    retriever: Callable[
        [str, int],
        list[dict[str, Any]],
    ]
    | None,
) -> list[dict[str, Any]]:
    """Gọi retriever một lần và luôn tôn trọng cấu hình query expansion."""

    if retriever is None:
        hits = _default_retriever(
            text,
            config.top_k,
            use_query_expansion=config.use_query_expansion,
            max_query_variants=config.max_query_variants,
        )
    else:
        hits = retriever(text, config.top_k)

    if hits is None:
        return []

    if isinstance(hits, list):
        return hits

    return list(hits)


# ============================================================================
# PUBLIC API — ONE EVENT
# ============================================================================


def tim_vung_tho(
    event: dict[str, Any],
    *,
    config: TRR1Config | None = None,
    retriever: Callable[
        [str, int],
        list[dict[str, Any]],
    ]
    | None = None,
) -> TRR1Result:
    """
    Tìm coarse regions cho MỘT event.

    Parameters
    ----------
    event:
        Dict bắt buộc có:
            event_id
            text

        relation là optional.

    config:
        TR-R1 config.

    retriever:
        Dependency injection cho unit test.

        Signature:
            retriever(text, top_k) -> list[hit]

    Returns
    -------
    TRR1Result
    """

    if config is None:
        config = TRR1Config()

    event_id, text, relation = _parse_event(
        event
    )

    hits = _retrieve_hits(
        text,
        config,
        retriever,
    )

    regions = _gom_vung(
        hits,
        config=config,
    )

    return TRR1Result(
        event_id=event_id,
        text=text,
        relation=relation,
        regions=tuple(
            regions
        ),
    )


# ============================================================================
# PUBLIC API — MULTIPLE EVENTS
# ============================================================================


def tim_nhieu_su_kien(
    events: Iterable[dict[str, Any]],
    *,
    config: TRR1Config | None = None,
    retriever: Callable[
        [str, int],
        list[dict[str, Any]],
    ]
    | None = None,
) -> tuple[TRR1Result, ...]:
    """
    Tìm coarse regions cho nhiều event.

    Thứ tự output giữ nguyên thứ tự QueryPlan.
    """

    if config is None:
        config = TRR1Config()

    parsed_events = [
        _parse_event(event)
        for event in events
    ]

    hits_by_event = [
        _retrieve_hits(text, config, retriever)
        for _event_id, text, _relation in parsed_events
    ]

    video_scores = _video_consensus_scores(
        hits_by_event,
        rrf_k=config.video_rrf_k,
    )

    results: list[TRR1Result] = []

    for (
        event_id,
        text,
        relation,
    ), hits in zip(parsed_events, hits_by_event):
        regions = _gom_vung(
            hits,
            config=config,
            video_scores=video_scores,
        )

        results.append(
            TRR1Result(
                event_id=event_id,
                text=text,
                relation=relation,
                regions=tuple(regions),
            )
        )

    return tuple(
        results
    )


# ============================================================================
# SERIALIZATION
# ============================================================================


def coarse_region_to_dict(
    region: CoarseRegion,
) -> dict[str, Any]:
    """
    Serialize CoarseRegion.

    Contract CHỈ gồm:

        video_id
        start_time
        end_time
        score
        hits

    Không có frame_idx/best_frame_idx ở top-level.
    """

    return {
        "video_id": region.video_id,
        "start_time": float(
            region.start_time
        ),
        "end_time": float(
            region.end_time
        ),
        "score": float(
            region.score
        ),
        "hits": [
            dict(hit)
            for hit in region.hits
        ],
    }


def trr1_result_to_dict(
    result: TRR1Result,
) -> dict[str, Any]:
    """
    Serialize TRR1Result.
    """

    return {
        "event_id": result.event_id,
        "text": result.text,
        "relation": result.relation,
        "regions": [
            coarse_region_to_dict(
                region
            )
            for region in result.regions
        ],
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "TRR1Config",
    "CoarseRegion",
    "TRR1Result",
    "_build_test_hit",
    "_gom_vung",
    "_video_consensus_scores",
    "coarse_region_to_dict",
    "tim_vung_tho",
    "tim_nhieu_su_kien",
    "trr1_result_to_dict",
]
