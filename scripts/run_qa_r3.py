"""
QA-R3 — Production Candidate Exporter
======================================

Đọc candidate thật từ manifest QA-R1, dùng Rank Normalization của QA-R2,
chạy QA-R3 và ghi payload đầu vào cho QA-R4.

Mặc định:

    D:/aic-data/runs/dev_qa_r1_profile.json
        -> D:/aic-data/runs/qa_r3_candidates.json

Nếu chạy một query bằng ``--query-id``, output là đúng một object:

    {"query_id": "...", "candidates": [...]}

Nếu không truyền ``--query-id``, output là một JSON array gồm các object
có cùng contract trên, mỗi object tương ứng một query QA.

Runner này không sinh mock candidate. Semantic metadata của candidate chỉ
được dùng khi upstream thật sự cung cấp các field entities/actions/
attributes/relations; nếu chưa có, QA-R3 vẫn xếp hạng bằng relevance và
temporal diversity, đồng thời in cảnh báo rõ ràng.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.fusion_engine import (
    CandidateMetadata,
    MultiModalFusionEngine,
    SelectionMethod,
    SemanticQuery,
)
from scripts.score_normalization import NormMethod, normalize_scores


DEFAULT_R1_PATH = Path("D:/aic-data/runs/dev_qa_r1_profile.json")
DEFAULT_N01_PATH = Path("D:/aic-data/runs/n01_semantic_parsing.jsonl")
DEFAULT_OUTPUT_PATH = Path("D:/aic-data/runs/qa_r3_candidates.json")

POOL_KEYS = ("coverage_candidates", "reader_candidates")


class QAInputError(ValueError):
    """Input upstream không đáp ứng contract tối thiểu của runner."""


def _query_id(value: object) -> str:
    result = str(value).strip()
    if not result:
        raise QAInputError("query_id không được rỗng")
    return result


def _finite_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QAInputError(
            f"{field} phải là số hữu hạn, nhận {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise QAInputError(
            f"{field} phải là số hữu hạn, nhận {value!r}"
        )

    return result


def _string_set(value: object, *, field: str) -> set[str]:
    if value is None:
        return set()

    if not isinstance(value, list):
        raise QAInputError(
            f"{field} phải là JSON array"
        )

    result: set[str] = set()

    for item in value:
        label = str(item).strip()

        if label:
            result.add(label)

    return result


def load_r1_query_records(
    path: Path,
) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        raise QAInputError(
            f"Không tìm thấy QA-R1 manifest: {path}"
        )

    try:
        document = json.loads(
            path.read_text(
                encoding="utf-8-sig"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise QAInputError(
            f"Không đọc được QA-R1 manifest {path}: {exc}"
        ) from exc

    records = (
        document.get("query_records")
        if isinstance(document, dict)
        else None
    )

    if not isinstance(records, list):
        raise QAInputError(
            "QA-R1 manifest phải có query_records là JSON array"
        )

    result: dict[str, Mapping[str, Any]] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise QAInputError(
                f"query_records[{index}] phải là JSON object"
            )

        qid = _query_id(
            record.get("query_id", "")
        )

        if qid in result:
            raise QAInputError(
                f"QA-R1 có query_id trùng: {qid}"
            )

        result[qid] = record

    return result


def load_n01_qa_plans(
    path: Path,
) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        raise QAInputError(
            f"Không tìm thấy N01 semantic file: {path}"
        )

    result: dict[str, Mapping[str, Any]] = {}

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:

            for line_number, line in enumerate(
                stream,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise QAInputError(
                        f"N01 JSONL lỗi tại dòng "
                        f"{line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise QAInputError(
                        f"N01 dòng {line_number} "
                        "phải là JSON object"
                    )

                if record.get("task") != "qa":
                    continue

                qid = _query_id(
                    record.get("query_id", "")
                )

                if qid in result:
                    raise QAInputError(
                        f"N01 có query_id QA trùng: {qid}"
                    )

                result[qid] = record

    except OSError as exc:
        raise QAInputError(
            f"Không đọc được N01 semantic file "
            f"{path}: {exc}"
        ) from exc

    if not result:
        raise QAInputError(
            "N01 không chứa query task=qa nào"
        )

    return result


def _pick_candidate_pool(
    record: Mapping[str, Any],
) -> tuple[str, list[Mapping[str, Any]]]:

    for key in POOL_KEYS:
        value = record.get(key)

        if value:
            if (
                not isinstance(value, list)
                or not all(
                    isinstance(item, dict)
                    for item in value
                )
            ):
                raise QAInputError(
                    f"{key} phải là JSON array của object"
                )

            return key, value

    return "none", []


def _candidate_id(
    query_id: str,
    candidate: Mapping[str, Any],
    index: int,
) -> str:

    video_id = str(
        candidate.get("video_id", "")
    ).strip()

    if not video_id:
        raise QAInputError(
            f"Query {query_id}: "
            f"candidate[{index}] thiếu video_id"
        )

    try:
        n_value = int(
            candidate["n"]
        )

        frame_index = int(
            candidate["frame_idx"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:

        raise QAInputError(
            f"Query {query_id}: candidate[{index}] "
            "thiếu n/frame_idx hợp lệ"
        ) from exc

    if frame_index < 0:
        raise QAInputError(
            f"Query {query_id}: candidate[{index}] "
            "có frame_idx âm"
        )

    # query_id + video_id + n + frame_idx phân biệt được các frame
    # ngay cả khi frame_idx bị trùng trong frame map của bộ dữ liệu.
    return (
        f"{query_id}:"
        f"{video_id}:"
        f"{n_value}:"
        f"{frame_index}"
    )


def _relevance(
    candidate: Mapping[str, Any],
    *,
    query_id: str,
    index: int,
) -> float:

    for field in (
        "evidence_score",
        "score_fused",
    ):
        if candidate.get(field) is not None:
            return _finite_float(
                candidate[field],
                field=(
                    f"Query {query_id} "
                    f"candidate[{index}].{field}"
                ),
            )

    raise QAInputError(
        f"Query {query_id}: candidate[{index}] "
        "thiếu evidence_score/score_fused"
    )


def build_query_inputs(
    query_id: str,
    r1_record: Mapping[str, Any],
    semantic_plan: Mapping[str, Any],
) -> tuple[
    dict[str, float],
    dict[str, CandidateMetadata],
    SemanticQuery,
    str,
    bool,
    dict[str, int],
]:

    pool_key, rows = _pick_candidate_pool(
        r1_record
    )

    if not rows:
        raise QAInputError(
            f"Query {query_id}: "
            "QA-R1 không có candidate"
        )

    candidate_ids: list[str] = []
    raw_relevance: list[float] = []

    metadata_map: dict[
        str,
        CandidateMetadata,
    ] = {}

    # Giữ n từ upstream R1 theo candidate_id.
    #
    # Không dùng (video_id, frame_id) làm key vì candidate_id của R3
    # được thiết kế để phân biệt candidate bằng query_id + video_id + n
    # + frame_idx.
    candidate_n_map: dict[str, int] = {}

    has_candidate_semantics = False

    for index, candidate in enumerate(rows):

        if candidate.get(
            "status",
            "ok",
        ) != "ok":
            continue

        cid = _candidate_id(
            query_id,
            candidate,
            index,
        )

        try:
            n_value = int(
                candidate["n"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise QAInputError(
                f"Query {query_id}: candidate[{index}] "
                "thiếu n hợp lệ"
            ) from exc

        if cid in metadata_map:
            raise QAInputError(
                f"Query {query_id}: "
                f"candidate_id trùng: {cid}"
            )

        entities = _string_set(
            candidate.get("entities"),
            field="entities",
        )

        actions = _string_set(
            candidate.get("actions"),
            field="actions",
        )

        attributes = _string_set(
            candidate.get("attributes"),
            field="attributes",
        )

        relations = _string_set(
            candidate.get("relations"),
            field="relations",
        )

        has_candidate_semantics = (
            has_candidate_semantics
            or bool(
                entities
                or actions
                or attributes
                or relations
            )
        )

        try:
            frame_index = int(
                candidate["frame_idx"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise QAInputError(
                f"Query {query_id}: candidate[{index}] "
                "thiếu frame_idx hợp lệ"
            ) from exc

        timestamp = _finite_float(
            candidate.get(
                "pts_time",
                0.0,
            ),
            field=(
                f"Query {query_id} "
                f"candidate[{index}].pts_time"
            ),
        )

        candidate_ids.append(cid)

        raw_relevance.append(
            _relevance(
                candidate,
                query_id=query_id,
                index=index,
            )
        )

        metadata_map[cid] = CandidateMetadata(
            candidate_id=cid,
            video_id=str(
                candidate["video_id"]
            ).strip(),
            frame_index=frame_index,
            timestamp_sec=timestamp,
            entities=entities,
            actions=actions,
            attributes=attributes,
            relations=relations,
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        # CandidateMetadata / FusedCandidate của fusion_engine không
        # có field `n`.
        #
        # Vì vậy provenance `n` được giữ riêng tại R3 exporter,
        # keyed bằng candidate_id.
        # --------------------------------------------------------------

        candidate_n_map[cid] = n_value

    if not candidate_ids:
        raise QAInputError(
            f"Query {query_id}: "
            "không còn candidate status=ok"
        )

    normalized = normalize_scores(
        raw_relevance,
        NormMethod.RANK,
    )

    fused_scores = {
        cid: float(score)
        for cid, score in zip(
            candidate_ids,
            normalized,
            strict=True,
        )
    }

    query = SemanticQuery(
        entities=_string_set(
            semantic_plan.get("entities"),
            field="entities",
        ),
        actions=_string_set(
            semantic_plan.get("actions"),
            field="actions",
        ),
        attributes=_string_set(
            semantic_plan.get("attributes"),
            field="attributes",
        ),
        relations=_string_set(
            semantic_plan.get("relations"),
            field="relations",
        ),
    )

    return (
        fused_scores,
        metadata_map,
        query,
        pool_key,
        has_candidate_semantics,
        candidate_n_map,
    )


def build_r4_payload(
    *,
    query_id: str,
    candidates: Sequence[object],
    candidate_n_map: Mapping[str, int],
) -> dict[str, object]:
    """
    Serialize selected QA-R3 FusedCandidate objects thành contract
    đầu vào QA-R4.

    QA-R4 yêu cầu mỗi candidate có:

        video_id
        frame_id
        n
        score

    `n` không có trong FusedCandidate nên được khôi phục từ
    candidate_n_map bằng candidate_id.
    """

    payload_candidates: list[dict[str, object]] = []

    seen_candidate_ids: set[str] = set()

    for index, candidate in enumerate(candidates):

        candidate_id = str(
            getattr(
                candidate,
                "candidate_id",
                "",
            )
        ).strip()

        if not candidate_id:
            raise QAInputError(
                f"Query {query_id}: "
                f"selected candidate[{index}] "
                "thiếu candidate_id"
            )

        if candidate_id in seen_candidate_ids:
            raise QAInputError(
                f"Query {query_id}: duplicate "
                f"candidate_id: {candidate_id}"
            )

        seen_candidate_ids.add(
            candidate_id
        )

        if candidate_id not in candidate_n_map:
            raise QAInputError(
                f"Query {query_id}: không tìm thấy n "
                f"cho candidate_id={candidate_id!r}"
            )

        try:
            video_id = str(
                candidate.video_id
            ).strip()

            frame_id = int(
                candidate.frame_id
            )

            score = float(
                candidate.selection_score
            )

            n_value = int(
                candidate_n_map[candidate_id]
            )

        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise QAInputError(
                f"Query {query_id}: "
                f"selected candidate[{index}] "
                "không serialize được"
            ) from exc

        if not video_id:
            raise QAInputError(
                f"Query {query_id}: "
                f"selected candidate[{index}] "
                "thiếu video_id"
            )

        if frame_id < 0:
            raise QAInputError(
                f"Query {query_id}: "
                f"selected candidate[{index}] "
                "có frame_id âm"
            )

        payload_candidates.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "n": n_value,
                "score": score,
            }
        )

    return {
        "query_id": query_id,
        "candidates": payload_candidates,
    }


def run_qa_r3(
    *,
    r1_path: Path,
    n01_path: Path,
    output_path: Path,
    query_id: str | None = None,
    top_k: int = 12,
    selection_method: SelectionMethod = SelectionMethod.GREEDY,
) -> tuple[object, dict[str, object]]:

    if top_k <= 0:
        raise QAInputError(
            "top_k phải lớn hơn 0"
        )

    r1_records = load_r1_query_records(
        r1_path
    )

    semantic_plans = load_n01_qa_plans(
        n01_path
    )

    if query_id is not None:

        requested = _query_id(
            query_id
        )

        if requested not in r1_records:
            raise QAInputError(
                f"QA-R1 không có "
                f"query_id={requested}"
            )

        if requested not in semantic_plans:
            raise QAInputError(
                f"N01 không có QA "
                f"query_id={requested}"
            )

        query_ids: Sequence[str] = (
            requested,
        )

    else:

        query_ids = sorted(
            set(r1_records).intersection(
                semantic_plans
            )
        )

    if not query_ids:
        raise QAInputError(
            "QA-R1 và N01 không có "
            "query QA giao nhau"
        )

    engine = MultiModalFusionEngine()

    payloads: list[
        dict[str, object]
    ] = []

    pool_counts: dict[str, int] = {}

    queries_without_candidate_semantics = 0

    for qid in query_ids:

        (
            fused_scores,
            metadata_map,
            query,
            pool_key,
            has_semantics,
            candidate_n_map,
        ) = build_query_inputs(
            qid,
            r1_records[qid],
            semantic_plans[qid],
        )

        pool_counts[pool_key] = (
            pool_counts.get(
                pool_key,
                0,
            )
            + 1
        )

        if not has_semantics:
            queries_without_candidate_semantics += 1

        # --------------------------------------------------------------
        # Chạy selection trực tiếp để giữ lại
        # FusedCandidate.candidate_id.
        #
        # IMPORTANT:
        # fusion_engine.select_top_k() có signature:
        #
        #   select_top_k(
        #       fused_scores,
        #       metadata_map,
        #       query,
        #       *,
        #       top_k,
        #       selection_method,
        #   )
        #
        # Nó KHÔNG nhận query_id.
        #
        # candidate_id được FusedCandidate giữ nguyên, nên sau selection
        # ta dùng candidate_id để khôi phục n từ candidate_n_map.
        # --------------------------------------------------------------

        selected = engine.select_top_k(
            fused_scores=fused_scores,
            metadata_map=metadata_map,
            query=query,
            top_k=min(
                top_k,
                len(fused_scores),
            ),
            selection_method=selection_method,
        )

        payload = build_r4_payload(
            query_id=qid,
            candidates=selected,
            candidate_n_map=candidate_n_map,
        )

        candidates = payload.get(
            "candidates"
        )

        if not isinstance(
            candidates,
            list,
        ):
            raise QAInputError(
                f"Query {qid}: R3 payload "
                "không có candidates hợp lệ"
            )

        # build_r4_payload() đã serialize theo đúng thứ tự selected.
        # Không cần map lại bằng (video_id, frame_id), vì candidate_id
        # là identity chính xác của candidate.
        if len(candidates) != len(selected):
            raise QAInputError(
                f"Query {qid}: R3 payload count mismatch "
                f"({len(candidates)} != {len(selected)})"
            )

        payloads.append(
            payload
        )

    document: object = (
        payloads[0]
        if query_id is not None
        else payloads
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Atomic write.
    # --------------------------------------------------------------

    temp_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )

    temp_path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temp_path.replace(
        output_path
    )

    summary: dict[str, object] = {
        "queries": len(payloads),
        "candidates": sum(
            len(item["candidates"])
            for item in payloads
        ),
        "pool_counts": pool_counts,
        "queries_without_candidate_semantics": (
            queries_without_candidate_semantics
        ),
        "output_path": str(
            output_path
        ),
    }

    return document, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Xuất candidate thật QA-R3 "
            "sang contract đầu vào QA-R4."
        )
    )

    parser.add_argument(
        "--r1",
        type=Path,
        default=DEFAULT_R1_PATH,
    )

    parser.add_argument(
        "--n01",
        type=Path,
        default=DEFAULT_N01_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--query-id",
        help=(
            "Chỉ chạy một query và "
            "xuất một JSON object"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--method",
        choices=[
            method.value
            for method in SelectionMethod
        ],
        default=SelectionMethod.GREEDY.value,
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "In payload candidate ra terminal "
            "sau khi ghi file"
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:

    args = build_parser().parse_args(
        argv
    )

    try:

        document, summary = run_qa_r3(
            r1_path=args.r1,
            n01_path=args.n01,
            output_path=args.output,
            query_id=args.query_id,
            top_k=args.top_k,
            selection_method=SelectionMethod(
                args.method
            ),
        )

    except (
        QAInputError,
        ValueError,
        OSError,
    ) as exc:

        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )

        return 1

    print(
        "=" * 72
    )

    print(
        "QA-R3 PRODUCTION CANDIDATE EXPORT"
    )

    print(
        "=" * 72
    )

    print(
        f"Queries          : "
        f"{summary['queries']}"
    )

    print(
        f"Candidates       : "
        f"{summary['candidates']}"
    )

    print(
        f"Candidate pools  : "
        f"{summary['pool_counts']}"
    )

    print(
        f"Output           : "
        f"{summary['output_path']}"
    )

    if summary[
        "queries_without_candidate_semantics"
    ]:

        print(
            "[WARN] Upstream chưa có semantic metadata "
            "cho candidate ở "
            f"{summary['queries_without_candidate_semantics']} "
            "query; R3 đang dùng relevance + "
            "temporal diversity."
        )

    if summary[
        "pool_counts"
    ].get("reader_candidates"):

        print(
            "[WARN] QA-R1 manifest hiện chỉ lưu "
            "reader_candidates (Top-12). "
            "Muốn R3 chọn từ Top-50, QA-R1 phải lưu "
            "coverage_candidates."
        )

    if args.show:
        print(
            json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
