"""INT-01 — review một vertical slice Q&A qua ba contract độc lập.

==============================================================================
[CHỐT POLICY: SEMANTIC_K_HINT VÀ READER_K]
1. semantic_k_hint (100-500): Là Retrieval-K. Chỉ dùng làm gợi ý ngân sách 
   ứng viên cho khâu tìm kiếm (Retrieval).
2. Tuyệt đối KHÔNG dùng semantic_k_hint trực tiếp làm Reader-K. 
   Không gửi trực tiếp 100-500 frame vào VLM để tránh quá tải/timeout.
3. Nếu khâu QA-R4 cần truyền danh sách rút gọn cho Reader (QA-V2), phải tạo 
   field/policy riêng (ví dụ `reader_k = 1` hoặc `reader_k = 5`). Khâu Retrieval 
   có trách nhiệm rerank và cắt ngọn danh sách xuống đúng bằng reader_k.
==============================================================================

Luồng được kiểm:

    QA-S1 QueryPlan -> QA-R4/QA-R1 reader candidates -> QA-V2 reader

INT-01 là task kiến trúc/nghiệm thu. Script này không sửa thuật toán của các
owner. Mọi chênh lệch contract được ghi thành issue có ``owner`` rõ ràng trong
report JSON để trả đúng nhánh chịu trách nhiệm.

Mặc định Qwen chạy offline từ cache. Có thể dùng ``--reader skip`` để chỉ audit
handoff khi máy chưa chạy được model; chế độ đó không được tính là end-to-end
reader pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = "1.1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


@dataclass(slots=True, frozen=True)
class IntegrationIssue:
    severity: str
    owner: str
    stage: str
    code: str
    message: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _issue(
    severity: str,
    owner: str,
    stage: str,
    code: str,
    message: str,
    **evidence: Any,
) -> IntegrationIssue:
    return IntegrationIssue(
        severity=severity,
        owner=owner,
        stage=stage,
        code=code,
        message=message,
        evidence=evidence,
    )


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    if isinstance(model, Mapping):
        return dict(model)
    raise TypeError(f"Không thể chuyển {type(model).__name__} thành record")


def normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_qa_query(path: Path, query_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp câu hỏi: {path}")

    expected = str(query_id)
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON lỗi ở dòng {line_no}: {exc}") from exc
            if str(row.get("id", "")) == expected:
                if row.get("loai_truy_van") != "hoi_dap":
                    raise ValueError(
                        f"Query {expected} không phải hoi_dap: "
                        f"{row.get('loai_truy_van')!r}"
                    )
                if not str(row.get("cau_hoi", "")).strip():
                    raise ValueError(f"Query {expected} thiếu cau_hoi")
                return row

    raise KeyError(f"Không có query_id={expected!r} trong {path}")


def parse_query_plan(query: Mapping[str, Any]) -> dict[str, Any]:
    # Import tại đây để lệnh --help và unit test contract không bị phụ thuộc
    # model/retrieval nặng.
    from aic2026.semantic.parser import RuleBasedParser

    parser = RuleBasedParser()
    plan = parser.parse_qa(str(query["id"]), str(query["cau_hoi"]))
    return _model_dump(plan)


def load_qa_r1_profile(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy QA-R1 profile: {path}")
    with path.open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("QA-R1 profile phải là JSON object")
    return data


def find_qa_r1_record(
    profile: Mapping[str, Any],
    query_id: str,
) -> dict[str, Any]:
    records = profile.get("query_records")
    if not isinstance(records, list):
        raise ValueError("QA-R1 profile thiếu list query_records")

    expected = str(query_id)
    matches = [
        row
        for row in records
        if isinstance(row, dict) and str(row.get("query_id", "")) == expected
    ]
    if not matches:
        raise KeyError(f"QA-R1 không có query_record cho query_id={expected!r}")
    if len(matches) > 1:
        raise ValueError(f"QA-R1 có {len(matches)} record trùng query_id={expected!r}")
    return dict(matches[0])


def audit_query_plan(plan: Mapping[str, Any]) -> list[IntegrationIssue]:
    issues: list[IntegrationIssue] = []
    required = {
        "schema_version",
        "query_id",
        "task",
        "query_text",
        "intent",
        "answer_type",
        "preferred_modalities",
        "queries",
        "uncertainty",
        "status",
        "error",
    }
    missing = sorted(required - set(plan))
    if missing:
        issues.append(
            _issue(
                "P0",
                "Nghi (QA-S1)",
                "QueryPlan",
                "QA_S1_MISSING_FIELDS",
                "QueryPlan thiếu trường contract bắt buộc.",
                missing=missing,
            )
        )

    if plan.get("schema_version") != SCHEMA_VERSION:
        issues.append(
            _issue(
                "P0",
                "Nghi (QA-S1)",
                "QueryPlan",
                "QA_S1_SCHEMA_VERSION",
                "QueryPlan không dùng schema_version 1.1.",
                observed=plan.get("schema_version"),
            )
        )

    error = plan.get("error")
    if error is not None and not isinstance(error, dict):
        issues.append(
            _issue(
                "P1",
                "Nghi (QA-S1)",
                "QueryPlan",
                "QA_S1_ERROR_SHAPE",
                "Contract chung yêu cầu error là null hoặc object code/message.",
                observed_type=type(error).__name__,
            )
        )

    modalities = plan.get("preferred_modalities")
    if not isinstance(modalities, dict) or not modalities:
        issues.append(
            _issue(
                "P0",
                "Nghi (QA-S1)",
                "QueryPlan",
                "QA_S1_MODALITIES_INVALID",
                "preferred_modalities phải là object không rỗng.",
                observed=modalities,
            )
        )
    else:
        values = list(modalities.values())
        numeric = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in values
        )
        if not numeric or not math.isclose(
            sum(float(value) for value in values if isinstance(value, (int, float))),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            issues.append(
                _issue(
                    "P0",
                    "Nghi (QA-S1)",
                    "QueryPlan",
                    "QA_S1_MODALITY_SUM",
                    "Trọng số modality phải hữu hạn, không âm và có tổng bằng 1.",
                    observed=modalities,
                )
            )
    return issues


def audit_candidate(
    candidate: Mapping[str, Any],
    query_id: str,
) -> list[IntegrationIssue]:
    issues: list[IntegrationIssue] = []
    required = {
        "schema_version",
        "query_id",
        "event_id",
        "video_id",
        "n",
        "frame_idx",
        "pts_time",
        "stage",
        "source_hits",
        "source_ranks",
        "scores",
        "score_fused",
        "rank_final",
        "status",
        "error",
    }
    missing = sorted(required - set(candidate))
    if missing:
        issues.append(
            _issue(
                "P0",
                "Thi (QA-R1)",
                "CandidateRecord",
                "QA_R1_MISSING_FIELDS",
                "CandidateRecord thiếu trường cần để reader truy vết bằng chứng.",
                missing=missing,
            )
        )

    if str(candidate.get("query_id", "")) != str(query_id):
        issues.append(
            _issue(
                "P0",
                "Thi (QA-R1)",
                "CandidateRecord",
                "QA_R1_QUERY_ID_MISMATCH",
                "CandidateRecord không giữ nguyên query_id từ QueryPlan.",
                expected=str(query_id),
                observed=candidate.get("query_id"),
            )
        )

    for field_name in ("n", "frame_idx", "rank_final"):
        value = candidate.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool):
            issues.append(
                _issue(
                    "P0",
                    "Thi (QA-R1)",
                    "CandidateRecord",
                    "QA_R1_INTEGER_FIELD",
                    f"{field_name} phải là integer thật.",
                    field=field_name,
                    observed=value,
                )
            )

    if not candidate.get("source_hits") or not candidate.get("source_ranks"):
        issues.append(
            _issue(
                "P0",
                "Thi (QA-R1)",
                "CandidateRecord",
                "QA_R1_PROVENANCE_EMPTY",
                "CandidateRecord phải giữ source_hits và source_ranks.",
                source_hits=candidate.get("source_hits"),
                source_ranks=candidate.get("source_ranks"),
            )
        )

    optional_contract_fields = [
        name for name in ("semantic_coverage", "window") if name not in candidate
    ]
    if optional_contract_fields:
        issues.append(
            _issue(
                "P1",
                "Thi (QA-R1)",
                "CandidateRecord",
                "QA_R1_CONTRACT_FIELDS_NOT_EXPORTED",
                "CandidateRecord chưa xuất đủ field của contract kiến trúc.",
                missing=optional_contract_fields,
            )
        )

    error = candidate.get("error")
    if error is not None and not isinstance(error, dict):
        issues.append(
            _issue(
                "P1",
                "Thi (QA-R1)",
                "CandidateRecord",
                "QA_R1_ERROR_SHAPE",
                "Contract chung yêu cầu error là null hoặc object code/message.",
                observed_type=type(error).__name__,
            )
        )
    return issues


def audit_qa_r1_handoff(
    plan: Mapping[str, Any],
    profile: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[IntegrationIssue]:
    issues: list[IntegrationIssue] = []

    linkage_fields = {
        "query_plan",
        "query_plan_schema_version",
        "query_plan_digest",
        "semantic_k_hint_applied",
        "routing_weights_applied",
    }
    if not linkage_fields.intersection(record):
        issues.append(
            _issue(
                "P0",
                "Thi (QA-R1)",
                "QueryPlan -> CandidateRecord",
                "QA_R1_QUERY_PLAN_NOT_CONSUMED",
                "QA-R1 không ghi bằng chứng đã nhận/applied QueryPlan; INT-01 chỉ có thể nối theo query_id.",
                expected_any_of=sorted(linkage_fields),
                query_id=plan.get("query_id"),
            )
        )

    modalities = plan.get("preferred_modalities")
    if isinstance(modalities, dict) and modalities:
        dominant = max(modalities, key=lambda name: float(modalities[name]))
        retrieval_sources = set()
        rrf = profile.get("rrf")
        if isinstance(rrf, dict):
            retrieval_sources.update(
                key.removeprefix("weight_")
                for key in rrf
                if str(key).startswith("weight_")
            )
        if dominant not in retrieval_sources and dominant in {"ocr", "asr", "caption"}:
            unavailable = record.get("unavailable_modalities", [])
            if isinstance(unavailable, list) and dominant in unavailable:
                issues.append(
                    _issue(
                        "P1",
                        "Thi (QA-R1)",
                        "QueryPlan -> CandidateRecord",
                        "QA_R1_MODALITY_UNAVAILABLE",
                        "QA-R1 đã nhận QueryPlan nhưng chưa có index cho modality chính.",
                        dominant_modality=dominant,
                        retrieval_sources=sorted(retrieval_sources),
                    )
                )
            else:
                issues.append(
                    _issue(
                        "P0",
                        "Thi (QA-R1)",
                        "QueryPlan -> CandidateRecord",
                        "QA_R1_ROUTING_NOT_APPLIED",
                        "Modality chính của QueryPlan không xuất hiện trong cấu hình retrieval QA-R1.",
                        dominant_modality=dominant,
                        retrieval_sources=sorted(retrieval_sources),
                    )
                )
    return issues


def reader_candidates(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    # [ADAPTER QA-R4]: Đọc đúng field "candidates" mới, dự phòng "reader_candidates" của R1
    candidates = record.get("candidates") or record.get("reader_candidates")
    if not isinstance(candidates, list):
        raise ValueError("QA-R4/R1 query_record thiếu list candidates (hoặc reader_candidates)")
    return [dict(item) for item in candidates if isinstance(item, dict)]


def choose_candidate_image(
    candidates: Iterable[Mapping[str, Any]],
    image_resolver: Callable[[str, int], Path],
) -> tuple[dict[str, Any] | None, Path | None]:
    for candidate in candidates:
        video_id = str(candidate.get("video_id", ""))
        n = candidate.get("n")
        
        # [ADAPTER QA-R4]: Dùng đúng video_id + n để resolve ảnh
        if not video_id or not isinstance(n, int) or isinstance(n, bool):
            continue
            
        image_path = Path(image_resolver(video_id, n))
        if image_path.is_file():
            return dict(candidate), image_path
            
    return None, None


def adapt_reader_output(
    raw_output: Any,
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    latency_ms: float,
    model_id: str,
) -> tuple[dict[str, Any], list[IntegrationIssue]]:
    issues: list[IntegrationIssue] = []

    if isinstance(raw_output, Mapping):
        answer_raw = str(raw_output.get("answer_raw", raw_output.get("answer", ""))).strip()
        confidence_raw = raw_output.get("confidence")
        confidence_method = raw_output.get("confidence_method")
        has_confidence = (
            isinstance(confidence_raw, (int, float))
            and not isinstance(confidence_raw, bool)
            and math.isfinite(float(confidence_raw))
            and 0.0 <= float(confidence_raw) <= 1.0
        )
    else:
        answer_raw = str(raw_output).strip()
        confidence_raw = None
        confidence_method = None
        has_confidence = False

    if not answer_raw:
        issues.append(
            _issue(
                "P0",
                "Nguyên (QA-V2)",
                "reader -> EvidenceAnswer",
                "QA_V2_EMPTY_ANSWER",
                "Reader trả answer rỗng; contract production cấm answer rỗng.",
                model_id=model_id,
            )
        )
        answer_raw = ""

    if not has_confidence:
        issues.append(
            _issue(
                "P0",
                "Nguyên (QA-V2)",
                "reader -> EvidenceAnswer",
                "QA_V2_NO_CONFIDENCE",
                "QA-V2 hiện trả text oracle, chưa trả confidence để tạo EvidenceAnswer status=ok.",
                raw_output_type=type(raw_output).__name__,
            )
        )
    elif not confidence_method:
        issues.append(
            _issue(
                "P1",
                "Nguyên (QA-V2)",
                "reader -> EvidenceAnswer",
                "QA_V2_CONFIDENCE_METHOD_MISSING",
                "Reader có confidence nhưng chưa khai báo phương pháp tính.",
                model_id=model_id,
            )
        )

    if answer_raw and has_confidence:
        answer = answer_raw
        confidence = float(confidence_raw)
        status = "ok"
        fallback_used = False
        error = None
    else:
        # Không bịa confidence. Khi adapter thiếu contract từ owner reader, giữ
        # raw evidence nhưng xuất fallback hợp lệ theo schema chung.
        answer = "khong ro"
        confidence = 0.0
        status = "fallback"
        fallback_used = True
        error = {
            "code": "QA_V2_OUTPUT_INCOMPLETE",
            "message": "Reader chưa cung cấp answer + confidence hợp lệ.",
        }

    evidence = [
        {
            "modality": "vlm",
            "value": answer_raw,
            "confidence": float(confidence_raw) if has_confidence else None,
        },
        {
            "modality": "retrieval",
            "value": {
                "source_hits": list(candidate.get("source_hits", [])),
                "source_ranks": dict(candidate.get("source_ranks", {})),
                "scores": dict(candidate.get("scores", {})),
                "rank_final": candidate.get("rank_final"),
            },
            "confidence": candidate.get("score_fused"),
        },
    ]

    result = {
        "schema_version": SCHEMA_VERSION,
        "query_id": str(plan.get("query_id", "")),
        "answer": answer,
        "answer_raw": answer_raw,
        "answer_normalized": normalize_answer(answer),
        "confidence": confidence,
        "confidence_method": confidence_method,
        "answer_type": str(plan.get("answer_type", "short_text")),
        "video_id": str(candidate.get("video_id", "")),
        "frame_idx": candidate.get("frame_idx"),
        "model_id": model_id,
        "prompt_version": "qa_vi_1.1",
        "latency_ms": round(float(latency_ms), 3),
        "evidence": evidence,
        "status": status,
        "fallback_used": fallback_used,
        "error": error,
    }
    return result, issues


def summarize_issues(issues: Iterable[IntegrationIssue]) -> dict[str, Any]:
    issue_list = list(issues)
    by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in issue_list:
        by_owner[item.owner].append(item.to_dict())
    return {
        "total": len(issue_list),
        "p0": sum(item.severity == "P0" for item in issue_list),
        "p1": sum(item.severity == "P1" for item in issue_list),
        "by_owner": dict(sorted(by_owner.items())),
    }


def deduplicate_issues(issues: Iterable[IntegrationIssue]) -> list[IntegrationIssue]:
    """Giữ report gọn khi cùng lỗi schema lặp lại ở cả 12 candidate."""
    result: list[IntegrationIssue] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for item in issues:
        evidence_key = json.dumps(
            item.evidence,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        key = (item.severity, item.owner, item.stage, item.code, evidence_key)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def review_vertical_slice(
    query: Mapping[str, Any],
    profile: Mapping[str, Any],
    image_resolver: Callable[[str, int], Path],
    reader: Callable[[Path, str], Any] | None,
    model_id: str,
) -> dict[str, Any]:
    plan = parse_query_plan(query)
    record = find_qa_r1_record(profile, str(query["id"]))
    candidates = reader_candidates(record)

    issues: list[IntegrationIssue] = []
    issues.extend(audit_query_plan(plan))
    issues.extend(audit_qa_r1_handoff(plan, profile, record))
    for candidate in candidates:
        issues.extend(audit_candidate(candidate, str(query["id"])))

    selected, image_path = choose_candidate_image(candidates, image_resolver)
    evidence_answer: dict[str, Any] | None = None
    reader_executed = False

    if selected is None or image_path is None:
        issues.append(
            _issue(
                "P0",
                "Nguyên (N-03/QA-V2)",
                "CandidateRecord -> reader",
                "INT_01_NO_CANDIDATE_IMAGE",
                "Không reader candidate nào ánh xạ tới ảnh keyframe local.",
                candidate_count=len(candidates),
                videos=sorted({str(item.get("video_id", "")) for item in candidates}),
            )
        )
    elif reader is None:
        issues.append(
            _issue(
                "P0",
                "Ngân (INT-01)",
                "CandidateRecord -> reader",
                "INT_01_READER_SKIPPED",
                "Chế độ audit đã bỏ qua model; chưa đạt vertical slice end-to-end.",
                image=str(image_path),
            )
        )
    else:
        started = time.perf_counter()
        raw_output = reader(image_path, str(plan["query_text"]))
        latency_ms = (time.perf_counter() - started) * 1000.0
        reader_executed = True
        evidence_answer, reader_issues = adapt_reader_output(
            raw_output=raw_output,
            plan=plan,
            candidate=selected,
            latency_ms=latency_ms,
            model_id=model_id,
        )
        issues.extend(reader_issues)

    issues = deduplicate_issues(issues)
    summary = summarize_issues(issues)
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "INT-01",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_id": str(query["id"]),
        "question": str(query["cau_hoi"]),
        "status": "pass" if summary["p0"] == 0 and reader_executed else "reviewed_with_blockers",
        "reader_executed": reader_executed,
        "query_plan": plan,
        "qa_r1_record_summary": {
            key: value
            for key, value in record.items()
            if key != "reader_candidates"
        },
        "candidate_count": len(candidates),
        "selected_candidate": selected,
        "selected_image": str(image_path) if image_path is not None else None,
        "evidence_answer": evidence_answer,
        "issues": [item.to_dict() for item in issues],
        "issue_summary": summary,
        "acceptance": {
            "query_plan_created": bool(plan),
            "candidate_provenance_present": bool(
                selected and selected.get("source_hits") and selected.get("source_ranks")
            ),
            "reader_executed": reader_executed,
            "evidence_answer_schema_valid": bool(
                evidence_answer
                and evidence_answer.get("schema_version") == SCHEMA_VERSION
                and evidence_answer.get("answer")
                and isinstance(evidence_answer.get("frame_idx"), int)
            ),
            "no_p0_blocker": summary["p0"] == 0,
        },
    }


def build_qwen_reader(
    model_name: str,
    max_new_tokens: int,
) -> Callable[[Path, str], dict[str, Any]]:
    # Import đúng implementation QA-V2, không copy model code sang INT-01.
    from scripts.qwen_oracle import QwenReader, tao_prompt

    qwen = QwenReader(model_name, max_new_tokens=max_new_tokens)
    qwen.nap()

    def call(image_path: Path, question: str) -> dict[str, Any]:
        return qwen.hoi_co_confidence(image_path, tao_prompt(question))

    return call


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="INT-01 — review QueryPlan -> candidates -> QA-V2 reader"
    )
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--qa-r1-profile", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reader", choices=("qwen", "skip"), default="qwen")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--fail-on-blocker", action="store_true")
    args = parser.parse_args()

    # paths.py là nguồn duy nhất biết DATA_ROOT. Import sau argparse để --help
    # vẫn dùng được khi người mới chưa tạo .env.
    from src.aic2026.paths import DEV_QUERIES_PATH, RUNS_DIR, keyframe_image

    questions_path = args.questions or DEV_QUERIES_PATH
    profile_path = args.qa_r1_profile or (RUNS_DIR / "dev_qa_r1_profile.json")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    output_path = args.output or (
        RUNS_DIR / f"{timestamp}_INT-01" / "report" / "int_01_vertical_slice.json"
    )

    query = load_qa_query(questions_path, args.query_id)
    profile = load_qa_r1_profile(profile_path)
    reader = None
    if args.reader == "qwen":
        reader = build_qwen_reader(args.model, args.max_new_tokens)

    report = review_vertical_slice(
        query=query,
        profile=profile,
        image_resolver=keyframe_image,
        reader=reader,
        model_id=args.model if reader is not None else "not_executed",
    )
    write_report(report, output_path)

    summary = report["issue_summary"]
    print("=" * 76)
    print("INT-01 — QUERYPLAN -> CANDIDATES -> READER")
    print("=" * 76)
    print(f"Query              : {report['query_id']}")
    print(f"Candidates         : {report['candidate_count']}")
    print(f"Reader executed    : {report['reader_executed']}")
    print(f"P0 / P1            : {summary['p0']} / {summary['p1']}")
    print(f"Status             : {report['status']}")
    print(f"Report             : {output_path}")
    for owner, owner_issues in summary["by_owner"].items():
        codes = ", ".join(item["code"] for item in owner_issues)
        print(f"- {owner}: {codes}")

    if args.fail_on_blocker and summary["p0"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
