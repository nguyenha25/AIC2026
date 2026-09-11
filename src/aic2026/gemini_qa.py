"""Gemini cloud reader for AIC 2026 Q&A.

The cloud path uses only Python's standard library, so the pinned Windows/CPU
environment stays unchanged.  Every answer remains attached to the exact row
(video_id, frame_idx) that produced it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


DEFAULT_MODEL = "gemini-3.6-flash"
PROMPT_VERSION = "aic2026-gemini-qa-v1"
FALLBACK_ANSWER = "khong ro"
TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


class GeminiQAError(RuntimeError):
    """A safe, user-displayable failure that contains no API key."""


class GeminiHTTPError(GeminiQAError):
    def __init__(self, status: int, message: str):
        self.status = int(status)
        super().__init__(f"Gemini HTTP {self.status}: {message[:400]}")


@dataclass(frozen=True)
class GeminiAssessment:
    row_index: int
    candidate_id: int
    relevance: float
    answer: str
    confidence: float
    evidence: str


Transport = Callable[[str, Mapping[str, str], dict[str, Any], float], dict[str, Any]]
ImagePathResolver = Callable[[str, int], Path]
EvidenceProvider = Callable[[str, int, float], tuple[str, str]]


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _short_answer(value: Any, limit: int = 80) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return FALLBACK_ANSWER
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(" ", 1)[0]
    return (shortened or text[:limit]).strip(" ,.;:!?-") or FALLBACK_ANSWER


def _default_image_path(video_id: str, n: int) -> Path:
    """
    Gemini ưu tiên ảnh keyframe gốc.
    Nếu máy chưa tải shard keyframe thì dùng thumbnail đã có sẵn.
    """
    from .paths import keyframe_image, thumbnail_image

    keyframe = keyframe_image(video_id, n)
    if keyframe.is_file():
        return keyframe

    thumbnail = thumbnail_image(video_id, n)
    if thumbnail.is_file():
        return thumbnail

    # Trả keyframe path để caller có thể báo thiếu ảnh như trước.
    return keyframe


def _default_evidence(video_id: str, n: int, pts_time: float) -> tuple[str, str]:
    """Load concise OCR and ASR context for a candidate."""
    ocr_text = ""
    asr_text = ""
    try:
        from .qa_answer import _doc_ocr_cua_khung

        boxes = sorted(
            _doc_ocr_cua_khung(video_id, n),
            key=lambda box: float(box.get("conf", 0.0)),
            reverse=True,
        )
        seen: set[str] = set()
        snippets: list[str] = []
        for box in boxes:
            text = " ".join(str(box.get("text", "")).split())
            key = text.casefold()
            if text and key not in seen and float(box.get("conf", 0.0)) >= 0.25:
                seen.add(key)
                snippets.append(text)
        ocr_text = " | ".join(snippets)[:700]
    except Exception:
        pass

    try:
        from .paths import asr_file

        path = asr_file(video_id)
        nearby: list[tuple[float, str]] = []
        if path.is_file():
            for line in path.open("r", encoding="utf-8-sig"):
                try:
                    item = json.loads(line)
                    start = float(item.get("start", 0.0))
                    end = float(item.get("end", start))
                    text = " ".join(str(item.get("text", "")).split())
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if text and start - 8.0 <= pts_time <= end + 8.0:
                    distance = 0.0 if start <= pts_time <= end else min(
                        abs(pts_time - start), abs(pts_time - end)
                    )
                    nearby.append((distance, text))
        nearby.sort(key=lambda item: item[0])
        asr_text = " ".join(text for _, text in nearby[:5])[:1000]
    except Exception:
        pass
    return ocr_text, asr_text


def select_candidate_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_images: int = 12,
    max_videos: int = 4,
    max_per_video: int = 3,
    scan_limit: int = 50,
) -> list[int]:
    """Choose up to three nearby frames from each of four ranked videos."""
    if max_images <= 0 or not rows:
        return []
    visible = list(rows)[: max(1, int(scan_limit))]
    anchor_videos: list[str] = []
    for row in visible:
        video_id = str(row.get("video_id", ""))
        if video_id and video_id not in anchor_videos:
            anchor_videos.append(video_id)
        if len(anchor_videos) >= max(1, int(max_videos)):
            break

    selected: list[int] = []
    per_video: dict[str, int] = {}
    for index, row in enumerate(visible):
        video_id = str(row.get("video_id", ""))
        if video_id not in anchor_videos:
            continue
        if per_video.get(video_id, 0) >= max(1, int(max_per_video)):
            continue
        selected.append(index)
        per_video[video_id] = per_video.get(video_id, 0) + 1
        if len(selected) >= int(max_images):
            return selected

    already = set(selected)
    for index in range(len(visible)):
        if index not in already:
            selected.append(index)
        if len(selected) >= int(max_images):
            break
    return selected


def _http_transport(
    url: str,
    headers: Mapping[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlrequest.urlopen(req, timeout=float(timeout_seconds)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc.reason)
        raise GeminiHTTPError(exc.code, detail) from None
    except (urlerror.URLError, TimeoutError, socket.timeout) as exc:
        raise GeminiQAError(f"Gemini network error: {exc}") from None
    except json.JSONDecodeError as exc:
        raise GeminiQAError(f"Gemini returned invalid HTTP JSON: {exc}") from None


class GeminiQAReader:
    """One-request multimodal reader with local cache and bounded retries."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_images: int | None = None,
        timeout_seconds: float | None = None,
        retries: int | None = None,
        cache_dir: Path | str | None = None,
        transport: Transport | None = None,
        image_path_resolver: ImagePathResolver | None = None,
        evidence_provider: EvidenceProvider | None = None,
    ):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv("AIC_GEMINI_MODEL", DEFAULT_MODEL)).strip()
        self.model = self.model or DEFAULT_MODEL
        self.max_images = int(max_images or os.getenv("AIC_GEMINI_MAX_IMAGES", "12"))
        self.timeout_seconds = float(
            timeout_seconds or os.getenv("AIC_GEMINI_TIMEOUT_SECONDS", "25")
        )
        self.retries = int(
            retries if retries is not None else os.getenv("AIC_GEMINI_RETRIES", "2")
        )
        root = Path(os.getenv("DATA_ROOT", "D:/aic-data"))
        configured_cache = os.getenv("AIC_GEMINI_CACHE_DIR", "").strip()
        self.cache_dir = Path(
            cache_dir or configured_cache or root / "runs" / "gemini_qa_cache"
        )
        self.transport = transport or _http_transport
        self.image_path_resolver = image_path_resolver or _default_image_path
        self.evidence_provider = evidence_provider or _default_evidence

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _cache_key(self, question: str, candidates: Sequence[dict[str, Any]]) -> str:
        identity = {
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "question": question,
            "candidates": [
                {
                    "video_id": item["video_id"],
                    "n": item["n"],
                    "frame_idx": item["frame_idx"],
                    "image_size": item["image_path"].stat().st_size,
                    "image_mtime_ns": item["image_path"].stat().st_mtime_ns,
                }
                for item in candidates
            ],
        }
        raw = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, payload: dict[str, Any]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{key}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError:
            # A cache failure must not break the competition UI.
            return

    def _prepare_candidates(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        indices = select_candidate_indices(rows, max_images=self.max_images)
        prepared: list[dict[str, Any]] = []
        total_bytes = 0
        max_bytes = 18 * 1024 * 1024  # Inline request limit is 20 MB.

        for candidate_id, row_index in enumerate(indices):
            row = rows[row_index]
            hit = row.get("hit")
            video_id = str(row.get("video_id") or getattr(hit, "video_id", ""))
            n = _safe_int(getattr(hit, "n", row.get("n", -1)))
            frame_values = row.get("frame_ids") or [getattr(hit, "frame_idx", -1)]
            frame_idx = _safe_int(frame_values[0] if frame_values else -1)
            pts_time = float(getattr(hit, "pts_time", row.get("pts_time", 0.0)))
            if not video_id or n < 0 or frame_idx < 0:
                continue

            image_path = self.image_path_resolver(video_id, n)
            if not image_path.is_file():
                continue
            size = image_path.stat().st_size
            if prepared and total_bytes + size > max_bytes:
                break
            total_bytes += size
            ocr_text, asr_text = self.evidence_provider(video_id, n, pts_time)
            prepared.append({
                "candidate_id": candidate_id,
                "row_index": row_index,
                "video_id": video_id,
                "n": n,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "image_path": image_path,
                "ocr": ocr_text,
                "asr": asr_text,
            })
        return prepared

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "candidate_id": {"type": "INTEGER"},
                            "relevance": {"type": "NUMBER"},
                            "answer": {"type": "STRING"},
                            "confidence": {"type": "NUMBER"},
                            "evidence": {"type": "STRING"},
                        },
                        "required": [
                            "candidate_id", "relevance", "answer",
                            "confidence", "evidence",
                        ],
                    },
                }
            },
            "required": ["items"],
        }

    def _build_payload(
        self, question: str, candidates: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        prompt = (
            "Bạn đang xử lý Video Question Answering cho AIC 2026.\n"
            "Mỗi candidate_id là một ứng viên độc lập. Đánh giá đúng ảnh gắn "
            "ngay sau nhãn của candidate đó. Không lấy đáp án của candidate "
            "này gắn sang candidate khác.\n"
            "Với MỖI candidate: (1) chấm relevance 0..1 so với toàn bộ truy "
            "vấn; (2) trả lời tiếng Việt thật ngắn, ưu tiên 1-8 từ; (3) chỉ "
            "dùng ảnh, OCR, ASR được cung cấp; (4) nếu chưa đủ bằng chứng thì "
            "answer='khong ro'. Không thay đổi candidate_id.\n\n"
            f"TRUY VẤN: {question}\n"
        )
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for item in candidates:
            label = (
                f"candidate_id={item['candidate_id']}; "
                f"video_id={item['video_id']}; frame_idx={item['frame_idx']}\n"
                f"OCR: {item['ocr'] or '(không có)'}\n"
                f"ASR quanh thời điểm: {item['asr'] or '(không có)'}"
            )
            mime = mimetypes.guess_type(item["image_path"].name)[0] or "image/jpeg"
            encoded = base64.b64encode(item["image_path"].read_bytes()).decode("ascii")
            parts.extend([
                {"text": label},
                {"inline_data": {"mime_type": mime, "data": encoded}},
            ])

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
                "responseSchema": self._response_schema(),
            },
        }

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise GeminiQAError(
                "Thiếu GEMINI_API_KEY trong .env; đã chuyển về QA local."
            )
        model = urlparse.quote(self.model, safe="-._")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        last_error: Exception | None = None
        schema_compatibility_used = False
        # One extra loop slot is reserved for the schema-compatibility retry;
        # normal network retries remain capped by ``self.retries`` below.
        for attempt in range(max(0, self.retries) + 2):
            try:
                return self.transport(url, headers, payload, self.timeout_seconds)
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if (
                    isinstance(exc, GeminiHTTPError)
                    and exc.status == 400
                    and "schema" in message
                    and not schema_compatibility_used
                    and payload.get("generationConfig", {}).get("responseSchema")
                ):
                    # Older v1beta deployments accept JSON mode but not the
                    # schema field. Keep deterministic JSON and retry once.
                    payload = json.loads(json.dumps(payload))
                    payload["generationConfig"].pop("responseSchema", None)
                    schema_compatibility_used = True
                    continue
                transient = (
                    isinstance(exc, GeminiHTTPError)
                    and exc.status in TRANSIENT_HTTP_STATUS
                ) or isinstance(exc, (TimeoutError, socket.timeout)) or any(
                    marker in message
                    for marker in (
                        "timeout", "temporarily", "resource_exhausted", "network"
                    )
                )
                if not transient or attempt >= max(0, self.retries):
                    break
                time.sleep(min(4.0, float(2**attempt)))
        if isinstance(last_error, GeminiQAError):
            raise last_error
        raise GeminiQAError(f"Gemini request failed: {last_error}")

    @staticmethod
    def _extract_json(response: Mapping[str, Any]) -> dict[str, Any]:
        try:
            parts = response["candidates"][0]["content"]["parts"]
            text = "".join(str(part.get("text", "")) for part in parts).strip()
        except (KeyError, IndexError, TypeError):
            feedback = response.get("promptFeedback", {})
            raise GeminiQAError(f"Gemini không trả nội dung: {feedback}") from None

        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GeminiQAError(f"Gemini trả JSON không hợp lệ: {exc}") from None
        if not isinstance(value, dict):
            raise GeminiQAError("Gemini JSON phải là một object.")
        return value

    def assess(
        self, question: str, rows: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[int, GeminiAssessment], dict[str, Any]]:
        started = time.perf_counter()
        candidates = self._prepare_candidates(rows)
        if not candidates:
            raise GeminiQAError("Không có ảnh keyframe gốc cho candidate Gemini.")

        cache_key = self._cache_key(question, candidates)
        cached = self._read_cache(cache_key)
        cache_hit = cached is not None
        if cached is None:
            raw_response = self._call(self._build_payload(question, candidates))
            cached = self._extract_json(raw_response)
            self._write_cache(cache_key, cached)

        by_candidate = {item["candidate_id"]: item for item in candidates}
        assessments: dict[int, GeminiAssessment] = {}
        for raw in cached.get("items", []):
            if not isinstance(raw, dict):
                continue
            candidate_id = _safe_int(raw.get("candidate_id"))
            candidate = by_candidate.get(candidate_id)
            if candidate is None:
                continue
            assessment = GeminiAssessment(
                row_index=int(candidate["row_index"]),
                candidate_id=candidate_id,
                relevance=_clamp01(raw.get("relevance")),
                answer=_short_answer(raw.get("answer")),
                confidence=_clamp01(raw.get("confidence")),
                evidence=" ".join(str(raw.get("evidence", "")).split())[:300],
            )
            assessments[assessment.row_index] = assessment

        if not assessments:
            raise GeminiQAError("Gemini không trả assessment hợp lệ nào.")
        return assessments, {
            "provider": "gemini",
            "model": self.model,
            "selected_images": len(candidates),
            "assessed_rows": len(assessments),
            "cache_hit": cache_hit,
            "api_calls": 0 if cache_hit else 1,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "fallback_reason": None,
        }


def apply_gemini_assessments(
    rows: Sequence[Mapping[str, Any]],
    assessments: Mapping[int, GeminiAssessment],
    *,
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """Attach answers to the same rows; optionally rerank assessed rows."""
    output: list[dict[str, Any]] = [dict(row) for row in rows]
    for row_index, assessment in assessments.items():
        if not 0 <= int(row_index) < len(output):
            continue
        row = output[int(row_index)]
        row["gemini_relevance"] = assessment.relevance
        row["gemini_confidence"] = assessment.confidence
        row["gemini_evidence"] = assessment.evidence
        if assessment.answer.casefold() != FALLBACK_ANSWER:
            row["answer"] = assessment.answer
            row["answer_confidence"] = assessment.confidence
            row["answer_source"] = "gemini"
            row["answer_explanation"] = assessment.evidence or (
                "Gemini multimodal trên đúng candidate"
            )

    if not rerank or not assessments:
        return output

    assessed_indices = [index for index in range(len(output)) if index in assessments]
    assessed_indices.sort(
        key=lambda index: (
            0.70 * assessments[index].relevance + 0.30 / (1.0 + index),
            -index,
        ),
        reverse=True,
    )
    assessed_set = set(assessed_indices)
    return [output[index] for index in assessed_indices] + [
        row for index, row in enumerate(output) if index not in assessed_set
    ]


__all__ = [
    "DEFAULT_MODEL",
    "GeminiAssessment",
    "GeminiHTTPError",
    "GeminiQAError",
    "GeminiQAReader",
    "apply_gemini_assessments",
    "select_candidate_indices",
]
