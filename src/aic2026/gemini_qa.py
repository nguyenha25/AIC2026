"""Gemini cloud reader for AIC 2026 Q&A.

The cloud path uses only Python's standard library, so the pinned Windows/CPU
environment stays unchanged.  Every answer remains attached to the exact row
(video_id, frame_idx) that produced it.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import mimetypes
import os
import re
import socket
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


DEFAULT_MODEL = "gemini-3.6-flash"
PROMPT_VERSION = "aic2026-gemini-qa-v2.1-json-recovery"
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
EvidenceProvider = Callable[
    [str, int, float],
    tuple[str, str] | tuple[str, str, str],
]


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


def _default_evidence(
    video_id: str, n: int, pts_time: float
) -> tuple[str, str, str]:
    """Load concise OCR, ASR and caption context for a candidate."""
    ocr_text = ""
    asr_text = ""
    caption_text = ""
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

    try:
        from .paths import captions_file

        path = captions_file(video_id)
        nearby_captions: list[tuple[int, str]] = []
        if path.is_file():
            with path.open("r", encoding="utf-8-sig") as stream:
                for line in stream:
                    try:
                        item = json.loads(line)
                        item_n = int(item.get("n", -1))
                        text = " ".join(str(item.get("caption", "")).split())
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
                    if text and abs(item_n - int(n)) <= 2:
                        nearby_captions.append((abs(item_n - int(n)), text))
        nearby_captions.sort(key=lambda item: item[0])
        caption_text = " | ".join(text for _, text in nearby_captions[:3])[:700]
    except Exception:
        pass

    return ocr_text, asr_text, caption_text


def _fold_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").casefold())
    return " ".join(
        "".join(char for char in text if unicodedata.category(char) != "Mn").split()
    )


def _is_abstention(value: Any) -> bool:
    folded = _fold_text(value).strip(" .,:;!?-_'")
    return folded in {
        "khong ro",
        "khong biet",
        "khong xac dinh",
        "unknown",
        "unclear",
        "cannot determine",
    }


def _target_question(question: str) -> str:
    """Tách câu hỏi đích khỏi phần mô tả dài, nhưng vẫn giữ full query."""
    clean = " ".join(str(question).split())
    interrogatives = re.findall(r"[^.!?]*\?", clean)
    if interrogatives:
        return interrogatives[-1].strip()
    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", clean) if chunk.strip()]
    return chunks[-1] if chunks else clean


def _query_context(question: str) -> dict[str, Any]:
    """Semantic hints for the cloud reader; never reads ground truth."""
    context: dict[str, Any] = {"target_question": _target_question(question)}
    try:
        from .semantic.parser import RuleBasedParser

        plan = RuleBasedParser().parse_qa("gemini-ui", question)
        context.update({
            "intent": plan.intent,
            "answer_type": plan.answer_type,
            "entities": list(plan.entities),
            "attributes": list(plan.attributes),
            "actions": list(plan.actions),
            "temporal_relation": plan.temporal_relation,
        })
    except Exception:
        context.update({"intent": "general_qa", "answer_type": "short_text"})

    folded = _fold_text(context["target_question"])
    if re.search(r"\b(ai|nguoi nao|nhan vat nao)\b", folded):
        context["answer_type"] = "person_or_name"
    elif re.search(r"\b(mau gi|mau nao)\b", folded):
        context["answer_type"] = "color"
    elif re.search(r"\b(bao nhieu|may)\b", folded):
        context["answer_type"] = "number"
    elif re.search(r"\b(o dau|noi nao|dia diem nao)\b", folded):
        context["answer_type"] = "place"
    elif re.search(r"\b(vat gi|mon gi|con gi|loai gi|cai gi)\b", folded):
        context["answer_type"] = "object_or_class"
    return context


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
                    # OCR/ASR/caption có thể được dựng lại mà ảnh không đổi.
                    # Đưa evidence vào cache identity để không tái dùng đáp án
                    # cũ sau khi pipeline enrich được cải thiện.
                    "ocr": item["ocr"],
                    "asr": item["asr"],
                    "caption": item["caption"],
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
            evidence = tuple(self.evidence_provider(video_id, n, pts_time))
            ocr_text = str(evidence[0] or "") if len(evidence) >= 1 else ""
            asr_text = str(evidence[1] or "") if len(evidence) >= 2 else ""
            caption_text = str(evidence[2] or "") if len(evidence) >= 3 else ""
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
                "caption": caption_text,
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
        context = _query_context(question)
        prompt = (
            "Bạn là bộ đọc Video Question Answering cho AIC 2026. Hãy hiểu "
            "đúng ĐỐI TƯỢNG được hỏi, không chỉ trích chữ gần ảnh.\n\n"
            "QUY TRÌNH SUY LUẬN (không in các bước này):\n"
            "1. Tách phần mô tả dùng để tìm cảnh khỏi câu hỏi đích.\n"
            "2. Xác định đối tượng/thuộc tính/hành động cần trả lời theo "
            "QUERY_CONTEXT.\n"
            "3. Chọn đúng VIDEO_GROUP. Các ảnh trong cùng VIDEO_GROUP là một "
            "chuỗi theo thời gian và ĐƯỢC kết hợp để hiểu cảnh. Tuyệt đối "
            "không ghép bằng chứng giữa hai video khác nhau.\n"
            "4. Ảnh là bằng chứng chính; OCR, ASR và CAPTION là bằng chứng hỗ "
            "trợ, có thể sai. Đừng trả một chữ/số chỉ vì nó xuất hiện trong "
            "OCR nếu không trả lời đúng câu hỏi đích.\n"
            "5. Được dùng kiến thức phổ thông để nhận dạng hoặc trả lời một "
            "sự thật về người/vật/địa danh đã được ảnh, OCR hay ASR nhận diện "
            "đủ rõ. Không dùng kiến thức ngoài để bịa chi tiết riêng của cảnh "
            "hoặc đoán danh tính khi bằng chứng mơ hồ.\n\n"
            "ĐẦU RA: Với MỖI candidate_id, chấm relevance 0..1 và trả lời "
            "đúng loại đáp án. Câu trả lời tiếng Việt ngắn, tự nhiên, thường "
            "1-12 từ; câu kiến thức có thể dài hơn nếu cần. Các frame cùng "
            "video có thể nhận cùng đáp án khi toàn chuỗi hỗ trợ đáp án đó. "
            "Nếu không đủ bằng chứng, answer phải đúng chuỗi 'khong ro'. "
            "Không thay đổi candidate_id. evidence nêu bằng chứng quyết định "
            "và nói rõ nếu có dùng kiến thức phổ thông, tối đa 12 từ. Chỉ "
            "xuất JSON đúng schema, không dùng Markdown và không thêm lời "
            "giải thích ngoài JSON.\n\n"
            "QUERY_CONTEXT: "
            + json.dumps(context, ensure_ascii=False, sort_keys=True)
            + "\n"
            f"TRUY VẤN ĐẦY ĐỦ: {question}\n"
        )
        parts: list[dict[str, Any]] = [{"text": prompt}]

        # Giữ thứ tự video theo ranking ban đầu, nhưng xếp frame trong từng
        # video theo thời gian để Gemini thực sự thấy được ngữ cảnh chuyển động.
        groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        for item in candidates:
            groups.setdefault(str(item["video_id"]), []).append(item)

        for group_rank, (video_id, items) in enumerate(groups.items(), start=1):
            ordered = sorted(
                items,
                key=lambda item: (float(item["pts_time"]), int(item["frame_idx"])),
            )
            parts.append({
                "text": (
                    f"\n=== VIDEO_GROUP {group_rank}: video_id={video_id}; "
                    f"{len(ordered)} frame theo thứ tự thời gian ==="
                )
            })
            for temporal_index, item in enumerate(ordered, start=1):
                label = (
                    f"candidate_id={item['candidate_id']}; "
                    f"temporal_position={temporal_index}/{len(ordered)}; "
                    f"frame_idx={item['frame_idx']}; pts_time={item['pts_time']:.3f}s\n"
                    f"OCR: {item['ocr'] or '(không có)'}\n"
                    f"ASR quanh thời điểm: {item['asr'] or '(không có)'}\n"
                    f"CAPTION lân cận: {item['caption'] or '(không có)'}"
                )
                mime = (
                    mimetypes.guess_type(item["image_path"].name)[0]
                    or "image/jpeg"
                )
                encoded = base64.b64encode(
                    item["image_path"].read_bytes()
                ).decode("ascii")
                parts.extend([
                    {"text": label},
                    {"inline_data": {"mime_type": mime, "data": encoded}},
                ])

        return {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                # 12 assessment có thể vượt 2K token nếu evidence dài. Khi bị
                # cắt giữa object, json.loads báo Expecting value/EOF và UI
                # trước đây rơi thẳng về local.
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseSchema": self._response_schema(),
            },
        }

    @staticmethod
    def _strict_retry_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Tạo request thứ hai ngắn gọn hơn khi JSON lần đầu bị hỏng."""
        retry = json.loads(json.dumps(payload))
        parts = retry.get("contents", [{}])[0].get("parts", [])
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            parts[0]["text"] = (
                "LẦN TRƯỚC ĐẦU RA BỊ LỖI JSON. Lần này bắt buộc hoàn thành "
                "một JSON object hợp lệ đúng response schema. Không code "
                "fence, không chú thích, không dấu phẩy thừa. evidence tối "
                "đa 8 từ.\n\n" + str(parts[0]["text"])
            )
        generation = retry.setdefault("generationConfig", {})
        generation["temperature"] = 0
        generation["maxOutputTokens"] = max(
            4096, int(generation.get("maxOutputTokens", 4096))
        )
        return retry

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
            texts = [
                str(part.get("text", "")).strip()
                for part in parts
                if isinstance(part, Mapping) and str(part.get("text", "")).strip()
            ]
        except (KeyError, IndexError, TypeError):
            feedback = response.get("promptFeedback", {})
            raise GeminiQAError(f"Gemini không trả nội dung: {feedback}") from None

        if not texts:
            feedback = response.get("promptFeedback", {})
            raise GeminiQAError(f"Gemini không trả nội dung: {feedback}")

        def variants(raw: str) -> list[str]:
            raw = raw.strip().lstrip("\ufeff")
            values = [raw]
            if raw.startswith("```"):
                lines = raw.splitlines()[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                values.append("\n".join(lines).strip())
            first, last = raw.find("{"), raw.rfind("}")
            if 0 <= first < last:
                values.append(raw[first:last + 1])

            output: list[str] = []
            for value in values:
                if value and value not in output:
                    output.append(value)
                # Hai lỗi phổ biến khi schema bị model bỏ qua.
                repaired = re.sub(r",\s*([}\]])", r"\1", value)
                repaired = re.sub(r"(:\s*)\.(\d+)", r"\g<1>0.\2", repaired)
                if repaired and repaired not in output:
                    output.append(repaired)
            return output

        # JSON hoàn chỉnh thường nằm ở part cuối; thử từng part trước để không
        # nối thinking/prose với JSON, rồi mới thử toàn bộ chuỗi ghép.
        raw_candidates = list(reversed(texts)) + ["".join(texts)]
        last_error: Exception | None = None
        all_variants: list[str] = []
        for raw in raw_candidates:
            for text in variants(raw):
                if text not in all_variants:
                    all_variants.append(text)
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    last_error = exc
                    try:
                        value = ast.literal_eval(text)
                    except (ValueError, SyntaxError):
                        continue
                if isinstance(value, dict):
                    return value

        # Nếu output bị cắt ở object cuối, cứu các item đã hoàn tất. assess()
        # sẽ yêu cầu lại một lần để lấy đủ; phần này là lưới an toàn cuối cùng.
        decoder = json.JSONDecoder()
        recovered: dict[int, dict[str, Any]] = {}
        for text in all_variants:
            for match in re.finditer(r"\{", text):
                try:
                    value, _ = decoder.raw_decode(text[match.start():])
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict) or "candidate_id" not in value:
                    continue
                candidate_id = _safe_int(value.get("candidate_id"))
                if candidate_id >= 0:
                    recovered[candidate_id] = value
        if recovered:
            return {"items": [recovered[key] for key in sorted(recovered)]}

        detail = str(last_error or "không tìm thấy JSON object")
        finish_reason = ""
        try:
            finish_reason = str(response["candidates"][0].get("finishReason", ""))
        except (KeyError, IndexError, TypeError):
            pass
        suffix = f"; finishReason={finish_reason}" if finish_reason else ""
        raise GeminiQAError(f"Gemini trả JSON không hợp lệ: {detail}{suffix}")

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
        api_calls = 0
        json_recovery = "cache" if cache_hit else "none"
        if cached is None:
            payload = self._build_payload(question, candidates)
            raw_response = self._call(payload)
            api_calls = 1
            first_value: dict[str, Any] | None = None
            first_error: GeminiQAError | None = None
            try:
                first_value = self._extract_json(raw_response)
            except GeminiQAError as exc:
                first_error = exc

            expected_ids = {item["candidate_id"] for item in candidates}
            first_ids = {
                _safe_int(item.get("candidate_id"))
                for item in (first_value or {}).get("items", [])
                if isinstance(item, dict)
            }
            need_retry = first_error is not None or not expected_ids.issubset(first_ids)

            if need_retry:
                try:
                    retry_response = self._call(self._strict_retry_payload(payload))
                    api_calls += 1
                    cached = self._extract_json(retry_response)
                    json_recovery = "retry"
                except GeminiQAError as retry_error:
                    if first_value and first_ids:
                        cached = first_value
                        json_recovery = "partial_first_response"
                    else:
                        reason = first_error or GeminiQAError(
                            "Gemini không trả đủ candidate."
                        )
                        raise GeminiQAError(
                            f"{reason}; retry thất bại: {retry_error}"
                        ) from None
            else:
                cached = first_value
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
            "selected_videos": len({item["video_id"] for item in candidates}),
            "assessed_rows": len(assessments),
            "prompt_version": PROMPT_VERSION,
            "evidence": {
                "ocr": sum(bool(item["ocr"]) for item in candidates),
                "asr": sum(bool(item["asr"]) for item in candidates),
                "caption": sum(bool(item["caption"]) for item in candidates),
            },
            "cache_hit": cache_hit,
            "api_calls": 0 if cache_hit else api_calls,
            "json_recovery": json_recovery,
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
        if not _is_abstention(assessment.answer):
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
