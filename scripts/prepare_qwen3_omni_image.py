#!/usr/bin/env python3
"""Build the image-only prompt corpus used to distill a Qwen3-Omni Thinker draft.

The source corpora intentionally keep their original formats.  This script
samples the requested task mixture, materializes embedded or remote images,
normalizes every prompt to the structured Qwen3-Omni message format, and makes
media-grouped train/validation/test splits.  Source assistant answers are never
copied: ``scripts/response_regeneration/script.py`` regenerates them on policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import random
import re
import ssl
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

LOGGER = logging.getLogger("prepare_qwen3_omni_image")
Image.MAX_IMAGE_PIXELS = 100_000_000

BASE_QUOTAS = {
    # General visual understanding: 10k.
    "general_pixmo_ama": 3_500,
    "general_pixmo_cap": 3_000,
    "general_finevision": 3_500,
    # Interleaved and multi-image web documents: 4k.  OmniCorpus is gated, so
    # its former 1k is redistributed between OBELICS and MMC4.
    "interleaved_obelics": 2_500,
    "interleaved_mmc4": 1_500,
    # Detection and grounding: 5k.
    "detection_coco": 2_500,
    "detection_objects365": 1_250,
    "detection_openimages": 1_250,
    # Referring, pointing, and counting: 4k.
    "referring_refcoco": 2_000,
    "pointing_pixmo_points": 1_000,
    "counting_pixmo_count": 1_000,
    # OCR, documents, charts, and UI: 9k.
    "document_docling": 1_800,
    "document_ureader": 1_300,
    "document_visualweb": 1_300,
    "ui_screenqa": 700,
    "ui_groundui": 700,
    "chart_cosyn": 3_200,
    # Spatial, STEM, and multi-image reasoning: 4k.
    "reasoning_ai2d": 1_000,
    "reasoning_scienceqa": 1_000,
    "reasoning_nlvr2": 1_000,
    "reasoning_spotdiff": 1_000,
    # Visual code: 4k.
    "code_websight": 1_000,
    "code_synthcodenet": 1_000,
    "code_datikz": 1_000,
    "code_iconstack": 1_000,
}

COMPONENT_META = {
    "general_pixmo_ama": ("general", "open_qa"),
    "general_pixmo_cap": ("general", "detailed_description"),
    "general_finevision": ("general", "open_qa"),
    "interleaved_obelics": ("interleaved", "web_multi_image"),
    "interleaved_mmc4": ("interleaved", "web_multi_image"),
    "detection_coco": ("detection", "object_detection"),
    "detection_objects365": ("detection", "object_detection"),
    "detection_openimages": ("detection", "object_detection"),
    "referring_refcoco": ("referring", "referring_grounding"),
    "pointing_pixmo_points": ("referring", "pointing"),
    "counting_pixmo_count": ("referring", "counting"),
    "document_docling": ("document", "document_qa"),
    "document_ureader": ("document", "ocr_chart_qa"),
    "document_visualweb": ("document", "visual_document_qa"),
    "ui_screenqa": ("document", "screen_qa"),
    "ui_groundui": ("document", "ui_grounding"),
    "chart_cosyn": ("document", "chart_table_diagram_qa"),
    "reasoning_ai2d": ("reasoning", "diagram_reasoning"),
    "reasoning_scienceqa": ("reasoning", "science_reasoning"),
    "reasoning_nlvr2": ("reasoning", "multi_image_reasoning"),
    "reasoning_spotdiff": ("reasoning", "multi_image_comparison"),
    "code_websight": ("visual_code", "screenshot_to_html"),
    "code_synthcodenet": ("visual_code", "image_to_code"),
    "code_datikz": ("visual_code", "figure_to_tikz"),
    "code_iconstack": ("visual_code", "icon_to_svg"),
}

FINEVISION_CONFIGS = {
    "general_finevision": (
        "image_textualization(filtered)",
        "sharegpt4o",
        "sharegpt4v(coco)",
    ),
    "detection_objects365": ("objects365_qa",),
    "document_docling": ("DoclingMatix",),
    "document_ureader": ("ureader_qa_processed",),
    "document_visualweb": ("visualwebinstruct(filtered)",),
    "ui_screenqa": ("screenqa",),
    "ui_groundui": ("groundui",),
    "chart_cosyn": (
        "CoSyn_400k_chart",
        "CoSyn_400k_table",
        "CoSyn_400k_diagram",
    ),
    "reasoning_ai2d": ("ai2d_merged",),
    "reasoning_scienceqa": ("scienceqa",),
    "reasoning_nlvr2": ("nlvr2",),
    "reasoning_spotdiff": ("spot_the_diff",),
    "code_websight": ("websight",),
    "code_synthcodenet": ("SynthCodeNet",),
    "code_datikz": ("datikz",),
}

IMAGE_PLACEHOLDER_RE = re.compile(
    r"(?:<image(?:_\d+)?>|<\|image_pad\|>|<\|vision_start\|>|<\|vision_end\|>)",
    flags=re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"[ \t]+")
MAX_PROMPT_CHARS = 8_000
MAX_IMAGES_PER_SAMPLE = 4
MIN_INTERLEAVED_IMAGES = 2
MIN_WEB_METADATA_SIDE = 128
MIN_OBELICS_CONTEXT_CHARS = 80
MIN_MMC4_CONTEXT_CHARS = 60
MIN_MMC4_SIMILARITY = 0.2
EXPECTED_REFCOCO_FILES = 3


@dataclass(frozen=True)
class FineVisionCandidate:
    parquet_path: Path
    row_index: int
    key: str
    prompt: str
    source: str


@dataclass(frozen=True)
class UrlCandidate:
    key: str
    prompt: str
    source: str
    urls: tuple[str, ...]
    expected_hashes: tuple[str | None, ...]


@dataclass(frozen=True)
class DownloadedImage:
    path: Path
    sha256: str


def _digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _clean_prompt(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = IMAGE_PLACEHOLDER_RE.sub(" ", value).replace("\x00", " ")
    value = "\n".join(
        WHITESPACE_RE.sub(" ", line).strip() for line in value.splitlines()
    )
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value[:MAX_PROMPT_CHARS].strip()


def _iter_parquet_rows(
    files: Iterable[Path], columns: list[str], *, batch_size: int = 512
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        selected_columns = [column for column in columns if column in available]
        row_index = 0
        for batch in parquet.iter_batches(
            columns=selected_columns, batch_size=batch_size
        ):
            for row in batch.to_pylist():
                yield path, row_index, row
                row_index += 1


def _reservoir_sample(
    rows: Iterable[Any], limit: int, *, rng: random.Random
) -> list[Any]:
    selected: list[Any] = []
    seen = 0
    for row in rows:
        seen += 1
        if len(selected) < limit:
            selected.append(row)
            continue
        replacement = rng.randrange(seen)
        if replacement < limit:
            selected[replacement] = row
    rng.shuffle(selected)
    return selected


def _read_selected_rows(
    path: Path, row_indices: Iterable[int], columns: list[str]
) -> dict[int, dict[str, Any]]:
    wanted = sorted(set(row_indices))
    if not wanted:
        return {}
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    selected_columns = [column for column in columns if column in available]
    values: dict[int, dict[str, Any]] = {}
    cursor = 0
    wanted_pos = 0
    for batch in parquet.iter_batches(columns=selected_columns, batch_size=32):
        rows = batch.to_pylist()
        batch_end = cursor + len(rows)
        while wanted_pos < len(wanted) and wanted[wanted_pos] < batch_end:
            target = wanted[wanted_pos]
            if target >= cursor:
                values[target] = rows[target - cursor]
            wanted_pos += 1
        cursor = batch_end
        if wanted_pos >= len(wanted):
            break
    return values


def _scaled_quotas(total: int) -> dict[str, int]:
    base_total = sum(BASE_QUOTAS.values())
    if total <= 0:
        raise ValueError("--total must be positive")
    raw = {name: total * value / base_total for name, value in BASE_QUOTAS.items()}
    quotas = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    for name in sorted(
        raw, key=lambda item: (raw[item] - quotas[item], item), reverse=True
    ):
        if remainder <= 0:
            break
        quotas[name] += 1
        remainder -= 1
    return quotas


def _image_suffix(image_format: str | None) -> str:
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }.get((image_format or "").upper(), ".img")


def _validate_image_bytes(data: bytes, min_side: int) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"invalid image: {exc}") from exc
    if min(width, height) < min_side:
        raise ValueError(f"image is too small: {width}x{height}")
    return _image_suffix(image_format)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_local_image(path: Path, min_side: int) -> str:
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"invalid local image {path}: {exc}") from exc
    if min(width, height) < min_side:
        raise ValueError(f"local image is too small: {path} ({width}x{height})")
    return _hash_file(path)


def _store_embedded_image(
    data: bytes, media_root: Path, min_side: int
) -> DownloadedImage:
    suffix = _validate_image_bytes(data, min_side)
    sha256 = hashlib.sha256(data).hexdigest()
    path = media_root / "embedded" / sha256[:2] / f"{sha256}{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return DownloadedImage(path=path.resolve(), sha256=sha256)


def _record(
    *,
    component: str,
    key: str,
    source: str,
    prompt: str,
    images: list[DownloadedImage],
) -> dict[str, Any]:
    category, task_type = COMPONENT_META[component]
    content: list[dict[str, str]] = [
        {"type": "image", "path": str(image.path.resolve())} for image in images
    ]
    content.append({"type": "text", "text": prompt})
    ordered_hashes = "|".join(image.sha256 for image in images)
    media_id = f"image-{hashlib.sha256(ordered_hashes.encode()).hexdigest()}"
    sample_id = f"{component}-{_digest(f'{key}|{prompt}|{ordered_hashes}')}"
    return {
        "id": sample_id,
        "conversations": [{"role": "user", "content": content}],
        "metadata": {
            "source": source,
            "modality": "image",
            "category": category,
            "task_type": task_type,
            "component": component,
            "media_id": media_id,
        },
    }


class UrlImageDownloader:
    def __init__(
        self,
        media_root: Path,
        *,
        workers: int,
        timeout: float,
        max_bytes: int,
        min_side: int,
        error_path: Path,
    ) -> None:
        self.cache_root = media_root / "url"
        self.workers = workers
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.min_side = min_side
        self.error_path = error_path
        self.ssl_context = ssl.create_default_context()
        self.failed_urls: set[str] = set()
        if self.error_path.exists():
            with self.error_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as handle:
                for line in handle:
                    try:
                        error = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    url = error.get("url")
                    if isinstance(url, str):
                        self.failed_urls.add(url)

    def _cached(self, url: str, expected_sha256: str | None) -> DownloadedImage | None:
        url_digest = hashlib.sha256(url.encode()).hexdigest()
        parent = self.cache_root / url_digest[:2]
        matches = list(parent.glob(f"{url_digest}.*")) if parent.exists() else []
        for path in matches:
            try:
                data = path.read_bytes()
                _validate_image_bytes(data, self.min_side)
                sha256 = hashlib.sha256(data).hexdigest()
                if expected_sha256 and sha256 != expected_sha256:
                    LOGGER.debug(
                        "Cached image SHA differs from source metadata for %s", url
                    )
                return DownloadedImage(path=path.resolve(), sha256=sha256)
            except (OSError, ValueError):
                continue
        return None

    def _download(self, url: str, expected_sha256: str | None) -> DownloadedImage:
        cached = self._cached(url, expected_sha256)
        if cached is not None:
            return cached
        if urllib.parse.urlsplit(url).scheme.lower() not in {"http", "https"}:
            raise ValueError(f"unsupported image URL scheme: {url}")
        request = urllib.request.Request(  # noqa: S310 - URL scheme checked upstream
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Qwen3OmniDatasetBuilder/1.0)",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(  # noqa: S310 - only HTTP(S) candidates are built
            request, timeout=self.timeout, context=self.ssl_context
        ) as response:
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError(f"image exceeds {self.max_bytes} bytes")
                chunks.append(chunk)
        data = b"".join(chunks)
        suffix = _validate_image_bytes(data, self.min_side)
        sha256 = hashlib.sha256(data).hexdigest()
        if expected_sha256 and sha256 != expected_sha256:
            # Hosts such as Flickr often recompress the same visual asset.  A
            # byte-level mismatch is therefore diagnostic, not proof that the
            # source annotation no longer applies.
            LOGGER.debug(
                "Downloaded image SHA differs from source metadata for %s", url
            )
        url_digest = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_root / url_digest[:2] / f"{url_digest}{suffix}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        return DownloadedImage(path=path.resolve(), sha256=sha256)

    def download_many(
        self, requests: Iterable[tuple[str, str | None]]
    ) -> dict[str, DownloadedImage]:
        unique: dict[str, str | None] = {}
        for url, expected_sha256 in requests:
            unique.setdefault(url, expected_sha256)
        results: dict[str, DownloadedImage] = {}
        pending: dict[str, str | None] = {}
        for url, expected_sha256 in unique.items():
            cached = self._cached(url, expected_sha256)
            if cached is not None:
                results[url] = cached
            elif url not in self.failed_urls:
                pending[url] = expected_sha256
        errors: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self._download, url, expected_sha256): url
                for url, expected_sha256 in pending.items()
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    results[url] = future.result()
                except Exception as exc:  # noqa: BLE001 - endpoints fail independently
                    errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
        if errors:
            self.failed_urls.update(error["url"] for error in errors)
            self.error_path.parent.mkdir(parents=True, exist_ok=True)
            with self.error_path.open("a", encoding="utf-8") as output:
                for error in errors:
                    output.write(json.dumps(error, ensure_ascii=False) + "\n")
        return results


def _materialize_url_candidates(
    component: str,
    candidates: list[UrlCandidate],
    quota: int,
    downloader: UrlImageDownloader,
    *,
    minimum_images: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_media: set[str] = set()
    progress = tqdm(total=quota, desc=component, unit="sample")
    for offset in range(0, len(candidates), 128):
        batch = candidates[offset : offset + 128]
        requests = [
            (url, expected)
            for candidate in batch
            for url, expected in zip(
                candidate.urls, candidate.expected_hashes, strict=True
            )
        ]
        downloaded = downloader.download_many(requests)
        for candidate in batch:
            images = [downloaded[url] for url in candidate.urls if url in downloaded]
            if len(images) < minimum_images:
                continue
            images = images[:4]
            media_key = "|".join(image.sha256 for image in images)
            if media_key in seen_media:
                continue
            seen_media.add(media_key)
            records.append(
                _record(
                    component=component,
                    key=candidate.key,
                    source=candidate.source,
                    prompt=candidate.prompt,
                    images=images,
                )
            )
            progress.update(1)
            if len(records) >= quota:
                progress.close()
                return records
    progress.close()
    return records


def _finevision_candidates(
    root: Path, configs: tuple[str, ...], *, rng: random.Random
) -> list[FineVisionCandidate]:
    candidates: list[FineVisionCandidate] = []
    for config in configs:
        files = sorted((root / config).rglob("*.parquet"))
        for path, row_index, row in _iter_parquet_rows(
            files,
            ["texts", "source"],
        ):
            turns = row.get("texts")
            if not isinstance(turns, list) or not turns:
                continue
            valid_prompts = [
                _clean_prompt(turn.get("user"))
                for turn in turns
                if isinstance(turn, dict)
            ]
            valid_prompts = [prompt for prompt in valid_prompts if prompt]
            if not valid_prompts:
                continue
            prompt = valid_prompts[rng.randrange(len(valid_prompts))]
            candidates.append(
                FineVisionCandidate(
                    parquet_path=path,
                    row_index=row_index,
                    key=f"{config}:{path.name}:{row_index}",
                    prompt=prompt,
                    source=str(row.get("source") or config),
                )
            )
    rng.shuffle(candidates)
    return candidates


def _build_finevision(  # noqa: C901
    component: str,
    source_root: Path,
    media_root: Path,
    quota: int,
    *,
    seed: int,
    min_side: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = _finevision_candidates(
        source_root / "FineVision-selected", FINEVISION_CONFIGS[component], rng=rng
    )
    reserve = min(128, max(8, math.ceil(quota * 0.2)))
    selected = candidates[: min(len(candidates), quota + reserve)]
    by_file: dict[Path, list[FineVisionCandidate]] = defaultdict(list)
    for candidate in selected:
        by_file[candidate.parquet_path].append(candidate)
    materialized: dict[tuple[Path, int], list[DownloadedImage]] = {}
    for path, path_candidates in tqdm(
        sorted(by_file.items()), desc=f"materialize {component}", unit="shard"
    ):
        values = _read_selected_rows(
            path, (candidate.row_index for candidate in path_candidates), ["images"]
        )
        for candidate in path_candidates:
            raw_images = values.get(candidate.row_index, {}).get("images")
            if not isinstance(raw_images, list):
                continue
            images: list[DownloadedImage] = []
            for raw_image in raw_images[:4]:
                data = raw_image.get("bytes") if isinstance(raw_image, dict) else None
                if isinstance(data, memoryview):
                    data = data.tobytes()
                if not isinstance(data, bytes) or not data:
                    continue
                try:
                    images.append(_store_embedded_image(data, media_root, min_side))
                except ValueError:
                    continue
            if images:
                materialized[(path, candidate.row_index)] = images
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in selected:
        images = materialized.get((candidate.parquet_path, candidate.row_index))
        if not images:
            continue
        record = _record(
            component=component,
            key=candidate.key,
            source=candidate.source,
            prompt=candidate.prompt,
            images=images,
        )
        if record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        records.append(record)
        if len(records) >= quota:
            break
    return records


def _pixmo_candidates(  # noqa: C901
    source_root: Path,
    component: str,
    quota: int,
    *,
    seed: int,
) -> list[UrlCandidate]:
    rng = random.Random(seed)
    multiplier = 8
    if component == "general_pixmo_ama":
        root = source_root / "PixMo-AskModelAnything"
        columns = ["image_url", "image_sha256", "question"]

        def convert(row: dict[str, Any], index: int) -> UrlCandidate | None:
            prompt = _clean_prompt(row.get("question"))
            url = row.get("image_url")
            if not prompt or not isinstance(url, str):
                return None
            return UrlCandidate(
                key=f"ama:{row.get('image_sha256') or index}:{_digest(prompt)}",
                prompt=prompt,
                source="allenai/pixmo-ask-model-anything",
                urls=(url,),
                expected_hashes=(row.get("image_sha256"),),
            )

    elif component == "general_pixmo_cap":
        root = source_root / "PixMo-Cap"
        columns = ["image_url"]
        prompts = (
            "Describe this image in detail, including the scene, objects, "
            "attributes, and spatial relationships.",
            "Give a careful and comprehensive description of everything "
            "visible in this image.",
            "Explain what is happening in this image and mention important "
            "visual details.",
        )

        def convert(row: dict[str, Any], index: int) -> UrlCandidate | None:
            url = row.get("image_url")
            if not isinstance(url, str):
                return None
            prompt = prompts[index % len(prompts)]
            return UrlCandidate(
                key=f"cap:{_digest(url)}",
                prompt=prompt,
                source="allenai/pixmo-cap",
                urls=(url,),
                expected_hashes=(None,),
            )

    elif component == "pointing_pixmo_points":
        root = source_root / "PixMo-Points"
        columns = ["image_url", "image_sha256", "label"]

        def convert(row: dict[str, Any], index: int) -> UrlCandidate | None:
            del index
            url, label = row.get("image_url"), _clean_prompt(row.get("label"))
            if not isinstance(url, str) or not label:
                return None
            prompt = (
                f"Find every visible instance of {label} in the image. "
                "State how many there are "
                "and describe their approximate locations."
            )
            return UrlCandidate(
                key=(
                    f"points:{row.get('image_sha256') or _digest(url)}:{_digest(label)}"
                ),
                prompt=prompt,
                source="allenai/pixmo-points",
                urls=(url,),
                expected_hashes=(row.get("image_sha256"),),
            )

    elif component == "counting_pixmo_count":
        root = source_root / "PixMo-Count"
        columns = ["image_url", "image_sha256", "label"]

        def convert(row: dict[str, Any], index: int) -> UrlCandidate | None:
            del index
            url, label = row.get("image_url"), _clean_prompt(row.get("label"))
            if not isinstance(url, str) or not label:
                return None
            prompt = (
                f"How many {label} are visible in this image? "
                "Explain briefly how you counted them."
            )
            return UrlCandidate(
                key=f"count:{row.get('image_sha256') or _digest(url)}:{_digest(label)}",
                prompt=prompt,
                source="allenai/pixmo-count",
                urls=(url,),
                expected_hashes=(row.get("image_sha256"),),
            )

    else:
        raise ValueError(f"unsupported PixMo component: {component}")

    files = sorted(root.rglob("*.parquet"))

    def rows() -> Iterator[UrlCandidate]:
        for index, (_, _, row) in enumerate(_iter_parquet_rows(files, columns)):
            candidate = convert(row, index)
            if candidate is not None:
                yield candidate

    return _reservoir_sample(rows(), quota * multiplier, rng=rng)


def _obelics_candidates(  # noqa: C901
    source_root: Path, quota: int, *, seed: int
) -> list[UrlCandidate]:
    rng = random.Random(seed)
    files = sorted((source_root / "OBELICS-sample").rglob("*.parquet"))

    def rows() -> Iterator[UrlCandidate]:  # noqa: C901
        for _, row_index, row in _iter_parquet_rows(
            files, ["images", "texts", "metadata", "general_metadata"]
        ):
            images = row.get("images")
            texts = row.get("texts")
            if not isinstance(images, list) or not isinstance(texts, list):
                continue
            metadata: list[Any] = []
            with suppress(TypeError, json.JSONDecodeError):
                metadata = json.loads(row.get("metadata") or "[]")
            urls: list[str] = []
            for index, url in enumerate(images):
                if not isinstance(url, str) or not url.startswith(
                    ("http://", "https://")
                ):
                    continue
                item_meta = metadata[index] if index < len(metadata) else None
                if isinstance(item_meta, dict):
                    width = item_meta.get("original_width") or 0
                    height = item_meta.get("original_height") or 0
                    if width and height and min(width, height) < MIN_WEB_METADATA_SIDE:
                        continue
                if url not in urls:
                    urls.append(url)
                if len(urls) >= MAX_IMAGES_PER_SAMPLE:
                    break
            if len(urls) < MIN_INTERLEAVED_IMAGES:
                continue
            snippets = [_clean_prompt(text) for text in texts if _clean_prompt(text)]
            context = "\n\n".join(snippets[:4])[:4_000]
            if len(context) < MIN_OBELICS_CONTEXT_CHARS:
                continue
            page_url = ""
            with suppress(TypeError, json.JSONDecodeError):
                page_url = json.loads(row.get("general_metadata") or "{}").get(
                    "url", ""
                )
            prompt = (
                "These images appeared together in one web document. Describe "
                "the important content of the images and explain how they relate "
                "to the supplied context.\n\n"
                f"Web context:\n{context}"
            )
            yield UrlCandidate(
                key=f"obelics:{_digest(page_url or '|'.join(urls))}:{row_index}",
                prompt=prompt,
                source="HuggingFaceM4/OBELICS",
                urls=tuple(urls),
                expected_hashes=tuple(None for _ in urls),
            )

    return _reservoir_sample(rows(), quota * 8, rng=rng)


def _mmc4_candidates(  # noqa: C901
    source_root: Path, quota: int, *, seed: int
) -> list[UrlCandidate]:
    rng = random.Random(seed)
    files = sorted((source_root / "MMC4-Core" / "extracted").rglob("*.jsonl"))
    rng.shuffle(files)
    limit = quota * 6
    selected: list[UrlCandidate] = []
    for path in tqdm(files, desc="scan MMC4", unit="shard"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_index, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text_list = row.get("text_list")
                image_info = row.get("image_info")
                if not isinstance(text_list, list) or not isinstance(image_info, list):
                    continue
                usable: list[tuple[int, str]] = []
                for info in image_info:
                    if (
                        not isinstance(info, dict)
                        or float(info.get("matched_sim") or 0) < MIN_MMC4_SIMILARITY
                    ):
                        continue
                    raw_url = info.get("raw_url")
                    if not isinstance(raw_url, str):
                        continue
                    url = urllib.parse.urljoin(str(row.get("url") or ""), raw_url)
                    if not url.startswith(("http://", "https://")):
                        continue
                    usable.append((int(info.get("matched_text_index") or 0), url))
                usable = sorted(
                    {url: index for index, url in usable}.items(),
                    key=lambda x: x[1],
                )
                ordered = [
                    (index, url) for url, index in usable[:MAX_IMAGES_PER_SAMPLE]
                ]
                if len(ordered) < MIN_INTERLEAVED_IMAGES:
                    continue
                snippets = [
                    _clean_prompt(text_list[index])
                    for index, _ in ordered
                    if 0 <= index < len(text_list) and _clean_prompt(text_list[index])
                ]
                context = "\n\n".join(snippets)[:4_000]
                if len(context) < MIN_MMC4_CONTEXT_CHARS:
                    continue
                urls = tuple(url for _, url in ordered)
                prompt = (
                    "Use the images and their nearby webpage text to summarize "
                    "the visual "
                    "content and explain the relationship between the images.\n\n"
                    f"Nearby text:\n{context}"
                )
                selected.append(
                    UrlCandidate(
                        key=f"mmc4:{_digest(str(row.get('url')))}:{path.name}:{line_index}",
                        prompt=prompt,
                        source="jmhessel/mmc4-core-ff",
                        urls=urls,
                        expected_hashes=tuple(None for _ in urls),
                    )
                )
                if len(selected) >= limit:
                    rng.shuffle(selected)
                    return selected
    rng.shuffle(selected)
    return selected


def _local_image(path: Path, min_side: int) -> DownloadedImage | None:
    try:
        return DownloadedImage(
            path=path.resolve(), sha256=_validate_local_image(path, min_side)
        )
    except (OSError, ValueError):
        return None


def _coco_path(source_root: Path, image_id: int) -> Path | None:
    filename = f"{image_id:012d}.jpg"
    for split in ("train2017", "val2017"):
        path = source_root / "COCO2017" / split / filename
        if path.exists():
            return path
    return None


def _build_coco(
    component: str, source_root: Path, quota: int, *, seed: int, min_side: int
) -> list[dict[str, Any]]:
    annotation_path = (
        source_root / "COCO2017" / "annotations" / "instances_train2017.json"
    )
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in data["categories"]}
    by_image: dict[int, Counter[str]] = defaultdict(Counter)
    for annotation in data["annotations"]:
        if annotation.get("iscrowd"):
            continue
        name = categories.get(annotation.get("category_id"))
        if name:
            by_image[int(annotation["image_id"])][name] += 1
    del data
    candidates = [(image_id, counts) for image_id, counts in by_image.items() if counts]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    for image_id, counts in candidates:
        path = _coco_path(source_root, image_id)
        if path is None or (image := _local_image(path, min_side)) is None:
            continue
        label = sorted(counts, key=lambda name: (-counts[name], name))[0]
        if len(records) % 2:
            prompt = (
                f"Find all visible instances of {label}. State how many there "
                "are and describe "
                "their approximate locations in the image."
            )
        else:
            prompt = (
                "Identify the main visible object categories, estimate their "
                "counts, and describe "
                "their approximate spatial locations."
            )
        records.append(
            _record(
                component=component,
                key=f"coco:{image_id}",
                source="COCO2017",
                prompt=prompt,
                images=[image],
            )
        )
        if len(records) >= quota:
            break
    return records


def _build_openimages(
    component: str, source_root: Path, quota: int, *, seed: int, min_side: int
) -> list[dict[str, Any]]:
    root = source_root / "OpenImages-sample"
    image_paths = {path.stem: path for path in (root / "images").glob("*.jpg")}
    class_names: dict[str, str] = {}
    with (root / "metadata" / "class-descriptions-boxable.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for label_id, name in csv.reader(handle):
            class_names[label_id] = name
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    with (root / "metadata" / "train-annotations-bbox.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("ImageID")
            if image_id not in image_paths or row.get("IsGroupOf") == "1":
                continue
            name = class_names.get(row.get("LabelName", ""))
            if name:
                labels[image_id][name] += 1
    candidates = [(image_id, value) for image_id, value in labels.items() if value]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    for image_id, counts in candidates:
        image = _local_image(image_paths[image_id], min_side)
        if image is None:
            continue
        label = sorted(counts, key=lambda name: (-counts[name], name))[0]
        prompt = (
            f"Locate every visible {label} in this image. Give the count and "
            "describe the "
            "approximate position of each instance."
        )
        records.append(
            _record(
                component=component,
                key=f"openimages:{image_id}:{_digest(label)}",
                source="OpenImages",
                prompt=prompt,
                images=[image],
            )
        )
        if len(records) >= quota:
            break
    return records


def _iter_refcoco_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_line"] = line_index
            row["_source"] = path.stem
            yield row


def _build_refcoco(
    component: str, source_root: Path, quota: int, *, seed: int, min_side: int
) -> list[dict[str, Any]]:
    root = source_root / "RefCOCO"
    files = sorted(root.glob("*_train.json"))
    per_file = math.ceil(quota * 1.4 / max(len(files), 1))
    candidates: list[dict[str, Any]] = []
    for file_index, path in enumerate(files):
        rng = random.Random(seed + file_index)
        candidates.extend(
            _reservoir_sample(_iter_refcoco_rows(path), per_file, rng=rng)
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    records: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for row in candidates:
        try:
            image_id = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            continue
        prompt = _clean_prompt(conversations[0].get("value"))
        if not prompt or prompt in seen_prompts:
            continue
        path = _coco_path(source_root, image_id)
        if path is None or (image := _local_image(path, min_side)) is None:
            continue
        seen_prompts.add(prompt)
        records.append(
            _record(
                component=component,
                key=f"{row['_source']}:{row['_line']}:{image_id}",
                source=f"PaDT-MLLM/{row['_source']}",
                prompt=prompt,
                images=[image],
            )
        )
        if len(records) >= quota:
            break
    return records


def _build_iconstack(
    component: str,
    source_root: Path,
    media_root: Path,
    quota: int,
    *,
    seed: int,
    min_side: int,
) -> list[dict[str, Any]]:
    files = sorted((source_root / "IconStack-sample").rglob("*.parquet"))
    if not files:
        return []
    path = files[0]
    row_count = pq.ParquetFile(path).metadata.num_rows
    rng = random.Random(seed)
    reserve = min(128, max(8, math.ceil(quota * 0.2)))
    indices = rng.sample(range(row_count), min(row_count, quota + reserve))
    rows = _read_selected_rows(path, indices, ["id", "image", "caption"])
    records: list[dict[str, Any]] = []
    for index in indices:
        row = rows.get(index, {})
        data = row.get("image")
        if isinstance(data, memoryview):
            data = data.tobytes()
        if not isinstance(data, bytes):
            continue
        try:
            image = _store_embedded_image(data, media_root, min_side)
        except ValueError:
            continue
        caption = _clean_prompt(row.get("caption"))
        prompt = "Recreate this icon as clean SVG code."
        if caption:
            prompt += f" The intended icon description is: {caption}"
        records.append(
            _record(
                component=component,
                key=f"iconstack:{row.get('id') or index}",
                source="likaixin/IconStack-48M-Rendered-Train",
                prompt=prompt,
                images=[image],
            )
        )
        if len(records) >= quota:
            break
    return records


def _source_requirements(source_root: Path) -> dict[str, bool]:
    return {
        "FineVision-selected": any(
            (source_root / "FineVision-selected").rglob("*.parquet")
        ),
        "PixMo-AskModelAnything": any(
            (source_root / "PixMo-AskModelAnything").rglob("*.parquet")
        ),
        "PixMo-Cap": any((source_root / "PixMo-Cap").rglob("*.parquet")),
        "PixMo-Points": any((source_root / "PixMo-Points").rglob("*.parquet")),
        "PixMo-Count": any((source_root / "PixMo-Count").rglob("*.parquet")),
        "OBELICS-sample": any((source_root / "OBELICS-sample").rglob("*.parquet")),
        "MMC4-Core": any((source_root / "MMC4-Core" / "extracted").rglob("*.jsonl")),
        "COCO2017/train2017": (source_root / "COCO2017" / "train2017").is_dir(),
        "COCO2017/annotations": (
            source_root / "COCO2017" / "annotations" / "instances_train2017.json"
        ).is_file(),
        "RefCOCO": len(list((source_root / "RefCOCO").glob("*_train.json")))
        == EXPECTED_REFCOCO_FILES,
        "OpenImages-sample": (
            source_root
            / "OpenImages-sample"
            / "metadata"
            / "train-annotations-bbox.csv"
        ).is_file(),
        "IconStack-sample": any((source_root / "IconStack-sample").rglob("*.parquet")),
    }


def _split_records(
    records: list[dict[str, Any]], *, seed: int
) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "validation": [], "test": []}
    for record in records:
        media_id = record["metadata"]["media_id"]
        value = (
            int.from_bytes(
                hashlib.sha256(f"{seed}:{media_id}".encode()).digest()[:8], "big"
            )
            % 20
        )
        split = "test" if value == 0 else "validation" if value == 1 else "train"
        splits[split].append(record)
    rng = random.Random(seed)
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def _component_cache_valid(records: list[dict[str, Any]], quota: int) -> bool:
    if len(records) != quota:
        return False
    ids = [record.get("id") for record in records]
    if len(ids) != len(set(ids)):
        return False
    return all(
        Path(part["path"]).is_file()
        for record in records
        for message in record.get("conversations", [])
        for part in message.get("content", [])
        if part.get("type") == "image"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-workers", type=int, default=32)
    parser.add_argument("--download-timeout", type=float, default=25.0)
    parser.add_argument("--max-image-mib", type=int, default=25)
    parser.add_argument("--min-image-side", type=int, default=64)
    parser.add_argument("--validate-sources", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.total <= 0:
        parser.error("--total must be positive")
    if args.download_workers <= 0:
        parser.error("--download-workers must be positive")
    return args


def main() -> None:  # noqa: C901
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    requirements = _source_requirements(source_root)
    print(json.dumps({"sources": requirements}, ensure_ascii=False, indent=2))
    missing = [name for name, present in requirements.items() if not present]
    if missing:
        raise FileNotFoundError(f"Missing required image sources: {', '.join(missing)}")
    if args.validate_sources:
        print("All required source layouts are present.")
        return

    final_prompt_paths = [
        output_dir / "train.prompts.jsonl",
        output_dir / "validation.prompts.jsonl",
        output_dir / "test.prompts.jsonl",
    ]
    existing = [path for path in final_prompt_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace manifests: "
            + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    media_root = output_dir / "media"
    quotas = _scaled_quotas(args.total)
    (output_dir / "quotas.json").write_text(
        json.dumps(quotas, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    downloader = UrlImageDownloader(
        media_root,
        workers=args.download_workers,
        timeout=args.download_timeout,
        max_bytes=args.max_image_mib * 1024 * 1024,
        min_side=args.min_image_side,
        error_path=output_dir / "url-download.errors.jsonl",
    )

    by_component: dict[str, list[dict[str, Any]]] = {}
    components_dir = output_dir / "components"
    for component in BASE_QUOTAS:
        quota = quotas[component]
        component_path = components_dir / f"{component}.jsonl"
        if component_path.exists():
            cached_records = _read_jsonl(component_path)
            if _component_cache_valid(cached_records, quota):
                LOGGER.info("Reusing %s checkpoint: %d/%d", component, quota, quota)
                by_component[component] = cached_records
                continue
        LOGGER.info("Building %s (quota=%d)", component, quota)
        component_seed = args.seed + int(_digest(component, 8), 16)
        if component in FINEVISION_CONFIGS:
            records = _build_finevision(
                component,
                source_root,
                media_root,
                quota,
                seed=component_seed,
                min_side=args.min_image_side,
            )
        elif component in {
            "general_pixmo_ama",
            "general_pixmo_cap",
            "pointing_pixmo_points",
            "counting_pixmo_count",
        }:
            candidates = _pixmo_candidates(
                source_root, component, quota, seed=component_seed
            )
            records = _materialize_url_candidates(
                component, candidates, quota, downloader
            )
        elif component == "interleaved_obelics":
            records = _materialize_url_candidates(
                component,
                _obelics_candidates(source_root, quota, seed=component_seed),
                quota,
                downloader,
                minimum_images=2,
            )
        elif component == "interleaved_mmc4":
            records = _materialize_url_candidates(
                component,
                _mmc4_candidates(source_root, quota, seed=component_seed),
                quota,
                downloader,
                minimum_images=2,
            )
        elif component == "detection_coco":
            records = _build_coco(
                component,
                source_root,
                quota,
                seed=component_seed,
                min_side=args.min_image_side,
            )
        elif component == "detection_openimages":
            records = _build_openimages(
                component,
                source_root,
                quota,
                seed=component_seed,
                min_side=args.min_image_side,
            )
        elif component == "referring_refcoco":
            records = _build_refcoco(
                component,
                source_root,
                quota,
                seed=component_seed,
                min_side=args.min_image_side,
            )
        elif component == "code_iconstack":
            records = _build_iconstack(
                component,
                source_root,
                media_root,
                quota,
                seed=component_seed,
                min_side=args.min_image_side,
            )
        else:
            raise AssertionError(f"no builder for component {component}")
        by_component[component] = records
        _write_jsonl(component_path, records)
        LOGGER.info("Built %s: %d/%d", component, len(records), quota)

    deficits = {
        component: quotas[component] - len(records)
        for component, records in by_component.items()
        if len(records) < quotas[component]
    }
    records = [
        record for component in BASE_QUOTAS for record in by_component[component]
    ]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        duplicate_ids = sorted(
            sample_id for sample_id, count in Counter(ids).items() if count > 1
        )
        raise RuntimeError(
            f"Generated {len(duplicate_ids)} duplicate record IDs: {duplicate_ids[:10]}"
        )

    status = "complete" if not deficits else "incomplete"
    manifest_base = {
        "version": 1,
        "status": status,
        "seed": args.seed,
        "requested_total": args.total,
        "actual_total": len(records),
        "quotas": quotas,
        "counts_by_component": {
            component: len(by_component[component]) for component in BASE_QUOTAS
        },
        "deficits": deficits,
    }
    if deficits:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest_base, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Image corpus is incomplete; rerun to reuse cached images. Deficits: "
            + json.dumps(deficits, sort_keys=True)
        )

    splits = _split_records(records, seed=args.seed)
    for split, rows in splits.items():
        _write_jsonl(output_dir / f"{split}.prompts.jsonl", rows)
    manifest_base["splits"] = {split: len(rows) for split, rows in splits.items()}
    manifest_base["counts_by_category"] = dict(
        sorted(Counter(record["metadata"]["category"] for record in records).items())
    )
    manifest_base["counts_by_source"] = dict(
        sorted(Counter(record["metadata"]["source"] for record in records).items())
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_base, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest_base, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
