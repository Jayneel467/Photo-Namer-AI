"""Image validation, caption generation, and safe output naming."""

from __future__ import annotations

import re
import threading
import unicodedata
import warnings
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_BATCH_IMAGES = 20
MAX_BATCH_BYTES = 40 * 1024 * 1024

FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a supported, safe image."""


def validate_image_data(data: bytes) -> tuple[Image.Image, str]:
    """Decode an uploaded image and return an RGB image plus its real format.

    The content is sniffed and decoded by Pillow rather than trusting the upload
    extension or browser-provided MIME type.
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Each image must be 10 MB or smaller.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in FORMAT_EXTENSIONS:
                    raise ImageValidationError(
                        "Use a JPEG, PNG, or WEBP image."
                    )
                if probe.width * probe.height > MAX_IMAGE_PIXELS:
                    raise ImageValidationError(
                        "Images must be 40 megapixels or smaller."
                    )
                probe.verify()

            with Image.open(BytesIO(data)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                return oriented.convert("RGB"), image_format
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(
            "This file could not be decoded as a safe image."
        ) from exc


def caption_to_filename(caption: str, image_format: str, max_length: int = 72) -> str:
    """Turn generated caption text into a portable, path-safe filename."""
    extension = FORMAT_EXTENSIONS.get(image_format.upper())
    if extension is None:
        raise ValueError("Unsupported image format.")

    normalized = unicodedata.normalize("NFKD", caption)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text.lower()).strip("-")
    slug = slug[:max_length].strip("-") or "meaningful-photo"
    return f"{slug}{extension}"


def unique_filenames(names: Iterable[str]) -> list[str]:
    """Resolve duplicate output names without allowing directory components."""
    used: set[str] = set()
    resolved: list[str] = []

    for name in names:
        safe_name = Path(name).name
        stem = Path(safe_name).stem or "meaningful-photo"
        suffix = Path(safe_name).suffix.lower()
        candidate = safe_name
        version = 2
        while candidate.casefold() in used:
            candidate = f"{stem}-{version}{suffix}"
            version += 1
        used.add(candidate.casefold())
        resolved.append(candidate)

    return resolved


class LocalBlipCaptioner:
    """Thread-safe wrapper around a locally hosted BLIP image-caption model."""

    def __init__(self, processor: Any, model: Any, torch_module: Any, device: str):
        self._processor = processor
        self._model = model
        self._torch = torch_module
        self._device = device
        self._inference_lock = threading.Lock()

    def caption(self, image: Image.Image) -> str:
        """Generate one concise caption. Image content is never sent to an API."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._inference_lock, self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
                do_sample=False,
            )
        caption = self._processor.decode(output[0], skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("The model returned an empty caption.")
        return caption


def load_captioner(model_id: str = "Salesforce/blip-image-captioning-base") -> LocalBlipCaptioner:
    """Load the Hugging Face model on the best available local device."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return LocalBlipCaptioner(processor, model, torch, device)