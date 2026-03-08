from __future__ import annotations

import base64
import dataclasses
import io
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclasses.dataclass(frozen=True)
class OCRConfig:
    model_name: str = "microsoft/trocr-base-printed"
    device: str = "cpu"
    batch_size: int = 4
    backend: str = "trocr"           # "trocr" | "gemini_page" | "openai_page" | "groq_page"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash-lite"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str = ""
    groq_model: str = "llama-3.2-11b-vision-preview"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OCRConfig":
        return OCRConfig(
            model_name=str(d.get("model_name", "microsoft/trocr-base-printed")),
            device=str(d.get("device", "cpu")),
            batch_size=int(d.get("batch_size", 4)),
            backend=str(d.get("backend", "trocr")),
            gemini_api_key=str(d.get("gemini_api_key", os.environ.get("GEMINI_API_KEY", ""))),
            gemini_model=str(d.get("gemini_model", "gemini-2.0-flash-lite")),
            openai_api_key=str(d.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))),
            openai_model=str(d.get("openai_model", "gpt-4o-mini")),
            groq_api_key=str(d.get("groq_api_key", os.environ.get("GROQ_API_KEY", ""))),
            groq_model=str(d.get("groq_model", "llama-3.2-11b-vision-preview")),
        )


@dataclasses.dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: Optional[float]


# ---------------------------------------------------------------------------
# TrOCR Engine (line-level)
# ---------------------------------------------------------------------------

class TrOCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(cfg.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            cfg.model_name, low_cpu_mem_usage=False,
        )
        for module in self.model.modules():
            for attr_name, tensor in list(vars(module).items()):
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "meta":
                    setattr(module, attr_name, torch.zeros_like(tensor, device=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def recognize(self, images: Sequence[Image.Image]) -> List[OCRResult]:
        results: List[OCRResult] = []
        bs = max(1, int(self.cfg.batch_size))
        for i in range(0, len(images), bs):
            batch = images[i : i + bs]
            rgb_batch = [img.convert("RGB") if img.mode != "RGB" else img for img in batch]
            inputs = self.processor(images=rgb_batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            gen = self.model.generate(
                pixel_values, max_new_tokens=128,
                output_scores=True, return_dict_in_generate=True,
            )
            texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)
            confs: List[Optional[float]] = [None] * len(texts)
            if gen.scores:
                token_ids = gen.sequences[:, 1 : 1 + len(gen.scores)]
                probs = []
                for t, step_logits in enumerate(gen.scores):
                    lp = torch.log_softmax(step_logits, dim=-1)
                    probs.append(lp.gather(-1, token_ids[:, t].unsqueeze(-1)).squeeze(-1))
                mean_p = torch.stack(probs, dim=1).mean(dim=1).exp().clamp(0, 1)
                confs = [float(v) for v in mean_p.detach().cpu().tolist()]
            for t, c in zip(texts, confs):
                results.append(OCRResult(text=t, confidence=c))
        return results


# ---------------------------------------------------------------------------
# Shared page prompt
# ---------------------------------------------------------------------------

_PAGE_PROMPT = """\
You are transcribing an early modern printed Spanish document (16th-17th century).
The image shows one full page of printed text.

Rules:
- Transcribe ALL text visible on the page, line by line, in reading order.
- Preserve the original historical spelling and orthography EXACTLY as printed.
- Preserve line breaks — each printed line should be a separate line in your output.
- Do NOT translate, modernize, summarize, or add any commentary.
- Ignore decorative elements, page borders, and marginalia unless they contain text.
- Output ONLY the transcribed text, nothing else.

Transcription:"""


# ---------------------------------------------------------------------------
# Gemini Page OCR Engine
# ---------------------------------------------------------------------------

def _get_genai():
    try:
        from google import genai
        from google.genai import types
        return genai, types, "new"
    except ImportError:
        pass
    try:
        import google.generativeai as genai_old
        return genai_old, None, "old"
    except ImportError:
        raise ImportError("Install with: pip install google-genai")


class GeminiPageOCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        if not cfg.gemini_api_key:
            raise ValueError("Gemini API key required.")
        genai, types, self._sdk = _get_genai()
        if self._sdk == "new":
            self._client = genai.Client(api_key=cfg.gemini_api_key)
        else:
            genai.configure(api_key=cfg.gemini_api_key)
            self._model_old = genai.GenerativeModel(cfg.gemini_model)
        self._genai = genai

    def recognize_page(self, page_image: Image.Image) -> List[str]:
        img_rgb = page_image.convert("RGB")
        if img_rgb.width > 2000:
            ratio = 2000 / img_rgb.width
            img_rgb = img_rgb.resize((2000, int(img_rgb.height * ratio)), Image.LANCZOS)
            print(f"  [Gemini] Resized to {img_rgb.size}")

        text = ""
        for attempt in range(3):
            try:
                if self._sdk == "new":
                    response = self._client.models.generate_content(
                        model=self.cfg.gemini_model,
                        contents=[_PAGE_PROMPT, img_rgb],
                    )
                    text = response.text or ""
                else:
                    response = self._model_old.generate_content([_PAGE_PROMPT, img_rgb])
                    text = response.text or ""
                break
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    wait = 30 * (attempt + 1)
                    print(f"  [Gemini] Quota error, waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        print(f"  [Gemini] All retries exhausted: {e}")
                else:
                    print(f"  [Gemini] API error: {e}")
                    break

        lines = list(text.splitlines())
        print(f"  [Gemini] Got {len(lines)} lines, {len(text)} chars")
        return lines

    def recognize(self, images: Sequence[Image.Image]) -> List[OCRResult]:
        return [OCRResult(text="", confidence=None) for _ in images]


# ---------------------------------------------------------------------------
# OpenAI Page OCR Engine
# ---------------------------------------------------------------------------

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OpenAIPageOCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        if not cfg.openai_api_key:
            raise ValueError("OpenAI API key required. Set ocr.openai_api_key in config.")
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=cfg.openai_api_key)
        except ImportError:
            raise ImportError("Install with: pip install openai")

    def recognize_page(self, page_image: Image.Image) -> List[str]:
        img_rgb = page_image.convert("RGB")
        # Resize to max 2000px wide to manage token cost
        if img_rgb.width > 2000:
            ratio = 2000 / img_rgb.width
            img_rgb = img_rgb.resize((2000, int(img_rgb.height * ratio)), Image.LANCZOS)
            print(f"  [OpenAI] Resized to {img_rgb.size}")

        b64 = _pil_to_base64(img_rgb)

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.openai_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64}",
                                        "detail": "high",
                                    },
                                },
                                {"type": "text", "text": _PAGE_PROMPT},
                            ],
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""
                lines = list(text.splitlines())
                print(f"  [OpenAI] Got {len(lines)} lines, {len(text)} chars")
                return lines
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 20 * (attempt + 1)
                    print(f"  [OpenAI] Rate limit, waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        print(f"  [OpenAI] All retries exhausted: {e}")
                        return []
                else:
                    print(f"  [OpenAI] API error: {e}")
                    return []
        return []

    def recognize(self, images: Sequence[Image.Image]) -> List[OCRResult]:
        return [OCRResult(text="", confidence=None) for _ in images]


# ---------------------------------------------------------------------------
# Groq Page OCR Engine (free tier, vision model)
# ---------------------------------------------------------------------------

class GroqPageOCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        if not cfg.groq_api_key:
            raise ValueError("Groq API key required. Set ocr.groq_api_key in config.")
        try:
            from groq import Groq
            self._client = Groq(api_key=cfg.groq_api_key)
        except ImportError:
            raise ImportError("Install with: pip install groq")

    def recognize_page(self, page_image: Image.Image) -> List[str]:
        img_rgb = page_image.convert("RGB")
        # Groq vision giới hạn kích thước ảnh — resize về max 1568px
        max_dim = 1568
        if img_rgb.width > max_dim or img_rgb.height > max_dim:
            ratio = min(max_dim / img_rgb.width, max_dim / img_rgb.height)
            img_rgb = img_rgb.resize(
                (int(img_rgb.width * ratio), int(img_rgb.height * ratio)), Image.LANCZOS
            )
            print(f"  [Groq] Resized to {img_rgb.size}")

        b64 = _pil_to_base64(img_rgb)
        text = ""

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.groq_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                                {"type": "text", "text": _PAGE_PROMPT},
                            ],
                        }
                    ],
                    max_tokens=4096,
                    temperature=0.0,
                )
                text = response.choices[0].message.content or ""
                lines = [l for l in text.splitlines()]
                print(f"  [Groq] Got {len(lines)} lines, {len(text)} chars")
                return lines
            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 20 * (attempt + 1)
                    print(f"  [Groq] Rate limit, waiting {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        print(f"  [Groq] All retries exhausted: {e}")
                        return []
                else:
                    print(f"  [Groq] API error: {e}")
                    return []
        return []

    def recognize(self, images: Sequence[Image.Image]) -> List[OCRResult]:
        return [OCRResult(text="", confidence=None) for _ in images]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ocr_engine(cfg: OCRConfig):
    if cfg.backend == "groq_page":
        return GroqPageOCREngine(cfg)
    if cfg.backend == "openai_page":
        return OpenAIPageOCREngine(cfg)
    if cfg.backend == "gemini_page":
        return GeminiPageOCREngine(cfg)
    if cfg.backend == "gemini_lines":
        raise NotImplementedError("Use backend: gemini_page or openai_page instead.")
    return TrOCREngine(cfg)


# ---------------------------------------------------------------------------
# Crop utility
# ---------------------------------------------------------------------------

def crop_to_pil(gray: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox_xywh
    crop = gray[y : y + h, x : x + w]
    return Image.fromarray(crop).convert("RGB")