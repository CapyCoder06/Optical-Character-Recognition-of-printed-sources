from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


@dataclasses.dataclass(frozen=True)
class OCRConfig:
    model_name: str = "microsoft/trocr-base-printed"
    device: str = "cpu"  # "cpu" | "cuda"
    batch_size: int = 4

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "OCRConfig":
        return OCRConfig(
            model_name=str(d.get("model_name", "microsoft/trocr-base-printed")),
            device=str(d.get("device", "cpu")),
            batch_size=int(d.get("batch_size", 4)),
        )


@dataclasses.dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: Optional[float]  # token-level mean probability proxy when available


class TrOCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(cfg.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def recognize(self, images: Sequence[Image.Image]) -> List[OCRResult]:
        results: List[OCRResult] = []
        bs = max(1, int(self.cfg.batch_size))

        for i in range(0, len(images), bs):
            batch = images[i : i + bs]

            rgb_batch = []
            for img in batch:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                rgb_batch.append(img)

            inputs = self.processor(images=rgb_batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            gen = self.model.generate(
                pixel_values,
                max_new_tokens=128,
                output_scores=True,
                return_dict_in_generate=True,
            )

            texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)

            confs: List[Optional[float]] = [None] * len(texts)
            # Confidence proxy: mean token probability of generated sequence.
            # gen.scores is a list of logits for each generated step (excluding the first decoder token).
            if gen.scores is not None and len(gen.scores) > 0:
                # sequences shape: [batch, seq_len]
                seq = gen.sequences
                # Align scores steps to token ids in sequences. Typically, scores[t] predicts seq[:, t+1].
                token_ids = seq[:, 1 : 1 + len(gen.scores)]
                probs = []
                for t, step_logits in enumerate(gen.scores):
                    step_log_probs = torch.log_softmax(step_logits, dim=-1)
                    step_token = token_ids[:, t].unsqueeze(-1)
                    probs.append(step_log_probs.gather(-1, step_token).squeeze(-1))
                # probs: list of [batch] log-probs
                logp = torch.stack(probs, dim=1)  # [batch, steps]
                mean_logp = logp.mean(dim=1)
                mean_p = mean_logp.exp().clamp(0.0, 1.0)
                confs = [float(v) for v in mean_p.detach().cpu().tolist()]

            for t, c in zip(texts, confs):
                results.append(OCRResult(text=t, confidence=c))

        return results


def crop_to_pil(gray: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = bbox_xywh
    crop = gray[y : y + h, x : x + w]
    # Handle both grayscale (2D) and already-colour (3D) arrays;
    # TrOCR processor requires 3-channel RGB input.
    pil = Image.fromarray(crop)
    return pil.convert("RGB")