from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiffusionConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    use_controlnet: bool = True
    learning_rate: float = 1e-5


class CarotidDiffusionAugmentor:
    """Wrapper for fine-tuning latent diffusion + ControlNet for carotid data."""

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.pipeline: Any | None = None

    def build_pipeline(self) -> None:
        """Initialize diffusion pipeline and optional ControlNet components."""
        raise NotImplementedError("Implement pipeline creation with diffusers/ControlNet.")

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Run one optimization step and return scalar losses."""
        raise NotImplementedError("Implement CDA training step.")
