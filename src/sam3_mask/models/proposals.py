from __future__ import annotations

from PIL import Image

from sam3_mask.models.base import ProposalProvider
from sam3_mask.utils.geometry import Box


class NoopProposalProvider(ProposalProvider):
    def propose(self, image: Image.Image, labels: list[str]) -> list[Box]:
        return []
