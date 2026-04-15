from __future__ import annotations

import numpy as np
import torch

from vipe.utils.cameras import CameraType
from vipe.utils.misc import unpack_optional

from .base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType


class ExternalDepthModel(DepthEstimationModel):
    """Depth model backed by a precomputed .npy tensor.

    Accepted shapes:
    - (N, H, W)
    - (N, 1, H, W)
    """

    def __init__(self, path: str):
        super().__init__()
        depth_data = np.load(path)
        if depth_data.ndim == 4:
            depth_data = depth_data[:, 0]  # (N, H, W)
        if depth_data.ndim != 3:
            raise ValueError(f"External depth must be rank-3 or rank-4, got shape={depth_data.shape}")
        self.depth_data = depth_data
        self.n_frames = depth_data.shape[0]
        self._cursor = 0

    def _depth_for_index(self, frame_idx: int, image_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        idx = min(frame_idx, self.n_frames - 1)
        depth = torch.from_numpy(self.depth_data[idx]).to(device=device)
        if depth.shape != image_hw:
            depth = torch.nn.functional.interpolate(
                depth[None, None], size=image_hw, mode="bilinear", align_corners=False
            )[0, 0]
        return depth

    # Kept for compatibility with SimpleDepthProcessor.
    def infer(self, image: torch.Tensor, frame_idx: int) -> DepthEstimationResult:
        image_hw = tuple(image.shape[:2])
        depth = self._depth_for_index(frame_idx, image_hw, image.device)
        return DepthEstimationResult(metric_depth=depth)

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        rgb = unpack_optional(src.rgb)
        if rgb.dim() == 3:
            depth = self._depth_for_index(self._cursor, tuple(rgb.shape[:2]), rgb.device)
            self._cursor += 1
            return DepthEstimationResult(metric_depth=depth)

        if rgb.dim() != 4:
            raise ValueError(f"Expected rgb rank 3 or 4, got rank={rgb.dim()}")

        batch_depth = []
        for i in range(rgb.shape[0]):
            batch_depth.append(self._depth_for_index(self._cursor + i, tuple(rgb.shape[1:3]), rgb.device))
        self._cursor += rgb.shape[0]
        return DepthEstimationResult(metric_depth=torch.stack(batch_depth, dim=0))

    @property
    def supported_camera_types(self) -> list[CameraType]:
        return [CameraType.PINHOLE]

    @property
    def depth_type(self) -> DepthType:
        return DepthType.METRIC_DEPTH
