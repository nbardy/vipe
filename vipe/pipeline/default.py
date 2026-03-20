# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import pickle
import torch

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig

from vipe.slam.interface import SLAMOutput
from vipe.slam.system import SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    ProcessedVideoStream,
    VideoStream,
    MultiviewVideoList,
    FrameAttribute,
    StreamProcessor
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (
    AdaptiveDepthProcessor,
    GeoCalibIntrinsicsProcessor,
    MultiviewDepthProcessor,
    TrackAnythingProcessor,
)


logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            if depth_align_model.startswith("mvd_"):
                post_processors.append(MultiviewDepthProcessor(slam_output, model=depth_align_model))
            else:
                post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for view_idx, (output_stream, artifact_path) in enumerate(zip(output_streams, artifact_paths)):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                
                # Save sparse tracks if available
                if slam_output.sparse_tracks is not None:
                    logger.info(f"Saving sparse tracks and flowet ensemble to {artifact_path.sparse_tracks_path}")
                    with artifact_path.sparse_tracks_path.open("wb") as f:
                        pickle.dump(slam_output.sparse_tracks.observations, f)
                    
                    # Compute Flowet Ensemble
                    try:
                        from training.flowet_ensemble_engine import FlowetEnsembleEngine
                        from training.init_utils import unproject_tracks_to_3d
                        
                        T = len(output_stream)
                        depths_list = []
                        for i, f in enumerate(output_stream):
                            if f.metric_depth is None:
                                H, W = output_stream[0].rgb.shape[0], output_stream[0].rgb.shape[1]
                                depths_list.append(torch.zeros((H, W), device=f.pose.device()))
                            else:
                                depths_list.append(f.metric_depth)
                        
                        depths = torch.stack(depths_list).unsqueeze(1)
                        poses = torch.stack([f.pose.matrix() for f in output_stream])
                        
                        K_list = []
                        for f in output_stream:
                            fx, fy, cx, cy = f.intrinsics[0], f.intrinsics[1], f.intrinsics[2], f.intrinsics[3]
                            K_list.append(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=depths.device, dtype=torch.float32))
                        intrinsics = torch.stack(K_list)
                        
                        X_tracks = unproject_tracks_to_3d(slam_output.sparse_tracks.observations[view_idx], depths, poses, intrinsics)
                        
                        if X_tracks.shape[0] == 0:
                            logger.info("Sparse tracks empty, falling back to dense flow tracks...")
                            H, W = depths.shape[2], depths.shape[3]
                            step = 32
                            u0, v0 = torch.meshgrid(torch.arange(0, W, step, device=depths.device), 
                                                   torch.arange(0, H, step, device=depths.device), indexing='xy')
                            u_curr, v_curr = u0.flatten(), v0.flatten()
                            N_dense = u_curr.shape[0]
                            X_tracks = torch.full((N_dense, T, 3), float('nan'), device=depths.device)
                            for t in range(T):
                                z = depths[t, 0, v_curr.long(), u_curr.long()]
                                valid = (z > 0) & (z < 1000.0)
                                if not valid.any(): continue
                                K = intrinsics[t]
                                fx, fy, cx, cy = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item()
                                p_cam = torch.stack([(u_curr[valid]-cx)/fx*z[valid], (v_curr[valid]-cy)/fy*z[valid], z[valid]], dim=1)
                                c2w = poses[t]
                                X_tracks[valid, t] = torch.matmul(p_cam, c2w[:3, :3].T) + c2w[:3, 3]

                        logger.info(f"Flowet Ensemble fitting {X_tracks.shape[0]} particles over {T} frames")
                        engine = FlowetEnsembleEngine(T=T, device=depths.device)
                        ensemble = engine.process_batch(X_tracks)
                        
                        ensemble_path = artifact_path.base_path / "vipe" / f"{artifact_path.artifact_name}_flowet_ensemble.pt"
                        torch.save(ensemble, ensemble_path)
                        logger.info(f"Saved Flowet Ensemble to {ensemble_path}")
                        
                    except Exception as e:
                        logger.error(f"Flowet Ensemble computation failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
