import numpy as np
import torch
from vipe.streams.base import StreamProcessor, VideoFrame, FrameAttribute
from vipe.ext.lietorch import SE3

class ExternalPoseIntrinsicsProcessor(StreamProcessor):
    def __init__(self, pose_path: str, intrinsics_path: str):
        super().__init__()
        self.poses = np.load(pose_path)
        self.intrinsics = np.load(intrinsics_path)
        self.n_frames = self.poses.shape[0]

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.POSE, FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        idx = min(frame_idx, self.n_frames - 1)
        
        # Load Pose
        pose_mat = self.poses[idx]
        if pose_mat.shape == (3, 4):
            temp = np.eye(4)
            temp[:3, :] = pose_mat
            pose_mat = temp
            
        # ViPE expects c2w SE3
        frame.pose = SE3(torch.from_numpy(pose_mat).float().cuda())
        
        # Load Intrinsics
        K = self.intrinsics[idx]
        # ViPE expects [fx, fy, cx, cy]
        frame.intrinsics = torch.as_tensor([K[0,0], K[1,1], K[0,2], K[1,2]]).float().cuda()
        
        from vipe.utils.cameras import CameraType
        frame.camera_type = CameraType.PINHOLE
        
        return frame
