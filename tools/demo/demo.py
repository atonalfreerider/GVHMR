import cv2
import torch
import argparse
import json
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_smpl_skeleton_3d
from hmr4d.utils.geo_transform import project_p2d

from hmr4d.utils.preproc import Extractor
from hmr4d.utils.geo.hmr_cam import get_bbx_xyxy_from_keypoints, estimate_K
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from tqdm import tqdm
from einops import einsum
import numpy as np
import os

CRF = 23  # 17 is lossless, every +6 halves the mp4 size

def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument("--pose_json", type=str, default=None, help="Path to the pose JSON file")
    parser.add_argument("--slam_results", type=str, default=None, help="Path to the SLAM results file in TUM format")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"

    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
        ]

        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        if args.pose_json is not None:
            overrides.append(f"+pose_json={args.pose_json}")
        if args.slam_results is not None:
            overrides.append(f"+slam_results={args.slam_results}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    # Add new paths
    cfg.paths.composite_video = os.path.join(cfg.output_dir, f"{cfg.video_name}_composite.mp4")
    cfg.paths.keypoints_3d_json = os.path.join(cfg.output_dir, f"{cfg.video_name}_keypoints_3d.json")
    cfg.paths.camera_poses_json = os.path.join(cfg.output_dir, f"{cfg.video_name}_camera_poses.json")

    return cfg


def load_and_format_pose_data(json_path, image_size):
    with open(json_path, 'r') as f:
        pose_data = json.load(f)

    num_frames = len(pose_data)
    keypoints_list = []
    bbx_xys_list = []
    bbx_xyxy_list = []

    for frame_idx in range(num_frames):
        frame = pose_data.get(str(frame_idx), {})

        if frame and 'keypoints' in frame:
            keypoints = torch.tensor(frame['keypoints'], dtype=torch.float32).reshape(1, -1, 3)
        else:
            keypoints = torch.zeros((1, 17, 3), dtype=torch.float32)

        bbx_xyxy = get_bbx_xyxy_from_keypoints(keypoints, image_size)
        x_min, y_min, x_max, y_max = bbx_xyxy[0]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        size = max(x_max - x_min, y_max - y_min)

        bbx_xys = torch.tensor([center_x, center_y, size], dtype=torch.float32)

        keypoints_list.append(keypoints[0])  # Remove the batch dimension
        bbx_xys_list.append(bbx_xys)
        bbx_xyxy_list.append(bbx_xyxy[0])  # Remove the batch dimension

    keypoints = torch.stack(keypoints_list)
    bbx_xys = torch.stack(bbx_xys_list)
    bbx_xyxy = torch.stack(bbx_xyxy_list)

    formatted_data = {
        "bbx_xyxy": bbx_xyxy,
        "bbx_xys": bbx_xys,
        "vitpose": keypoints
    }

    return formatted_data


@torch.no_grad()
def run_preprocess(cfg):
    video_path = cfg.video_path
    paths = cfg.paths

    # Get video dimensions
    length, width, height = get_video_lwh(video_path)
    image_size = (height, width)

    if cfg.pose_json:
        json_path = cfg.pose_json
        formatted_data = load_and_format_pose_data(json_path, image_size)
        bbx_xyxy = formatted_data["bbx_xyxy"]
        bbx_xys = formatted_data["bbx_xys"]
        vitpose = formatted_data["vitpose"]

        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        torch.save(vitpose, paths.vitpose)
        Log.info(f"[Preprocess] Saved vitpose data to {paths.vitpose}")
    else:
        raise ValueError("No pose JSON provided")

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get DPVO results
    if cfg.slam_results:
        Log.info(f"[Preprocess] Loading SLAM results from {cfg.slam_results}")
        slam_results = np.loadtxt(cfg.slam_results)
        
        if slam_results.shape[1] != 8:
            raise ValueError(f"Expected SLAM results with 8 columns, but got {slam_results.shape[1]}")
        
        # Ensure the SLAM results have the same number of frames as the video
        video_length = get_video_lwh(cfg.video_path)[0]
        if slam_results.shape[0] != video_length:
            Log.warning(f"SLAM results ({slam_results.shape[0]} frames) do not match video length ({video_length} frames)")
            
            if slam_results.shape[0] * 2 == video_length:
                Log.info("SLAM results appear to be for every other frame. Duplicating each result.")
                slam_results = np.repeat(slam_results, 2, axis=0)
            elif slam_results.shape[0] > video_length:
                slam_results = slam_results[:video_length]
            else:
                # Linear interpolation
                from scipy.interpolate import interp1d
                x = np.linspace(0, 1, slam_results.shape[0])
                x_new = np.linspace(0, 1, video_length)
                f = interp1d(x, slam_results, axis=0, kind='linear')
                slam_results = f(x_new)
        
        torch.save(slam_results, paths.slam)
        Log.info(f"[Preprocess] Saved SLAM results with shape {slam_results.shape}")

def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)

    traj = torch.load(cfg.paths.slam)
    Log.info(f"SLAM trajectory shape: {traj.shape}")
    
    if traj.shape[1] != 8:
        raise ValueError(f"Expected SLAM trajectory with 8 columns, but got {traj.shape[1]}")
    
    traj_quat = torch.from_numpy(traj[:, [4, 5, 6, 7]])  # TUM format: x y z qx qy qz qw
    R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    Log.info(f"R_w2c shape: {R_w2c.shape}")
    Log.info(f"K_fullimg shape: {K_fullimg.shape}")

    bbx_xys = torch.load(paths.bbx)["bbx_xys"]
    kp2d = torch.load(paths.vitpose)
    f_imgseq = torch.load(paths.vit_features)

    Log.info(f"bbx_xys shape: {bbx_xys.shape}")
    Log.info(f"kp2d shape: {kp2d.shape}")
    Log.info(f"f_imgseq shape: {f_imgseq.shape}")

    cam_angvel = compute_cam_angvel(R_w2c)
    Log.info(f"cam_angvel shape: {cam_angvel.shape}")

    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_imgseq": f_imgseq,
    }
    return data

def get_global_cameras(verts, device="cuda", distance=5, position=(0.0, 0.0, 5.0), tilt_degrees=0):
    """This always put object at the center of view"""
    verts = verts.to(device)
    positions = torch.tensor([position], device=device).repeat(len(verts), 1)
    targets = verts.mean(1)

    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions

    # Create a rotation that looks at the target and orients the camera upright
    forward = -directions
    up = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(len(verts), 1)
    right = torch.cross(up, forward, dim=-1)
    up = torch.cross(forward, right, dim=-1)

    # Normalize vectors
    forward = forward / torch.norm(forward, dim=-1, keepdim=True)
    right = right / torch.norm(right, dim=-1, keepdim=True)
    up = up / torch.norm(up, dim=-1, keepdim=True)

    # Create rotation matrices
    R = torch.stack([right, up, forward], dim=-1)

    # Apply tilt rotation
    tilt_rad = torch.deg2rad(torch.tensor(tilt_degrees, device=device))
    tilt_rotation = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(tilt_rad), -torch.sin(tilt_rad)],
        [0, torch.sin(tilt_rad), torch.cos(tilt_rad)]
    ], device=device)

    R = torch.matmul(tilt_rotation, R)

    # Convert to 6D rotation representation for stability
    rotation_6d = matrix_to_rotation_6d(R)

    translation = -torch.bmm(R, positions.unsqueeze(-1)).squeeze(-1)

    return rotation_6d, translation

def render_composite(cfg):
    composite_video_path = Path(cfg.paths.composite_video)
    if composite_video_path.exists():
        Log.info(f"[Render Composite] Video already exists at {composite_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(device)
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(device)

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    joints_glob = einsum(J_regressor, pred_ay_verts, "j v, l v i -> l j i")  # (L, J, 3)

    # Get global cameras
    rotation_6d, T_world2cam = get_global_cameras(pred_ay_verts, device=device, tilt_degrees=-20)
    
    # Convert 6D rotation back to matrix when needed
    R_world2cam = rotation_6d_to_matrix(rotation_6d)

    # Invert the rotation matrices to reverse the camera motion
    R_world2cam_inverted = R_world2cam.transpose(1, 2)

    # Fix the camera translation to the initial position
    T_world2cam_fixed = T_world2cam[0].unsqueeze(0).repeat(T_world2cam.shape[0], 1)

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0].to(device)
    focal_length = K[0, 0].item()

    reader = get_video_reader(video_path)
    writer = get_writer(composite_video_path, fps=30, crf=CRF)

    # Prepare data for JSON export
    keypoints_3d = []
    camera_poses = []

    # Define world axes
    world_axes = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device)

    for i, img_raw in tqdm(enumerate(reader), total=length, desc=f"Rendering Composite"):
        # Transform world axes to camera space using inverted rotation
        axes_cam = (R_world2cam_inverted[i] @ (world_axes.T - T_world2cam_fixed[i].unsqueeze(1))).T
        
        # Project axes to image space
        axes_2d = project_p2d(axes_cam.unsqueeze(0), K.unsqueeze(0))
        axes_2d = axes_2d.squeeze(0).cpu().numpy().astype(int)

        # Draw 3D SMPL skeleton on image using inverted rotation
        img = draw_smpl_skeleton_3d(img_raw, joints_glob[i].cpu().numpy(), K.cpu().numpy(),
                                    R_world2cam_inverted[i].cpu().numpy(), T_world2cam_fixed[i].cpu().numpy())

        # Ensure axes_2d points are within image bounds
        h, w = img.shape[:2]
        axes_2d = np.clip(axes_2d, 0, [w - 1, h - 1])

        # Draw world axes
        cv2.line(img, tuple(axes_2d[0]), tuple(axes_2d[1]), (0, 0, 255), 2)  # X-axis (Red)
        cv2.line(img, tuple(axes_2d[0]), tuple(axes_2d[2]), (0, 255, 0), 2)  # Y-axis (Green)
        cv2.line(img, tuple(axes_2d[0]), tuple(axes_2d[3]), (255, 0, 0), 2)  # Z-axis (Blue)

        writer.write_frame(img)

        # Store 3D keypoints for JSON export
        frame_keypoints = joints_glob[i].cpu().numpy()
        frame_keypoints[:, 0] *= -1  # Negate x values
        keypoints_3d.append([
            {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])} for kp in frame_keypoints
        ])

        # Store camera pose for JSON export using inverted rotation
        camera_poses.append({
            "position": {
                "x": float(T_world2cam_fixed[i, 0].cpu()),
                "y": float(T_world2cam_fixed[i, 1].cpu()),
                "z": float(T_world2cam_fixed[i, 2].cpu())
            },
            "rotation": R_world2cam_inverted[i].cpu().numpy().tolist(),
            "focal_length": float(focal_length)
        })

    writer.close()
    reader.close()

    # Export 3D keypoints to JSON
    with open(cfg.paths.keypoints_3d_json, 'w') as f:
        json.dump(keypoints_3d, f, indent=2)

    # Export camera poses to JSON
    with open(cfg.paths.camera_poses_json, 'w') as f:
        json.dump(camera_poses, f, indent=2)

    Log.info(f"Composite video saved to: {cfg.paths.composite_video}")
    Log.info(f"3D keypoints saved to: {cfg.paths.keypoints_3d_json}")
    Log.info(f"Camera poses saved to: {cfg.paths.camera_poses_json}")

if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        torch.save(pred, paths.hmr4d_results)

    # ===== Render ===== #
    render_composite(cfg)

    Log.info(f"Composite video saved to: {paths.composite_video}")
    Log.info(f"3D keypoints saved to: {paths.keypoints_3d_json}")
    Log.info(f"Camera poses saved to: {paths.camera_poses_json}")

