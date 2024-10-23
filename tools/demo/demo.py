import torch
import argparse
import json
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_writer,
    get_video_reader,
)

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
    parser.add_argument("--pose_json", type=str, default=None, help="Path to the pose JSON file")
    parser.add_argument("--slam_results", type=str, default=None, help="Path to the SLAM results file in TUM format")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"

    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}"
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
    cfg.paths.keypoints_3d_json = os.path.join(cfg.output_dir, f"{Path(cfg.pose_json).stem}_keypoints_3d.json")

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

    traj_quat = torch.from_numpy(traj[:, [4, 5, 6, 7]])  # TUM format: x y z qx qy qz qw
    R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    bbx_xys = torch.load(paths.bbx)["bbx_xys"]
    kp2d = torch.load(paths.vitpose)
    f_imgseq = torch.load(paths.vit_features)

    cam_angvel = compute_cam_angvel(R_w2c)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys,
        "kp2d": kp2d,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "f_imgseq": f_imgseq,
    }
    return data

def export_json(cfg):
    pred = torch.load(cfg.paths.hmr4d_results)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(device)
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(device)

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    joints_glob = einsum(J_regressor, pred_ay_verts, "j v, l v i -> l j i")  # (L, J, 3)

    # Prepare data for JSON export
    keypoints_3d = []

    for i in tqdm(range(len(joints_glob)), desc="Rendering Composite"):
        # Store 3D keypoints for JSON export
        frame_keypoints = joints_glob[i].cpu().numpy()
        frame_keypoints[:, 0] *= -1  # Negate x values
        keypoints_3d.append([
            {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])} for kp in frame_keypoints
        ])

    # Export 3D keypoints to JSON
    with open(cfg.paths.keypoints_3d_json, 'w') as f:
        json.dump(keypoints_3d, f, indent=2)

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
        pred = model.predict(data, static_cam=not Path(paths.slam).exists())
        pred = detach_to_cpu(pred)
        torch.save(pred, paths.hmr4d_results)

    export_json(cfg)
    Log.info(f"3D keypoints saved to: {paths.keypoints_3d_json}")

