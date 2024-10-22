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
    get_video_lwhfps,
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
BATCH_LENGTH_SECONDS = 120  # Default batch length in seconds

def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--output_root", type=str, default="./outputs/demo", help="by default to outputs/demo")
    parser.add_argument("--pose_json", type=str, default=None, help="Path to the pose 2D JSON file, eg from YOLO pose output")
    parser.add_argument("--visual_odometry_json", type=str, default=None, help="Path to the visual odometry results json")
    parser.add_argument("--batch_length", type=int, default=BATCH_LENGTH_SECONDS, help="Batch length in seconds")
    args = parser.parse_args()

    # Input
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"

    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"batch_length={args.batch_length}"
        ]

        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        if args.pose_json is not None:
            overrides.append(f"+pose_json={args.pose_json}")
        if args.visual_odometry_json is not None:
            overrides.append(f"+visual_odometry_json={args.visual_odometry_json}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Add new paths
    cfg.video_path = video_path
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
        "yolopose": keypoints
    }

    return formatted_data

@torch.no_grad()
def run_preprocess(cfg, batch_video_path, formatted_data, start_frame, end_frame):
    paths = cfg.paths

    # Extract the relevant segment of pose data for the batch
    bbx_xyxy_full = formatted_data["bbx_xyxy"]
    bbx_xys_full = formatted_data["bbx_xys"]
    yolopose_full = formatted_data["yolopose"]

    bbx_xyxy = bbx_xyxy_full[start_frame:end_frame]
    bbx_xys = bbx_xys_full[start_frame:end_frame]
    keypoints = yolopose_full[start_frame:end_frame]

    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
    torch.save(keypoints, paths.vitpose)
    Log.info(f"[Preprocess] Saved vitpose data to {paths.vitpose}")

    # Get vit features
    extractor = Extractor()
    vit_features = extractor.extract_video_features(batch_video_path, bbx_xys)
    torch.save(vit_features, paths.vit_features)
    del extractor

def load_data_dict(cfg, length, width, height, traj):
    paths = cfg.paths

    R_w2c = quaternion_to_matrix(traj).mT
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

def process_batch(cfg, model, start_frame:int, end_frame:int, batch_index:int, fps:int, formatted_data, vo_results, image_size):
    # Copy relevant frames to a temporary video file
    video_filename = f"file_batch_{batch_index}.mp4"
    temp_video_path = os.path.join(cfg.output_dir, video_filename)
    reader = get_video_reader(cfg.video_path)
    writer = get_writer(temp_video_path, fps=fps, crf=CRF)

    for frame_idx, img in enumerate(tqdm(reader, total=end_frame - start_frame, desc=f"Copy Batch {batch_index}")):
        if start_frame <= frame_idx < end_frame:
            writer.write_frame(img)

    writer.close()
    reader.close()

    # Run preprocessing for the batch
    run_preprocess(cfg, temp_video_path, formatted_data, start_frame, end_frame)

    quaternion_array = vo_results[start_frame:end_frame]

    # Load data and run HMR4D
    data = load_data_dict(cfg, end_frame - start_frame, image_size[0], image_size[1], quaternion_array)
    pred = model.predict(data, static_cam=len(quaternion_array) > 0)
    pred = detach_to_cpu(pred)

    # Save HMR results for the batch
    batch_hmr_results_path = cfg.paths.hmr4d_results.replace(".pt", f"_batch_{batch_index}.pt")
    torch.save(pred, batch_hmr_results_path)

    # Clean up temporary video file
    os.remove(temp_video_path)

def stitch_batches_in_memory(cfg, num_batches:int):
    """
    Stitch the batches together and hold the results in memory.
    """
    all_transformed_keypoints_3d = []

    # Convert batch to global joints
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx = make_smplx("supermotion").to(device)
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").to(device)
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").to(device)

    for batch_index in range(num_batches):
        # Load current batch results
        batch_hmr_results_path = cfg.paths.hmr4d_results.replace(".pt", f"_batch_{batch_index}.pt")
        batch = torch.load(batch_hmr_results_path)

        smplx_out = smplx(**to_cuda(batch["smpl_params_global"]))
        pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
        joints_glob = einsum(J_regressor, pred_ay_verts, "j v, l v i -> l j i")  # (L, J, 3)

        # Append transformed joints to the list
        for i in range(len(joints_glob)):
            frame_keypoints = joints_glob[i].cpu().numpy()
            frame_keypoints[:, 0] *= -1  # Negate x values
            all_transformed_keypoints_3d.append([
                {"x": float(kp[0]), "y": float(kp[1]), "z": float(kp[2])} for kp in frame_keypoints
            ])

    return all_transformed_keypoints_3d

def json_to_rotations(data):
    """
    Extract quaternion rotations [qx, qy, qz, qw] from the provided sequential data (without keys).
    The data format is expected to be a list of lists where each sublist contains [x, y, z, q_x, q_y, q_z, q_w].
    """

    # Get the number of poses (based on the number of entries in the data list)
    n_poses = len(data)

    # Initialize an array for the quaternion rotations
    rotations = np.zeros((n_poses, 4))  # Array for quaternion rotations [q_x, q_y, q_z, q_w]

    # Iterate through the data and extract quaternion components
    for i, pose_data in enumerate(data):
        rotations[i, 0] = pose_data[3]  # q_x
        rotations[i, 1] = pose_data[4]  # q_y
        rotations[i, 2] = pose_data[5]  # q_z
        rotations[i, 3] = pose_data[6]  # q_w

    # Convert the NumPy array to a PyTorch tensor
    rotations_tensor = torch.tensor(rotations, dtype=torch.float32)

    return rotations_tensor

def load_json(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    # Determine video length, fps, and number of batches
    length, width, height, fps = get_video_lwhfps(cfg.video_path)
    image_size = (height, width)
    batch_length_frames = cfg.batch_length * fps
    num_batches = int((length + batch_length_frames - 1) // batch_length_frames)  # Use integer division

    formatted_data = load_and_format_pose_data(cfg.pose_json, image_size)
    vo_results = None
    if cfg.visual_odometry_json:
        vo_results = json_to_rotations(load_json(cfg.visual_odometry_json))

    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    # Process each batch
    for batch_index in range(num_batches):
        Log.info(f"Processing Batch {batch_index}")
        start_frame = int(batch_index * batch_length_frames)
        end_frame = int(min((batch_index + 1) * batch_length_frames, length))
        process_batch(cfg, model, start_frame, end_frame, batch_index, int(fps), formatted_data, vo_results, image_size)

    # Stitch all batches together in memory
    all_transformed_keypoints_3d = stitch_batches_in_memory(cfg, num_batches)

    # Serialize to JSON
    with open(cfg.paths.keypoints_3d_json, 'w') as f:
        json.dump(all_transformed_keypoints_3d, f, indent=2)

    Log.info(f"3D keypoints saved to: {paths.keypoints_3d_json}")
