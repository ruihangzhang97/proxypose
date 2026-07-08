import argparse
import yaml
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

from inference.diffsynth_patches import patch_wan_dit
from inference.render_proxy import CubeTrajectoryRenderer, camera_space_point_to_proxy
from inference.utils import load_video, load_config, save_video, downsample_video
from inference.model_utils import encode_video_to_latents, decode_latents_to_video


def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


def load_model(config, device="cuda"):
    """Load the Proxy Video Generation model from checkpoint."""
    if config.get("lora_repo"):
        print(f"Downloading LoRA from HuggingFace: {config['lora_repo']}/{config['lora_filename']}")
        checkpoint_path = hf_hub_download(
            repo_id=config["lora_repo"],
            filename=config["lora_filename"],
        )
    else:
        checkpoint_path = config["checkpoint_path"]
        assert Path(checkpoint_path).exists(), f"Checkpoint not found at {checkpoint_path}"

    model_config_path = Path(checkpoint_path).parent / "config.yaml"
    model_config_yaml = load_config(model_config_path) if model_config_path.exists() else {}

    if config.get("prompt"):
        model_config_yaml["prompt"] = config["prompt"]

    local_model_path = str(Path(config["local_model_path"]).expanduser())
    model_id = config["model_id"]

    download_source = config.get("download_source", "huggingface")

    def _mc(pattern, **kwargs):
        return ModelConfig(
            model_id=model_id,
            origin_file_pattern=pattern,
            local_model_path=local_model_path,
            download_source=download_source,
            **kwargs,
        )

    model_configs = [
        _mc("diffusion_pytorch_model*.safetensors"),
        _mc("Wan2.1_VAE.pth"),
    ]
    tokenizer_config = _mc("google/umt5-xxl/")

    if config["offload_text_encoder"]:
        model_configs.append(_mc("models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"))
    else:
        model_configs.append(_mc("models_t5_umt5-xxl-enc-bf16.pth"))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )

    pipe.dit.has_image_input = True
    pipe.dit.use_token_concat = True
    pipe.dit.require_vae_embedding = False

    print(f"Loading LoRA checkpoint from {checkpoint_path} ...")
    pipe.load_lora(pipe.dit, checkpoint_path, alpha=config["lora_alpha"])
    print(f"LoRA checkpoint loaded (alpha={config['lora_alpha']})")

    patch_wan_dit(pipe.dit)

    if "prompt" in model_config_yaml:
        prompt = model_config_yaml["prompt"]
        print(f"Encoding prompt: '{prompt}'")
        prompt_context = encode_prompt(pipe, prompt).cpu()
    else:
        prompt_context = torch.zeros((1, 256, 4096), dtype=torch.bfloat16)

    pipe.prompt_context = prompt_context
    pipe.text_encoder = None
    pipe.tokenizer = None

    return pipe


def fov_to_intrinsics(fov_deg, h=512, w=512):
    """Convert FOV in degrees to camera intrinsic matrix."""
    fov_rad = np.radians(fov_deg)
    focal_length = (w / 2) / np.tan(fov_rad / 2)
    intrinsics = torch.tensor([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    return intrinsics, focal_length


def render_proxy(scale, cam_rt, object_rt, intrinsics, image_size=512, device="cuda"):
    """Render the proxy mesh at the given rotation and translation."""
    cube_renderer = CubeTrajectoryRenderer(scale, image_size=image_size, device=device)
    bgr_image, _ = cube_renderer.render_frame_rt(cam_rt, object_rt, intrinsics)
    return bgr_image


@torch.no_grad()
def ddim_sample(pipe, z_input, z_start, num_steps=50, noise_offset=500, cfg=1.0, device="cuda", progress_bar=True):
    """DDIM sampling with partial noise on the first frame, matching the training setup.

    Args:
        pipe:         Loaded WanVideoPipeline with prompt_context attribute.
        z_input:      VAE latents for the source video [1, C, F, H, W].
        z_start:      VAE latents for the first proxy frame [1, C, 1, H, W].
        num_steps:    Number of DDIM denoising steps.
        noise_offset: Timestep offset applied to the first frame (matching training).
        cfg:          Classifier-free guidance scale over source-video conditioning
                      (1.0 = no guidance, training default ≈ 1.0 at eval time).
        device:       Torch device string.
        progress_bar: Whether to display a tqdm progress bar.

    Returns:
        z_gen: Generated proxy video latents [1, C, F, H, W].
    """
    num_partial = 1
    context = pipe.prompt_context.to(device)

    z_clean_f1 = z_start
    noise_f1 = torch.randn_like(z_clean_f1)
    z_gen = torch.randn_like(z_input[:, :, num_partial:, :, :])

    pipe.scheduler.set_timesteps(num_steps)
    time_iter = tqdm(pipe.scheduler.timesteps) if progress_bar else pipe.scheduler.timesteps

    for timestep in time_iter:
        t_partial = max(timestep.item() - noise_offset, 0.0)
        z_f1_cond = (
            pipe.scheduler.add_noise(z_clean_f1, noise_f1, t_partial)
            if t_partial > 0 else z_clean_f1
        )

        t_partial_tensor = torch.tensor([t_partial], dtype=pipe.torch_dtype, device=device)
        
        x_input = torch.cat([z_f1_cond, z_gen], dim=2)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            flow_pred = pipe.dit(
                x=x_input,
                timestep=timestep.unsqueeze(0).to(device),
                context=context,
                y=z_input,
                clip_feature=None,
                num_clean_x_frames=num_partial,
                frame1_timestep=t_partial_tensor,
            )

            if cfg != 1.0:
                z_f1_uncond = pipe.scheduler.add_noise(z_clean_f1, noise_f1, timestep)
                t_uncond_tensor = torch.tensor([timestep.item()], dtype=pipe.torch_dtype, device=device)
                flow_uncond = pipe.dit(
                    x=torch.cat([z_f1_uncond, z_gen], dim=2),
                    timestep=timestep.unsqueeze(0).to(device),
                    context=context,
                    y=torch.zeros_like(z_input),
                    clip_feature=None,
                    num_clean_x_frames=0,
                    frame1_timestep=t_uncond_tensor,
                )
                flow_pred = flow_uncond + cfg * (flow_pred - flow_uncond)

        denoised_full = pipe.scheduler.step(flow_pred, timestep, x_input)
        z_gen = denoised_full[:, :, num_partial:, :, :]

    return torch.cat([z_clean_f1, z_gen], dim=2)


def generate_proxy_video(pipe, input_video, first_frame, config, device="cuda", seed=None):
    """Run proxy video generation on the input video.

    Args:
        pipe:        Loaded WanVideoPipeline.
        input_video: List of RGB frames (H, W, 3) uint8.
        first_frame: Single RGB frame used as the proxy first-frame condition.
        config:      Generation config dict (resolution, num_steps, noise_offset, proxy_scale).
        device:      Torch device string.
        seed:        Optional random seed for reproducibility.

    Returns:
        gen_video: List of RGB frames (H, W, 3) uint8 at the original input resolution.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    h, w = input_video[0].shape[:2]

    input_video = downsample_video(input_video, config["resolution"])
    first_frame = downsample_video([first_frame], config["resolution"])

    z_cond = encode_video_to_latents(pipe, input_video, device=device)
    z_start = encode_video_to_latents(pipe, first_frame, device=device)

    z_gen = ddim_sample(
        pipe,
        z_cond,
        z_start=z_start,
        num_steps=config["num_steps"],
        noise_offset=config["noise_offset"],
        cfg=config.get("cfg_scale", 1.0),
        device=device
    )

    gen_video = decode_latents_to_video(pipe, z_gen, device=device)
    gen_video = downsample_video(gen_video, target_size=(h, w))

    return gen_video


def prompt_to_proxy(prompt, intrinsics, image_size, proxy_scale, device="cpu"):
    """Render a proxy cube image from a pixel-space prompt.

    Args:
        prompt:      Dict with keys 'type' (str) and 'coordinates' (tuple of ints).
        intrinsics:  (3, 3) numpy array camera intrinsic matrix K.
        image_size:  (H, W) tuple of the image dimensions.
        proxy_scale: Relative size of the proxy cube on screen.
        device:      Torch device string.

    Returns:
        proxy_img: BGR numpy array (H, W, 3) of the rendered proxy cube.
    """
    object_rt, scale = camera_space_point_to_proxy(
        intrinsics=intrinsics,
        image_size=image_size[0],
        pixel_xy=prompt["coordinates"],
        depth=1.0,
        proxy_scale=proxy_scale,
    )
    proxy_img = render_proxy(scale * 0.5, torch.eye(4), object_rt, intrinsics, device=device)
    return proxy_img


DEFAULT_INTRINSICS = torch.tensor([
    [660.0, 0.0, 256.0],
    [0.0, 660.0, 256.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Proxy video generation")
    parser.add_argument("--config_path", type=str, required=True, help="Path to generation config YAML")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output video")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--prompt", type=int, nargs=2, required=True, metavar=("U", "V"),
                        help="Pixel coordinate (u v) of the target object in the first frame")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    config = load_config(args.config_path)
    video = load_video(args.video_path)

    print(f"Video loaded: {len(video)} frames")
    print(f"Loading model ...")
    pipe = load_model(config, device=args.device)

    prompt = {"type": "point", "coordinates": args.prompt}
    print(f"Using prompt: {prompt}")

    h, w = video[0].shape[:2]
    focal_deg = config.get("default_focal_deg", 60)
    focal_length_px = 0.5 * w / np.tan(np.radians(focal_deg / 2))
    intrinsics = np.array([
        [focal_length_px, 0, w / 2],
        [0, focal_length_px, h / 2],
        [0, 0, 1],
    ])

    proxy_img = prompt_to_proxy(prompt, intrinsics, image_size=(h, w),
                                proxy_scale=config["proxy_scale"], device=args.device)

    gen_video = generate_proxy_video(pipe, video, proxy_img, config, device=args.device, seed=args.seed)

    output_frames = []
    for i in range(len(gen_video)):
        side_by_side = np.concatenate([video[i][..., [2, 1, 0]], gen_video[i][..., [2, 1, 0]]], axis=1)
        output_frames.append(side_by_side)

    save_video(output_frames, args.output_path, fps=24)
    print(f"Saved {len(output_frames)} frames to {args.output_path}")


if __name__ == "__main__":
    main()
