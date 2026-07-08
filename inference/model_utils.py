import torch
import numpy as np


def encode_video_to_latents(pipe, video, device="cuda"):
    """Encode video frames to latents using the VAE."""
    arrays = [np.array(img).astype(np.float32) / 255.0 * 2 - 1 for img in video]
    video_array = np.stack(arrays, axis=0)  # [F, H, W, C]
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2)  # [C, F, H, W]
    video_tensor = video_tensor.unsqueeze(0)  # [1, C, F, H, W]
    
    # Get VAE dtype (bfloat16)
    vae_dtype = next(pipe.vae.parameters()).dtype
    video_tensor = video_tensor.to(device, dtype=vae_dtype)
    
    with torch.no_grad():
        z_cond = pipe.vae.encode(video_tensor, device=device)
    
    return z_cond


def decode_latents_to_video(pipe, latents, device="cuda"):
    """Decode latents back to video frames using the VAE."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        video = pipe.vae.decode(latents.bfloat16(), device=device).float().cpu()
    
    # Convert to numpy [F, H, W, C]
    video = video[0].permute(1, 2, 3, 0).numpy()
    
    # Denormalize from [-1, 1] to [0, 255]
    video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    
    return video