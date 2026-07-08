import types

import torch
from einops import rearrange

def _patched_forward(self, x, timestep, context, clip_feature=None, y=None,
                     use_gradient_checkpointing=False,
                     use_gradient_checkpointing_offload=False,
                     num_clean_x_frames: int = 0,
                     frame1_timestep=None,
                     **kwargs):
    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

    context = self.text_embedding(context)

    if self.has_image_input and getattr(self, 'use_token_concat', False) and y is not None:
        # Patchify both streams independently: (B, D, f, h, w)
        y_tokens = self.patch_embedding(y)
        x_tokens = self.patch_embedding(x)

        B, D, f, h, w = x_tokens.shape
        N = f * h * w

        # Flatten: (B, D, f, h, w) → (B, N, D)
        y_tokens = rearrange(y_tokens, 'b d f h w -> b (f h w) d')
        x_tokens = rearrange(x_tokens, 'b d f h w -> b (f h w) d')

        # Conditioning first, then target: [B, 2N, D]
        x = torch.cat([y_tokens, x_tokens], dim=1)

        # RoPE frequencies 
        base_freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(N, 1, -1).to(x.device)

        freqs_y = base_freqs.clone()
        freqs_x = base_freqs.clone()
        freqs_x[..., -1] = -1.0  

        freqs = torch.cat([freqs_y, freqs_x], dim=0)  # [2N, 1, D_freq]

        # Per-token timestep modulation
        t_cond   = torch.zeros_like(timestep)  # y tokens see t=0 (clean)
        t_target = timestep                    # x tokens see t=T

        t_cond_emb   = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_cond).to(x.dtype))
        t_target_emb = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_target).to(x.dtype))

        t_cond_mod   = self.time_projection(t_cond_emb).unflatten(1, (6, self.dim))    # [B, 6, D]
        t_target_mod = self.time_projection(t_target_emb).unflatten(1, (6, self.dim))  # [B, 6, D]

        # y tokens all get t=0
        t_cond_mod_expanded = t_cond_mod.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, 6, D]

        # x tokens: first num_clean_x_frames get frame1_timestep, rest get t_target
        if num_clean_x_frames > 0 and num_clean_x_frames < f:
            n_clean = num_clean_x_frames * h * w
            n_noisy = N - n_clean
            if frame1_timestep is not None:
                t_f1_emb = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, frame1_timestep).to(x.dtype))
                t_f1_mod = self.time_projection(t_f1_emb).unflatten(1, (6, self.dim))
                t_x_f1 = t_f1_mod.unsqueeze(1).expand(-1, n_clean, -1, -1)
            else:
                t_x_f1 = t_cond_mod.unsqueeze(1).expand(-1, n_clean, -1, -1)
            t_x_noisy = t_target_mod.unsqueeze(1).expand(-1, n_noisy, -1, -1)
            t_target_mod_expanded = torch.cat([t_x_f1, t_x_noisy], dim=1)  # [B, N, 6, D]
        else:
            t_target_mod_expanded = t_target_mod.unsqueeze(1).expand(-1, N, -1, -1)

        # Full per-token modulation: [B, 2N, 6, D]
        t_mod = torch.cat([t_cond_mod_expanded, t_target_mod_expanded], dim=1)
        t = t_target_emb 

        if clip_feature is not None and hasattr(self, 'img_emb'):
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

    else:
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            if clip_feature is not None and hasattr(self, 'img_emb'):
                clip_embdding = self.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)

        x_emb = self.patch_embedding(x)
        B, D, f, h, w = x_emb.shape
        N = f * h * w
        x = rearrange(x_emb, 'b d f h w -> b (f h w) d')

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(N, 1, -1).to(x.device)

    # Transformer blocks
    for block in self.blocks:
        if use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                block, x, context, t_mod, freqs, use_reentrant=False)
        else:
            x = block(x, context, t_mod, freqs)

    if self.has_image_input and getattr(self, 'use_token_concat', False) and y is not None:
        x = x[:, N:, :]

    x = self.head(x, t)
    x = self.unpatchify(x, (f, h, w))
    return x


def patch_wan_dit(dit):
    dit.forward = types.MethodType(_patched_forward, dit)
