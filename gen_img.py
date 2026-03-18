"""
Generate images using ControlNet for each sample in step1_seen.json.

For each sample:
  1. Adjust predicted deg_num and rag_num by {-delta, 0, +delta} → 9 combos
  2. Generate an image for each combo using the trained ControlNet
  3. Save each generated image to  outputs/{sample_id}/{gen_id}.png
"""

import sys
import os

os.environ["NNPACK_DISABLE"] = "1"          # disable NNPACK entirely
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" # suppress C++ warnings

import json
import itertools
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import einops

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# ══════════════════════════════════════════════════════════════════════════
# Configurable parameters
# ══════════════════════════════════════════════════════════════════════════
DEG_DELTA = 40           # offset applied to heading (deg_num)
RAG_DELTA = 0           # offset applied to range (rag_num)
DEG_OFFSETS = [-DEG_DELTA, 0, DEG_DELTA]  # candidate offsets for heading
RAG_OFFSETS = [-RAG_DELTA, 0, RAG_DELTA]  # candidate offsets for range

STEP1_JSON   = './step1_seen.json'
MAX_SAMPLES  = 2916      # only use the first N samples (set to None to use all)
MODEL_CONFIG = './models/cldm_v15.yaml'
CKPT_PATH    = './lightning_logs/version_0/checkpoints/last.ckpt'
IMG_DIR      = './pairUAV/tours/'
TEST_DIR     = './pairUAV/test/'
OUTPUT_DIR   = './outputs/'

DDIM_STEPS   = 20
GUIDANCE_SCALE = 9.0
SEED         = 42
IMAGE_RES    = 512      # images are already 512×512
BATCH_SIZE   = 9        # how many images to generate in parallel (set ≤ combo count)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ══════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════

def make_prompt(heading: float, rng: float) -> str:
    """Build the text prompt from heading (deg) and range (m) values."""
    h_dir = "right" if heading >= 0 else "left"
    r_dir = "forward" if rng >= 0 else "backward"
    return (
        f"Aerial drone view after turning {h_dir} {abs(heading):.0f} degrees "
        f"and moving {r_dir} {abs(rng):.0f} meters"
    )


def load_image(path: str) -> np.ndarray:
    """Read an image (BGR→RGB, uint8)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resolve_json_path(json_path: str) -> str:
    """Map json_path from the prediction file to the actual local path.
    e.g. /root/pairUAV/1_test/0000/01_01.json → ./pairUAV/test/0000/01_01.json
    """
    # Extract the tour_id and filename from the path
    parts = json_path.replace('\\', '/').split('/')
    # Find the pattern: .../<tour_id>/<filename>.json
    filename = parts[-1]
    tour_id = parts[-2]
    local_path = os.path.join(TEST_DIR, tour_id, filename)
    if os.path.exists(local_path):
        return local_path
    # Fallback: try original path
    if os.path.exists(json_path):
        return json_path
    raise FileNotFoundError(
        f"Cannot resolve json_path: {json_path}  (tried {local_path})"
    )


def wrap_angle(deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (deg + 180) % 360 - 180


@torch.no_grad()
def generate_images_batch(model, ddim_sampler, source_np: np.ndarray,
                         prompts: list) -> list:
    """Run ControlNet inference for a batch of prompts sharing the same source
    image.  Returns a list of RGB uint8 numpy images."""
    n = len(prompts)
    # Prepare control (source image normalised to [0,1]), repeat for batch
    control = torch.from_numpy(source_np.copy()).float() / 255.0
    control = control.unsqueeze(0)                          # 1,H,W,C
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    control = control.repeat(n, 1, 1, 1)                    # N,C,H,W
    control = control.to(DEVICE)

    H, W = source_np.shape[:2]

    # Conditioning
    cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning(prompts)]
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [model.get_learned_conditioning([""] * n)]
    }
    shape = (4, H // 8, W // 8)

    model.control_scales = [1.0] * 13
    samples, _ = ddim_sampler.sample(
        DDIM_STEPS, n, shape, cond,
        verbose=False, eta=0.0,
        unconditional_guidance_scale=GUIDANCE_SCALE,
        unconditional_conditioning=un_cond
    )

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5
    ).cpu().numpy().clip(0, 255).astype(np.uint8)

    return [x_samples[i] for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load step-1 predictions ──────────────────────────────────────────
    with open(STEP1_JSON, 'r') as f:
        step1 = json.load(f)

    n_total = len(step1['json_path'])
    if MAX_SAMPLES is not None and MAX_SAMPLES < n_total:
        for k in step1:
            step1[k] = step1[k][:MAX_SAMPLES]
    n_samples = len(step1['json_path'])
    print(f"Loaded {n_total} samples from {STEP1_JSON}, using first {n_samples}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model on {DEVICE} ...")
    model = create_model(MODEL_CONFIG).cpu()
    model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'))
    model = model.to(DEVICE).eval()
    # The FrozenCLIPEmbedder stores device as a plain string attribute
    # (default "cuda") which is NOT updated by model.to().  Fix it here.
    if hasattr(model, 'cond_stage_model') and hasattr(model.cond_stage_model, 'device'):
        model.cond_stage_model.device = DEVICE
    ddim_sampler = DDIMSampler(model)

    # Monkey-patch DDIMSampler.register_buffer to respect DEVICE instead of
    # hardcoded cuda (the original code unconditionally moves tensors to cuda).
    _device = torch.device(DEVICE)
    def _register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            attr = attr.to(_device)
        setattr(self, name, attr)
    import types
    ddim_sampler.register_buffer = types.MethodType(_register_buffer, ddim_sampler)

    print(f"Model loaded on {DEVICE}.")

    from pytorch_lightning import seed_everything
    seed_everything(SEED)

    combos = list(itertools.product(DEG_OFFSETS, RAG_OFFSETS))

    for idx in range(n_samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{n_samples}")
        print(f"{'='*60}")

        pred_deg = step1['pred_deg_num'][idx]
        pred_rag = step1['pred_rag_num'][idx]
        json_path = step1['json_path'][idx]

        # Load pair JSON to get source image path
        local_json = resolve_json_path(json_path)
        with open(local_json, 'r') as f:
            pair = json.load(f)

        source_path = os.path.join(IMG_DIR, pair['image_a'])
        source_img = load_image(source_path)   # RGB uint8

        print(f"  pred_deg={pred_deg:.2f}  pred_rag={pred_rag:.2f}")
        print(f"  source={source_path}")

        # Create output directory for this sample
        sample_dir = os.path.join(OUTPUT_DIR, f'{idx:04d}')
        os.makedirs(sample_dir, exist_ok=True)

        # ── Build all prompts for this sample ────────────────────────────
        combo_info = []
        for ci, (d_off, r_off) in enumerate(combos):
            adj_deg = wrap_angle(pred_deg + d_off)
            adj_rag = pred_rag + r_off
            prompt = make_prompt(adj_deg, adj_rag)
            combo_info.append((ci, d_off, r_off, adj_deg, adj_rag, prompt))
            print(f"  [{ci+1}/{len(combos)}] deg_off={d_off:+.0f} rag_off={r_off:+.0f}  "
                  f"→ deg={adj_deg:.2f} rag={adj_rag:.2f}")
            print(f"        prompt: {prompt}")

        # ── Generate in batches ──────────────────────────────────────────
        for batch_start in range(0, len(combo_info), BATCH_SIZE):
            batch = combo_info[batch_start:batch_start + BATCH_SIZE]
            prompts = [info[5] for info in batch]

            print(f"  >> Generating batch [{batch_start+1}–{batch_start+len(batch)}] "
                  f"({len(batch)} images) ...")
            gen_imgs = generate_images_batch(
                model, ddim_sampler, source_img, prompts
            )

            for (ci, d_off, r_off, adj_deg, adj_rag, prompt), g_img in zip(batch, gen_imgs):
                save_path = os.path.join(sample_dir, f'{ci:02d}.png')
                cv2.imwrite(save_path, cv2.cvtColor(g_img, cv2.COLOR_RGB2BGR))
                print(f"        saved → {save_path}")

    print(f"\nDone. Generated images saved under {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
