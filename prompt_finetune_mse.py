"""
Prompt fine-tuning for ControlNet via grid search over deg/rag offsets.

For each sample in 1_step1_seen.json:
  1. Adjust predicted deg_num and rag_num by {-delta, 0, +delta} → 9 combos
  2. Read the pre-generated image for each combo from outputs/
  3. Select the combo whose output has the lowest pixel MSE with the target
  4. Report MAE before (original prediction) and after (best combo)
"""

import sys
import os
import json
import itertools
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════
# Configurable parameters
# ══════════════════════════════════════════════════════════════════════════
DEG_DELTA = 40           # offset applied to heading (deg_num)  — must match gen_img.py
RAG_DELTA = 0           # offset applied to range (rag_num)   — must match gen_img.py
DEG_OFFSETS = [-DEG_DELTA, 0, DEG_DELTA]  # candidate offsets for heading
RAG_OFFSETS = [-RAG_DELTA, 0, RAG_DELTA]  # candidate offsets for range

STEP1_JSON     = './step1_seen.json'
MAX_SAMPLES    = 2915       # only use the first N samples (set to None to use all)
IMG_DIR        = './pairUAV/tours/'
TEST_DIR       = './pairUAV/test/'
GENERATED_DIR  = './outputs/'       # pre-generated images from gen_img.py
OUTPUT_DIR     = './finetune_results_mse/'

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


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute pixel-wise MSE between two RGB uint8 images (resized to match if needed)."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def wrap_angle(deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (deg + 180) % 360 - 180


def angular_distance(a: float, b: float) -> float:
    """Circular distance between two angles in degrees (result in [0, 180])."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def compute_mae(pred_deg, pred_rag, true_deg, true_rag):
    """Compute per-sample MAE for deg (circular) and rag separately."""
    mae_deg = angular_distance(pred_deg, true_deg)
    mae_rag = abs(pred_rag - true_rag)
    return mae_deg, mae_rag


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

    # ── Accumulators for overall MAE ─────────────────────────────────────
    before_deg_errors, before_rag_errors = [], []
    after_deg_errors,  after_rag_errors  = [], []

    results_summary = []

    for idx in range(n_samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{n_samples}")
        print(f"{'='*60}")

        pred_deg = step1['pred_deg_num'][idx]
        pred_rag = step1['pred_rag_num'][idx]
        true_deg = step1['true_deg_num'][idx]
        true_rag = step1['true_rag_num'][idx]
        json_path = step1['json_path'][idx]

        # Load pair JSON to get image paths
        local_json = resolve_json_path(json_path)
        with open(local_json, 'r') as f:
            pair = json.load(f)

        source_path = os.path.join(IMG_DIR, pair['image_a'])
        target_path = os.path.join(IMG_DIR, pair['image_b'])
        source_img = load_image(source_path)   # RGB uint8
        target_img = load_image(target_path)    # RGB uint8

        print(f"  pred_deg={pred_deg:.2f}  pred_rag={pred_rag:.2f}")
        print(f"  true_deg={true_deg:.2f}  true_rag={true_rag:.2f}")
        print(f"  source={source_path}  target={target_path}")

        # ── Load 9 pre-generated variants from outputs/ ─────────────────
        combos = list(itertools.product(DEG_OFFSETS, RAG_OFFSETS))  # (deg_off, rag_off)
        gen_images = []
        prompts = []
        adjusted_vals = []
        sample_gen_dir = os.path.join(GENERATED_DIR, f'{idx:04d}')

        for ci, (d_off, r_off) in enumerate(combos):
            adj_deg = wrap_angle(pred_deg + d_off)
            adj_rag = pred_rag + r_off
            prompt = make_prompt(adj_deg, adj_rag)
            prompts.append(prompt)
            adjusted_vals.append((adj_deg, adj_rag))

            gen_path = os.path.join(sample_gen_dir, f'{ci:02d}.png')
            print(f"  [{ci+1}/9] deg_off={d_off:+.0f} rag_off={r_off:+.0f}  "
                  f"→ deg={adj_deg:.2f} rag={adj_rag:.2f}")
            print(f"        prompt: {prompt}")
            print(f"        loading: {gen_path}")

            gen_img = load_image(gen_path)
            gen_images.append(gen_img)

        # ── Compute MSE for each variant against target ──────────────────
        mse_scores = []
        for ci, gen_img in enumerate(gen_images):
            m = compute_mse(gen_img, target_img)
            mse_scores.append(m)
            print(f"  MSE[{ci}] (deg_off={combos[ci][0]:+.0f}, "
                  f"rag_off={combos[ci][1]:+.0f}): {m:.4f}")

        best_idx = int(np.argmin(mse_scores))
        best_deg, best_rag = adjusted_vals[best_idx]
        best_mse = mse_scores[best_idx]
        print(f"\n  ★ Best combo index={best_idx}  "
              f"deg_off={combos[best_idx][0]:+.0f}  "
              f"rag_off={combos[best_idx][1]:+.0f}  "
              f"MSE={best_mse:.4f}")
        print(f"    adjusted_deg={best_deg:.2f}  adjusted_rag={best_rag:.2f}")

        # ── MAE before / after ───────────────────────────────────────────
        mae_deg_before, mae_rag_before = compute_mae(
            pred_deg, pred_rag, true_deg, true_rag)
        mae_deg_after,  mae_rag_after  = compute_mae(
            best_deg, best_rag, true_deg, true_rag)

        before_deg_errors.append(mae_deg_before)
        before_rag_errors.append(mae_rag_before)
        after_deg_errors.append(mae_deg_after)
        after_rag_errors.append(mae_rag_after)

        print(f"  MAE before:  deg={mae_deg_before:.2f}  rag={mae_rag_before:.2f}  "
              f"total={mae_deg_before+mae_rag_before:.2f}")
        print(f"  MAE after :  deg={mae_deg_after:.2f}   rag={mae_rag_after:.2f}   "
              f"total={mae_deg_after+mae_rag_after:.2f}")

        results_summary.append({
            'sample_idx': idx,
            'json_path': json_path,
            'pred_deg': pred_deg, 'pred_rag': pred_rag,
            'true_deg': true_deg, 'true_rag': true_rag,
            'best_deg': best_deg, 'best_rag': best_rag,
            'best_deg_offset': combos[best_idx][0],
            'best_rag_offset': combos[best_idx][1],
            'best_mse': best_mse,
            'mse_scores': mse_scores,
            'mae_deg_before': mae_deg_before,
            'mae_rag_before': mae_rag_before,
            'mae_deg_after': mae_deg_after,
            'mae_rag_after': mae_rag_after,
        })

        # ── Visualisation: 3×3 grid + source + target ───────────────────
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle(
            f"Sample {idx+1}  |  "
            f"pred=({pred_deg:.1f}°, {pred_rag:.1f}m)  "
            f"true=({true_deg:.1f}°, {true_rag:.1f}m)  "
            f"best=({best_deg:.1f}°, {best_rag:.1f}m)  "
            f"MSE={best_mse:.4f}",
            fontsize=14
        )

        # Column 0 row 0: source
        axes[0, 0].imshow(source_img)
        axes[0, 0].set_title("Source (image_a)", fontsize=10)
        axes[0, 0].axis('off')

        # Column 0 row 1: target
        axes[1, 0].imshow(target_img)
        axes[1, 0].set_title("Target (image_b)", fontsize=10)
        axes[1, 0].axis('off')

        # Column 0 row 2: best generated
        axes[2, 0].imshow(gen_images[best_idx])
        axes[2, 0].set_title(f"Best (MSE={best_mse:.4f})", fontsize=10)
        axes[2, 0].axis('off')

        # Columns 1-3, rows 0-2: the 9 generated images (3×3 grid)
        for ci in range(9):
            row = ci // 3
            col = ci % 3 + 1  # shift to columns 1..3
            d_off, r_off = combos[ci]
            adj_deg, adj_rag = adjusted_vals[ci]
            border_color = 'red' if ci == best_idx else 'white'
            axes[row, col].imshow(gen_images[ci])
            title = (f"d={d_off:+.0f} r={r_off:+.0f}\n"
                     f"({adj_deg:.1f}°,{adj_rag:.1f}m)\n"
                     f"MSE={mse_scores[ci]:.4f}")
            axes[row, col].set_title(title, fontsize=8,
                                     color='red' if ci == best_idx else 'black')
            axes[row, col].axis('off')
            if ci == best_idx:
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)

        # Column 4: MSE bar chart
        axes[0, 4].barh(range(9),
                        mse_scores,
                        color=['red' if i == best_idx else 'steelblue'
                               for i in range(9)])
        labels = [f"d={c[0]:+.0f},r={c[1]:+.0f}" for c in combos]
        axes[0, 4].set_yticks(range(9))
        axes[0, 4].set_yticklabels(labels, fontsize=7)
        axes[0, 4].set_xlabel('MSE')
        axes[0, 4].set_title('MSE scores (lower is better)')

        # MAE comparison bar chart
        x_labels = ['deg_before', 'deg_after', 'rag_before', 'rag_after']
        x_vals = [mae_deg_before, mae_deg_after, mae_rag_before, mae_rag_after]
        colors = ['salmon', 'green', 'salmon', 'green']
        axes[1, 4].bar(x_labels, x_vals, color=colors)
        axes[1, 4].set_title('MAE comparison')
        axes[1, 4].set_ylabel('MAE')

        axes[2, 4].axis('off')

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, f'sample_{idx:04d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved visualisation → {fig_path}")

    # ══════════════════════════════════════════════════════════════════════
    # Overall summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")

    avg_deg_before = np.mean(before_deg_errors)
    avg_rag_before = np.mean(before_rag_errors)
    avg_deg_after  = np.mean(after_deg_errors)
    avg_rag_after  = np.mean(after_rag_errors)
    avg_total_before = avg_deg_before + avg_rag_before
    avg_total_after  = avg_deg_after  + avg_rag_after

    print(f"  Mean MAE (deg) before: {avg_deg_before:.4f}")
    print(f"  Mean MAE (deg) after : {avg_deg_after:.4f}")
    print(f"  Mean MAE (rag) before: {avg_rag_before:.4f}")
    print(f"  Mean MAE (rag) after : {avg_rag_after:.4f}")
    print(f"  Mean MAE (total) before: {avg_total_before:.4f}")
    print(f"  Mean MAE (total) after : {avg_total_after:.4f}")
    print(f"  DEG_DELTA: {DEG_DELTA}, RAG_DELTA: {RAG_DELTA}")

    # Save summary JSON
    summary = {
        'deg_delta': DEG_DELTA,
        'rag_delta': RAG_DELTA,
        'n_samples': n_samples,
        'avg_mae_deg_before': avg_deg_before,
        'avg_mae_deg_after': avg_deg_after,
        'avg_mae_rag_before': avg_rag_before,
        'avg_mae_rag_after': avg_rag_after,
        'avg_mae_total_before': avg_total_before,
        'avg_mae_total_after': avg_total_after,
        'per_sample': results_summary,
    }
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {summary_path}")

    # Overall comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"MAE Before vs After Prompt Fine-tuning [MSE] (deg_delta={DEG_DELTA}, rag_delta={RAG_DELTA})")

    categories = ['deg', 'rag', 'total']
    before_vals = [avg_deg_before, avg_rag_before, avg_total_before]
    after_vals  = [avg_deg_after,  avg_rag_after,  avg_total_after]

    x = np.arange(len(categories))
    w = 0.35
    axes[0].bar(x - w/2, before_vals, w, label='Before', color='salmon')
    axes[0].bar(x + w/2, after_vals,  w, label='After',  color='seagreen')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel('Mean MAE')
    axes[0].set_title('Mean MAE comparison')
    axes[0].legend()

    # Per-sample total MAE
    total_before = [d + r for d, r in zip(before_deg_errors, before_rag_errors)]
    total_after  = [d + r for d, r in zip(after_deg_errors,  after_rag_errors)]
    axes[1].plot(range(n_samples), total_before, 'o-', label='Before', color='salmon')
    axes[1].plot(range(n_samples), total_after,  's-', label='After',  color='seagreen')
    axes[1].set_xlabel('Sample index')
    axes[1].set_ylabel('Total MAE (deg+rag)')
    axes[1].set_title('Per-sample total MAE')
    axes[1].legend()

    plt.tight_layout()
    overall_fig_path = os.path.join(OUTPUT_DIR, 'overall_comparison.png')
    plt.savefig(overall_fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Overall figure saved → {overall_fig_path}")


if __name__ == '__main__':
    main()
