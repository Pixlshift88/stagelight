"""
STAGELIGHT v0.9.5
Fresh start. One goal: node opens panel automatically.
"""

import sys, json, subprocess, shutil
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QFileDialog, QMessageBox
)
from PySide6.QtCore  import Qt, QThread, Signal, QTimer
from PySide6.QtGui   import QColor, QPalette

BASE_DIR      = Path(__file__).parent.resolve()
CONFIG_FILE   = BASE_DIR / "config.json"
MANIFEST_FILE = BASE_DIR / "install_manifest.json"

APP_NAME = "STAGELIGHT"
APP_VER  = "0.9.5"
COMFY_PORT = 8188

# ══════════════════════════════════════════════════════════════
#  PLUGIN FILES — everything written to ComfyUI/custom_nodes/stagelight/
# ══════════════════════════════════════════════════════════════

PLUGIN_FILES = {}

# ── __init__.py ───────────────────────────────────────────────────────────────
PLUGIN_FILES["__init__.py"] = """\
from .node import StagelightNode

NODE_CLASS_MAPPINGS         = {"StagelightNode": StagelightNode}
NODE_DISPLAY_NAME_MAPPINGS  = {"StagelightNode": "STAGELIGHT"}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
print("[STAGELIGHT] v0.9.5 loaded.")
"""

# ── node.py — the Python node ─────────────────────────────────────────────────
PLUGIN_FILES["node.py"] = """\
# STAGELIGHT v0.9.5 - SmartNormal generator + three-source normal blend + cleanup outputs

import torch
import numpy as np

_dpt_proc       = None
_dpt_model      = None
_rembg_session  = None
_bae_detector   = None

def _load_depth():
    global _dpt_proc, _dpt_model
    if _dpt_model is None:
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        print("[STAGELIGHT] Loading depth model...")
        _dpt_proc  = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        _dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        _dpt_model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _dpt_model = _dpt_model.to(dev)
        print(f"[STAGELIGHT] Depth model on {dev}.")
    return _dpt_proc, _dpt_model


def _load_rembg():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        print("[STAGELIGHT] Loading rembg subject model (first run ~175MB)...")
        _rembg_session = new_session("u2net")
        print("[STAGELIGHT] rembg ready.")
    return _rembg_session


def _load_bae():
    global _bae_detector
    if _bae_detector is None:
        try:
            from controlnet_aux import NormalBaeDetector
            print("[STAGELIGHT] Loading BAE normal model (first run ~500MB)...")
            _bae_detector = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _bae_detector = _bae_detector.to(dev)
            print(f"[STAGELIGHT] BAE normal model on {dev}.")
        except Exception as e:
            print(f"[STAGELIGHT] BAE unavailable ({e}). Using depth-gradient fallback.")
            _bae_detector = "FALLBACK"
    return _bae_detector


def estimate_depth(pil_img):
    proc, model = _load_depth()
    dev = next(model.parameters()).device
    ow, oh = pil_img.size
    inp = proc(images=pil_img, return_tensors="pt")
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        pred = model(**inp).predicted_depth
    dr = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(oh, ow),
        mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    mn, mx = dr.min(), dr.max()
    return ((dr - mn) / (mx - mn + 1e-8)).clip(0, 1).astype(np.float32)


def estimate_subject_mask(pil_img):
    from rembg import remove
    session = _load_rembg()
    out = remove(pil_img, session=session)
    alpha = np.array(out.split()[-1], dtype=np.float32) / 255.0
    return alpha.astype(np.float32)


def estimate_normal_bae(pil_img, target_hw, supersample=2.0):
    det = _load_bae()
    if det == "FALLBACK":
        return None
    try:
        from PIL import Image as PILImage
        th, tw = target_hw
        ss = max(1.0, supersample)
        res_target = int(round(max(th, tw) * ss))
        res = min(max(res_target, 512), 1536)
        try:
            result = det(pil_img, detect_resolution=res, image_resolution=res)
        except TypeError:
            result = det(pil_img)
        if result.size != (tw, th):
            result = result.resize((tw, th), PILImage.LANCZOS)
        arr = np.asarray(result).astype(np.float32) / 255.0
        return arr
    except Exception as e:
        print(f"[STAGELIGHT] BAE normal failed: {e}.")
        return None


def smart_normal(np_img_01, blur_sigma=1.0, strength=4.0, detail_pow=1.0, equalize=False):
    # SmartNormal-style normal generation: Sobel gradients on luminance.
    # np_img_01: (H, W, 3) float32 in [0,1]
    # blur_sigma: pre-blur to smooth noise (0 disables)
    # strength: gradient amplification - higher = more pronounced normals
    # detail_pow: gamma applied to luminance (>1 darkens midtones, <1 brightens) - crunches values
    # equalize: histogram-style stretch to use full luminance range
    R, G, B = np_img_01[:,:,0], np_img_01[:,:,1], np_img_01[:,:,2]
    gray = (0.299*R + 0.587*G + 0.114*B).astype(np.float32)

    if equalize:
        mn, mx = gray.min(), gray.max()
        if mx - mn > 1e-6:
            gray = (gray - mn) / (mx - mn)

    if abs(detail_pow - 1.0) > 1e-3:
        gray = np.power(np.clip(gray, 1e-8, 1.0), detail_pow).astype(np.float32)

    if blur_sigma > 0.05:
        try:
            from scipy.ndimage import gaussian_filter
            gray = gaussian_filter(gray, sigma=blur_sigma, mode='reflect').astype(np.float32)
        except ImportError:
            pass

    # Sobel kernels via np.gradient (close enough to Sobel for our purposes)
    gy, gx = np.gradient(gray)
    nx = -gx * strength
    ny =  gy * strength  # flipped y to match standard normal map convention (Y up)
    nz = np.ones_like(gray, dtype=np.float32)
    length = np.sqrt(nx*nx + ny*ny + nz*nz).clip(1e-8, None)
    nx /= length
    ny /= length
    nz /= length
    return np.dstack([(nx + 1) * 0.5, (ny + 1) * 0.5, (nz + 1) * 0.5]).astype(np.float32)


def depth_to_normal(depth, strength=4.0):
    dx = np.gradient(depth * strength, axis=1).astype(np.float32)
    dy = np.gradient(depth * strength, axis=0).astype(np.float32)
    h, w = depth.shape
    nx = -dx
    ny = -dy
    nz = np.ones((h, w), dtype=np.float32)
    length = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= length
    ny /= length
    nz /= length
    return np.dstack([(nx + 1) * 0.5, (ny + 1) * 0.5, (nz + 1) * 0.5]).astype(np.float32)


def blend_three_normals(n_smart, n_bae, n_depth, w_smart, w_bae, w_depth):
    # Weighted blend in signed normal space, then renormalize
    total = max(w_smart + w_bae + w_depth, 1e-6)
    w_smart /= total; w_bae /= total; w_depth /= total
    s = (n_smart * 2.0 - 1.0) if n_smart is not None else None
    b = (n_bae   * 2.0 - 1.0) if n_bae   is not None else None
    d = (n_depth * 2.0 - 1.0) if n_depth is not None else None
    acc = None
    def add(acc, vec, weight):
        if vec is None or weight <= 0:
            return acc
        contrib = vec * weight
        return contrib if acc is None else acc + contrib
    acc = add(acc, s, w_smart)
    acc = add(acc, b, w_bae)
    acc = add(acc, d, w_depth)
    if acc is None:
        # Fallback - all weights zero or all sources missing - use neutral
        h, w = (n_smart if n_smart is not None else (n_bae if n_bae is not None else n_depth)).shape[:2]
        acc = np.zeros((h, w, 3), dtype=np.float32)
        acc[:,:,2] = 1.0
    length = np.sqrt((acc * acc).sum(axis=2, keepdims=True)).clip(1e-8, None)
    acc = acc / length
    return ((acc + 1.0) * 0.5).astype(np.float32)


def boost_normal_contrast(normal_01, strength):
    if 0.999 <= strength <= 1.001:
        return normal_01
    neutral = np.array([0.5, 0.5, 1.0], dtype=np.float32).reshape(1, 1, 3)
    deviation = normal_01 - neutral
    boosted = neutral + deviation * strength
    n = boosted * 2.0 - 1.0
    length = np.sqrt((n * n).sum(axis=2, keepdims=True)).clip(1e-8, None)
    n = n / length
    return ((n + 1.0) * 0.5).astype(np.float32)


def boost_high_frequency(normal_01, hf_strength, sigma=4.0):
    if 0.999 <= hf_strength <= 1.001:
        return normal_01
    try:
        from scipy.ndimage import gaussian_filter
        lowpass = np.zeros_like(normal_01)
        for c in range(3):
            lowpass[:,:,c] = gaussian_filter(normal_01[:,:,c], sigma=sigma, mode='reflect')
    except ImportError:
        return normal_01
    highpass = normal_01 - lowpass
    boosted = lowpass + highpass * hf_strength
    n = boosted * 2.0 - 1.0
    length = np.sqrt((n * n).sum(axis=2, keepdims=True)).clip(1e-8, None)
    n = n / length
    return ((n + 1.0) * 0.5).astype(np.float32)


def depth_to_world_position(depth, disp_amount=0.5):
    # Match the GPU viewport's coordinate system exactly.
    # The viewport mesh is a flat plane:
    #   - width  2.0 * aspect, x in [-aspect, +aspect]
    #   - height 2.0,           y in [-1, +1]  (top row = +1, bottom row = -1)
    #   - z      = depth * disp_amount (subject pops toward camera = +Z)
    # Camera sits at (0, 0, +3) by default looking at origin (set elsewhere).
    h, w = depth.shape
    aspect = w / h
    xs = np.linspace(-aspect, aspect, w, dtype=np.float32)
    ys = np.linspace( 1.0, -1.0, h, dtype=np.float32)  # row 0 = top = +1
    xv, yv = np.meshgrid(xs, ys)
    z = depth * disp_amount
    return np.dstack([xv, yv, z]).astype(np.float32)


def phong_full(normal_01, world_pos, view_pos, light_pos, target_pos,
               intensity, cone_reach, falloff_curve, spec_strength, spec_shiny,
               light_type, spot_cos_inner, spot_cos_outer, fixed_intensity=True,
               light_size=0.0):
    # Returns (diffuse_intensity HxW, specular_intensity HxW)
    # light_type: 0=point, 1=spot, 2=directional
    N = normal_01 * 2.0 - 1.0
    N = N / (np.linalg.norm(N, axis=2, keepdims=True).clip(1e-8, None))

    if light_type == 2:
        # Directional: light direction is constant from light_pos toward target
        Ldir = target_pos - light_pos
        Ldir = Ldir / (np.linalg.norm(Ldir) + 1e-8)
        L = -Ldir[np.newaxis, np.newaxis, :]
        L = np.broadcast_to(L, world_pos.shape).astype(np.float32)
        attn = np.ones(world_pos.shape[:2], dtype=np.float32) * intensity
    else:
        # Point or spot: per-pixel direction from surface to light
        LV = light_pos[np.newaxis, np.newaxis, :] - world_pos
        dist = np.linalg.norm(LV, axis=2, keepdims=True).clip(1e-8, None)
        L = LV / dist
        d = dist[:, :, 0]
        if fixed_intensity:
            # Art-directable: brightness constant within reach, hard zero past it
            within = (d <= cone_reach).astype(np.float32)
            attn = within * intensity
        else:
            # Physical-ish radial falloff
            clamped = np.clip(d / max(cone_reach, 0.001), 0.0, 1.0)
            radial = np.power(np.clip(1.0 - clamped * clamped, 0.0, 1.0), falloff_curve).astype(np.float32)
            attn = radial * intensity

    # Wrapped lighting: light_size > 0 softens the terminator (approximates area light)
    raw_NdotL = np.sum(N * L, axis=2)
    w = max(float(light_size), 0.0)
    NdotL = np.clip((raw_NdotL + w) / (1.0 + w), 0.0, 1.0)

    # Spotlight cone
    if light_type == 1:
        # Light forward direction = from light position toward target
        Ldir = target_pos - light_pos
        Ldir_norm = np.linalg.norm(Ldir) + 1e-8
        Ldir = Ldir / Ldir_norm
        # cos(angle) between -L (light-to-surface) and Ldir
        # -L because L is surface-to-light, we want light-to-surface
        minusL = -L
        cosTheta = np.sum(minusL * Ldir[np.newaxis, np.newaxis, :], axis=2)
        # Smooth cone: full at cos > inner, zero at cos < outer
        cone = np.clip((cosTheta - spot_cos_outer) / max(spot_cos_inner - spot_cos_outer, 1e-3), 0.0, 1.0).astype(np.float32)
        attn = attn * cone

    diffuse = (NdotL * attn).astype(np.float32)

    # Specular (Blinn-Phong: half vector dot N)
    if spec_strength > 0.001:
        Vv = view_pos[np.newaxis, np.newaxis, :] - world_pos
        Vv = Vv / (np.linalg.norm(Vv, axis=2, keepdims=True).clip(1e-8, None))
        H = L + Vv
        H = H / (np.linalg.norm(H, axis=2, keepdims=True).clip(1e-8, None))
        NdotH = np.sum(N * H, axis=2).clip(0, 1)
        # Mask specular by NdotL > 0 to avoid backside highlights
        mask = (NdotL > 0).astype(np.float32)
        specular = (np.power(NdotH, max(spec_shiny, 1.0)) * mask * attn * spec_strength).astype(np.float32)
    else:
        specular = np.zeros_like(diffuse)

    return diffuse, specular


def phong_diffuse(normal_01, world_pos, light_pos, intensity, falloff):
    # Backward-compat wrapper kept for any older callers
    N = normal_01 * 2.0 - 1.0
    N = N / (np.linalg.norm(N, axis=2, keepdims=True).clip(1e-8, None))
    LV = light_pos[np.newaxis, np.newaxis, :] - world_pos
    dist = np.linalg.norm(LV, axis=2, keepdims=True).clip(1e-8, None)
    L = LV / dist
    NdotL = np.sum(N * L, axis=2).clip(0, 1)
    attn = 1.0 / (1.0 + falloff * (dist[:, :, 0] ** 2))
    return (NdotL * attn * intensity).astype(np.float32)


def apply_depth_cutoff(depth, mask, cutoff, feather):
    if cutoff >= 0.999:
        return mask
    feather = max(feather, 1e-3)
    upper = cutoff + feather
    visibility = np.clip((upper - depth) / feather, 0.0, 1.0).astype(np.float32)
    return (mask * visibility).astype(np.float32)


# FUTURE ROADMAP (v0.8.x+): multi-light support.
# Each light entry should be a dict with its own:
#   - position (x, y, z)
#   - color (r, g, b)
#   - intensity, falloff
#   - PER-LIGHT separation attributes (bg_depth_scale, bg_visibility)
#   - PER-LIGHT depth cutoff (depth_cutoff, cutoff_feather)


class StagelightNode:

    @classmethod
    def INPUT_TYPES(cls):
        def f(d, mn, mx, st):
            return ("FLOAT", {"default": d, "min": mn, "max": mx, "step": st, "display": "slider"})
        return {
            "required": {
                "image":             ("IMAGE",),
                "light_x":           f(-0.5, -3.0,  3.0, 0.05),
                "light_y":           f( 1.0, -3.0,  3.0, 0.05),
                "light_z":           f( 2.0, -3.0,  5.0, 0.05),
                "intensity":         f( 1.0,  0.0, 15.0, 0.05),
                "cone_reach":        f( 5.0,  0.5, 20.0, 0.1),
                "falloff_curve":     f( 2.0,  0.5,  4.0, 0.1),
                # v0.8.8: this param is now used as EXPOSURE (multiplicative final pass).
                # Kept name "ambient" for backward compat with old saved workflows.
                "ambient":           f( 1.0,  0.0,  2.0, 0.05),
                "gamma":             f( 1.0,  0.5,  2.5, 0.05),
                "light_r":           f( 1.0,  0.0,  1.0, 0.01),
                "light_g":           f( 1.0,  0.0,  1.0, 0.01),
                "light_b":           f( 1.0,  0.0,  1.0, 0.01),
                "bg_depth_scale":    f( 0.24, 0.0,  1.0, 0.01),
                "depth_cutoff":      f( 1.0,  0.0,  1.0, 0.01),
                "cutoff_feather":    f( 0.1,  0.01, 0.5, 0.01),
                "bg_visibility":     f( 0.0,  0.0,  1.0, 0.01),
                # Three-source normal blend - smart-heavy default for AI-gen imagery
                "weight_smart":      f( 0.7,  0.0,  1.0, 0.01),
                "weight_bae":        f( 0.2,  0.0,  1.0, 0.01),
                "weight_depth":      f( 0.1,  0.0,  1.0, 0.01),
                # SmartNormal controls
                "smart_strength":    f( 2.9,  0.5, 12.0, 0.1),
                "smart_blur":        f( 0.0,  0.0,  4.0, 0.1),
                "smart_detail":      f( 0.3,  0.3,  3.0, 0.05),
                # Post-process on the blended normal
                "normal_strength":   f( 1.0,  0.5,  6.0, 0.1),
                "hf_detail":         f( 1.0,  1.0,  5.0, 0.1),
                "bae_supersample":   f( 1.0,  1.0,  3.0, 0.25),
                "use_bae_normal":    ("BOOLEAN", {"default": True}),
                "smart_equalize":    ("BOOLEAN", {"default": False}),
                # Specular highlights (Phong)
                "spec_strength":     f( 0.3,  0.0,  2.0, 0.05),
                "spec_shiny":        f( 16.0, 1.0, 128.0, 1.0),
                # Spotlight params (target point + cone angle)
                "target_x":          f( 0.0, -3.0,  3.0, 0.05),
                "target_y":          f( 0.0, -3.0,  3.0, 0.05),
                "target_z":          f( 0.0, -3.0,  3.0, 0.05),
                "spot_angle_deg":    f( 60.0, 1.0, 180.0, 1.0),
                "spot_softness":     f( 0.7,  0.0,  1.0, 0.01),
                # Light type: 0=point, 1=spot, 2=directional
                "light_type":        ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "fixed_intensity":   ("BOOLEAN", {"default": True}),
                "light_size":        f( 0.0,  0.0,  1.0, 0.01),
                # Displacement amount used to compute world-space Z. Should match the
                # viewport's VIEW POP slider so the lighting math sees the same surface
                # geometry the user is interacting with.
                "disp_amount":       f( 0.5,  0.0,  1.5, 0.05),
                # Multi-light JSON (v0.8.3): when non-empty, contains an array of light
                # objects. If present, all enabled lights accumulate in the render. If
                # empty, falls back to the legacy single-light parameters above (so old
                # workflows still render correctly).
                "lights_json":       ("STRING", {"default": "", "multiline": False}),
                # === Background mask (v0.8.6) ===
                # === Background mask (v0.8.7) ===
                # mask_cutoff: depth threshold (0-1). Pixels below this are hidden.
                # mask_falloff: smoothstep transition width around cutoff.
                # mask_invert: 1 = flip which side is hidden (cuts subject instead of BG).
                "mask_cutoff":       f( 0.05, 0.0,  1.0, 0.01),
                "mask_falloff":      f( 0.0,  0.0,   0.5, 0.005),
                "mask_invert":       ("BOOLEAN", {"default": False}),
                # === Grade chain (v0.8.9, Nuke-style) ===
                "exposure_stops":    f( 0.0, -5.0,  5.0, 0.05),
                "grade_gain":        f( 1.0,  0.5,  2.0, 0.01),
                "grade_lift":        f( 0.0, -0.2,  0.2, 0.005),
                # === Shadow control (v0.8.11) ===
                # shadow_strength: 1 = full shadow (unlit areas dark), 0 = pass-through.
                "shadow_strength":   f( 1.0,  0.0,  1.0, 0.01),
                # === Grade tints (v0.8.13) - per-channel multipliers ===
                # Defaults all 1.0 = neutral white tint = identity grade.
                "grade_gain_r":      f( 1.0,  0.0,  2.0, 0.01),
                "grade_gain_g":      f( 1.0,  0.0,  2.0, 0.01),
                "grade_gain_b":      f( 1.0,  0.0,  2.0, 0.01),
                "grade_lift_r":      f( 1.0,  0.0,  2.0, 0.01),
                "grade_lift_g":      f( 1.0,  0.0,  2.0, 0.01),
                "grade_lift_b":      f( 1.0,  0.0,  2.0, 0.01),
                "grade_gamma_r":     f( 1.0,  0.0,  2.0, 0.01),
                "grade_gamma_g":     f( 1.0,  0.0,  2.0, 0.01),
                "grade_gamma_b":     f( 1.0,  0.0,  2.0, 0.01),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "depth_map", "normal_map", "relit_image", "subject_mask", "final_alpha", "relit_composite", "lighting_only")
    FUNCTION     = "run"
    CATEGORY     = "STAGELIGHT"
    DESCRIPTION  = "VFX relight. SmartNormal + BAE + Depth-gradient three-source blend."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    def run(self, image,
            light_x, light_y, light_z,
            intensity, cone_reach, falloff_curve, ambient, gamma,
            light_r, light_g, light_b,
            bg_depth_scale, depth_cutoff, cutoff_feather, bg_visibility,
            weight_smart, weight_bae, weight_depth,
            smart_strength, smart_blur, smart_detail,
            normal_strength, hf_detail, bae_supersample,
            use_bae_normal, smart_equalize,
            spec_strength, spec_shiny,
            target_x, target_y, target_z,
            spot_angle_deg, spot_softness, light_type, fixed_intensity, disp_amount, light_size,
            lights_json="",
            mask_cutoff=0.05, mask_falloff=0.05, mask_invert=False,
            exposure_stops=0.0, grade_gain=1.0, grade_lift=0.0,
            shadow_strength=1.0,
            grade_gain_r=1.0, grade_gain_g=1.0, grade_gain_b=1.0,
            grade_lift_r=1.0, grade_lift_g=1.0, grade_lift_b=1.0,
            grade_gamma_r=1.0, grade_gamma_g=1.0, grade_gamma_b=1.0):
        from PIL import Image as PILImage

        out_image, out_depth, out_normal, out_relit, out_mask, out_alpha = [],[],[],[],[],[]
        # v0.9.0 cleanup outputs
        out_composite, out_lighting = [], []

        light_pos = np.array([light_x, light_y, light_z], dtype=np.float32)
        light_col = np.array([light_r, light_g, light_b], dtype=np.float32)

        for i in range(image.shape[0]):
            np_img_uint = (image[i].cpu().numpy() * 255).astype(np.uint8)
            np_img_01 = image[i].cpu().numpy().astype(np.float32)
            pil_img = PILImage.fromarray(np_img_uint)
            H, W = np_img_uint.shape[:2]

            # 1. Depth + subject mask
            depth_full = estimate_depth(pil_img)
            subject = estimate_subject_mask(pil_img)

            # 2. Composite depth (subject vs bg scaling)
            if bg_depth_scale < 0.999:
                bg_region = 1.0 - subject
                bg_depth_values = depth_full * bg_region
                bg_mean = (bg_depth_values.sum() / (bg_region.sum() + 1e-8))
                bg_flat = np.full_like(depth_full, bg_mean)
                bg_blend = bg_flat * (1.0 - bg_depth_scale) + depth_full * bg_depth_scale
                depth = depth_full * subject + bg_blend * (1.0 - subject)
            else:
                depth = depth_full

            # 3. Three normal sources
            n_smart = smart_normal(
                np_img_01,
                blur_sigma=smart_blur,
                strength=smart_strength,
                detail_pow=smart_detail,
                equalize=smart_equalize
            ) if weight_smart > 0.001 else None

            n_bae = None
            if use_bae_normal and weight_bae > 0.001:
                n_bae = estimate_normal_bae(pil_img, (H, W), supersample=bae_supersample)

            n_depth = depth_to_normal(depth, strength=4.0) if weight_depth > 0.001 else None

            # 4. Three-way blend
            normal_rgb = blend_three_normals(
                n_smart, n_bae, n_depth,
                weight_smart, weight_bae, weight_depth
            )

            # 5. Post-process
            normal_rgb = boost_high_frequency(normal_rgb, hf_detail, sigma=4.0)
            normal_rgb = boost_normal_contrast(normal_rgb, normal_strength)

            # 6. Phong relight
            world_pos = depth_to_world_position(depth, disp_amount=float(disp_amount))
            view_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)

            # === Multi-light handling (v0.8.3) ===
            # Try to parse lights_json. If it has entries, accumulate Phong from all
            # enabled lights. Otherwise fall back to legacy single-light behavior.
            import json as _json
            lights_list = []
            try:
                if lights_json and lights_json.strip():
                    parsed = _json.loads(lights_json)
                    if isinstance(parsed, list):
                        lights_list = [L for L in parsed if isinstance(L, dict) and L.get("enabled", True)]
            except Exception as _e:
                lights_list = []
                print(f"[STAGELIGHT] lights_json parse failed, using legacy single-light: {_e}")

            # Accumulate diffuse + specular contributions
            diff_color_total = np.zeros((normal_rgb.shape[0], normal_rgb.shape[1], 3), dtype=np.float32)
            spec_color_total = np.zeros((normal_rgb.shape[0], normal_rgb.shape[1], 3), dtype=np.float32)

            if lights_list:
                # New path: multi-light accumulation
                for L in lights_list:
                    lp = np.array([
                        float(L.get("light_x", 0.0)),
                        float(L.get("light_y", 0.0)),
                        float(L.get("light_z", 0.0)),
                    ], dtype=np.float32)
                    tp = np.array([
                        float(L.get("target_x", 0.0)),
                        float(L.get("target_y", 0.0)),
                        float(L.get("target_z", -1.0)),
                    ], dtype=np.float32)
                    lcol = np.array([
                        float(L.get("light_r", 1.0)),
                        float(L.get("light_g", 1.0)),
                        float(L.get("light_b", 1.0)),
                    ], dtype=np.float32)
                    L_intensity = float(L.get("intensity", 2.0))
                    L_reach     = float(L.get("falloff", 5.0))
                    L_curve     = float(L.get("fcv", 2.0))
                    L_spec      = float(L.get("spec_strength", 0.3))
                    L_shiny     = float(L.get("spec_shiny", 16))
                    L_type      = int(L.get("light_type", 1))
                    L_fixed     = bool(L.get("fixed_intensity", True))
                    L_size      = float(L.get("light_size", 0.0))
                    L_cone_ang  = float(L.get("cone_angle", 60))
                    L_cone_soft = float(L.get("cone_soft", 0.7))
                    angle_rad = np.radians(L_cone_ang * 0.5)
                    cos_outer = np.cos(angle_rad)
                    cos_inner = np.cos(angle_rad * (1.0 - L_cone_soft * 0.5))
                    diffuse_i, specular_i = phong_full(
                        normal_rgb, world_pos, view_pos, lp, tp,
                        L_intensity, L_reach, L_curve, L_spec, L_shiny,
                        L_type, float(cos_inner), float(cos_outer),
                        L_fixed, L_size
                    )
                    diff_color_total += np.dstack([diffuse_i * lcol[0], diffuse_i * lcol[1], diffuse_i * lcol[2]]).astype(np.float32)
                    spec_color_total += np.dstack([specular_i * lcol[0], specular_i * lcol[1], specular_i * lcol[2]]).astype(np.float32)
            else:
                # Legacy path: single light using individual params
                angle_rad = np.radians(spot_angle_deg * 0.5)
                cos_outer = np.cos(angle_rad)
                cos_inner = np.cos(angle_rad * (1.0 - spot_softness * 0.5))
                target_pos = np.array([target_x, target_y, target_z], dtype=np.float32)
                diffuse, specular = phong_full(
                    normal_rgb, world_pos, view_pos, light_pos, target_pos,
                    intensity, cone_reach, falloff_curve, spec_strength, spec_shiny,
                    int(light_type), float(cos_inner), float(cos_outer),
                    bool(fixed_intensity),
                    float(light_size)
                )
                diff_color_total = np.dstack([
                    diffuse * light_col[0],
                    diffuse * light_col[1],
                    diffuse * light_col[2],
                ]).astype(np.float32)
                spec_color_total = np.dstack([
                    specular * light_col[0],
                    specular * light_col[1],
                    specular * light_col[2],
                ]).astype(np.float32)

            # v0.8.9: Nuke-style grade chain matches the GLSL shader exactly.
            # Order: result * gain + lift -> pow(result, 1/gamma) -> result * 2^exposure
            # Small fixed 0.05 ambient floor (so unlit pixels arent fully black).
            # v0.8.11: HYBRID lighting. SHADOW STRENGTH controls darkening of unlit areas.
            # shadowFactor = 1 - shadow_strength * (1 - clamp(diffuse, 0, 1))
            # At shadow_strength=1: unlit = baseColor*0 (full shadow). At 0: unlit = baseColor.
            lit_amount = np.linalg.norm(diff_color_total, axis=2)
            lit_amount = np.clip(lit_amount, 0.0, 1.0)
            shadow_factor = 1.0 - float(shadow_strength) * (1.0 - lit_amount)
            shadow_factor_3 = np.dstack([shadow_factor, shadow_factor, shadow_factor]).astype(np.float32)
            relit = np_img_01 * shadow_factor_3 + np_img_01 * diff_color_total + spec_color_total
            # v0.8.13: per-channel grade. Each grade param multiplied by a per-channel tint.
            gain_v = np.array([grade_gain * float(grade_gain_r), grade_gain * float(grade_gain_g), grade_gain * float(grade_gain_b)], dtype=np.float32)
            lift_v = np.array([grade_lift * float(grade_lift_r), grade_lift * float(grade_lift_g), grade_lift * float(grade_lift_b)], dtype=np.float32)
            gamma_v = np.array([gamma * float(grade_gamma_r), gamma * float(grade_gamma_g), gamma * float(grade_gamma_b)], dtype=np.float32)
            relit = relit * gain_v + lift_v
            relit = np.maximum(relit, 0.0)
            gamma_v_safe = np.maximum(gamma_v, 0.001)
            relit = np.power(relit, 1.0 / gamma_v_safe).astype(np.float32)
            relit = relit * (2.0 ** float(exposure_stops))
            relit = np.clip(relit, 0.0, 1.0).astype(np.float32)

            # 7. Compositing alpha (v0.8.7: MASK CUTOFF + INVERT)
            # Same formula as the GPU shader for parity:
            #   alpha = smoothstep(mask_cutoff - falloff, mask_cutoff + falloff, depth)
            #   if mask_invert: alpha = 1 - alpha
            mfth = max(float(mask_falloff), 0.001)
            mc = float(mask_cutoff)
            t = np.clip((depth_full - (mc - mfth)) / max(2.0 * mfth, 1e-6), 0.0, 1.0)
            final_alpha = (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep
            if mask_invert:
                final_alpha = (1.0 - final_alpha).astype(np.float32)
            alpha3 = np.dstack([final_alpha, final_alpha, final_alpha])

            relit_final = relit * alpha3
            orig_final  = np_img_01 * alpha3  # kept for backward compat; not used as output 0 anymore

            # === v0.9.0 cleanup-ready outputs ===
            # relit_composite: relit subject placed back over the original background.
            # This is the primary input to a downstream img2img cleanup pass (e.g.,
            # Z Image Turbo at low denoise). The foreground is relit, the background
            # is the original image untouched, blended via the rembg subject mask.
            subj_3 = np.dstack([subject, subject, subject]).astype(np.float32)
            relit_composite = relit * subj_3 + np_img_01 * (1.0 - subj_3)
            relit_composite = np.clip(relit_composite, 0.0, 1.0).astype(np.float32)

            # lighting_only: pure additive lighting contribution (diffuse + specular)
            # without the shadow-darkening or base color factor. Useful for additive
            # comp workflows: composite as `original * shadow_mask + lighting_only`.
            # Masked by the subject so background stays zero.
            lighting_only = (np_img_01 * diff_color_total + spec_color_total) * subj_3
            lighting_only = np.clip(lighting_only, 0.0, 1.0).astype(np.float32)

            depth_rgb  = np.dstack([depth, depth, depth]).astype(np.float32)
            normal_vis = normal_rgb

            # v0.9.3: output 0 is the UNMASKED original image. The viewport uses this
            # as its color texture for the displaced 3D mesh; if we sent the depth-alpha-
            # masked version (orig_final), the viewport went dark on every auto-render.
            # The depth alpha is still applied to relit_image (output 3) which is the
            # actual relit result. Subject mask and final alpha are still available on
            # outputs 4 and 5 for downstream consumers.
            out_image.append(torch.from_numpy(np_img_01.astype(np.float32)))
            out_depth.append(torch.from_numpy(depth_rgb.astype(np.float32)))
            out_normal.append(torch.from_numpy(normal_vis.astype(np.float32)))
            out_relit.append(torch.from_numpy(relit_final.astype(np.float32)))
            out_composite.append(torch.from_numpy(relit_composite))
            out_lighting.append(torch.from_numpy(lighting_only))
            out_mask.append(torch.from_numpy(subject.astype(np.float32)))
            out_alpha.append(torch.from_numpy(final_alpha.astype(np.float32)))

        return (
            torch.stack(out_image),
            torch.stack(out_depth),
            torch.stack(out_normal),
            torch.stack(out_relit),
            torch.stack(out_mask),
            torch.stack(out_alpha),
            torch.stack(out_composite),
            torch.stack(out_lighting),
        )
"""

# ── web/stagelight.js — the frontend ─────────────────────────────────────────
PLUGIN_FILES["web/stagelight.js"] = """\
/**
 * STAGELIGHT v0.5.2
 * Three.js viewport. Orbit drag. Zoom scroll. Generate Depth.
 */

import { app } from "../../scripts/app.js";

function loadThree(cb) {
    if (window.THREE) { cb(window.THREE); return; }
    var s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    s.onload  = function() { cb(window.THREE); };
    s.onerror = function() { cb(null); };
    document.head.appendChild(s);
}

function queuePrompt(prompt, cb) {
    fetch("http://127.0.0.1:8188/prompt", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({prompt:prompt})
    }).then(function(r){return r.json();}).then(function(d){cb(null,d);}).catch(function(e){cb(e,null);});
}

function pollHistory(pid, timeout, cb) {
    var t = Date.now();
    function poll() {
        if (Date.now()-t > timeout) { cb(new Error("timeout"),null); return; }
        fetch("http://127.0.0.1:8188/history/"+pid)
        .then(function(r){return r.json();})
        .then(function(d){
            if (d[pid] && d[pid].outputs) cb(null, d[pid].outputs);
            else setTimeout(poll, 800);
        }).catch(function(){ setTimeout(poll, 800); });
    }
    poll();
}

function viewUrl(fn, sub, type) {
    return "http://127.0.0.1:8188/view?filename="+encodeURIComponent(fn)+
           "&subfolder="+encodeURIComponent(sub||"")+"&type="+encodeURIComponent(type||"temp");
}

var _scene=null,_camera=null,_renderer=null,_mesh=null,_animId=null,_lightGizmo=null,_lightAxes=null;
var _loadGen=0;  // monotonic counter for loadMesh calls; stale async callbacks check this
var _tmode="t";  // "t"=translate (W), "r"=rotate (E), "s"=scale (R)
// When target toggle is OFF, the target physically snaps to a position along the
// current cone axis and "parents" to the light - moving with it. This vector stores
// the local-space offset from light to target. While target is off, every time the
// light moves, the target sliders are updated to (lightPos + _targetOffset). When
// target turns back on, _targetOffset is null and sliders are user-controlled.
var _targetOffset=null;  // {x,y,z} world-space offset (light to target), or null when target is on

// === Multi-light data model (v0.8.0) ===
// Each light is a plain object holding all per-light values. _lights is the array;
// _selectedLight is the index of the currently active light. The right-panel sliders
// always read/write _lights[_selectedLight]. When user clicks a different row in the
// outliner, we copy that light's values back into the sliders.
//
// IMPORTANT v0.8.0 SCOPE NOTE: only the *selected* light renders. Adding more lights
// stores their data but doesn't accumulate them in the shader yet. v0.8.1 will add
// real shader accumulation. This reduced scope keeps v0.8.0 low-risk - if anything
// breaks, the existing render pipeline is untouched.
var _lights = [];           // array of light state objects (see makeDefaultLight)
var _selectedLight = 0;     // index into _lights
var _maxLights = 10;        // shader-side hard cap (also dictates uniform array size)
// Per-light gizmo color palette - distinct hues so user can tell lights apart.
// Active light's gizmo uses the bright version; inactive lights show as small
// colored marker spheres in the viewport.
var _lightColors = [
    "#ffd060",  // 1: yellow (default - first light, matches existing gizmo)
    "#60d0ff",  // 2: cyan
    "#ff60c0",  // 3: magenta-pink
    "#60ff80",  // 4: green
    "#ff9040",  // 5: orange
    "#a060ff",  // 6: purple
    "#ff5050",  // 7: red
    "#80e0e0",  // 8: teal
    "#e0e070",  // 9: lime
    "#ffffff"   // 10: white
];
function lightColorForIndex(i){
    return _lightColors[i % _lightColors.length];
}

function makeDefaultLight(){
    return {
        name: "Light",                  // caller will dedupe via uniqueLightName()
        enabled: true,                  // eye-icon toggle
        light_x: 0.0, light_y: 0.0, light_z: 0.0,    // spawn at origin per user spec
        target_x: 0.0, target_y: 0.0, target_z: -1.0,
        intensity: 2.0,
        falloff: 5.0, fcv: 2.0,         // attenuation reach + curve
        light_r: 1.0, light_g: 1.0, light_b: 1.0,
        spec_strength: 0.3, spec_shiny: 16,
        cone_angle: 60, cone_soft: 0.7,
        light_type: 1,                  // 0=POINT 1=SPOT 2=DIR
        fixed_intensity: true,
        light_size: 0.0,
        target_visible: false,          // matches v0.7.x default (off)
        target_offset: null             // per-light frozen offset
    };
}
// Generate a name unique within the current _lights array. If `desired` is taken,
// append " 1", " 2", etc. until we find a free one. Maya-style.
function uniqueLightName(desired, excludeIdx){
    if(typeof excludeIdx !== "number") excludeIdx = -1;
    var taken = {};
    for(var i = 0; i < _lights.length; i++){
        if(i === excludeIdx) continue;
        taken[_lights[i].name] = true;
    }
    if(!taken[desired]) return desired;
    var n = 1;
    while(taken[desired + " " + n]) n++;
    return desired + " " + n;
}
var _orbitTheta=0;  // horizontal angle (around Y axis)
var _orbitPhi=0;    // vertical angle (around X axis)
var _orbitDist=3.0; // camera distance from origin
var _gizmoScale=1.0; // gizmo visual size multiplier
var _panX=0.0, _panY=0.0; // pan target offset (middle-mouse drag)


function cleanupThree() {
    if(_animId){cancelAnimationFrame(_animId);_animId=null;}
    // Remove document-level event listeners that openPanel registered
    if(window._slDocHandlers){
        try {
            document.removeEventListener("mousemove", window._slDocHandlers.move);
            document.removeEventListener("mouseup", window._slDocHandlers.up);
        } catch(e){}
        window._slDocHandlers = null;
    }
    // Remove splitter listeners
    if(window._slSplitHandlers){
        window._slSplitHandlers.forEach(function(h){
            try {
                document.removeEventListener("mousemove", h.move);
                document.removeEventListener("mouseup", h.up);
            } catch(e){}
        });
        window._slSplitHandlers = null;
    }
    // Remove pin-resize listeners
    if(window._slPinResizeHandlers){
        window._slPinResizeHandlers.forEach(function(h){
            try {
                document.removeEventListener("mousemove", h.move);
                document.removeEventListener("mouseup", h.up);
            } catch(e){}
        });
        window._slPinResizeHandlers = null;
    }
    // Remove transform-mode keydown handler (try both window AND document because
    // older builds bound to document; new builds bind to window)
    if(window._slTmodeHandler){
        try { window.removeEventListener("keydown", window._slTmodeHandler, true); } catch(e){}
        try { document.removeEventListener("keydown", window._slTmodeHandler, true); } catch(e){}
        // Also remove the panel-level listener if a panel ref is stashed
        if(window._slTmodePanel){
            try { window._slTmodePanel.removeEventListener("keydown", window._slTmodeHandler, true); } catch(e){}
        }
        window._slTmodeHandler = null;
        window._slTmodePanel = null;
    }
    // Dispose mesh resources to free GPU memory
    if(_mesh){
        if(_mesh.geometry) _mesh.geometry.dispose();
        if(_mesh.material){
            // Dispose textures attached to shader uniforms
            if(_mesh.material.uniforms){
                ["colorMap","depthMap","normalMap"].forEach(function(k){
                    var u = _mesh.material.uniforms[k];
                    if(u && u.value && u.value.dispose) try{ u.value.dispose(); }catch(e){}
                });
            }
            _mesh.material.dispose();
        }
        _mesh = null;
    }
    if(_renderer){_renderer.dispose();_renderer=null;}
    _scene=null;_camera=null;_lightGizmo=null;_lightAxes=null;
    // Dispose marker meshes
    if(window._slMarkers){
        window._slMarkers.forEach(function(m){
            if(m){
                if(m.geometry) m.geometry.dispose();
                if(m.material) m.material.dispose();
            }
        });
        window._slMarkers = null;
    }
    // Clear globals
    if(window._slUpdateLighting) window._slUpdateLighting = null;
    if(window._slUpdateCamera) window._slUpdateCamera = null;
    if(window._slSyncGizmo) window._slSyncGizmo = null;
    if(window._slUpdateDisplacement) window._slUpdateDisplacement = null;
}

function loadMesh(colorUrl, depthUrl, normalUrl, statusEl) {
    if(!window.THREE||!_renderer||!_scene) return;
    var THREE=window.THREE;
    if(statusEl){statusEl.textContent="Building 3D view...";statusEl.style.color="#e8a020";}
    // Capture the generation for THIS loadMesh call. If the user clicks RENDER again
    // before our textures finish loading, _loadGen increments and our checkReady
    // becomes stale (myGen !== _loadGen) and aborts cleanly.
    _loadGen += 1;
    var myGen = _loadGen;
    var loader=new THREE.TextureLoader();
    var colorTex=null, depthTex=null, normalTex=null;
    var pending = 1 + (depthUrl?1:0) + (normalUrl?1:0);
    function checkReady(){
        pending -= 1;
        if(pending > 0) return;
        // Stale-callback guard: another loadMesh ran ahead of us. Discard our textures
        // so they don't leak, and bail without touching the live mesh.
        if(myGen !== _loadGen){
            [colorTex, depthTex, normalTex].forEach(function(t){
                if(t && t.dispose) try{ t.dispose(); }catch(e){}
            });
            return;
        }
        var W = (colorTex.image.naturalWidth||colorTex.image.width||512);
        var H = (colorTex.image.naturalHeight||colorTex.image.height||512);
        // Stash references to the OLD mesh's textures so we can dispose them
        // AFTER swapping in the new mesh. We only dispose textures that are not
        // still referenced by the new mesh's uniforms (defensive - shouldn't happen
        // with fresh TextureLoader.load() but protects against shared/cached refs).
        var oldMesh = _mesh;
        var oldGeom = oldMesh ? oldMesh.geometry : null;
        var oldMat = oldMesh ? oldMesh.material : null;
        var oldTextures = [];
        if(oldMat && oldMat.uniforms){
            ["colorMap","depthMap","normalMap"].forEach(function(k){
                var u = oldMat.uniforms[k];
                if(u && u.value) oldTextures.push(u.value);
            });
        }
        if(oldMesh){
            _scene.remove(oldMesh);
            _mesh = null;
        }

        // High-resolution displaced plane
        var seg = 256;
        var aspect = W/H;

        // Custom shader: vertex displaces along Z based on depth*amount*alpha
        // Fragment shows color textured, but discards alpha-cut pixels
        var vertShader = [
            "uniform sampler2D depthMap;",
            "uniform float dispAmount;",
            "uniform bool useDepth;",
            "varying vec2 vUv;",
            "varying float vDepth;",
            "varying vec3 vWorldPos;",
            "void main(){",
            "  vUv = uv;",
            "  vec3 pos = position;",
            "  if(useDepth){",
            "    float d = texture2D(depthMap, uv).r;",
            "    pos.z += d * dispAmount;",
            "    vDepth = d;",
            "  } else {",
            "    vDepth = 1.0;",
            "  }",
            "  vec4 wp = modelMatrix * vec4(pos, 1.0);",
            "  vWorldPos = wp.xyz;",
            "  gl_Position = projectionMatrix * viewMatrix * wp;",
            "}"
        ].join("\\n");

        var fragShader = [
            "uniform sampler2D colorMap;",
            "uniform sampler2D normalMap;",
            "uniform sampler2D depthMap;",
            // === MASK BY DEPTH (v0.8.6) ===
            // === Depth mask (v0.8.7) ===
            // maskCutoff: depth threshold. Pixels with vDepth below this are hidden,
            //   above are shown. As you raise the cutoff, more pixels get hidden
            //   starting from the lowest-depth (background) end.
            // maskFalloff: smoothstep transition width around the cutoff.
            // maskInvert: when 1, flips which side is hidden (cuts subject instead of BG).
            "uniform float maskCutoff;",
            "uniform float maskFalloff;",
            "uniform int maskInvert;",
            // === Multi-light arrays (v0.8.3) ===
            // numLights tells the loop how many array entries to consume. Max 10 to keep
            // shader compile predictable. WebGL ES 2 requires constant loop bounds.
            "uniform int numLights;",
            "uniform vec3 lightPos[10];",
            "uniform vec3 lightTarget[10];",
            "uniform vec3 lightColor[10];",
            "uniform float lightIntensity[10];",
            "uniform float coneReach[10];",
            "uniform float falloffCurve[10];",
            "uniform float specStrength[10];",
            "uniform float specShiny[10];",
            "uniform float spotCosInner[10];",
            "uniform float spotCosOuter[10];",
            "uniform int lightType[10];",
            "uniform int fixedIntensityFlag[10];",  // 0/1 (bool arrays are messy in GLSL)
            "uniform float lightSize[10];",
            "uniform vec3 cameraPosition2;",
            // === Grade controls (v0.8.9) - Nuke-style chain ===
            // exposure: stops (1 stop = 2x brightness). 0 = neutral.
            // lift: adds to all channels (raises blacks). 0 = neutral.
            // gain: multiplies all channels (boosts whites). 1.0 = neutral.
            // gamma: pow curve on midtones. 1.0 = neutral.
            // Chain order (Nuke standard): result * gain + lift; pow(result, 1/gamma); result *= 2^exposure
            // === Grade controls (v0.8.13) - Nuke-style chain with per-channel tints ===
            // exposure: stops (scalar, no tint).
            // lift, gain, gamma: vec3 - per-channel grading. White (1,1,1) = neutral
            // for gain/gamma; (0,0,0) = neutral for lift. Color wheels in the UI tint these.
            "uniform float exposure;",
            "uniform vec3 gamma;",
            "uniform vec3 lift;",
            "uniform vec3 gain;",
            // === Shadow control (v0.8.11) ===
            // shadowStrength: 0 = pass-through (no shadows, current additive model),
            //                 1 = full shadow (unlit areas go fully dark).
            // Hybrid math: lit areas use additive, unlit areas use multiplicative.
            "uniform float shadowStrength;",
            "uniform int viewMode;",       // 0=color, 1=depth, 2=normal, 3=relit
            "uniform bool useNormalMap;",
            "varying vec2 vUv;",
            "varying float vDepth;",
            "varying vec3 vWorldPos;",
            "void main(){",
            // === Simple depth-cutoff mask with invert ===
            // Below maskCutoff = hidden (alpha=0), above = visible (alpha=1), with
            // a smoothstep transition of width maskFalloff. INVERT flips the result.
            "  float fwidth = max(maskFalloff, 0.001);",
            "  float edgeAlpha = smoothstep(maskCutoff - fwidth, maskCutoff + fwidth, vDepth);",
            "  if(maskInvert == 1) edgeAlpha = 1.0 - edgeAlpha;",
            "  vec3 baseColor = texture2D(colorMap, vUv).rgb;",
            // VIEW MODE 0: just color - no lighting
            "  if(viewMode == 0){ gl_FragColor = vec4(baseColor, edgeAlpha); return; }",
            // VIEW MODE 1: depth grayscale
            "  if(viewMode == 1){ float d = texture2D(depthMap, vUv).r; gl_FragColor = vec4(d, d, d, edgeAlpha); return; }",
            // VIEW MODE 2: normal as RGB
            "  if(viewMode == 2){ vec3 nn = texture2D(normalMap, vUv).rgb; gl_FragColor = vec4(nn, edgeAlpha); return; }",
            // VIEW MODE 3: real-time relight - accumulate over all enabled lights
            "  vec3 N = vec3(0.0, 0.0, 1.0);",
            "  if(useNormalMap){",
            "    N = texture2D(normalMap, vUv).rgb * 2.0 - 1.0;",
            "    N = normalize(N);",
            "  }",
            "  vec3 V = normalize(cameraPosition2 - vWorldPos);",
            // Per-light accumulators - sum contributions from every enabled light
            "  vec3 totalDiffuse = vec3(0.0);",
            "  vec3 totalSpecular = vec3(0.0);",
            // Loop over up to 10 lights (max array size). numLights cuts the loop early.
            // GLSL ES 2 requires constant bounds; the runtime check inside lets us short-circuit.
            "  for(int i = 0; i < 10; i++){",
            "    if(i >= numLights) break;",
            // Light direction
            "    vec3 L;",
            "    float attn = lightIntensity[i];",
            "    if(lightType[i] == 2){",
            // Directional
            "      L = normalize(lightPos[i] - lightTarget[i]);",
            "    } else {",
            // Point/Spot - per-pixel direction from surface to light
            "      vec3 LV = lightPos[i] - vWorldPos;",
            "      float dist = length(LV);",
            "      L = LV / max(dist, 0.0001);",
            "      if(fixedIntensityFlag[i] == 1){",
            "        attn *= step(dist, coneReach[i]);",
            "      } else {",
            "        float clamped = clamp(dist / max(coneReach[i], 0.001), 0.0, 1.0);",
            "        float radial = pow(clamp(1.0 - clamped*clamped, 0.0, 1.0), falloffCurve[i]);",
            "        attn *= radial;",
            "      }",
            "    }",
            // Spot cone
            "    if(lightType[i] == 1){",
            "      vec3 lightForward = normalize(lightTarget[i] - lightPos[i]);",
            "      float cosTheta = dot(-L, lightForward);",
            "      float cone = clamp((cosTheta - spotCosOuter[i]) / max(spotCosInner[i] - spotCosOuter[i], 0.001), 0.0, 1.0);",
            "      attn *= cone;",
            "    }",
            // Wrapped lighting
            "    float w = lightSize[i];",
            "    float NdotL = max((dot(N, L) + w) / (1.0 + w), 0.0);",
            "    totalDiffuse += lightColor[i] * NdotL * attn;",
            // Specular (Blinn-Phong)
            "    vec3 H = normalize(L + V);",
            "    float NdotH = max(dot(N, H), 0.0);",
            "    float spec = pow(NdotH, max(specShiny[i], 1.0)) * step(0.001, NdotL);",
            "    totalSpecular += lightColor[i] * spec * specStrength[i] * attn;",
            "  }",
            // Composite: ambient floor + accumulated diffuse + accumulated specular
            // Composite: hybrid lighting (v0.8.11)
            // shadow_factor blends between full pass-through (shadowStrength=0) and
            // full dark-when-unlit (shadowStrength=1). totalDiffuse is the per-pixel
            // accumulated light energy from all lights.
            // - lit areas: get added brightness (additive)
            // - unlit areas: get darkened by shadowStrength
            "  float litAmount = clamp(length(totalDiffuse), 0.0, 1.0);",
            "  float shadowFactor = 1.0 - shadowStrength * (1.0 - litAmount);",
            "  vec3 result = baseColor * shadowFactor + baseColor * totalDiffuse + totalSpecular;",
            // === Grade chain (Nuke order: gain -> lift -> gamma -> exposure) ===
            "  result = result * gain + lift;",
            "  result = pow(max(result, vec3(0.0)), vec3(1.0) / max(gamma, vec3(0.001)));",
            "  result *= pow(2.0, exposure);",
            "  gl_FragColor = vec4(result, edgeAlpha);",
            "}"
        ].join("\\n");

        var dispAmount = parseFloat((document.getElementById("sl-vdp")||{value:0.5}).value);
        var useDepth = depthTex !== null;
        var useNormalMap = normalTex !== null;

        // initial view mode
        var activeBtn = document.querySelector(".sl-vbtn-active");
        var initialView = activeBtn ? parseInt(activeBtn.getAttribute("data-view")) : 3;

        // initial light position from sliders
        var ipx = parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value);
        var ipy = parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value);
        var ipz = parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value);
        var itx = parseFloat((document.getElementById("sl-tx")||{value: 0.0}).value);
        var ity = parseFloat((document.getElementById("sl-ty")||{value: 0.0}).value);
        var itz = parseFloat((document.getElementById("sl-tz")||{value: 0.0}).value);
        var iIn = parseFloat((document.getElementById("sl-int")||{value:2.0}).value);
        var iFa = parseFloat((document.getElementById("sl-fal")||{value:5.0}).value);
        var iFc = parseFloat((document.getElementById("sl-fcv")||{value:2.0}).value);
        var iEx = parseFloat((document.getElementById("sl-exp")||{value:0.0}).value);
        var iGn = parseFloat((document.getElementById("sl-gain")||{value:1.0}).value);
        var iLf = parseFloat((document.getElementById("sl-lift")||{value:0.0}).value);
        var iGm = parseFloat((document.getElementById("sl-gam")||{value:1.0}).value);
        var iShd = parseFloat((document.getElementById("sl-shd")||{value:1.0}).value);
        var iSp = parseFloat((document.getElementById("sl-spc")||{value:0.3}).value);
        var iSh = parseFloat((document.getElementById("sl-shn")||{value:16}).value);
        var iCa = parseFloat((document.getElementById("sl-cone")||{value:60}).value);
        var iSo = parseFloat((document.getElementById("sl-soft")||{value:0.7}).value);
        var iLt = (function(){var s=document.getElementById("sl-ltype");return s?parseInt(s.value):0;})();
        var iLr = parseFloat((document.getElementById("sl-lr")||{value:1.0}).value);
        var iLg = parseFloat((document.getElementById("sl-lg")||{value:0.95}).value);
        var iLb = parseFloat((document.getElementById("sl-lb")||{value:0.85}).value);
        var halfA = (iCa * 0.5) * Math.PI / 180.0;
        var spotCosOuter = Math.cos(halfA);
        var spotCosInner = Math.cos(halfA * (1.0 - iSo * 0.5));

        // Initialize multi-light uniform arrays (size 10). All slots are filled with
        // safe defaults; only entries 0..numLights-1 actually contribute (loop in shader).
        var arrLightPos = []; var arrLightTarget = []; var arrLightColor = [];
        var arrLightIntensity = []; var arrConeReach = []; var arrFalloffCurve = [];
        var arrSpecStrength = []; var arrSpecShiny = [];
        var arrSpotCosInner = []; var arrSpotCosOuter = [];
        var arrLightType = []; var arrFixedFlag = []; var arrLightSize = [];
        for(var li = 0; li < 10; li++){
            arrLightPos.push(new THREE.Vector3(0, 0, 0));
            arrLightTarget.push(new THREE.Vector3(0, 0, -1));
            arrLightColor.push(new THREE.Vector3(1, 1, 1));
            arrLightIntensity.push(0);  // 0 = no contribution from unused slots
            arrConeReach.push(5);
            arrFalloffCurve.push(2);
            arrSpecStrength.push(0);
            arrSpecShiny.push(16);
            arrSpotCosInner.push(0.5);
            arrSpotCosOuter.push(0.4);
            arrLightType.push(0);
            arrFixedFlag.push(1);
            arrLightSize.push(0);
        }
        var mat = new THREE.ShaderMaterial({
            uniforms: {
                colorMap:        { value: colorTex },
                depthMap:        { value: depthTex || colorTex },
                normalMap:       { value: normalTex || colorTex },
                dispAmount:      { value: dispAmount },
                maskCutoff:      { value: parseFloat((document.getElementById("sl-mask")||{value:0.05}).value) },
                maskFalloff:     { value: parseFloat((document.getElementById("sl-mfth")||{value:0.05}).value) },
                maskInvert:      { value: ((document.getElementById("sl-minv")||{checked:false}).checked) ? 1 : 0 },
                cameraPosition2: { value: new THREE.Vector3(0,0,3) },
                exposure:        { value: iEx },
                gamma:           { value: new THREE.Vector3(iGm, iGm, iGm) },
                lift:            { value: new THREE.Vector3(iLf, iLf, iLf) },
                gain:            { value: new THREE.Vector3(iGn, iGn, iGn) },
                shadowStrength:  { value: iShd },
                // Multi-light arrays
                numLights:       { value: 1 },
                lightPos:        { value: arrLightPos },
                lightTarget:     { value: arrLightTarget },
                lightColor:      { value: arrLightColor },
                lightIntensity:  { value: arrLightIntensity },
                coneReach:       { value: arrConeReach },
                falloffCurve:    { value: arrFalloffCurve },
                specStrength:    { value: arrSpecStrength },
                specShiny:       { value: arrSpecShiny },
                spotCosInner:    { value: arrSpotCosInner },
                spotCosOuter:    { value: arrSpotCosOuter },
                lightType:       { value: arrLightType },
                fixedIntensityFlag: { value: arrFixedFlag },
                lightSize:       { value: arrLightSize },
                viewMode:        { value: initialView },
                useDepth:        { value: useDepth },
                useNormalMap:    { value: useNormalMap }
            },
            transparent: true,
            vertexShader: vertShader,
            fragmentShader: fragShader,
            side: THREE.DoubleSide
        });

        _mesh = new THREE.Mesh(
            new THREE.PlaneGeometry(2.0*aspect, 2.0, seg, seg),
            mat
        );
        _mesh.position.set(0,0,0);
        _mesh._displaceMat = mat;  // expose for live updates
        _scene.add(_mesh);

        // Function to update displacement amount live (slider drag)
        // Master live-update function: pumps multi-light uniform arrays from _lights state
        window._slUpdateLighting = function(){
            if(!_mesh || !_mesh._displaceMat) return;
            var u = _mesh._displaceMat.uniforms;
            var g = function(id, def){ var e=document.getElementById(id); return e ? parseFloat(e.value) : def; };
            u.dispAmount.value     = g("sl-vdp", 0.5);
            u.maskCutoff.value     = g("sl-mask", 0.05);
            u.maskFalloff.value    = g("sl-mfth", 0.05);
            var minvBox = document.getElementById("sl-minv");
            u.maskInvert.value     = (minvBox && minvBox.checked) ? 1 : 0;
            u.exposure.value       = g("sl-exp", 0.0);
            // Grade tints (v0.8.13): combine scalar slider value with per-channel tint color.
            // Tint defaults to white (no tint). User can set via color wheel.
            // For LIFT: result = lift_value * tint (so neutral white = lift_value, no color shift)
            // For GAIN: result = gain_value * tint (white = neutral 1.0 across channels)
            // For GAMMA: result = gamma_value * tint (white = uniform gamma curve)
            var lifTint = (window._slGradeTints && window._slGradeTints.lift) || [1,1,1];
            var ganTint = (window._slGradeTints && window._slGradeTints.gain) || [1,1,1];
            var gamTint = (window._slGradeTints && window._slGradeTints.gamma) || [1,1,1];
            var gamVal = g("sl-gam", 1.0);
            var lifVal = g("sl-lift", 0.0);
            var ganVal = g("sl-gain", 1.0);
            u.gamma.value.set(gamVal * gamTint[0], gamVal * gamTint[1], gamVal * gamTint[2]);
            u.lift.value.set(lifVal * lifTint[0], lifVal * lifTint[1], lifVal * lifTint[2]);
            u.gain.value.set(ganVal * ganTint[0], ganVal * ganTint[1], ganVal * ganTint[2]);
            u.shadowStrength.value = g("sl-shd", 1.0);
            // First, commit the live sliders into the SELECTED light entry so the array
            // reflects current edits (otherwise the user's drag wouldn't be visible).
            if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
            // Now build the uniform arrays from _lights (only enabled lights count).
            var enabledLights = _lights.filter(function(L){ return L.enabled; });
            var n = Math.min(enabledLights.length, 10);
            u.numLights.value = n;
            for(var i = 0; i < 10; i++){
                if(i < n){
                    var L = enabledLights[i];
                    u.lightPos.value[i].set(L.light_x, L.light_y, L.light_z);
                    u.lightTarget.value[i].set(L.target_x, L.target_y, L.target_z);
                    u.lightColor.value[i].set(L.light_r, L.light_g, L.light_b);
                    u.lightIntensity.value[i] = L.intensity;
                    u.coneReach.value[i] = L.falloff;
                    u.falloffCurve.value[i] = L.fcv;
                    u.specStrength.value[i] = L.spec_strength;
                    u.specShiny.value[i] = L.spec_shiny;
                    var coneA = L.cone_angle * 0.5 * Math.PI / 180.0;
                    u.spotCosOuter.value[i] = Math.cos(coneA);
                    u.spotCosInner.value[i] = Math.cos(coneA * (1.0 - L.cone_soft * 0.5));
                    u.lightType.value[i] = L.light_type;
                    u.fixedIntensityFlag.value[i] = L.fixed_intensity ? 1 : 0;
                    u.lightSize.value[i] = L.light_size;
                } else {
                    // Zero out unused slots (defensive: shouldn't matter since loop honors numLights)
                    u.lightIntensity.value[i] = 0;
                }
            }
            // View mode from active tab button
            var activeBtn = document.querySelector(".sl-vbtn-active");
            u.viewMode.value = activeBtn ? parseInt(activeBtn.getAttribute("data-view")) : 3;
            // Camera position - lookup from Three.js camera
            if(_camera) u.cameraPosition2.value.copy(_camera.position);
            // Request a frame draw (dirty-flag rendering)
            if(window._slRequestRender) window._slRequestRender();
        };
        // Backward compat name
        window._slUpdateDisplacement = window._slUpdateLighting;

        if(window._slUpdateCamera)window._slUpdateCamera();
        // v0.9.4 CRITICAL: pump the current _lights array into the freshly-built mesh
        // uniforms. Without this, the new mesh ships with zeroed light intensities and
        // the viewport goes black until the user moves a slider. With auto-render firing
        // on every slider tick, the viewport would flicker dark → bright → dark.
        if(window._slUpdateLighting) window._slUpdateLighting();
        var msg=document.getElementById("sl-msg");
        if(msg) msg.style.display="none";
        if(statusEl){
            var note = useDepth ? " - 3D displaced" : " - flat (no depth)";
            statusEl.textContent = W+"x"+H+note+" - drag to orbit";
            statusEl.style.color="#44dd88";
        }
        // New mesh ready - request a frame draw
        if(window._slRequestRender) window._slRequestRender();

        // Now that the new mesh is in the scene with its own fresh textures,
        // dispose the OLD mesh's resources. Important: only dispose textures that
        // are NOT referenced by the new mesh's uniforms (defensive against any
        // shared/cached texture objects).
        var newRefs = [];
        if(mat && mat.uniforms){
            ["colorMap","depthMap","normalMap"].forEach(function(k){
                var u = mat.uniforms[k];
                if(u && u.value) newRefs.push(u.value);
            });
        }
        if(oldGeom) try{ oldGeom.dispose(); }catch(e){}
        oldTextures.forEach(function(tex){
            if(tex && newRefs.indexOf(tex) === -1 && tex.dispose){
                try{ tex.dispose(); }catch(e){}
            }
        });
        if(oldMat) try{ oldMat.dispose(); }catch(e){}
    }
    loader.load(colorUrl, function(t){ colorTex = t; checkReady(); }, undefined, function(){
        if(statusEl){statusEl.textContent="Color load failed";statusEl.style.color="#cc4444";}
    });
    if(depthUrl){
        loader.load(depthUrl, function(t){ depthTex = t; checkReady(); }, undefined, function(){
            depthTex = null; checkReady();
        });
    }
    if(normalUrl){
        loader.load(normalUrl, function(t){ normalTex = t; checkReady(); }, undefined, function(){
            normalTex = null; checkReady();
        });
    }
}

function openPanel(node) {
    var existing=document.getElementById("sl-panel");
    if(existing){existing.remove();cleanupThree();return;}
    // Stash node ref so RESET button can reopen the panel
    window._slLastNode = node;

    // Local alias to the dirty-flag render request (defined once Three.js is ready)
    function requestRender(){ if(window._slRequestRender) window._slRequestRender(); }

    // === Settings persistence (localStorage) ===
    // Single key holding all panel state. Survives panel close/reopen and ComfyUI page
    // refresh. Does NOT survive browser cache clear or different machines - for that
    // we will add EXPORT/IMPORT JSON in v0.7.11+.
    var STORAGE_KEY = "stagelight_settings_v1";
    function loadSavedSettings(){
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            if(!raw) return null;
            return JSON.parse(raw);
        } catch(e){ return null; }
    }
    function saveCurrentSettings(){
        try {
            var s = {};
            // Every slider and number-input
            var sliderIds = ["sl-lx","sl-ly","sl-lz","sl-int","sl-fal","sl-fcv","sl-shd","sl-exp","sl-gain","sl-lift","sl-gam",
                "sl-lr","sl-lg","sl-lb","sl-mask","sl-mfth","sl-nrm","sl-nhf","sl-nss",
                "sl-gsz","sl-wsm","sl-wba","sl-wdp","sl-sst","sl-sbl","sl-sdt","sl-vdp",
                "sl-spc","sl-shn","sl-tx","sl-ty","sl-tz","sl-cone","sl-soft","sl-lsz","sl-gop"];
            sliderIds.forEach(function(id){
                var el = document.getElementById(id);
                if(el) s[id] = el.value;
            });
            // Checkboxes
            ["sl-bae","sl-camlock","sl-gizmos","sl-conevis","sl-fixed","sl-tgtvis","sl-minv"].forEach(function(id){
                var el = document.getElementById(id);
                if(el) s[id] = el.checked;
            });
            // Hidden light-type
            var lt = document.getElementById("sl-ltype");
            if(lt) s["sl-ltype"] = lt.value;
            // Color picker
            var cp = document.getElementById("sl-cpick");
            if(cp) s["sl-cpick"] = cp.value;
            // Transform mode
            s["_tmode"] = _tmode;
            // Panel layout: panel size + splitter (left/right column) widths
            var pnl = document.getElementById("sl-panel");
            if(pnl){
                s["panel_w"] = pnl.style.width;
                s["panel_h"] = pnl.style.height;
            }
            var ctlL = document.getElementById("sl-ctl-l");
            if(ctlL) s["ctl_l_w"] = ctlL.style.width;
            var ctlR = document.getElementById("sl-ctl-r");
            if(ctlR) s["ctl_r_w"] = ctlR.style.width;
            // Active view tab
            var actV = document.querySelector(".sl-vbtn-active");
            if(actV) s["view"] = actV.getAttribute("data-view");
            // Section collapse states (so collapsed sections stay collapsed)
            s["sections"] = {};
            document.querySelectorAll(".sl-sec-body").forEach(function(b){
                s.sections[b.id] = b.style.display !== "none";  // true = open
            });
            // Multi-light state: commit current sliders to active light, then save the array
            if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
            s["_lights"] = _lights;
            s["_selectedLight"] = _selectedLight;
            // v0.8.13: grade tint colors for LIFT/GAIN/GAMMA color wheels
            s["_gradeTints"] = window._slGradeTints || { lift:[1,1,1], gain:[1,1,1], gamma:[1,1,1] };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
        } catch(e){ /* localStorage may be full or disabled - fail silently */ }
    }
    // Apply saved settings to the DOM. Called after panel HTML is built.
    function applySavedSettings(s){
        if(!s) return;
        // Sliders + numeric inputs - assign value AND update the readout label
        Object.keys(s).forEach(function(id){
            if(typeof id !== "string" || id.indexOf("sl-") !== 0) return;
            var el = document.getElementById(id);
            if(!el) return;
            if(el.type === "checkbox"){
                el.checked = !!s[id];
            } else if(el.type === "range" || el.type === "number" || el.type === "color" || el.type === "hidden"){
                el.value = s[id];
                var vl = document.getElementById(id + "-v");
                if(vl) vl.textContent = parseFloat(s[id]).toFixed(2);
            }
        });
        // Transform mode
        if(s._tmode === "t" || s._tmode === "r" || s._tmode === "s"){
            _tmode = s._tmode;
            // Visual button update will be applied later when wiring is set up
        }
        // Panel size
        var pnl = document.getElementById("sl-panel");
        if(pnl){
            if(s.panel_w) pnl.style.width = s.panel_w;
            if(s.panel_h) pnl.style.height = s.panel_h;
        }
        var ctlL = document.getElementById("sl-ctl-l");
        if(ctlL && s.ctl_l_w) ctlL.style.width = s.ctl_l_w;
        var ctlR = document.getElementById("sl-ctl-r");
        if(ctlR && s.ctl_r_w) ctlR.style.width = s.ctl_r_w;
        // View tab
        if(s.view){
            document.querySelectorAll(".sl-vbtn").forEach(function(b){
                var active = b.getAttribute("data-view") === s.view;
                b.classList.toggle("sl-vbtn-active", active);
                b.style.background = active ? "#2a1a04" : "transparent";
                b.style.color = active ? "#e8a020" : "#666";
            });
        }
        // Section collapse states
        if(s.sections){
            Object.keys(s.sections).forEach(function(bid){
                var body = document.getElementById(bid);
                if(!body) return;
                var open = !!s.sections[bid];
                body.style.display = open ? "" : "none";
                // Find the matching arrow span via hdr
                var hdr = document.querySelector(\'[data-target="\' + bid + \'"]\');
                if(hdr){
                    var arrow = hdr.querySelector(".sl-sec-arrow");
                    if(arrow) arrow.textContent = open ? "\u25bc" : "\u25b6";
                }
            });
        }
    }
    // Debounced save - avoid hammering localStorage on every slider tick
    var _saveTimer = null;
    function scheduleSave(){
        if(_saveTimer) clearTimeout(_saveTimer);
        _saveTimer = setTimeout(function(){
            saveCurrentSettings();
            // v0.9.1: Also sync panel state to the StagelightNode's widget values
            // so ComfyUI's "Run" button (not just our RENDER button) uses current state.
            // Without this, loading the JSON workflow + hitting Run uses empty widget defaults.
            syncPanelStateToNode();
            // Snapshot for undo after save completes (debounced - so a slider drag
            // produces ONE snapshot when the user stops moving it, not 30 mid-drag).
            if(window._slSnapshotState) window._slSnapshotState();
        }, 400);
    }
    // Expose so other handlers (gizmo drag, button clicks etc) can request a save
    window._slScheduleSave = scheduleSave;

    // v0.9.1: Sync the current panel state to the StagelightNode's widget values
    // in the ComfyUI graph. This is what makes ComfyUI's "Run" button produce the
    // same output as our RENDER button - both now use the panel's current state.
    function syncPanelStateToNode(){
        try {
            // Get ComfyUI's app reference - may not exist if extension hasn't fully loaded
            var app = window.app || (window.LiteGraph && window.LiteGraph.app);
            if(!app || !app.graph) return;
            // Find all StagelightNode instances in the graph
            var nodes = app.graph._nodes || [];
            for(var ni = 0; ni < nodes.length; ni++){
                var n = nodes[ni];
                if(!n || n.type !== "StagelightNode") continue;
                if(!n.widgets) continue;
                // For each widget, look up its value from panel state and set it
                var ws = n.widgets;
                function setW(name, val){
                    for(var i = 0; i < ws.length; i++){
                        if(ws[i].name === name){ ws[i].value = val; return; }
                    }
                }
                // Pull current panel slider values to push into the node widgets
                var $v = function(id, dflt){
                    var el = document.getElementById(id);
                    return el ? parseFloat(el.value) : dflt;
                };
                var $c = function(id, dflt){
                    var el = document.getElementById(id);
                    return el ? el.checked : dflt;
                };
                // Light position
                setW("light_x",        $v("sl-lx", -0.5));
                setW("light_y",        $v("sl-ly", 1.0));
                setW("light_z",        $v("sl-lz", 2.0));
                setW("intensity",      $v("sl-int", 1.0));
                setW("cone_reach",     $v("sl-fal", 5.0));
                setW("falloff_curve",  $v("sl-fcv", 2.0));
                setW("gamma",          $v("sl-gam", 1.0));
                setW("light_r",        $v("sl-lr", 1.0));
                setW("light_g",        $v("sl-lg", 1.0));
                setW("light_b",        $v("sl-lb", 1.0));
                // v0.9.3: REMOVED bg_depth_scale sync. There is no UI slider for it,
                // and sl-vdp (the DEPTH AMOUNT slider, range up to 1.5) was incorrectly
                // being written here, causing "Value 1.5 bigger than max of 1.0" validation
                // errors. bg_depth_scale now stays at its INPUT_TYPES default (0.24).
                setW("weight_smart",   $v("sl-wsm", 0.7));
                setW("weight_bae",     $v("sl-wba", 0.2));
                setW("weight_depth",   $v("sl-wdp", 0.1));
                setW("smart_strength", $v("sl-sst", 2.9));
                setW("smart_blur",     $v("sl-sbl", 0.0));
                setW("smart_detail",   $v("sl-sdt", 0.3));
                setW("normal_strength",$v("sl-nrm", 1.0));
                setW("hf_detail",      $v("sl-nhf", 1.0));
                setW("spec_strength",  $v("sl-spc", 0.3));
                setW("spec_shiny",     $v("sl-shn", 16));
                setW("target_x",       $v("sl-tx", 0.0));
                setW("target_y",       $v("sl-ty", 0.0));
                setW("target_z",       $v("sl-tz", 0.0));
                setW("spot_angle_deg", $v("sl-cone", 60));
                setW("spot_softness",  $v("sl-soft", 0.7));
                setW("light_size",     $v("sl-lsz", 0.0));
                setW("disp_amount",    $v("sl-vdp", 0.5));
                setW("mask_cutoff",    $v("sl-mask", 0.05));
                setW("mask_falloff",   $v("sl-mfth", 0.0));
                setW("exposure_stops", $v("sl-exp", 0.0));
                setW("grade_gain",     $v("sl-gain", 1.0));
                setW("grade_lift",     $v("sl-lift", 0.0));
                setW("shadow_strength",$v("sl-shd", 1.0));
                // Booleans
                setW("use_bae_normal", $c("sl-bae", true));
                setW("mask_invert",    $c("sl-minv", false));
                var fxEl = document.getElementById("sl-fixed");
                if(fxEl) setW("fixed_intensity", fxEl.checked);
                // Tints from color wheel
                if(window._slGradeTints){
                    var t = window._slGradeTints;
                    if(t.gain){
                        setW("grade_gain_r", t.gain[0]);
                        setW("grade_gain_g", t.gain[1]);
                        setW("grade_gain_b", t.gain[2]);
                    }
                    if(t.lift){
                        setW("grade_lift_r", t.lift[0]);
                        setW("grade_lift_g", t.lift[1]);
                        setW("grade_lift_b", t.lift[2]);
                    }
                    if(t.gamma){
                        setW("grade_gamma_r", t.gamma[0]);
                        setW("grade_gamma_g", t.gamma[1]);
                        setW("grade_gamma_b", t.gamma[2]);
                    }
                }
                // CRITICAL: lights_json - serialize current _lights array
                setW("lights_json", JSON.stringify(_lights || []));
            }
        } catch(e) {
            // Don't break the panel if ComfyUI's graph API changes
            console.warn("[STAGELIGHT] syncPanelStateToNode failed:", e);
        }
    }
    // Expose globally so panel RENDER button and other handlers can also call it
    window._slSyncToNode = syncPanelStateToNode;

    var panel=document.createElement("div");
    panel.id="sl-panel";
    panel.tabIndex = -1;  // focusable but not in tab order; needed for keyboard events to work reliably
    panel.style.cssText="position:fixed;top:52px;right:16px;width:920px;height:600px;"+
        "min-width:780px;min-height:440px;background:#111;border:1px solid #2a2a2a;"+
        "border-radius:6px;z-index:9999;font-family:Consolas,monospace;color:#ccc;"+
        "display:flex;flex-direction:column;box-shadow:0 10px 40px rgba(0,0,0,0.85);user-select:none;outline:none;";

    function slRow(id, lbl, def, mn, mx, st){
        return '<div style="display:flex;align-items:center;margin-bottom:4px;">'+
            '<span style="width:64px;color:#888;font-size:8px;letter-spacing:1px;">'+lbl+'</span>'+
            '<input id="'+id+'" type="range" min="'+mn+'" max="'+mx+'" step="'+st+'" value="'+def+'" '+
            'style="flex:1;accent-color:#e8a020;height:12px;margin:0 6px;">'+
            '<span id="'+id+'-v" class="sl-num" data-slider="'+id+'" '+
            'style="width:44px;color:#e8a020;font-size:9px;text-align:right;cursor:text;" '+
            'title="Click to type a value (overrides slider range)">'+
            parseFloat(def).toFixed(2)+'</span></div>';
    }

    // Grade-row variant: identical to slRow but adds a 12x12 color square on the right.
    // Clicking the square opens the HSV color wheel anchored to the click. The tint key
    // (lift/gain/gamma) is used to look up + store the chosen color in window._slGradeTints.
    function slRowTinted(id, lbl, def, mn, mx, st, tintKey){
        return '<div style="display:flex;align-items:center;margin-bottom:4px;">'+
            '<span style="width:54px;color:#888;font-size:8px;letter-spacing:1px;">'+lbl+'</span>'+
            '<input id="'+id+'" type="range" min="'+mn+'" max="'+mx+'" step="'+st+'" value="'+def+'" '+
            'style="flex:1;accent-color:#e8a020;height:12px;margin:0 4px;">'+
            '<span id="'+id+'-v" class="sl-num" data-slider="'+id+'" '+
            'style="width:38px;color:#e8a020;font-size:9px;text-align:right;cursor:text;" '+
            'title="Click to type a value (overrides slider range)">'+
            parseFloat(def).toFixed(2)+'</span>'+
            '<button class="sl-tint-sw" data-tint="'+tintKey+'" id="sl-tint-'+tintKey+'" '+
            'style="width:14px;height:14px;border:1px solid #444;background:#ffffff;cursor:pointer;'+
            'margin-left:4px;padding:0;border-radius:2px;" title="Click for color tint"></button>'+
            '</div>';
    }

    function slSection(id, title, body, collapsed){
        // Collapsible section - header click toggles body visibility.
        var arrow = collapsed ? "\\u25b6" : "\\u25bc";  // right vs down
        var bodyDisp = collapsed ? "display:none;" : "";
        return '<div class="sl-section" data-sec="'+id+'" style="margin-bottom:8px;border:1px solid #1a1a1a;border-radius:3px;overflow:hidden;">'+
            '<div class="sl-sec-hdr" data-target="sl-sec-'+id+'" style="background:#1a1208;color:#e8a020;font-size:8px;letter-spacing:2px;padding:5px 8px;cursor:pointer;display:flex;align-items:center;user-select:none;">'+
                '<span class="sl-sec-arrow" style="margin-right:6px;color:#e8a020;font-size:9px;">'+arrow+'</span>'+
                '<span style="flex:1;">'+title+'</span>'+
            '</div>'+
            '<div id="sl-sec-'+id+'" class="sl-sec-body" style="padding:6px 8px;background:#0a0a0a;'+bodyDisp+'">'+
                body+
            '</div>'+
        '</div>';
    }

    panel.innerHTML=
        '<div id="sl-hdr" style="display:flex;align-items:center;padding:0 14px;height:40px;'+
        'background:#0a0a0a;border-bottom:1px solid #1e1e1e;border-radius:6px 6px 0 0;cursor:grab;flex-shrink:0;">'+
        '<span style="color:#fff;font-size:13px;font-weight:bold;letter-spacing:3px;">STAGELIGHT</span>'+
        '<span style="color:#e8a020;font-size:8px;letter-spacing:3px;margin-left:10px;">v0.9.5</span>'+
        '<span style="flex:1"></span>'+
        '<button id="sl-min" style="background:transparent;color:#555;border:none;font-size:14px;cursor:pointer;padding:0 6px;">&#8212;</button>'+
        '<button id="sl-x"   style="background:transparent;color:#555;border:none;font-size:14px;cursor:pointer;padding:0 4px;">&#x2715;</button>'+
        '</div>'+
        '<div style="flex:1;display:flex;min-height:0;">'+
            // === LEFT PANEL: geometry + normals ===
            '<div id="sl-ctl-l" style="width:230px;background:#0d0d0d;'+
                'padding:8px 10px;overflow-y:auto;font-size:10px;flex-shrink:0;">'+
                // BACKGROUND: combines all subject-cutout / background controls into
                // one panel with subheaders. Renamed sliders for clarity:
                //   PUSH BACK    (was BG DEPTH)   - pushes BG plane back in Z
                //   SUBJECT BLEND (was BG SHOW)   - how much BG contributes to render
                //   DEPTH CUT / EDGE SOFT         - the actual mask cutoff
                //   VIEWPORT TINT (was VIEW BG)   - viewport-only color overlay
                // v0.8.7: simple depth-cutoff mask + INVERT checkbox.
                // Drag MASK CUTOFF up to hide more of the background-side pixels.
                // Toggle INVERT to flip which side is hidden.
                // DEPTH CONTROLS (was BACKGROUND): all depth-mask + displacement controls
                slSection("background", "DEPTH CONTROLS",
                    slRow("sl-mask","MASK CUTOFF",  0.05, 0, 1, 0.01)+
                    slRow("sl-mfth","MASK FALLOFF", 0.0, 0.0, 0.5, 0.005)+
                    slRow("sl-vdp", "DEPTH AMOUNT", 0.60, 0.0, 1.5, 0.05)+
                    '<label style="display:flex;align-items:center;gap:6px;color:#e8a020;font-size:9px;letter-spacing:1px;cursor:pointer;margin-top:6px;">'+
                    '<input id="sl-minv" type="checkbox" style="accent-color:#e8a020;cursor:pointer;">INVERT'+
                    '<span style="color:#555;font-size:8px;">(flip cut side)</span>'+
                    '</label>')+
                // DISPLAY: pure viewport tweaks (no effect on rendered output)
                slSection("display", "DISPLAY",
                    slRow("sl-gsz","GIZMO SZ", 1.0, 0.3, 4.0, 0.1))+
                slSection("composite", "COMPOSITE",
                    // Nuke-style grade chain. EXPOSURE is in stops (1 stop = 2x).
                    // GAIN multiplies, LIFT adds, GAMMA is the midtone curve.
                    // SHADOW controls how dark unlit areas become (1 = full shadow,
                    // 0 = pass-through with no darkening, just additive lighting).
                    slRow("sl-shd", "SHADOW STRENGTH", 1.0, 0.0, 1.0, 0.01)+
                    slRow("sl-exp", "EXPOSURE", 0.0, -5.0, 5.0, 0.05)+
                    slRowTinted("sl-gain","GAIN",  1.0,  0.5, 2.0, 0.01, "gain")+
                    slRowTinted("sl-lift","LIFT",  0.0, -0.2, 0.2, 0.005, "lift")+
                    slRowTinted("sl-gam", "GAMMA", 1.0,  0.5, 2.5, 0.05, "gamma")+
                    // Reset just the four grade sliders to their neutral identity values
                    '<div style="display:flex;justify-content:flex-end;margin-top:6px;">'+
                    '<button id="sl-comp-reset" style="background:#2a0a0a;color:#cc6644;border:1px solid #441616;'+
                    'padding:3px 10px;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;'+
                    'letter-spacing:1px;" title="Reset EXPOSURE, GAIN, LIFT, GAMMA to neutral defaults">RESET SETTINGS</button>'+
                    '</div>')+
                // SURFACE NORMALS - combines BLEND + SMART + POST into one panel with subheaders.
                // Lives at the BOTTOM of the left column per user spec.
                slSection("normals", "SURFACE NORMALS",
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin-bottom:3px;">BLEND</div>'+
                    slRow("sl-wsm","SMART",  0.7, 0.0, 1.0, 0.01)+
                    slRow("sl-wba","BAE",    0.2, 0.0, 1.0, 0.01)+
                    slRow("sl-wdp","DEPTH",  0.1, 0.0, 1.0, 0.01)+
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin:8px 0 3px 0;border-top:1px solid #1a1a1a;padding-top:6px;">SMART</div>'+
                    slRow("sl-sst","STRENGTH",  2.9, 0.5, 12.0, 0.1)+
                    slRow("sl-sbl","BLUR",      0.0, 0.0, 4.0, 0.1)+
                    slRow("sl-sdt","DETAIL",    0.3, 0.3, 3.0, 0.05)+
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin:8px 0 3px 0;border-top:1px solid #1a1a1a;padding-top:6px;">POST</div>'+
                    slRow("sl-nrm","BOOST",     1.0, 0.5, 6.0, 0.1)+
                    slRow("sl-nhf","HF DETAIL", 1.0, 1.0, 5.0, 0.1)+
                    slRow("sl-nss","BAE SS",    1.0, 1.0, 3.0, 0.25),
                    true)+  // collapsed by default
            '</div>'+
            // === LEFT SPLITTER ===
            '<div id="sl-split-l" style="width:5px;background:#1e1e1e;cursor:ew-resize;flex-shrink:0;'+
                'transition:background 0.15s;" title="Drag to resize"></div>'+
            // === CENTER VIEWPORT ===
            '<div id="sl-vp" style="flex:1;background:#060606;position:relative;overflow:hidden;cursor:grab;min-width:300px;">'+
                '<canvas id="sl-canvas" style="display:block;width:100%;height:100%;"></canvas>'+
                '<div id="sl-viewbar" style="position:absolute;top:8px;left:8px;display:flex;gap:2px;'+
                'background:rgba(10,10,10,0.85);padding:3px;border:1px solid #222;border-radius:4px;'+
                'font-family:Consolas;font-size:9px;letter-spacing:1px;z-index:5;">'+
                '<button data-view="0" class="sl-vbtn" style="background:transparent;color:#666;border:none;padding:3px 8px;cursor:pointer;">COLOR</button>'+
                '<button data-view="1" class="sl-vbtn" style="background:transparent;color:#666;border:none;padding:3px 8px;cursor:pointer;">DEPTH</button>'+
                '<button data-view="2" class="sl-vbtn" style="background:transparent;color:#666;border:none;padding:3px 8px;cursor:pointer;">NORMAL</button>'+
                '<button data-view="3" class="sl-vbtn sl-vbtn-active" style="background:#2a1a04;color:#e8a020;border:none;padding:3px 8px;cursor:pointer;">RELIT</button>'+
                '</div>'+
                '<div id="sl-tmodebar" style="position:absolute;top:8px;right:8px;display:flex;gap:2px;'+
                'background:rgba(10,10,10,0.85);padding:3px;border:1px solid #222;border-radius:4px;'+
                'font-family:Consolas;font-size:9px;letter-spacing:1px;z-index:5;" title="Hotkeys: W=move | E=rotate | R=scale">'+
                '<button data-tmode="t" class="sl-tmbtn sl-tmbtn-active" style="background:#2a1a04;color:#e8a020;border:none;padding:3px 10px;cursor:pointer;" title="Translate (W)">MOVE</button>'+
                '<button data-tmode="r" class="sl-tmbtn" style="background:transparent;color:#666;border:none;padding:3px 10px;cursor:pointer;" title="Rotate (E)">ROTATE</button>'+
                '<button data-tmode="s" class="sl-tmbtn" style="background:transparent;color:#666;border:none;padding:3px 10px;cursor:pointer;" title="Scale / Softness (R)">SCALE</button>'+
                '</div>'+
                '<div id="sl-msg" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'+
                'color:#2a2a2a;font-size:10px;letter-spacing:2px;text-align:center;pointer-events:none;">'+
                'PLACE LIGHT  \u00b7  RENDER RELIT</div>'+
            '</div>'+
            // === RIGHT SPLITTER ===
            '<div id="sl-split-r" style="width:5px;background:#1e1e1e;cursor:ew-resize;flex-shrink:0;'+
                'transition:background 0.15s;" title="Drag to resize"></div>'+
            // === RIGHT PANEL: light controls ===
            // Outer container is flex column - non-scrolling. Inside: pinned outliner
            // header + scrollable attribute body. This way the outliner stays visible
            // while attribute sections scroll behind it (Unreal-style details panel).
            '<div id="sl-ctl-r" style="width:230px;background:#0d0d0d;'+
                'font-size:10px;flex-shrink:0;display:flex;flex-direction:column;overflow:hidden;">'+
                // PINNED HEADER - outliner sits here, never scrolls
                '<div id="sl-ctl-r-pinned" style="padding:8px 10px 4px 10px;flex-shrink:0;'+
                    'background:#0d0d0d;position:relative;">'+
                    slSection("outliner", "OUTLINER",
                        '<div id="sl-outliner-list" style="display:flex;flex-direction:column;gap:2px;margin-bottom:6px;max-height:140px;overflow-y:auto;"></div>'+
                        '<div style="display:flex;gap:4px;">'+
                        '<button id="sl-light-add" style="flex:1;background:#0a2a14;color:#5acc7a;border:1px solid #1f4a2f;padding:4px 0;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;letter-spacing:1px;" title="Add a new light at origin">+ ADD</button>'+
                        '</div>')+
                '</div>'+
                // Drag handle for resizing the pinned outliner area
                '<div id="sl-pin-resize" style="height:5px;background:#1a1a1a;cursor:ns-resize;flex-shrink:0;'+
                    'border-top:1px solid #050505;border-bottom:1px solid #050505;transition:background 0.15s;" title="Drag to resize outliner"></div>'+
                // SCROLLABLE BODY - all attribute sections live here
                '<div id="sl-ctl-r-body" style="padding:6px 10px 8px 10px;overflow-y:auto;flex:1;">'+
                // POSITION: combined light + target with internal subheaders
                slSection("lpos", "POSITION",
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin-bottom:3px;">LIGHT</div>'+
                    slRow("sl-lx","X", -0.5, -3, 3, 0.05)+
                    slRow("sl-ly","Y",  1.0, -3, 3, 0.05)+
                    slRow("sl-lz","Z",  2.0, -3, 5, 0.05)+
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin:8px 0 3px 0;border-top:1px solid #1a1a1a;padding-top:6px;">TARGET</div>'+
                    slRow("sl-tx","X", 0.0, -3, 3, 0.05)+
                    slRow("sl-ty","Y", 0.0, -3, 3, 0.05)+
                    slRow("sl-tz","Z", 0.0, -3, 3, 0.05),
                    true)+  // collapsed by default
                // LIGHT ATTRIBUTES: combines TYPE + COLOR + SETTINGS + CONE in one panel
                // with subheaders. CONE is wrapped in a div that hides for non-SPOT lights.
                slSection("lattr", "LIGHT ATTRIBUTES",
                    // --- TYPE subheader
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin-bottom:3px;">TYPE</div>'+
                    '<div style="display:flex;gap:4px;margin-bottom:6px;">'+
                    '<button class="sl-ltbtn" data-lt="0" style="flex:1;background:#2a1a04;color:#e8a020;border:1px solid #1a1a1a;padding:5px 0;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;">POINT</button>'+
                    '<button class="sl-ltbtn" data-lt="1" style="flex:1;background:transparent;color:#666;border:1px solid #1a1a1a;padding:5px 0;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;">SPOT</button>'+
                    '<button class="sl-ltbtn" data-lt="2" style="flex:1;background:transparent;color:#666;border:1px solid #1a1a1a;padding:5px 0;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;">DIR</button>'+
                    '</div>'+
                    '<input id="sl-ltype" type="hidden" value="0">'+
                    // --- COLOR subheader
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin:8px 0 3px 0;border-top:1px solid #1a1a1a;padding-top:6px;">COLOR</div>'+
                    '<div style="display:flex;align-items:center;margin-bottom:6px;">'+
                    '<span style="width:64px;color:#888;font-size:8px;letter-spacing:1px;">PICKER</span>'+
                    // Custom color swatch button - opens the HSV wheel anchored to itself
                    '<button id="sl-cwell" style="flex:1;height:22px;background:#ffffff;'+
                    'border:1px solid #333;border-radius:3px;cursor:pointer;margin:0 6px;padding:0;" '+
                    'title="Click to open color wheel"></button>'+
                    // Hidden native input kept for backward compat with saved state
                    '<input id="sl-cpick" type="hidden" value="#ffffff">'+
                    '<span style="width:44px;color:#e8a020;font-size:8px;text-align:right;letter-spacing:1px;">WHEEL</span>'+
                    '</div>'+
                    slRow("sl-lr","R", 1.0, 0, 1, 0.01)+
                    slRow("sl-lg","G", 1.0, 0, 1, 0.01)+
                    slRow("sl-lb","B", 1.0, 0, 1, 0.01)+
                    // --- SETTINGS subheader (was its own LIGHT panel)
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin:8px 0 3px 0;border-top:1px solid #1a1a1a;padding-top:6px;">SETTINGS</div>'+
                    '<label style="display:flex;align-items:center;gap:6px;color:#e8a020;font-size:9px;letter-spacing:1px;cursor:pointer;margin-bottom:6px;">'+
                    '<input id="sl-fixed" type="checkbox" checked style="accent-color:#e8a020;cursor:pointer;">FIXED INTENSITY'+
                    '<span style="color:#555;font-size:8px;">(constant within reach)</span>'+
                    '</label>'+
                    slRow("sl-int","INTENSITY", 1.0, 0, 15, 0.05)+
                    slRow("sl-fal","ATTEN", 5.0, 0.5, 20, 0.1)+
                    slRow("sl-fcv","CURVE", 2.0, 0.5, 4.0, 0.1)+
                    slRow("sl-lsz","SCALE", 0.0, 0.0, 1.0, 0.01)+
                    // --- CONE subheader, wrapped so we can hide for non-SPOT lights
                    '<div id="sl-cone-wrap">'+
                    '<div style="color:#666;font-size:8px;letter-spacing:1px;margin:8px 0 3px 0;border-top:1px solid #1a1a1a;padding-top:6px;">CONE</div>'+
                    slRow("sl-cone","CONE \u00b0", 60, 1, 180, 1)+
                    slRow("sl-soft","CONE FT", 0.7, 0, 1, 0.01)+
                    '</div>',
                    true)+  // LIGHT ATTRIBUTES collapsed by default
                slSection("spec", "SPECULAR",
                    slRow("sl-spc","SPEC STR", 0.3, 0.0, 2.0, 0.05)+
                    slRow("sl-shn","SHINY",    16, 1, 128, 1),
                    true)+  // collapsed
                '</div>'+ // close sl-ctl-r-body (scrollable attributes)
            '</div>'+ // close sl-ctl-r (right panel container)
        '</div>'+
        '<div id="sl-foot" style="display:flex;align-items:center;padding:0 14px;height:44px;'+
        'background:#0a0a0a;border-top:1px solid #1a1a1a;border-radius:0 0 6px 6px;flex-shrink:0;gap:10px;">'+
        '<button id="sl-gen" style="background:#0d3820;color:#44dd88;border:1px solid #1a9050;'+
        'border-radius:3px;padding:7px 16px;cursor:pointer;font-family:Consolas;font-size:10px;'+
        'letter-spacing:1px;font-weight:bold;">RENDER</button>'+
        '<label style="display:flex;align-items:center;gap:4px;color:#888;font-size:9px;letter-spacing:1px;cursor:pointer;" title="When ON, normal-map and depth sliders auto-trigger a render after 600ms.">'+
        '<input id="sl-autorender" type="checkbox" checked style="accent-color:#44dd88;cursor:pointer;">AUTO</label>'+
        '<label style="display:flex;align-items:center;gap:4px;color:#888;font-size:9px;letter-spacing:1px;cursor:pointer;">'+
        '<input id="sl-bae" type="checkbox" checked style="accent-color:#e8a020;cursor:pointer;">BAE (NORMAL MAP)</label>'+
        '<label style="display:flex;align-items:center;gap:4px;color:#888;font-size:9px;letter-spacing:1px;cursor:pointer;">'+
        '<input id="sl-camlock" type="checkbox" style="accent-color:#e8a020;cursor:pointer;">CAM LOCK</label>'+
        '<label style="display:flex;align-items:center;gap:4px;color:#888;font-size:9px;letter-spacing:1px;cursor:pointer;">'+
        '<input id="sl-gizmos" type="checkbox" checked style="accent-color:#e8a020;cursor:pointer;">GIZMOS</label>'+
        '<label style="display:flex;align-items:center;gap:4px;color:#888;font-size:9px;letter-spacing:1px;cursor:pointer;">'+
        '<input id="sl-conevis" type="checkbox" checked style="accent-color:#e8a020;cursor:pointer;">CONE</label>'+
        '<span style="color:#666;font-size:9px;letter-spacing:1px;">GIZMO OP</span>'+
        '<input id="sl-gop" type="range" min="0" max="1" step="0.01" value="0.5" '+
        'style="width:60px;accent-color:#e8a020;height:10px;margin:0 4px;" title="Gizmo opacity (all light handles + markers)">'+
        '<label style="display:flex;align-items:center;gap:4px;color:#888;font-size:9px;letter-spacing:1px;cursor:pointer;">'+
        '<input id="sl-tgtvis" type="checkbox" style="accent-color:#e8a020;cursor:pointer;">TARGET</label>'+
        '<button id="sl-frame" style="background:#0a1a2a;color:#5a9acc;border:1px solid #1f3a52;'+
        'padding:3px 10px;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;'+
        'margin-left:8px;letter-spacing:1px;" title="Frame camera on subject (F)">FRAME</button>'+
        '<button id="sl-reset" style="background:#2a0a0a;color:#cc6644;border:1px solid #441616;'+
        'padding:3px 10px;font-family:Consolas;font-size:9px;cursor:pointer;border-radius:3px;'+
        'margin-left:4px;letter-spacing:1px;" title="Reset all settings to defaults">RESET</button>'+
        '<div id="sl-status" style="flex:1;font-size:9px;color:#444;letter-spacing:1px;text-align:right;">'+
        'Node #'+(node?node.id:"?")+' · click RGB handles for single axis · sphere for free drag</div>'+
        '</div>'+
        '<div id="sl-rsz" style="position:absolute;bottom:2px;right:2px;width:16px;height:16px;'+
        'cursor:nwse-resize;color:#333;font-size:14px;line-height:1;text-align:center;">&#8689;</div>';

    document.body.appendChild(panel);

    // Restore previous panel state from localStorage (if any)
    var _savedSettings = loadSavedSettings();
    if(_savedSettings) applySavedSettings(_savedSettings);

    // === Multi-light initialization (v0.8.0) ===
    // If saved settings contain a _lights array, restore it. Otherwise migrate the
    // current slider values (from single-light v0.7.x save) into a single _lights[0].
    _lights = [];
    _selectedLight = 0;
    // v0.8.13: restore grade tints (or initialize to neutral)
    if(_savedSettings && _savedSettings._gradeTints){
        window._slGradeTints = _savedSettings._gradeTints;
    } else if(!window._slGradeTints){
        window._slGradeTints = { lift:[1,1,1], gain:[1,1,1], gamma:[1,1,1] };
    }
    if(_savedSettings && _savedSettings._lights && _savedSettings._lights.length > 0){
        // Restore saved multi-light state
        _lights = _savedSettings._lights;
        _selectedLight = Math.max(0, Math.min(_lights.length - 1, _savedSettings._selectedLight || 0));
        // Normalize names: any light without a unique name gets "Light N" assigned.
        // Handles legacy state from v0.8.3 where all lights might say just "Light".
        for(var ni = 0; ni < _lights.length; ni++){
            if(!_lights[ni].name || _lights[ni].name === "Light"){
                _lights[ni].name = "Light " + (ni + 1);
            }
        }
        // Push the selected light's values into the sliders
        populateSlidersFromLight();
    } else {
        // Migrate from single-light state: create one light from current slider values
        var L = makeDefaultLight();
        L.light_x = parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value);
        L.light_y = parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value);
        L.light_z = parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value);
        L.target_x = parseFloat((document.getElementById("sl-tx")||{value: 0.0}).value);
        L.target_y = parseFloat((document.getElementById("sl-ty")||{value: 0.0}).value);
        L.target_z = parseFloat((document.getElementById("sl-tz")||{value: 0.0}).value);
        L.intensity = parseFloat((document.getElementById("sl-int")||{value: 2.0}).value);
        L.falloff = parseFloat((document.getElementById("sl-fal")||{value: 5.0}).value);
        L.fcv = parseFloat((document.getElementById("sl-fcv")||{value: 2.0}).value);
        L.light_r = parseFloat((document.getElementById("sl-lr")||{value: 1.0}).value);
        L.light_g = parseFloat((document.getElementById("sl-lg")||{value: 1.0}).value);
        L.light_b = parseFloat((document.getElementById("sl-lb")||{value: 1.0}).value);
        L.spec_strength = parseFloat((document.getElementById("sl-spc")||{value: 0.3}).value);
        L.spec_shiny = parseFloat((document.getElementById("sl-shn")||{value: 16}).value);
        L.cone_angle = parseFloat((document.getElementById("sl-cone")||{value: 60}).value);
        L.cone_soft = parseFloat((document.getElementById("sl-soft")||{value: 0.7}).value);
        L.light_size = parseFloat((document.getElementById("sl-lsz")||{value: 0.0}).value);
        var fxEl = document.getElementById("sl-fixed");
        L.fixed_intensity = fxEl ? fxEl.checked : true;
        var ltyEl = document.getElementById("sl-ltype");
        L.light_type = ltyEl ? parseInt(ltyEl.value) : 1;
        var tvEl = document.getElementById("sl-tgtvis");
        L.target_visible = tvEl ? tvEl.checked : false;
        L.name = "Light 1";  // first migrated light gets a real default name
        _lights.push(L);
    }
    renderOutliner();
    // Hide/show conditional sections based on the active light's type
    if(window._slUpdateConditionalSections) window._slUpdateConditionalSections();

    // If TARGET was restored as OFF, derive a parent-child offset from the saved
    // light + target positions so the relationship is preserved across panel reopens.
    (function(){
        var tb = document.getElementById("sl-tgtvis");
        if(!tb || tb.checked) { _targetOffset = null; return; }
        var lx = parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value);
        var ly = parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value);
        var lz = parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value);
        var tx = parseFloat((document.getElementById("sl-tx")||{value:0.0}).value);
        var ty = parseFloat((document.getElementById("sl-ty")||{value:0.0}).value);
        var tz = parseFloat((document.getElementById("sl-tz")||{value:0.0}).value);
        // Use the actual saved offset (not normalized) so the target stays exactly
        // where it was last seen, not snapped to a different distance.
        _targetOffset = {x: tx - lx, y: ty - ly, z: tz - lz};
    })();

    // --- Panel splitters: drag to resize columns
    function setupSplitter(splitId, panelId, side){
        var split = document.getElementById(splitId);
        var pnl = document.getElementById(panelId);
        if(!split || !pnl) return;
        var resizing = false, startX = 0, startW = 0;
        split.addEventListener("mouseenter", function(){
            split.style.background = "#e8a020";
        });
        split.addEventListener("mouseleave", function(){
            if(!resizing) split.style.background = "#1e1e1e";
        });
        split.addEventListener("mousedown", function(e){
            resizing = true;
            startX = e.clientX;
            startW = pnl.offsetWidth;
            split.style.background = "#e8a020";
            document.body.style.cursor = "ew-resize";
            document.body.style.userSelect = "none";
            e.preventDefault();
            e.stopPropagation();
        });
        var onMove = function(e){
            if(!resizing) return;
            var dx = e.clientX - startX;
            // For left splitter: dragging right = wider left panel.
            // For right splitter: dragging right = narrower right panel (so flip sign).
            var newW = (side === "left") ? (startW + dx) : (startW - dx);
            // Min 120px (so numbers stay visible), max 500px (so viewport stays usable)
            newW = Math.max(120, Math.min(500, newW));
            pnl.style.width = newW + "px";
        };
        var onUp = function(){
            if(!resizing) return;
            resizing = false;
            split.style.background = "#1e1e1e";
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
            scheduleSave();
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        // Stash handlers for cleanup
        if(!window._slSplitHandlers) window._slSplitHandlers = [];
        window._slSplitHandlers.push({move: onMove, up: onUp});
    }
    setupSplitter("sl-split-l", "sl-ctl-l", "left");
    setupSplitter("sl-split-r", "sl-ctl-r", "right");

    // --- Outliner resize: vertical drag handle below the pinned area
    // Drags up/down to change the outliner section's max-height. The list inside
    // the outliner becomes more or less spacious as you resize.
    (function(){
        var handle = document.getElementById("sl-pin-resize");
        var list = document.getElementById("sl-outliner-list");
        if(!handle || !list) return;
        var resizing = false, startY = 0, startH = 0;
        handle.addEventListener("mouseenter", function(){ handle.style.background = "#e8a020"; });
        handle.addEventListener("mouseleave", function(){ if(!resizing) handle.style.background = "#1a1a1a"; });
        handle.addEventListener("mousedown", function(e){
            resizing = true;
            startY = e.clientY;
            startH = parseInt(list.style.maxHeight) || 140;
            handle.style.background = "#e8a020";
            document.body.style.cursor = "ns-resize";
            document.body.style.userSelect = "none";
            e.preventDefault();
            e.stopPropagation();
        });
        var onMove = function(e){
            if(!resizing) return;
            var dy = e.clientY - startY;
            var newH = Math.max(40, Math.min(500, startH + dy));
            list.style.maxHeight = newH + "px";
        };
        var onUp = function(){
            if(!resizing) return;
            resizing = false;
            handle.style.background = "#1a1a1a";
            document.body.style.cursor = "";
            document.body.style.userSelect = "";
            // Save the height so it persists across panel reopens
            try {
                var saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
                saved.outliner_h = parseInt(list.style.maxHeight) || 140;
                localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
            } catch(e){}
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        // Stash for cleanup
        if(!window._slPinResizeHandlers) window._slPinResizeHandlers = [];
        window._slPinResizeHandlers.push({move: onMove, up: onUp});
        // Restore saved height
        try {
            var saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
            if(saved.outliner_h) list.style.maxHeight = saved.outliner_h + "px";
        } catch(e){}
    })();

    // --- Collapsible section headers
    document.querySelectorAll(".sl-sec-hdr").forEach(function(hdr){
        hdr.addEventListener("click", function(){
            var bodyId = hdr.getAttribute("data-target");
            var body = document.getElementById(bodyId);
            var arrow = hdr.querySelector(".sl-sec-arrow");
            if(!body) return;
            var collapsed = body.style.display === "none";
            body.style.display = collapsed ? "" : "none";
            if(arrow) arrow.textContent = collapsed ? "\u25bc" : "\u25b6";
            scheduleSave();
        });
    });

    // --- auto-render debounce
    var _autoTimer=null;
    // AUTO was removed in v0.7.5 - the GPU shader handles all live lighting.
    // This stub remains so any code path that calls it still works. AUTO will return
    // for projection / AI cleanup workflows in v0.8.x+ when those features need a
    // Python re-render to be triggered automatically.
    // v0.9.2: triggerAutoRender is now a real debounced re-render dispatcher.
    // Called by sliders that affect Python-computed pipeline (normal map, depth).
    // Debounce ~600ms so dragging a slider produces ONE render when you stop, not 30 mid-drag.
    // Respects the AUTO checkbox: if off, sliders update silently and require a manual RENDER.
    var _autoRenderTimer = null;
    function triggerAutoRender(){
        var toggle = document.getElementById("sl-autorender");
        if(toggle && !toggle.checked) return;
        if(_autoRenderTimer) clearTimeout(_autoRenderTimer);
        _autoRenderTimer = setTimeout(function(){
            var btn = document.getElementById("sl-gen");
            // If render button is disabled, a render is already in flight.
            // Skip - the next slider change will retry.
            if(btn && !btn.disabled) btn.click();
        }, 600);
    }
    // Set of slider IDs that affect Python-side computation (normal map / depth).
    // Other sliders (lights, grade chain, mask) update live in GLSL and don't need re-render.
    var _pythonSliders = {
        "sl-nrm":1, "sl-nhf":1,         // normal_strength, hf_detail
        "sl-sst":1, "sl-sbl":1, "sl-sdt":1,   // smart_strength, smart_blur, smart_detail
        "sl-wsm":1, "sl-wba":1, "sl-wdp":1,   // weight_smart/bae/depth
        "sl-vdp":1                       // disp_amount (DEPTH AMOUNT)
    };

    // --- slider live updates (all GPU-live now; v0.8.6 removed Python-only BG sliders)
    ["sl-lx","sl-ly","sl-lz","sl-int","sl-fal","sl-fcv","sl-shd","sl-exp","sl-gain","sl-lift","sl-gam","sl-lr","sl-lg","sl-lb","sl-mask","sl-mfth","sl-nrm","sl-nhf","sl-nss","sl-gsz","sl-wsm","sl-wba","sl-wdp","sl-sst","sl-sbl","sl-sdt","sl-vdp","sl-spc","sl-shn","sl-tx","sl-ty","sl-tz","sl-cone","sl-soft","sl-lsz"]
        .forEach(function(id){
            var el=document.getElementById(id);
            var vl=document.getElementById(id+"-v");
            if(el&&vl){
                el.addEventListener("input",function(){
                    vl.textContent=parseFloat(el.value).toFixed(2);
                    // Parent-child: if this is a light-position slider AND target is
                    // off, slide the target with us so the offset stays constant.
                    if((id === "sl-lx" || id === "sl-ly" || id === "sl-lz") && _targetOffset){
                        if(window._slSetTargetFromOffset) window._slSetTargetFromOffset();
                    }
                    if(window._slSyncGizmo)window._slSyncGizmo();
                    if(window._slUpdateLighting)window._slUpdateLighting();
                    // Multi-light: commit current slider state to the selected light
                    if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
                    scheduleSave();
                    // v0.9.2: Python-side sliders need a re-render to see changes
                    // (normal map regen, depth re-pipeline). Debounced.
                    if(_pythonSliders[id]) triggerAutoRender();
                });
            }
        });

    // --- color picker <-> RGB sliders two-way sync
    function hexToRgb01(hex){
        var m=hex.replace("#","");
        return [parseInt(m.substr(0,2),16)/255,parseInt(m.substr(2,2),16)/255,parseInt(m.substr(4,2),16)/255];
    }
    function rgb01ToHex(r,g,b){
        function c(x){var n=Math.max(0,Math.min(255,Math.round(x*255))).toString(16);return n.length<2?"0"+n:n;}
        return "#"+c(r)+c(g)+c(b);
    }
    var cpick=document.getElementById("sl-cpick");
    if(cpick){
        cpick.addEventListener("input",function(){
            var rgb=hexToRgb01(cpick.value);
            ["sl-lr","sl-lg","sl-lb"].forEach(function(id,i){
                var el=document.getElementById(id);
                var vl=document.getElementById(id+"-v");
                if(el){el.value=rgb[i];if(vl)vl.textContent=rgb[i].toFixed(2);}
            });
            if(window._slUpdateLighting)window._slUpdateLighting();
            scheduleSave();
        });
        cpick.addEventListener("change",function(){triggerAutoRender();scheduleSave();});
        function syncPickerFromSliders(){
            var r=parseFloat(document.getElementById("sl-lr").value);
            var g=parseFloat(document.getElementById("sl-lg").value);
            var b=parseFloat(document.getElementById("sl-lb").value);
            cpick.value=rgb01ToHex(r,g,b);
        }
        ["sl-lr","sl-lg","sl-lb"].forEach(function(id){
            var el=document.getElementById(id);
            if(el)el.addEventListener("input",syncPickerFromSliders);
        });
        syncPickerFromSliders();
    }

    // --- BAE toggle also triggers auto-render
    var baeBox=document.getElementById("sl-bae");
    if(baeBox)baeBox.addEventListener("change",function(){triggerAutoRender();scheduleSave();});

    // --- LIGHT TYPE buttons (POINT/SPOT/DIR)
    document.querySelectorAll(".sl-ltbtn").forEach(function(btn){
        btn.addEventListener("click",function(){
            document.querySelectorAll(".sl-ltbtn").forEach(function(b){
                b.style.background="transparent";b.style.color="#666";
            });
            btn.style.background="#2a1a04";btn.style.color="#e8a020";
            var lt = btn.getAttribute("data-lt");
            var hidden=document.getElementById("sl-ltype");
            if(hidden)hidden.value=lt;
            if(window._slUpdateLighting)window._slUpdateLighting();
            if(window._slSyncGizmo)window._slSyncGizmo();
            if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
            // Refresh outliner row (so type label updates) and conditional sections
            if(window._slRenderOutliner) window._slRenderOutliner();
            if(window._slUpdateConditionalSections) window._slUpdateConditionalSections();
            scheduleSave();
        });
    });
    // Apply restored light type visually (the hidden input was loaded from localStorage)
    (function(){
        var hidden=document.getElementById("sl-ltype");
        if(!hidden) return;
        var lt = hidden.value;
        document.querySelectorAll(".sl-ltbtn").forEach(function(b){
            var active = b.getAttribute("data-lt") === lt;
            b.style.background = active ? "#2a1a04" : "transparent";
            b.style.color = active ? "#e8a020" : "#666";
        });
    })();

    // --- GIZMOS visibility toggle (master)
    var gizBox=document.getElementById("sl-gizmos");
    if(gizBox)gizBox.addEventListener("change",function(){
        if(window._slSyncGizmo)window._slSyncGizmo();
        scheduleSave();
    });

    // --- CONE visibility toggle
    var coneBox=document.getElementById("sl-conevis");
    if(coneBox)coneBox.addEventListener("change",function(){
        if(window._slSyncGizmo)window._slSyncGizmo();
        scheduleSave();
    });

    // --- TARGET visibility toggle
    var tgtBox=document.getElementById("sl-tgtvis");
    if(tgtBox)tgtBox.addEventListener("change",function(){
        // Going from ON -> OFF: snap the target to a parent-child offset from the
        // light (1 unit forward along current aim). Every subsequent light move will
        // also move the target by this offset, keeping aim direction constant.
        // Going OFF -> ON: clear the offset; user controls target with sliders.
        if(!tgtBox.checked){
            // Compute current aim direction (from light to target) and use it as offset.
            // Normalize to a fixed length (1.0) so the snap distance is predictable.
            var lx = parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value);
            var ly = parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value);
            var lz = parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value);
            var tx = parseFloat((document.getElementById("sl-tx")||{value:0.0}).value);
            var ty = parseFloat((document.getElementById("sl-ty")||{value:0.0}).value);
            var tz = parseFloat((document.getElementById("sl-tz")||{value:0.0}).value);
            var dx = tx - lx, dy = ty - ly, dz = tz - lz;
            var len = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if(len > 0.0001){
                // Use a fixed offset distance of 1.0 along the current aim direction.
                // This gives the target a predictable resting position centered in the cone.
                _targetOffset = {x: dx/len, y: dy/len, z: dz/len};
            } else {
                // Light coincident with target - default to forward (-Z toward subject)
                _targetOffset = {x: 0, y: 0, z: -1};
            }
            // Snap target sliders to lightPos + offset NOW so the math is consistent
            // with what the user will see going forward.
            setTargetFromOffset();
        } else {
            _targetOffset = null;
        }
        if(window._slSyncGizmo)window._slSyncGizmo();
        if(window._slUpdateLighting)window._slUpdateLighting();
        if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
        scheduleSave();
    });

    // Helper: when _targetOffset is set, push target sliders to lightPos + offset.
    // Called whenever the light position changes (slider input or gizmo drag).
    function setTargetFromOffset(){
        if(!_targetOffset) return;
        var lx = parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value);
        var ly = parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value);
        var lz = parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value);
        var newTx = lx + _targetOffset.x;
        var newTy = ly + _targetOffset.y;
        var newTz = lz + _targetOffset.z;
        // Clamp to slider ranges so we don't go out of bounds (sliders are -3..3)
        newTx = Math.max(-3, Math.min(3, newTx));
        newTy = Math.max(-3, Math.min(3, newTy));
        newTz = Math.max(-3, Math.min(3, newTz));
        ["sl-tx","sl-ty","sl-tz"].forEach(function(id, i){
            var el = document.getElementById(id);
            var vl = document.getElementById(id+"-v");
            if(!el) return;
            var v = [newTx, newTy, newTz][i];
            el.value = v;
            if(vl) vl.textContent = v.toFixed(2);
        });
    }
    // Expose globally so light position handlers (slider input AND gizmo drag) can call it
    window._slSetTargetFromOffset = setTargetFromOffset;

    // ====================================================================
    // === Multi-light commit/populate helpers (v0.8.0)
    // commitSlidersToLight: reads slider values into _lights[_selectedLight]
    // populateSlidersFromLight: pushes _lights[_selectedLight] values into sliders
    // ====================================================================
    function _slv(id, def){
        var el = document.getElementById(id);
        if(!el) return def;
        return (el.type === "checkbox") ? el.checked : parseFloat(el.value);
    }
    function _slvSet(id, v){
        var el = document.getElementById(id);
        if(!el) return;
        if(el.type === "checkbox"){ el.checked = !!v; }
        else {
            el.value = v;
            var vl = document.getElementById(id+"-v");
            if(vl) vl.textContent = parseFloat(v).toFixed(2);
        }
    }
    function commitSlidersToLight(){
        if(!_lights[_selectedLight]) return;
        var L = _lights[_selectedLight];
        L.light_x = _slv("sl-lx", L.light_x);
        L.light_y = _slv("sl-ly", L.light_y);
        L.light_z = _slv("sl-lz", L.light_z);
        L.target_x = _slv("sl-tx", L.target_x);
        L.target_y = _slv("sl-ty", L.target_y);
        L.target_z = _slv("sl-tz", L.target_z);
        L.intensity = _slv("sl-int", L.intensity);
        L.falloff = _slv("sl-fal", L.falloff);
        L.fcv = _slv("sl-fcv", L.fcv);
        L.light_r = _slv("sl-lr", L.light_r);
        L.light_g = _slv("sl-lg", L.light_g);
        L.light_b = _slv("sl-lb", L.light_b);
        L.spec_strength = _slv("sl-spc", L.spec_strength);
        L.spec_shiny = _slv("sl-shn", L.spec_shiny);
        L.cone_angle = _slv("sl-cone", L.cone_angle);
        L.cone_soft = _slv("sl-soft", L.cone_soft);
        L.light_size = _slv("sl-lsz", L.light_size);
        L.fixed_intensity = _slv("sl-fixed", L.fixed_intensity);
        var ltEl = document.getElementById("sl-ltype");
        if(ltEl) L.light_type = parseInt(ltEl.value);
        var tgtVis = document.getElementById("sl-tgtvis");
        if(tgtVis) L.target_visible = tgtVis.checked;
        // Capture per-light frozen offset
        L.target_offset = _targetOffset ? {x:_targetOffset.x, y:_targetOffset.y, z:_targetOffset.z} : null;
    }
    function populateSlidersFromLight(){
        if(!_lights[_selectedLight]) return;
        var L = _lights[_selectedLight];
        _slvSet("sl-lx", L.light_x);
        _slvSet("sl-ly", L.light_y);
        _slvSet("sl-lz", L.light_z);
        _slvSet("sl-tx", L.target_x);
        _slvSet("sl-ty", L.target_y);
        _slvSet("sl-tz", L.target_z);
        _slvSet("sl-int", L.intensity);
        _slvSet("sl-fal", L.falloff);
        _slvSet("sl-fcv", L.fcv);
        _slvSet("sl-lr", L.light_r);
        _slvSet("sl-lg", L.light_g);
        _slvSet("sl-lb", L.light_b);
        _slvSet("sl-spc", L.spec_strength);
        _slvSet("sl-shn", L.spec_shiny);
        _slvSet("sl-cone", L.cone_angle);
        _slvSet("sl-soft", L.cone_soft);
        _slvSet("sl-lsz", L.light_size);
        _slvSet("sl-fixed", L.fixed_intensity);
        var ltEl = document.getElementById("sl-ltype");
        if(ltEl) ltEl.value = L.light_type;
        // Update visible LIGHT TYPE buttons
        document.querySelectorAll(".sl-ltbtn").forEach(function(b){
            var active = parseInt(b.getAttribute("data-lt")) === L.light_type;
            b.style.background = active ? "#2a1a04" : "transparent";
            b.style.color = active ? "#e8a020" : "#666";
        });
        // Target visibility checkbox
        var tgtVis = document.getElementById("sl-tgtvis");
        if(tgtVis) tgtVis.checked = !!L.target_visible;
        // Restore frozen offset
        _targetOffset = L.target_offset ? {x:L.target_offset.x, y:L.target_offset.y, z:L.target_offset.z} : null;
        // Sync hidden color input + visible custom swatch to RGB sliders
        var cp = document.getElementById("sl-cpick");
        if(cp){
            function c(n){var h=Math.round(Math.max(0,Math.min(1,n))*255).toString(16);return h.length<2?"0"+h:h;}
            cp.value = "#" + c(L.light_r) + c(L.light_g) + c(L.light_b);
        }
        // Update the visible swatch button on the picker (v0.8.13 custom color wheel)
        if(window._slSyncCwellSwatch) window._slSyncCwellSwatch();
        // Trigger downstream updates
        if(window._slSyncGizmo) window._slSyncGizmo();
        if(window._slUpdateLighting) window._slUpdateLighting();
        if(window._slUpdateConditionalSections) window._slUpdateConditionalSections();
        if(window._slRequestRender) window._slRequestRender();
    }
    window._slCommitSlidersToLight = commitSlidersToLight;
    window._slPopulateSlidersFromLight = populateSlidersFromLight;

    // Conditional section visibility: SPOT CONE only shows for SPOT lights.
    // Called after light selection change or light-type change.
    function updateConditionalSections(){
        var coneWrap = document.getElementById("sl-cone-wrap");
        if(!coneWrap) return;
        var ltEl = document.getElementById("sl-ltype");
        var lt = ltEl ? parseInt(ltEl.value) : 0;
        coneWrap.style.display = (lt === 1) ? "" : "none";  // 1 = SPOT
    }
    window._slUpdateConditionalSections = updateConditionalSections;

    // ====================================================================
    // === Outliner: render the light list, wire add/select/delete
    // ====================================================================
    function renderOutliner(){
        var list = document.getElementById("sl-outliner-list");
        if(!list) return;
        var html = "";
        for(var i = 0; i < _lights.length; i++){
            var L = _lights[i];
            var sel = (i === _selectedLight);
            var bg = sel ? "#2a1a04" : "transparent";
            var fg = sel ? "#e8a020" : "#888";
            var bd = sel ? "#5a3a14" : "#222";
            var typeLabel = L.light_type === 0 ? "POINT" : (L.light_type === 1 ? "SPOT" : "DIR");
            // Eye icon: filled when enabled, hollow when disabled
            var eyeChar = L.enabled ? "\u25CF" : "\u25CB";
            var eyeColor = L.enabled ? "#5acc7a" : "#444";
            // Disable delete when only one light remains
            var canDelete = _lights.length > 1;
            html += '<div data-row-idx="' + i + '" class="sl-out-row" '+
                'style="display:flex;align-items:center;background:' + bg + ';border:1px solid ' + bd + ';'+
                'padding:3px 5px;border-radius:3px;cursor:pointer;color:' + fg + ';font-size:9px;letter-spacing:0.5px;">'+
                // gizmo color swatch (placeholder yellow for v0.8.0 - per-light colors arrive in v0.8.1)
                // gizmo color swatch reflects per-light color
                '<div style="width:8px;height:8px;background:' + lightColorForIndex(i) + ';border-radius:50%;margin-right:6px;flex-shrink:0;"></div>'+
                '<span class="sl-out-name" data-light-idx="' + i + '" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:text;" title="Double-click to rename">' + L.name + '</span>'+
                '<span style="color:#666;margin-right:6px;font-size:8px;">' + typeLabel + '</span>'+
                '<button class="sl-out-eye" data-light-idx="' + i + '" '+
                    'style="background:transparent;border:none;color:' + eyeColor + ';cursor:pointer;font-size:11px;padding:0 4px;line-height:1;" '+
                    'title="Toggle visibility">' + eyeChar + '</button>'+
                '<button class="sl-out-del" data-light-idx="' + i + '" '+
                    'style="background:transparent;border:none;color:' + (canDelete ? "#cc6644" : "#222") + ';cursor:' + (canDelete ? "pointer" : "not-allowed") + ';font-size:11px;padding:0 4px;line-height:1;" '+
                    (canDelete ? '' : 'disabled') + ' title="Delete light">\u2715</button>'+
                '</div>';
        }
        list.innerHTML = html;
        // Wire row click for selection (avoiding eye/delete buttons)
        list.querySelectorAll(".sl-out-row").forEach(function(row){
            row.addEventListener("click", function(e){
                if(e.target.classList.contains("sl-out-eye")) return;
                if(e.target.classList.contains("sl-out-del")) return;
                var idx = parseInt(row.getAttribute("data-row-idx"));
                if(isNaN(idx) || idx === _selectedLight) return;
                // Commit current slider state to the OLD selected light first
                commitSlidersToLight();
                // Switch selection
                _selectedLight = idx;
                // Push the NEW light's values into the sliders
                populateSlidersFromLight();
                // Refresh outliner highlighting + markers + lighting
                renderOutliner();
                if(window._slSyncGizmo) window._slSyncGizmo();
                if(window._slUpdateLighting) window._slUpdateLighting();
                scheduleSave();
            });
        });
        // Wire eye toggles
        // Wire double-click rename on the name span. Replaces the span with an
        // <input>; on Enter or blur we save the name (deduped via uniqueLightName).
        list.querySelectorAll(".sl-out-name").forEach(function(span){
            span.addEventListener("dblclick", function(e){
                e.stopPropagation();
                var idx = parseInt(span.getAttribute("data-light-idx"));
                if(isNaN(idx) || !_lights[idx]) return;
                var oldName = _lights[idx].name;
                // Build an inline text input replacing the span
                var input = document.createElement("input");
                input.type = "text";
                input.value = oldName;
                input.style.cssText = "flex:1;background:#000;color:#fff;border:1px solid #e8a020;font-family:Consolas;font-size:9px;padding:1px 3px;outline:none;";
                input.maxLength = 32;
                span.parentNode.replaceChild(input, span);
                input.focus();
                input.select();
                var committed = false;
                function commit(save){
                    if(committed) return;
                    committed = true;
                    var newName = save ? input.value.trim() : oldName;
                    if(!newName) newName = oldName;
                    // Apply uniqueness (in case another light already has this name)
                    _lights[idx].name = uniqueLightName(newName, idx);
                    // Re-render outliner
                    if(window._slRenderOutliner) window._slRenderOutliner();
                    scheduleSave();
                }
                input.addEventListener("keydown", function(ke){
                    if(ke.key === "Enter"){ ke.preventDefault(); commit(true); }
                    else if(ke.key === "Escape"){ ke.preventDefault(); commit(false); }
                    ke.stopPropagation();  // don't let our hotkey handler eat W/E/R
                });
                input.addEventListener("blur", function(){ commit(true); });
            });
        });

        list.querySelectorAll(".sl-out-eye").forEach(function(btn){
            btn.addEventListener("click", function(e){
                e.stopPropagation();
                var idx = parseInt(btn.getAttribute("data-light-idx"));
                if(isNaN(idx) || !_lights[idx]) return;
                _lights[idx].enabled = !_lights[idx].enabled;
                renderOutliner();
                if(window._slSyncGizmo) window._slSyncGizmo();
                if(window._slUpdateLighting) window._slUpdateLighting();
                if(window._slRequestRender) window._slRequestRender();
                scheduleSave();
            });
        });
        // Wire delete buttons
        list.querySelectorAll(".sl-out-del").forEach(function(btn){
            btn.addEventListener("click", function(e){
                e.stopPropagation();
                if(_lights.length <= 1) return;  // never delete last light
                var idx = parseInt(btn.getAttribute("data-light-idx"));
                if(isNaN(idx) || !_lights[idx]) return;
                // No confirm dialog - user has Ctrl-Z to undo if it was a mistake
                _lights.splice(idx, 1);
                // Adjust selection if needed
                if(_selectedLight >= _lights.length) _selectedLight = _lights.length - 1;
                if(_selectedLight < 0) _selectedLight = 0;
                populateSlidersFromLight();
                renderOutliner();
                if(window._slSyncGizmo) window._slSyncGizmo();
                if(window._slUpdateLighting) window._slUpdateLighting();
                scheduleSave();
            });
        });
    }
    window._slRenderOutliner = renderOutliner;

    // ADD button handler
    var addBtn = document.getElementById("sl-light-add");
    if(addBtn) addBtn.addEventListener("click", function(){
        if(_lights.length >= _maxLights){
            alert("Maximum " + _maxLights + " lights reached.");
            return;
        }
        // Commit current slider state before switching
        commitSlidersToLight();
        // Add a new light at origin (per user spec). Default name uses unique
        // numbering so adds become "Light 1", "Light 2", etc.
        var nl = makeDefaultLight();
        nl.name = uniqueLightName("Light");
        _lights.push(nl);
        _selectedLight = _lights.length - 1;
        populateSlidersFromLight();
        renderOutliner();
        if(window._slSyncGizmo) window._slSyncGizmo();
        if(window._slUpdateLighting) window._slUpdateLighting();
        scheduleSave();
    });

    // --- RESET button: clears localStorage + reloads panel with defaults
    var resetBtn = document.getElementById("sl-reset");
    if(resetBtn) resetBtn.addEventListener("click", function(){
        if(!confirm("Reset all settings to defaults? This will close and reopen the panel.")) return;
        try { localStorage.removeItem(STORAGE_KEY); } catch(e){}
        // Clear undo history too (no point in undoing back to pre-reset state)
        window._slUndoStack = [];
        window._slUndoPtr = -1;
        // Close & reopen the panel - openPanel detects existing panel and toggles
        var panel = document.getElementById("sl-panel");
        if(panel){ panel.remove(); cleanupThree(); }
        // Re-run openPanel for the same node. Stored on window for access here.
        if(window._slLastNode){
            // Slight delay so cleanup completes
            setTimeout(function(){ openPanel(window._slLastNode); }, 50);
        }
    });

    // --- FRAME button: reset camera to default framing on subject
    var frameBtn = document.getElementById("sl-frame");
    if(frameBtn) frameBtn.addEventListener("click", function(){
        if(window._slFrameSubject) window._slFrameSubject();
    });

    // --- GIZMO opacity slider (live update for all gizmo materials + markers)
    // Applies to: translate handles (xyz arrows + center sphere), rotate rings,
    // scale cube, target gizmo handles, cone (at reduced ratio so it stays subtle),
    // and the gray marker spheres for inactive lights.
    function applyGizmoOpacity(value){
        var v = Math.max(0, Math.min(1, value));
        if(_lightGizmo){
            // Translate handles
            if(_lightGizmo._handleRefs){
                ["x","y","z","free"].forEach(function(a){
                    var r = _lightGizmo._handleRefs[a];
                    if(r && r.mat){ r.mat.transparent = true; r.mat.opacity = v; }
                });
            }
            // Target handles
            if(_lightGizmo._targetHandleRefs){
                ["tgt-x","tgt-y","tgt-z","tgt-free"].forEach(function(a){
                    var r = _lightGizmo._targetHandleRefs[a];
                    if(r && r.mat){ r.mat.transparent = true; r.mat.opacity = v; }
                });
            }
            // Rotate rings
            if(_lightGizmo._rotateRefs){
                ["rx","ry","rz"].forEach(function(a){
                    var r = _lightGizmo._rotateRefs[a];
                    if(r && r.mat){ r.mat.transparent = true; r.mat.opacity = v; }
                });
            }
            // Scale cube
            if(_lightGizmo._scaleCubeMat){
                _lightGizmo._scaleCubeMat.transparent = true;
                _lightGizmo._scaleCubeMat.opacity = v;
            }
            // Cone gets a fraction so it stays subtle even at high gizmo opacity
            if(_lightGizmo._coneMesh && _lightGizmo._coneMesh.material){
                _lightGizmo._coneMesh.material.opacity = v * 0.2;
            }
        }
        // Inactive-light marker spheres
        if(window._slMarkers){
            window._slMarkers.forEach(function(m){
                if(m && m.material){
                    m.material.transparent = true;
                    m.material.opacity = v;
                }
            });
        }
    }
    // Expose so other code (gizmo build, sync) can call this to keep opacity consistent
    window._slApplyGizmoOpacity = applyGizmoOpacity;

    var gopSlider = document.getElementById("sl-gop");
    if(gopSlider) gopSlider.addEventListener("input", function(){
        applyGizmoOpacity(parseFloat(gopSlider.value));
        if(window._slRequestRender) window._slRequestRender();
        scheduleSave();
    });

    // --- RESET SETTINGS button (COMPOSITE panel)
    // Resets just the five grade sliders to their neutral identity values.
    // No confirm dialog - user has Ctrl-Z to recover if it was unintentional.
    var compResetBtn = document.getElementById("sl-comp-reset");
    if(compResetBtn) compResetBtn.addEventListener("click", function(){
        var defs = {"sl-shd": 1.0, "sl-exp": 0.0, "sl-gain": 1.0, "sl-lift": 0.0, "sl-gam": 1.0};
        Object.keys(defs).forEach(function(id){
            var el = document.getElementById(id);
            var vl = document.getElementById(id+"-v");
            if(el){
                el.value = defs[id];
                if(vl) vl.textContent = defs[id].toFixed(2);
                // Trigger same handlers as a slider drag would
                el.dispatchEvent(new Event("input", {bubbles: true}));
            }
        });
        // v0.8.13: also reset the grade tint colors back to neutral white
        if(window._slGradeTints){
            window._slGradeTints.lift  = [1,1,1];
            window._slGradeTints.gain  = [1,1,1];
            window._slGradeTints.gamma = [1,1,1];
        }
        // Refresh the tint swatch button colors back to white
        ["lift","gain","gamma"].forEach(function(k){
            var tb = document.getElementById("sl-tint-" + k);
            if(tb) tb.style.background = "#ffffff";
        });
        // Push the neutral tints into shader
        if(window._slUpdateLighting) window._slUpdateLighting();
    });

    // --- FIXED INTENSITY toggle (live update)
    var fxBox=document.getElementById("sl-fixed");
    if(fxBox)fxBox.addEventListener("change",function(){
        if(window._slUpdateLighting)window._slUpdateLighting();
        if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
        scheduleSave();
    });

    // --- MASK INVERT toggle (live update via shader uniform)
    var minvBox = document.getElementById("sl-minv");
    if(minvBox) minvBox.addEventListener("change", function(){
        if(window._slUpdateLighting) window._slUpdateLighting();
        scheduleSave();
    });

    // --- Click-to-edit numeric overrides on slider readouts
    // Click any orange number next to a slider -> input appears. Type any value
    // (even outside the slider's normal range), Enter or blur to commit.
    // Escape cancels. Supports negative values.
    panel.addEventListener("click", function(e){
        var span = e.target;
        if(!span.classList || !span.classList.contains("sl-num")) return;
        var sliderId = span.getAttribute("data-slider");
        var slider = document.getElementById(sliderId);
        if(!slider) return;
        var oldVal = parseFloat(slider.value);
        // Build inline input replacing the span
        var input = document.createElement("input");
        input.type = "text";
        input.value = oldVal.toFixed(2);
        input.style.cssText = "width:50px;background:#000;color:#e8a020;border:1px solid #e8a020;font-family:Consolas;font-size:9px;padding:1px 3px;outline:none;text-align:right;";
        // Hide span, insert input after it
        span.style.display = "none";
        span.parentNode.insertBefore(input, span.nextSibling);
        input.focus();
        input.select();
        var done = false;
        function commit(save){
            if(done) return;
            done = true;
            if(save){
                var v = parseFloat(input.value);
                if(!isNaN(v)){
                    // Allow values outside the slider's range. Slider clamps visually
                    // but the actual value isnt clamped - shader/Python will use it.
                    slider.value = v;
                    span.textContent = v.toFixed(2);
                    // Trigger same handlers as a drag
                    slider.dispatchEvent(new Event("input", {bubbles: true}));
                }
            }
            // Restore span, remove input
            input.parentNode.removeChild(input);
            span.style.display = "";
        }
        input.addEventListener("keydown", function(ke){
            if(ke.key === "Enter"){ ke.preventDefault(); commit(true); }
            else if(ke.key === "Escape"){ ke.preventDefault(); commit(false); }
            ke.stopPropagation();  // don't trigger W/E/R hotkeys
        });
        input.addEventListener("blur", function(){ commit(true); });
    });

    // --- Transform mode buttons (W/E/R) + hotkeys
    function setTransformMode(mode){
        if(mode !== "t" && mode !== "r" && mode !== "s") return;
        _tmode = mode;
        document.querySelectorAll(".sl-tmbtn").forEach(function(b){
            var active = b.getAttribute("data-tmode") === mode;
            b.classList.toggle("sl-tmbtn-active", active);
            b.style.background = active ? "#2a1a04" : "transparent";
            b.style.color = active ? "#e8a020" : "#666";
        });
        if(window._slSyncGizmo) window._slSyncGizmo();
        scheduleSave();
    }
    document.querySelectorAll(".sl-tmbtn").forEach(function(btn){
        btn.addEventListener("click",function(){
            setTransformMode(btn.getAttribute("data-tmode"));
        });
    });
    // Apply the loaded _tmode (if any) to the visible buttons now that they\'re wired
    setTransformMode(_tmode);
    // Hotkey handler. Only fire when focus is in the panel and not on an input/textarea.
    // ====================================================================
    // === Undo / Redo system (v0.8.10) ===
    // Snapshots the full saved-settings state on commits. Ctrl-Z pops the
    // stack and restores. Ctrl-Shift-Z (or Ctrl-Y) redoes. Cap at 50 entries.
    // ====================================================================
    var UNDO_MAX = 50;
    // Stash on window so undo state survives across openPanel re-entries
    if(!window._slUndoStack) window._slUndoStack = [];
    if(typeof window._slUndoPtr !== "number") window._slUndoPtr = -1;
    // Suppression flag so restoring state doesn't trigger a snapshot of itself
    if(typeof window._slUndoSuppressed !== "boolean") window._slUndoSuppressed = false;

    // Build a state snapshot using the same shape as localStorage save.
    // We deep-clone via JSON so future mutations don't leak into the stack.
    function buildSnapshot(){
        if(window._slUndoSuppressed) return null;
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            if(!raw){
                // Force a save first so we have something to snapshot
                if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
                saveCurrentSettings();
                raw = localStorage.getItem(STORAGE_KEY);
            }
            return raw;  // string is already an immutable clone
        } catch(e){ return null; }
    }

    // Push a new state. Drops anything past _slUndoPtr (no longer reachable via redo).
    // Skips no-op snapshots (same as last entry).
    function snapshotState(){
        if(window._slUndoSuppressed) return;
        var snap = buildSnapshot();
        if(!snap) return;
        // If we're not at the tail, slice off the redo tail before pushing
        if(window._slUndoPtr < window._slUndoStack.length - 1){
            window._slUndoStack = window._slUndoStack.slice(0, window._slUndoPtr + 1);
        }
        // Skip duplicate (no real change since last snapshot)
        var lastEntry = window._slUndoStack[window._slUndoStack.length - 1];
        if(lastEntry === snap) return;
        window._slUndoStack.push(snap);
        // Cap the stack
        if(window._slUndoStack.length > UNDO_MAX){
            window._slUndoStack = window._slUndoStack.slice(window._slUndoStack.length - UNDO_MAX);
        }
        window._slUndoPtr = window._slUndoStack.length - 1;
    }
    window._slSnapshotState = snapshotState;

    // Restore a snapshot by writing it to localStorage and re-applying.
    // Suppression flag prevents any save->snapshot loop during restoration.
    function restoreSnapshot(snapStr){
        if(!snapStr) return;
        window._slUndoSuppressed = true;
        try {
            localStorage.setItem(STORAGE_KEY, snapStr);
            var s = JSON.parse(snapStr);
            applySavedSettings(s);
            // Re-hydrate multi-light state from the snapshot
            if(s._lights && s._lights.length > 0){
                _lights = s._lights;
                _selectedLight = Math.max(0, Math.min(_lights.length - 1, s._selectedLight || 0));
                if(window._slPopulateSlidersFromLight) window._slPopulateSlidersFromLight();
                if(window._slRenderOutliner) window._slRenderOutliner();
            }
            // Sync visuals
            if(window._slSyncGizmo) window._slSyncGizmo();
            if(window._slUpdateLighting) window._slUpdateLighting();
            if(window._slUpdateConditionalSections) window._slUpdateConditionalSections();
            if(window._slRequestRender) window._slRequestRender();
        } catch(e){
            console.warn("[STAGELIGHT] undo restore failed:", e);
        }
        window._slUndoSuppressed = false;
    }

    function doUndo(){
        if(window._slUndoPtr <= 0){
            console.log("[STAGELIGHT] nothing to undo");
            return;
        }
        window._slUndoPtr -= 1;
        restoreSnapshot(window._slUndoStack[window._slUndoPtr]);
        console.log("[STAGELIGHT] undo ->", window._slUndoPtr + "/" + (window._slUndoStack.length - 1));
    }
    function doRedo(){
        if(window._slUndoPtr >= window._slUndoStack.length - 1){
            console.log("[STAGELIGHT] nothing to redo");
            return;
        }
        window._slUndoPtr += 1;
        restoreSnapshot(window._slUndoStack[window._slUndoPtr]);
        console.log("[STAGELIGHT] redo ->", window._slUndoPtr + "/" + (window._slUndoStack.length - 1));
    }
    window._slDoUndo = doUndo;
    window._slDoRedo = doRedo;

    // ====================================================================
    // === HSV Color Wheel (v0.8.13)
    // A custom color picker that opens anchored to the click point. Replaces
    // the native <input type="color"> for the main light color, AND drives
    // the LIFT/GAIN/GAMMA grade tints in the COMPOSITE panel.
    // ====================================================================
    (function setupColorWheel(){
        // Initialize the grade tint state on first openPanel. Survives across
        // panel reopens by living on window.
        if(!window._slGradeTints) window._slGradeTints = { lift:[1,1,1], gain:[1,1,1], gamma:[1,1,1] };

        // Build the wheel popup once and keep it hidden until needed
        var wheel = document.getElementById("sl-color-wheel");
        if(wheel) wheel.remove();
        wheel = document.createElement("div");
        wheel.id = "sl-color-wheel";
        wheel.style.cssText = "position:fixed;display:none;background:#111;border:1px solid #333;"+
            "border-radius:6px;padding:10px;z-index:99999;font-family:Consolas,monospace;"+
            "box-shadow:0 8px 32px rgba(0,0,0,0.85);user-select:none;width:212px;";
        wheel.innerHTML =
            '<canvas id="sl-cw-wheel" width="180" height="180" style="display:block;margin:0 auto 8px auto;cursor:crosshair;"></canvas>'+
            '<canvas id="sl-cw-bri" width="180" height="12" style="display:block;margin:0 auto 8px auto;cursor:ew-resize;border-radius:2px;"></canvas>'+
            '<div style="display:flex;gap:4px;align-items:center;margin-bottom:6px;">'+
                '<span style="color:#666;font-size:8px;width:18px;">HEX</span>'+
                '<input id="sl-cw-hex" type="text" value="#ffffff" maxlength="7" '+
                'style="flex:1;background:#000;color:#e8a020;border:1px solid #333;font-family:Consolas;font-size:9px;padding:3px 4px;outline:none;">'+
                '<button id="sl-cw-reset" style="background:#1a1a1a;color:#888;border:1px solid #333;'+
                'font-family:Consolas;font-size:8px;padding:3px 6px;cursor:pointer;border-radius:2px;" title="Reset to white">RST</button>'+
            '</div>'+
            '<div style="display:flex;gap:4px;align-items:center;">'+
                '<span style="color:#666;font-size:8px;width:12px;">R</span>'+
                '<input id="sl-cw-r" type="number" min="0" max="255" value="255" style="width:40px;background:#000;color:#e8a020;border:1px solid #333;font-family:Consolas;font-size:9px;padding:2px 3px;outline:none;">'+
                '<span style="color:#666;font-size:8px;width:12px;">G</span>'+
                '<input id="sl-cw-g" type="number" min="0" max="255" value="255" style="width:40px;background:#000;color:#e8a020;border:1px solid #333;font-family:Consolas;font-size:9px;padding:2px 3px;outline:none;">'+
                '<span style="color:#666;font-size:8px;width:12px;">B</span>'+
                '<input id="sl-cw-b" type="number" min="0" max="255" value="255" style="width:40px;background:#000;color:#e8a020;border:1px solid #333;font-family:Consolas;font-size:9px;padding:2px 3px;outline:none;">'+
            '</div>';
        document.body.appendChild(wheel);

        // Current state of the wheel: which target it's editing (light color or grade tint),
        // current HSV values, current callback to fire on change.
        var cwState = {
            mode: null,        // "light" or "lift" or "gain" or "gamma"
            tintKey: null,
            h: 0, s: 0, v: 1,
            r: 1, g: 1, b: 1,
            anchorX: 0, anchorY: 0
        };

        // --- HSV / RGB conversion helpers (standard formulas) ---
        function hsv2rgb(h, s, v){
            var c = v * s;
            var x = c * (1 - Math.abs(((h * 6) % 2) - 1));
            var m = v - c;
            var r=0, g=0, b=0;
            if(h < 1/6){ r=c; g=x; b=0; }
            else if(h < 2/6){ r=x; g=c; b=0; }
            else if(h < 3/6){ r=0; g=c; b=x; }
            else if(h < 4/6){ r=0; g=x; b=c; }
            else if(h < 5/6){ r=x; g=0; b=c; }
            else { r=c; g=0; b=x; }
            return [r+m, g+m, b+m];
        }
        function rgb2hsv(r, g, b){
            var max = Math.max(r, g, b), min = Math.min(r, g, b);
            var h=0, s=0, v=max;
            var d = max - min;
            s = max === 0 ? 0 : d / max;
            if(d === 0){ h = 0; }
            else if(max === r){ h = ((g - b) / d) / 6; if(h < 0) h += 1; }
            else if(max === g){ h = ((b - r) / d + 2) / 6; }
            else { h = ((r - g) / d + 4) / 6; }
            return [h, s, v];
        }
        function rgbToHex(r, g, b){
            function c(n){ var h=Math.round(Math.max(0,Math.min(1,n))*255).toString(16); return h.length<2?"0"+h:h; }
            return "#" + c(r) + c(g) + c(b);
        }
        function hexToRgb(hex){
            hex = hex.replace("#","").trim();
            if(hex.length === 3) hex = hex.split("").map(function(c){return c+c;}).join("");
            if(hex.length !== 6) return null;
            var n = parseInt(hex, 16);
            if(isNaN(n)) return null;
            return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
        }

        // --- Draw the HSV wheel (hue around circle, sat along radius) ---
        // Cached pixel buffer so we don't recompute every interaction
        var wheelCanvas = wheel.querySelector("#sl-cw-wheel");
        var wheelCtx = wheelCanvas.getContext("2d");
        var briCanvas = wheel.querySelector("#sl-cw-bri");
        var briCtx = briCanvas.getContext("2d");
        var WHEEL_SIZE = 180, WHEEL_R = 88;
        var wheelImageCache = null;

        function drawWheel(){
            if(wheelImageCache){
                wheelCtx.putImageData(wheelImageCache, 0, 0);
            } else {
                var img = wheelCtx.createImageData(WHEEL_SIZE, WHEEL_SIZE);
                var data = img.data;
                var cx = WHEEL_SIZE/2, cy = WHEEL_SIZE/2;
                for(var y = 0; y < WHEEL_SIZE; y++){
                    for(var x = 0; x < WHEEL_SIZE; x++){
                        var dx = x - cx, dy = y - cy;
                        var dist = Math.sqrt(dx*dx + dy*dy);
                        var idx = (y * WHEEL_SIZE + x) * 4;
                        if(dist <= WHEEL_R){
                            var angle = Math.atan2(dy, dx);
                            var hue = (angle / (2 * Math.PI)) + 0.5;
                            if(hue >= 1) hue -= 1;
                            var sat = Math.min(1, dist / WHEEL_R);
                            var rgb = hsv2rgb(hue, sat, 1.0);
                            data[idx]   = Math.round(rgb[0] * 255);
                            data[idx+1] = Math.round(rgb[1] * 255);
                            data[idx+2] = Math.round(rgb[2] * 255);
                            data[idx+3] = 255;
                        } else {
                            data[idx]=0; data[idx+1]=0; data[idx+2]=0; data[idx+3]=0;
                        }
                    }
                }
                wheelImageCache = img;
                wheelCtx.putImageData(img, 0, 0);
            }
            // Crosshair marker for current h,s position
            var markAngle = (cwState.h - 0.5) * 2 * Math.PI;
            var markDist = cwState.s * WHEEL_R;
            var mx = WHEEL_SIZE/2 + Math.cos(markAngle) * markDist;
            var my = WHEEL_SIZE/2 + Math.sin(markAngle) * markDist;
            wheelCtx.strokeStyle = "#000";
            wheelCtx.lineWidth = 2;
            wheelCtx.beginPath(); wheelCtx.arc(mx, my, 5, 0, 2 * Math.PI); wheelCtx.stroke();
            wheelCtx.strokeStyle = "#fff";
            wheelCtx.lineWidth = 1;
            wheelCtx.beginPath(); wheelCtx.arc(mx, my, 5, 0, 2 * Math.PI); wheelCtx.stroke();
        }
        function drawBri(){
            // Brightness bar: gradient from black to full-hue at current h,s
            var grad = briCtx.createLinearGradient(0, 0, 180, 0);
            var rgb = hsv2rgb(cwState.h, cwState.s, 1.0);
            grad.addColorStop(0, "#000");
            grad.addColorStop(1, "rgb("+Math.round(rgb[0]*255)+","+Math.round(rgb[1]*255)+","+Math.round(rgb[2]*255)+")");
            briCtx.fillStyle = grad;
            briCtx.fillRect(0, 0, 180, 12);
            // Marker for current v
            var mx = cwState.v * 180;
            briCtx.strokeStyle = "#000"; briCtx.lineWidth = 2;
            briCtx.beginPath(); briCtx.moveTo(mx, 0); briCtx.lineTo(mx, 12); briCtx.stroke();
            briCtx.strokeStyle = "#fff"; briCtx.lineWidth = 1;
            briCtx.beginPath(); briCtx.moveTo(mx, 0); briCtx.lineTo(mx, 12); briCtx.stroke();
        }

        function syncInputs(){
            var rgb = hsv2rgb(cwState.h, cwState.s, cwState.v);
            cwState.r = rgb[0]; cwState.g = rgb[1]; cwState.b = rgb[2];
            wheel.querySelector("#sl-cw-hex").value = rgbToHex(cwState.r, cwState.g, cwState.b);
            wheel.querySelector("#sl-cw-r").value = Math.round(cwState.r * 255);
            wheel.querySelector("#sl-cw-g").value = Math.round(cwState.g * 255);
            wheel.querySelector("#sl-cw-b").value = Math.round(cwState.b * 255);
        }

        function applyToTarget(){
            var r = cwState.r, g = cwState.g, b = cwState.b;
            if(cwState.mode === "light"){
                // Update the R/G/B sliders that drive the active light color, plus the
                // (now hidden) native color input for legacy compat.
                ["sl-lr","sl-lg","sl-lb"].forEach(function(id, i){
                    var el = document.getElementById(id);
                    var vl = document.getElementById(id+"-v");
                    var val = [r, g, b][i];
                    if(el){ el.value = val; }
                    if(vl) vl.textContent = parseFloat(val).toFixed(2);
                });
                var cp = document.getElementById("sl-cpick");
                if(cp) cp.value = rgbToHex(r, g, b);
                // Update the visible swatch on the picker button too
                var sw = document.getElementById("sl-cwell");
                if(sw) sw.style.background = rgbToHex(r, g, b);
                if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
                if(window._slUpdateLighting) window._slUpdateLighting();
            } else if(cwState.tintKey){
                window._slGradeTints[cwState.tintKey] = [r, g, b];
                // Update the small tint square button
                var tb = document.getElementById("sl-tint-" + cwState.tintKey);
                if(tb) tb.style.background = rgbToHex(r, g, b);
                if(window._slUpdateLighting) window._slUpdateLighting();
            }
            if(window._slScheduleSave) window._slScheduleSave();
        }

        function setFromRgb(r, g, b){
            var hsv = rgb2hsv(r, g, b);
            cwState.h = hsv[0]; cwState.s = hsv[1]; cwState.v = hsv[2];
            cwState.r = r; cwState.g = g; cwState.b = b;
            drawWheel(); drawBri(); syncInputs();
            applyToTarget();
        }

        // --- Wheel mouse interaction ---
        var wheelDragging = false;
        function pickFromWheel(e){
            var rect = wheelCanvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            var dx = x - WHEEL_SIZE/2, dy = y - WHEEL_SIZE/2;
            var dist = Math.sqrt(dx*dx + dy*dy);
            var angle = Math.atan2(dy, dx);
            cwState.h = (angle / (2 * Math.PI)) + 0.5;
            if(cwState.h >= 1) cwState.h -= 1;
            cwState.s = Math.min(1, dist / WHEEL_R);
            drawWheel(); drawBri();
            var rgb = hsv2rgb(cwState.h, cwState.s, cwState.v);
            cwState.r = rgb[0]; cwState.g = rgb[1]; cwState.b = rgb[2];
            syncInputs();
            applyToTarget();
        }
        wheelCanvas.addEventListener("mousedown", function(e){
            wheelDragging = true; pickFromWheel(e); e.preventDefault();
        });
        document.addEventListener("mousemove", function(e){
            if(wheelDragging) pickFromWheel(e);
        });
        document.addEventListener("mouseup", function(){ wheelDragging = false; });

        // --- Brightness bar mouse interaction ---
        var briDragging = false;
        function pickFromBri(e){
            var rect = briCanvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            cwState.v = Math.max(0, Math.min(1, x / 180));
            drawWheel(); drawBri();
            var rgb = hsv2rgb(cwState.h, cwState.s, cwState.v);
            cwState.r = rgb[0]; cwState.g = rgb[1]; cwState.b = rgb[2];
            syncInputs();
            applyToTarget();
        }
        briCanvas.addEventListener("mousedown", function(e){
            briDragging = true; pickFromBri(e); e.preventDefault();
        });
        document.addEventListener("mousemove", function(e){
            if(briDragging) pickFromBri(e);
        });
        document.addEventListener("mouseup", function(){ briDragging = false; });

        // --- Hex input ---
        wheel.querySelector("#sl-cw-hex").addEventListener("input", function(e){
            var rgb = hexToRgb(e.target.value);
            if(rgb) setFromRgb(rgb[0], rgb[1], rgb[2]);
        });
        // --- RGB inputs ---
        ["r","g","b"].forEach(function(ch){
            wheel.querySelector("#sl-cw-"+ch).addEventListener("input", function(){
                var r = parseFloat(wheel.querySelector("#sl-cw-r").value) / 255;
                var g = parseFloat(wheel.querySelector("#sl-cw-g").value) / 255;
                var b = parseFloat(wheel.querySelector("#sl-cw-b").value) / 255;
                if(!isNaN(r) && !isNaN(g) && !isNaN(b)) setFromRgb(r, g, b);
            });
        });
        // --- Reset to white ---
        wheel.querySelector("#sl-cw-reset").addEventListener("click", function(){
            setFromRgb(1, 1, 1);
        });

        // --- Open / close ---
        function openWheel(mode, tintKey, anchorEl, initialRgb){
            cwState.mode = mode;
            cwState.tintKey = tintKey;
            if(initialRgb){
                var hsv = rgb2hsv(initialRgb[0], initialRgb[1], initialRgb[2]);
                cwState.h = hsv[0]; cwState.s = hsv[1]; cwState.v = hsv[2];
                cwState.r = initialRgb[0]; cwState.g = initialRgb[1]; cwState.b = initialRgb[2];
            }
            // Position the wheel near the anchor element. Clamp to viewport.
            var rect = anchorEl.getBoundingClientRect();
            var W = 232, H = 280;
            var left = rect.right + 6;
            var top = rect.top - 8;
            if(left + W > window.innerWidth) left = rect.left - W - 6;
            if(left < 4) left = 4;
            if(top + H > window.innerHeight) top = window.innerHeight - H - 4;
            if(top < 4) top = 4;
            wheel.style.left = left + "px";
            wheel.style.top  = top + "px";
            wheel.style.display = "block";
            drawWheel(); drawBri(); syncInputs();
        }
        function closeWheel(){ wheel.style.display = "none"; cwState.mode = null; cwState.tintKey = null; }

        // Click outside wheel = close (but not if click is on a trigger that opens the wheel)
        document.addEventListener("mousedown", function(e){
            if(wheel.style.display === "none") return;
            if(wheel.contains(e.target)) return;
            // Allow clicks on triggers to re-open with new mode without flashing close+open
            if(e.target.classList && (e.target.classList.contains("sl-tint-sw") || e.target.id === "sl-cwell")) return;
            closeWheel();
        });
        // Escape closes
        document.addEventListener("keydown", function(e){
            if(e.key === "Escape" && wheel.style.display !== "none"){
                closeWheel(); e.stopPropagation();
            }
        });

        window._slOpenColorWheel = openWheel;
        window._slCloseColorWheel = closeWheel;

        // --- Wire the LIGHT color picker button (custom replacement for native input) ---
        // Initial sync of the swatch button color from the active light's RGB
        function syncCwellSwatch(){
            var sw = document.getElementById("sl-cwell");
            if(!sw) return;
            var r = parseFloat((document.getElementById("sl-lr")||{value:1}).value);
            var g = parseFloat((document.getElementById("sl-lg")||{value:1}).value);
            var b = parseFloat((document.getElementById("sl-lb")||{value:1}).value);
            sw.style.background = rgbToHex(r, g, b);
        }
        var cwell = document.getElementById("sl-cwell");
        if(cwell){
            syncCwellSwatch();
            cwell.addEventListener("click", function(e){
                var r = parseFloat((document.getElementById("sl-lr")||{value:1}).value);
                var g = parseFloat((document.getElementById("sl-lg")||{value:1}).value);
                var b = parseFloat((document.getElementById("sl-lb")||{value:1}).value);
                openWheel("light", null, cwell, [r, g, b]);
                e.stopPropagation();
            });
        }
        window._slSyncCwellSwatch = syncCwellSwatch;

        // --- Wire the three grade tint squares ---
        document.querySelectorAll(".sl-tint-sw").forEach(function(btn){
            var tk = btn.getAttribute("data-tint");
            // Initial swatch color from stored tint state
            var t = window._slGradeTints[tk] || [1,1,1];
            btn.style.background = rgbToHex(t[0], t[1], t[2]);
            btn.addEventListener("click", function(e){
                var cur = window._slGradeTints[tk] || [1,1,1];
                openWheel("tint", tk, btn, cur);
                e.stopPropagation();
            });
        });
    })();


    // Take an initial snapshot of the current state so the very first undo
    // has somewhere to go (otherwise the first edit's "before" state is lost).
    setTimeout(function(){ snapshotState(); }, 100);

    // === Snapshot hooks at commit points ===
    // On slider release (change event fires once per drag, distinct from input)
    document.querySelectorAll("#sl-panel input[type=range]").forEach(function(el){
        el.addEventListener("change", snapshotState);
    });
    // On checkbox toggles
    document.querySelectorAll("#sl-panel input[type=checkbox]").forEach(function(el){
        el.addEventListener("change", snapshotState);
    });
    // On color picker change
    var cpick = document.getElementById("sl-cpick");
    if(cpick) cpick.addEventListener("change", snapshotState);

    var _tmodeKeyHandler = function(e){
        // Don\'t hijack keys if user is typing in a field
        var t = e.target;
        if(t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
        // Don\'t hijack keys if panel isn\'t open
        if(!document.getElementById("sl-panel")) return;
        var key = e.key && e.key.toLowerCase();
        // Ctrl-Z = undo, Ctrl-Shift-Z = redo, Ctrl-Y = redo (Windows convention)
        if((e.ctrlKey || e.metaKey) && !e.altKey){
            if(key === "z" && !e.shiftKey){
                if(window._slDoUndo) window._slDoUndo();
                e.preventDefault();
                e.stopPropagation();
                if(typeof e.stopImmediatePropagation === "function") e.stopImmediatePropagation();
                return;
            }
            if((key === "z" && e.shiftKey) || key === "y"){
                if(window._slDoRedo) window._slDoRedo();
                e.preventDefault();
                e.stopPropagation();
                if(typeof e.stopImmediatePropagation === "function") e.stopImmediatePropagation();
                return;
            }
            // Other Ctrl combos pass through normally
            return;
        }
        // Don\'t fire single-key W/E/R/F on modifier combos
        if(e.ctrlKey || e.metaKey || e.altKey) return;
        if(key === "w" || key === "e" || key === "r" || key === "f"){
            console.log("[STAGELIGHT] hotkey:", key);
            if(key === "w") setTransformMode("t");
            else if(key === "e") setTransformMode("r");
            else if(key === "r") setTransformMode("s");
            else if(key === "f"){
                // Frame: reset camera to default framing on subject
                if(window._slFrameSubject) window._slFrameSubject();
            }
            // Aggressively stop propagation: prevent ComfyUI canvas/window handlers
            // from also firing on the same keystroke.
            e.preventDefault();
            e.stopPropagation();
            if(typeof e.stopImmediatePropagation === "function") e.stopImmediatePropagation();
        }
    };
    // Cleanup any prior handler from a previous panel open (window OR document)
    if(window._slTmodeHandler){
        try { window.removeEventListener("keydown", window._slTmodeHandler, true); } catch(e){}
        try { document.removeEventListener("keydown", window._slTmodeHandler, true); } catch(e){}
        // Also remove from any prior panel that might still exist
        try {
            if(window._slTmodePanel){
                window._slTmodePanel.removeEventListener("keydown", window._slTmodeHandler, true);
            }
        } catch(e){}
    }
    // Bind to WINDOW at capture phase (works when nothing else intercepts)
    window.addEventListener("keydown", _tmodeKeyHandler, true);
    // ALSO bind to the panel element directly. When panel has focus, this handler
    // fires before any window-level listener that other extensions might have added.
    panel.addEventListener("keydown", _tmodeKeyHandler, true);
    // Auto-focus the panel when mouse enters it - so keyboard input goes here.
    // This means hovering the panel "captures" the keyboard for our purposes.
    panel.addEventListener("mouseenter", function(){
        try { panel.focus({preventScroll: true}); } catch(e){ try { panel.focus(); } catch(e2){} }
    });
    window._slTmodeHandler = _tmodeKeyHandler;
    window._slTmodePanel = panel;
    console.log("[STAGELIGHT] hotkey handler bound (window+panel, capture). Hover panel + press W/E/R/F.");

    // --- view toggle buttons (instant - no render needed, just shader uniform)
    document.querySelectorAll(".sl-vbtn").forEach(function(btn){
        btn.addEventListener("click",function(){
            document.querySelectorAll(".sl-vbtn").forEach(function(b){
                b.classList.remove("sl-vbtn-active");
                b.style.background="transparent";b.style.color="#666";
            });
            btn.classList.add("sl-vbtn-active");
            btn.style.background="#2a1a04";btn.style.color="#e8a020";
            // No re-render - just update view mode uniform (instant)
            if(window._slUpdateLighting)window._slUpdateLighting();
            scheduleSave();
        });
    });

    var vp=document.getElementById("sl-vp");
    var foot=document.getElementById("sl-foot");

    document.getElementById("sl-x").onclick=function(){panel.remove();cleanupThree();};

    var mini=false;
    document.getElementById("sl-min").onclick=function(){
        mini=!mini;
        vp.style.display=mini?"none":"";
        foot.style.display=mini?"none":"";
        panel.style.height=mini?"40px":"";
        document.getElementById("sl-min").innerHTML=mini?"&#9650;":"&#8212;";
    };

    var drag=false,resz=false,orb=false,ox=0,oy=0,rx=0,ry=0,rw=0,rh=0;

    document.getElementById("sl-hdr").addEventListener("mousedown",function(e){
        if(e.target.tagName==="BUTTON") return;
        drag=true;ox=e.clientX-panel.offsetLeft;oy=e.clientY-panel.offsetTop;e.preventDefault();
    });
    document.getElementById("sl-rsz").addEventListener("mousedown",function(e){
        resz=true;rx=e.clientX;ry=e.clientY;rw=panel.offsetWidth;rh=panel.offsetHeight;
        e.preventDefault();e.stopPropagation();
    });
    var draggingGizmo=false, gizmoPlane=null, gizmoAxis="free", gizmoStartPos=null, gizmoStartHit=null, gizmoStartTarget=null, gizmoStartScale=0, gizmoStartOffset=null;
    function screenToVP(e){
        var rect=vp.getBoundingClientRect();
        return {x:((e.clientX-rect.left)/rect.width)*2-1, y:-((e.clientY-rect.top)/rect.height)*2+1};
    }
    function raycastGizmoHit(e){
        // Returns {axis:"x"|"y"|"z"|"free", object} or null
        if(!window.THREE||!_lightGizmo||!_camera||!_lightGizmo._handles)return null;
        var p=screenToVP(e);
        var rc=new THREE.Raycaster();
        rc.setFromCamera(new THREE.Vector2(p.x,p.y),_camera);
        var hits=rc.intersectObjects(_lightGizmo._handles,true);
        if(hits.length===0)return null;
        // Find first hit that has an axis tag
        for(var i=0;i<hits.length;i++){
            var o=hits[i].object;
            while(o && o.userData && !o.userData.axis){o=o.parent;}
            if(o && o.userData && o.userData.axis){
                return {axis:o.userData.axis, object:o};
            }
        }
        return null;
    }

    // Hover highlight on handles (light + target)
    var _lastHover=null;
    vp.addEventListener("mousemove",function(e){
        if(draggingGizmo||drag||resz||orb)return;
        var h=raycastGizmoHit(e);
        var axis=h?h.axis:null;
        if(axis===_lastHover)return;
        _lastHover=axis;
        // reset all handles to original color
        if(_lightGizmo._handleRefs){
            ["x","y","z"].forEach(function(a){
                var r=_lightGizmo._handleRefs[a];
                if(r&&r.mat)r.mat.color.setHex(r.group.userData.origColor);
            });
        }
        if(_lightGizmo._targetHandleRefs){
            ["x","y","z"].forEach(function(a){
                var r=_lightGizmo._targetHandleRefs[a];
                if(r&&r.mat)r.mat.color.setHex(r.group.userData.origColor);
            });
        }
        if(_lightGizmo._rotateRefs){
            ["x","y","z"].forEach(function(a){
                var r=_lightGizmo._rotateRefs[a];
                if(r&&r.mat)r.mat.color.setHex(r.group.userData.origColor);
            });
        }
        // Reset scale cube to neutral white
        if(_lightGizmo._scaleCubeMat){
            _lightGizmo._scaleCubeMat.color.setHex(0xffffff);
        }
        // brighten the hovered handle
        if(axis){
            var refs = null;
            var key = axis;
            if(axis === "x" || axis === "y" || axis === "z"){
                refs = _lightGizmo._handleRefs;
            } else if(axis === "tgt-x" || axis === "tgt-y" || axis === "tgt-z"){
                refs = _lightGizmo._targetHandleRefs;
                key = axis.replace("tgt-","");
            } else if(axis === "rot-x" || axis === "rot-y" || axis === "rot-z"){
                refs = _lightGizmo._rotateRefs;
                key = axis.replace("rot-","");
            } else if(axis === "scale" && _lightGizmo._scaleCubeMat){
                _lightGizmo._scaleCubeMat.color.setHex(0xe8a020);  // orange highlight for cube
            }
            if(refs && refs[key]) refs[key].mat.color.setHex(0xffffff);
        }
        // Cursor: pointer over gizmo handle OR over a marker; grab otherwise
        var overMarker = false;
        if(!axis && window._slMarkers){
            overMarker = (raycastMarkerHit(e) >= 0);
        }
        vp.style.cursor = (axis || overMarker) ? "pointer" : "grab";
    });

    var panning=false;
    // Raycast for inactive light markers (the small gray spheres). Returns the
    // light index that was clicked, or -1.
    function raycastMarkerHit(e){
        if(!window.THREE || !_camera) return -1;
        var markers = window._slMarkers;
        if(!markers || markers.length === 0) return -1;
        var pickable = markers.filter(function(m){ return m && m.visible; });
        if(pickable.length === 0) return -1;
        var p = screenToVP(e);
        var rc = new THREE.Raycaster();
        rc.setFromCamera(new THREE.Vector2(p.x, p.y), _camera);
        var hits = rc.intersectObjects(pickable, false);
        if(hits.length === 0) return -1;
        var idx = hits[0].object.userData.lightIdx;
        return (typeof idx === "number") ? idx : -1;
    }

    vp.addEventListener("mousedown",function(e){
        // Middle-mouse-button = pan the camera
        if(e.button===1){
            panning=true;
            ox=e.clientX;oy=e.clientY;
            vp.style.cursor="move";
            e.preventDefault();
            return;
        }
        // Marker click - select that light in the outliner. After selection,
        // fall through so the user's drag immediately moves the now-active
        // light's gizmo (Maya-style click-and-drag-to-select-and-move).
        var markerIdx = raycastMarkerHit(e);
        if(markerIdx >= 0 && markerIdx !== _selectedLight){
            // Commit current edits to the OLD selected light
            if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
            _selectedLight = markerIdx;
            if(window._slPopulateSlidersFromLight) window._slPopulateSlidersFromLight();
            if(window._slRenderOutliner) window._slRenderOutliner();
            if(window._slSyncGizmo) window._slSyncGizmo();
            if(window._slUpdateLighting) window._slUpdateLighting();
            // Don\'t return - let the user continue with a drag if they're holding
            // mouse down. The newly-selected light's gizmo center is at the same
            // 3D position as the marker they just clicked, so the gizmo center
            // hit-test will succeed on the next raycast attempt below.
        }
        // Try gizmo - always draggable, camera lock doesn\'t block it
        var hit=raycastGizmoHit(e);
        if(hit){
            draggingGizmo=true;
            gizmoAxis=hit.axis;
            var p=screenToVP(e);
            var rc=new THREE.Raycaster();
            rc.setFromCamera(new THREE.Vector2(p.x,p.y),_camera);
            // Use target's position when target axis is dragged, else light's position
            var isTargetHit = (hit.axis === "tgt-free" || hit.axis === "tgt-x" || hit.axis === "tgt-y" || hit.axis === "tgt-z");
            var isRotateHit = (hit.axis === "rot-x" || hit.axis === "rot-y" || hit.axis === "rot-z");
            var isScaleHit  = (hit.axis === "scale");
            var gp = isTargetHit ? _lightGizmo._targetGroup.position.clone() : _lightGizmo._group.position.clone();
            gizmoStartPos=gp.clone();
            // Always capture target start position - rotate mode needs it
            if(_lightGizmo._targetGroup) gizmoStartTarget = _lightGizmo._targetGroup.position.clone();
            // Capture the starting offset for rotate-mode parent-child math
            if(_targetOffset){
                gizmoStartOffset = {x: _targetOffset.x, y: _targetOffset.y, z: _targetOffset.z};
            } else {
                gizmoStartOffset = null;
            }
            // Scale mode also needs to capture the starting scale value
            if(isScaleHit){
                var sl = document.getElementById("sl-lsz");
                gizmoStartScale = sl ? parseFloat(sl.value) : 0;
            }

            if(isRotateHit){
                // Rotation plane: perpendicular to the rotation axis, through the light position
                var rotAxis = new THREE.Vector3(
                    hit.axis === "rot-x" ? 1 : 0,
                    hit.axis === "rot-y" ? 1 : 0,
                    hit.axis === "rot-z" ? 1 : 0
                );
                gizmoPlane=new THREE.Plane().setFromNormalAndCoplanarPoint(rotAxis, gp);
                var tmpHit=new THREE.Vector3();
                rc.ray.intersectPlane(gizmoPlane,tmpHit);
                gizmoStartHit=tmpHit.clone();
            } else if(isScaleHit){
                // Scale plane: perpendicular to camera, through the light position.
                // Drag distance from light controls the SCALE value.
                var camDir=new THREE.Vector3();_camera.getWorldDirection(camDir);
                gizmoPlane=new THREE.Plane().setFromNormalAndCoplanarPoint(camDir.negate(),gp);
                var tmpHit=new THREE.Vector3();
                rc.ray.intersectPlane(gizmoPlane,tmpHit);
                gizmoStartHit=tmpHit.clone();
            } else {
                // Normalize axis for plane setup (translate mode)
                var planeAxis = isTargetHit ? hit.axis.replace("tgt-","") : hit.axis;

                if(planeAxis==="free"){
                    // plane perpendicular to camera
                    var camDir=new THREE.Vector3();_camera.getWorldDirection(camDir);
                    gizmoPlane=new THREE.Plane().setFromNormalAndCoplanarPoint(camDir.negate(),gp);
                    var tmpHit=new THREE.Vector3();
                    rc.ray.intersectPlane(gizmoPlane,tmpHit);
                    gizmoStartHit=tmpHit.clone();
                } else {
                    // For single axis: use plane that contains the axis and faces the camera as much as possible
                    var axisDir=new THREE.Vector3(
                        planeAxis==="x"?1:0,
                        planeAxis==="y"?1:0,
                        planeAxis==="z"?1:0
                    );
                    var camDir=new THREE.Vector3();_camera.getWorldDirection(camDir);
                    var planeNormal=new THREE.Vector3().crossVectors(axisDir,
                        new THREE.Vector3().crossVectors(camDir,axisDir)).normalize();
                    if(planeNormal.lengthSq()<0.001){
                        planeNormal.copy(camDir).negate();
                    }
                    gizmoPlane=new THREE.Plane().setFromNormalAndCoplanarPoint(planeNormal,gp);
                    var tmpHit=new THREE.Vector3();
                    rc.ray.intersectPlane(gizmoPlane,tmpHit);
                    gizmoStartHit=tmpHit.clone();
                }
            }

            vp.style.cursor="grabbing";
            e.preventDefault();e.stopPropagation();
            return;
        }
        // Otherwise orbit the scene - unless camera is locked
        var camLock=document.getElementById("sl-camlock");
        if(camLock&&camLock.checked)return;
        orb=true;ox=e.clientX;oy=e.clientY;vp.style.cursor="grabbing";e.preventDefault();
    });
    var _docMouseMove = function(e){
        if(drag){panel.style.left=(e.clientX-ox)+"px";panel.style.top=(e.clientY-oy)+"px";panel.style.right="auto";}
        if(resz){panel.style.width=Math.max(780,rw+(e.clientX-rx))+"px";panel.style.height=Math.max(440,rh+(e.clientY-ry))+"px";}
        if(draggingGizmo && window.THREE && _camera && _lightGizmo && gizmoPlane && gizmoStartHit && gizmoStartPos){
            var p=screenToVP(e);
            var rc=new THREE.Raycaster();
            rc.setFromCamera(new THREE.Vector2(p.x,p.y),_camera);
            var curHit=new THREE.Vector3();
            if(!rc.ray.intersectPlane(gizmoPlane,curHit)) return;

            // === ROTATE MODE: dragging a rotate ring ===
            // gizmoAxis is "rot-x" / "rot-y" / "rot-z". Compute the angle change between
            // start and current hit points (both projected onto the rotation plane), then
            // rotate the light+target as a unit around the light\'s position.
            if(gizmoAxis === "rot-x" || gizmoAxis === "rot-y" || gizmoAxis === "rot-z"){
                var pivot = _lightGizmo._group.position.clone();
                var axisVec = new THREE.Vector3(
                    gizmoAxis === "rot-x" ? 1 : 0,
                    gizmoAxis === "rot-y" ? 1 : 0,
                    gizmoAxis === "rot-z" ? 1 : 0
                );
                // Project start & current hits onto the plane perpendicular to axisVec at the pivot
                var startVec = gizmoStartHit.clone().sub(pivot);
                var curVec = curHit.clone().sub(pivot);
                // Remove the axis component so we\'re working in the rotation plane
                startVec.sub(axisVec.clone().multiplyScalar(startVec.dot(axisVec)));
                curVec.sub(axisVec.clone().multiplyScalar(curVec.dot(axisVec)));
                if(startVec.lengthSq() < 0.0001 || curVec.lengthSq() < 0.0001) return;
                startVec.normalize();
                curVec.normalize();
                // Signed angle from startVec to curVec around axisVec
                var cross = new THREE.Vector3().crossVectors(startVec, curVec);
                var sign = cross.dot(axisVec) > 0 ? 1 : -1;
                var dot = Math.max(-1, Math.min(1, startVec.dot(curVec)));
                var angle = Math.acos(dot) * sign;

                // Compute rotation
                // When target is OFF (parent-child mode), rotate the OFFSET around the
                // light - the light stays put and only its aim direction changes.
                // When target is ON, the original behavior - rotate target around light.
                if(_targetOffset && gizmoStartOffset){
                    // Rotate the START offset (captured at mousedown) by the cumulative
                    // angle. This way each mousemove computes the right total rotation
                    // rather than compounding incremental rotations.
                    var quat = new THREE.Quaternion().setFromAxisAngle(axisVec, angle);
                    var rotatedOffset = new THREE.Vector3(gizmoStartOffset.x, gizmoStartOffset.y, gizmoStartOffset.z);
                    rotatedOffset.applyQuaternion(quat);
                    // Update the live _targetOffset
                    _targetOffset.x = rotatedOffset.x;
                    _targetOffset.y = rotatedOffset.y;
                    _targetOffset.z = rotatedOffset.z;
                    // Update target slider values to lightPos + rotatedOffset
                    var newTgt = pivot.clone().add(rotatedOffset);
                    newTgt.x = Math.max(-3, Math.min(3, newTgt.x));
                    newTgt.y = Math.max(-3, Math.min(3, newTgt.y));
                    newTgt.z = Math.max(-3, Math.min(3, newTgt.z));
                    function setSliderR2(id,v){
                        var el=document.getElementById(id);
                        var vl=document.getElementById(id+"-v");
                        if(el){el.value=v;if(vl)vl.textContent=parseFloat(v).toFixed(2);}
                    }
                    if(_lightGizmo._targetGroup) _lightGizmo._targetGroup.position.copy(newTgt);
                    setSliderR2("sl-tx", newTgt.x);
                    setSliderR2("sl-ty", newTgt.y);
                    setSliderR2("sl-tz", newTgt.z);
                } else if(gizmoStartTarget){
                    // Original behavior: rotate target around light position
                    var tgt = gizmoStartTarget.clone();
                    var rel = tgt.sub(pivot);  // target relative to light
                    var quat = new THREE.Quaternion().setFromAxisAngle(axisVec, angle);
                    rel.applyQuaternion(quat);
                    var newTgt = pivot.clone().add(rel);
                    // Clamp to slider ranges
                    newTgt.x = Math.max(-3, Math.min(3, newTgt.x));
                    newTgt.y = Math.max(-3, Math.min(3, newTgt.y));
                    newTgt.z = Math.max(-3, Math.min(3, newTgt.z));
                    function setSliderR(id,v){
                        var el=document.getElementById(id);
                        var vl=document.getElementById(id+"-v");
                        if(el){el.value=v;if(vl)vl.textContent=parseFloat(v).toFixed(2);}
                    }
                    if(_lightGizmo._targetGroup) _lightGizmo._targetGroup.position.copy(newTgt);
                    setSliderR("sl-tx", newTgt.x);
                    setSliderR("sl-ty", newTgt.y);
                    setSliderR("sl-tz", newTgt.z);
                }
                if(window._slUpdateLighting) window._slUpdateLighting();
                requestRender();
                return;
            }
            // === SCALE MODE: dragging the cube ===
            // Drag distance from light position determines scale value [0..1].
            // We measure the projected distance on the camera-facing plane (the same
            // plane that gizmoPlane is set up to be, in the mousedown handler).
            if(gizmoAxis === "scale"){
                var lightPos = _lightGizmo._group.position.clone();
                // Measure how far the current hit is from the light, normalized to a
                // typical drag range. At drag distance 1.0 (in world units) -> SCALE=1.
                var distFromLight = curHit.clone().sub(lightPos).length();
                var startDistFromLight = gizmoStartHit.clone().sub(lightPos).length();
                // The cube starts at distance `0.5 + scale_initial * 0.5`. Use that as baseline.
                // Compute new scale value from the new cube distance.
                // We invert the formula: dist = 0.5 + sc*0.5, so sc = (dist - 0.5) * 2
                // But use the start-of-drag scale as an offset so dragging feels relative.
                var dragDelta = distFromLight - startDistFromLight;
                // Sensitivity: 1 world unit of drag = ~1.0 of scale change
                var newScale = (gizmoStartScale || 0) + dragDelta;
                newScale = Math.max(0, Math.min(1, newScale));
                var sl = document.getElementById("sl-lsz");
                var vl = document.getElementById("sl-lsz-v");
                if(sl){
                    sl.value = newScale;
                    if(vl) vl.textContent = newScale.toFixed(2);
                }
                if(window._slSyncGizmo) window._slSyncGizmo();
                if(window._slUpdateLighting) window._slUpdateLighting();
                requestRender();
                return;
            }
            // === TRANSLATE MODE (existing behavior) ===
            var delta=curHit.clone().sub(gizmoStartHit);
            var newPos=gizmoStartPos.clone();

            // Determine if this is a target-axis or light-axis drag
            var isTarget = (gizmoAxis === "tgt-free" || gizmoAxis === "tgt-x" || gizmoAxis === "tgt-y" || gizmoAxis === "tgt-z");
            var axis = gizmoAxis;
            if(isTarget) axis = axis.replace("tgt-",""); // normalize to free/x/y/z

            if(axis==="free"){
                newPos.add(delta);
            } else if(axis==="x"){
                newPos.x += delta.x;
            } else if(axis==="y"){
                newPos.y += delta.y;
            } else if(axis==="z"){
                newPos.z += delta.z;
            }

            // Clamp to slider ranges
            newPos.x=Math.max(-3,Math.min(3,newPos.x));
            newPos.y=Math.max(-3,Math.min(3,newPos.y));
            newPos.z=Math.max(-3,Math.min(5,newPos.z));

            function setSlider(id,v){
                var el=document.getElementById(id);
                var vl=document.getElementById(id+"-v");
                if(el){el.value=v;if(vl)vl.textContent=parseFloat(v).toFixed(2);}
            }

            if(isTarget){
                // Move the target gizmo
                _lightGizmo._targetGroup.position.copy(newPos);
                setSlider("sl-tx",newPos.x);setSlider("sl-ty",newPos.y);setSlider("sl-tz",newPos.z);
            } else {
                // Move the light gizmo (default behavior)
                _lightGizmo._group.position.copy(newPos);
                setSlider("sl-lx",newPos.x);setSlider("sl-ly",newPos.y);setSlider("sl-lz",newPos.z);
                // Parent-child: if target is off, slide target with the light AND
                // resync the visual target gizmo so the cone (which reads from the
                // gizmo's actual world position, not the sliders) follows correctly.
                if(_targetOffset && window._slSetTargetFromOffset){
                    window._slSetTargetFromOffset();
                    if(window._slSyncGizmo) window._slSyncGizmo();
                }
            }
            if(window._slUpdateLighting)window._slUpdateLighting();
            // Multi-light: commit the change to the active light's data model
            if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
            requestRender();
            return;
        }
        if(panning && _camera){
            var dx=(e.clientX-ox), dy=(e.clientY-oy);
            // Convert pixel delta to world units - scale with distance for natural feel
            var scale = _orbitDist * 0.002;
            _panX -= dx * scale;
            _panY += dy * scale;  // screen Y flipped
            ox=e.clientX;oy=e.clientY;
            updateCamera();
            requestRender();
            return;
        }
        if(orb&&_camera){
            var dx=(e.clientX-ox)*0.006,dy=(e.clientY-oy)*0.006;
            _orbitTheta-=dx;
            _orbitPhi=Math.max(-1.4,Math.min(1.4,_orbitPhi+dy));
            ox=e.clientX;oy=e.clientY;
            updateCamera();
            requestRender();
        }
    };
    document.addEventListener("mousemove", _docMouseMove);
    var _docMouseUp = function(){
        drag=false;resz=false;
        if(panning){panning=false;vp.style.cursor="grab";}
        if(orb){orb=false;vp.style.cursor="grab";}
        if(draggingGizmo){
            draggingGizmo=false;
            gizmoAxis="free";gizmoStartPos=null;gizmoStartHit=null;gizmoStartTarget=null;gizmoStartScale=0;gizmoStartOffset=null;
            vp.style.cursor="grab";
            _lastHover=null;
            // Snapshot for undo (gizmo drag is a commit)
            if(window._slSnapshotState) window._slSnapshotState();
            // reset handle colors
            if(_lightGizmo&&_lightGizmo._handleRefs){
                ["x","y","z"].forEach(function(a){
                    var r=_lightGizmo._handleRefs[a];
                    if(r&&r.mat)r.mat.color.setHex(r.group.userData.origColor);
                });
            }
            if(_lightGizmo&&_lightGizmo._targetHandleRefs){
                ["x","y","z"].forEach(function(a){
                    var r=_lightGizmo._targetHandleRefs[a];
                    if(r&&r.mat)r.mat.color.setHex(r.group.userData.origColor);
                });
            }
            if(_lightGizmo&&_lightGizmo._rotateRefs){
                ["x","y","z"].forEach(function(a){
                    var r=_lightGizmo._rotateRefs[a];
                    if(r&&r.mat)r.mat.color.setHex(r.group.userData.origColor);
                });
            }
            if(_lightGizmo&&_lightGizmo._scaleCubeMat){
                _lightGizmo._scaleCubeMat.color.setHex(0xffffff);
            }
            if(typeof triggerAutoRender==="function")triggerAutoRender();
            if(window._slScheduleSave) window._slScheduleSave();
        }
    };
    document.addEventListener("mouseup", _docMouseUp);
    // Stash refs on window so cleanupThree can remove them
    window._slDocHandlers = {move: _docMouseMove, up: _docMouseUp};
    vp.addEventListener("auxclick",function(e){if(e.button===1)e.preventDefault();});
    vp.addEventListener("wheel",function(e){
        if(_camera){_orbitDist=Math.max(0.5,Math.min(12,_orbitDist+e.deltaY*0.005));updateCamera();requestRender();}
        e.preventDefault();
    },{passive:false});

    loadThree(function(THREE){
        if(!THREE){document.getElementById("sl-msg").textContent="3D UNAVAILABLE";return;}
        var canvas=document.getElementById("sl-canvas");
        var w=vp.clientWidth||480,h=vp.clientHeight||360;
        _renderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true});
        _renderer.setSize(w,h);_renderer.setClearColor(0x060606,1);
        _scene=new THREE.Scene();
        _camera=new THREE.PerspectiveCamera(45,w/h,0.1,100);

        // --- Maya-style orbit: camera always looks at origin ---
        window._slUpdateCamera=function(){
            if(!_camera)return;
            var x=_orbitDist*Math.sin(_orbitTheta)*Math.cos(_orbitPhi);
            var y=_orbitDist*Math.sin(_orbitPhi);
            var z=_orbitDist*Math.cos(_orbitTheta)*Math.cos(_orbitPhi);
            _camera.position.set(x + _panX, y + _panY, z);
            _camera.lookAt(_panX, _panY, 0);
            // Update shader cameraPosition for specular calc
            if(_mesh && _mesh._displaceMat){
                _mesh._displaceMat.uniforms.cameraPosition2.value.copy(_camera.position);
            }
            if(window._slRequestRender) window._slRequestRender();
        };
        // alias so code above can call it
        window.updateCamera=window._slUpdateCamera;
        window._slUpdateCamera();

        // --- Frame subject: reset camera to default framing centered on subject
        // Subject lives at world origin (0,0,0) up to ~0.5 in Z due to displacement.
        // Default framing is straight-on view from +Z at distance 3.0.
        window._slFrameSubject = function(){
            _orbitTheta = 0;
            _orbitPhi = 0;
            _orbitDist = 3.0;
            _panX = 0.0;
            _panY = 0.0;
            window._slUpdateCamera();
            console.log("[STAGELIGHT] camera framed to subject");
        };

        _scene.add(new THREE.AmbientLight(0xffffff,0.4));

        // Placeholder image plane at origin (vertical, facing camera initially)
        _mesh=new THREE.Mesh(
            new THREE.PlaneGeometry(1.5,1.5,1,1),
            new THREE.MeshBasicMaterial({color:0x1a1a1a,transparent:true,opacity:0.3,side:THREE.DoubleSide})
        );
        _mesh.position.set(0,0,0);
        _scene.add(_mesh);

        // Floor grid - horizontal, below origin
        var floor=new THREE.GridHelper(6,12,0x2a2a2a,0x1a1a1a);
        floor.position.y=-1.0;
        _scene.add(floor);

        // Origin axes helper (permanent, so user sees world 0,0,0)
        var worldAxes=new THREE.AxesHelper(0.4);
        _scene.add(worldAxes);

        // --- light gizmo: Maya-style. Sphere + ring fixed size. Handles scale longer.
        var gizmoGroup=new THREE.Group();

        // Constant-size sphere pivot (center dot)
        var gizmoMat=new THREE.MeshBasicMaterial({color:0xffd060,transparent:true,opacity:0.95});
        _lightGizmo=new THREE.Mesh(new THREE.SphereGeometry(0.08,24,24),gizmoMat);
                gizmoGroup.add(_lightGizmo);

        // Constant-size ring, billboards to camera
        var ring=new THREE.Mesh(
            new THREE.RingGeometry(0.11,0.135,32),
            new THREE.MeshBasicMaterial({color:0xffd060,side:THREE.DoubleSide,transparent:true,opacity:0.65})
        );
                gizmoGroup.add(ring);

        // Scalable axes group - these are the "handles" that stretch
        var axesGroup=new THREE.Group();
        // Thicker hitboxes for easier clicking (invisible cylinders wrapping visible handles)
        function makeHandle(color, axis, rotate, pos){
            var group=new THREE.Group();
            var mat=new THREE.MeshBasicMaterial({color:color});
            var line=new THREE.Mesh(new THREE.CylinderGeometry(0.012,0.012,1.0,8), mat);
            line.position.y=0.5;
                        group.add(line);
            var tip=new THREE.Mesh(new THREE.ConeGeometry(0.04,0.12,12), mat);
            tip.position.y=1.0;
                        group.add(tip);
            // Invisible fat cylinder for easier hit detection
            var hitMat=new THREE.MeshBasicMaterial({visible:false});
            var hitbox=new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.05,1.2,8), hitMat);
            hitbox.position.y=0.6;
            hitbox.userData.axis=axis;
            hitbox.userData.isHandle=true;
            group.add(hitbox);
            // Also tag visible pieces so hover/pick still work even if we ever ray the visible side
            line.userData.axis=axis; line.userData.isHandle=true;
            tip.userData.axis=axis;  tip.userData.isHandle=true;
            group.userData.axis=axis;
            group.userData.handleMat=mat;    // for hover highlight
            group.userData.origColor=color;
            if(rotate) rotate(group);
            return {group:group, hitbox:hitbox, line:line, tip:tip, mat:mat};
        }
        var xH=makeHandle(0xff4040,"x",function(g){g.rotation.z=-Math.PI/2;});
        var yH=makeHandle(0x40ff40,"y",null);
        var zH=makeHandle(0x4080ff,"z",function(g){g.rotation.x=Math.PI/2;});
        axesGroup.add(xH.group);
        axesGroup.add(yH.group);
        axesGroup.add(zH.group);

        // Tag the center sphere so it\'s identifiable as "free plane drag"
        _lightGizmo.userData.axis="free";
        _lightGizmo.userData.isHandle=true;

        gizmoGroup.add(axesGroup);
        _lightAxes=axesGroup;
        // Master handle inventory by mode. _handles is rebuilt by syncGizmoFromSliders
        // when _tmode changes so only the active mode's handles are pickable.
        _lightGizmo._handlesByMode = {
            move:   [_lightGizmo, xH.hitbox, yH.hitbox, zH.hitbox],  // light center + axis arrows
            rotate: [_lightGizmo],                                    // just light center for now (rings added below)
            scale:  [_lightGizmo]                                     // light center + scale cube (added below)
        };
        _lightGizmo._handles = _lightGizmo._handlesByMode.move.slice();
        _lightGizmo._handleRefs={x:xH, y:yH, z:zH};

        // === Rotate gizmo: 3 colored circles around the light ===
        // Each circle is a torus (thin ring) with a wider invisible hitbox torus
        // so the user can grab it more easily without pixel-perfect aim.
        function makeRotateRing(color, axis, rotateFn){
            var grp=new THREE.Group();
            var mat=new THREE.MeshBasicMaterial({color:color, transparent:true, opacity:0.85});
            var ringMesh=new THREE.Mesh(new THREE.TorusGeometry(0.6, 0.005, 8, 64), mat);
            grp.add(ringMesh);
            // Wider invisible hitbox for easier picking
            var hitbox=new THREE.Mesh(
                new THREE.TorusGeometry(0.6, 0.06, 8, 32),
                new THREE.MeshBasicMaterial({visible:false})
            );
            hitbox.userData.axis=axis;
            hitbox.userData.isHandle=true;
            grp.add(hitbox);
            if(rotateFn) rotateFn(grp);
            grp.userData.origColor=color;
            return {group:grp, ring:ringMesh, hitbox:hitbox, mat:mat};
        }
        var rotGroup=new THREE.Group();
        // Red ring - rotates around X axis - lies in YZ plane
        var rxR=makeRotateRing(0xff4040, "rot-x", function(g){g.rotation.y=Math.PI/2;});
        // Green ring - rotates around Y axis - lies in XZ plane
        var ryR=makeRotateRing(0x40ff40, "rot-y", function(g){g.rotation.x=Math.PI/2;});
        // Blue ring - rotates around Z axis - lies in XY plane (default torus orientation)
        var rzR=makeRotateRing(0x4080ff, "rot-z", null);
        rotGroup.add(rxR.group);
        rotGroup.add(ryR.group);
        rotGroup.add(rzR.group);
        rotGroup.visible=false;  // hidden until E mode active
        gizmoGroup.add(rotGroup);
        _lightGizmo._rotateGroup=rotGroup;
        _lightGizmo._rotateRefs={x:rxR, y:ryR, z:rzR};
        // Register rotate hitboxes under the rotate-mode handles list (NOT the master
        // _handles list - that gets rebuilt dynamically by syncGizmoFromSliders)
        _lightGizmo._handlesByMode.rotate.push(rxR.hitbox, ryR.hitbox, rzR.hitbox);

        // === Scale gizmo: a single neutral white cube on a stalk ===
        // User drags the cube; distance from the light controls the SCALE slider.
        var scaleGroup=new THREE.Group();
        // Stalk - thin line from origin to cube position
        var stalkMat=new THREE.LineBasicMaterial({color:0xcccccc, transparent:true, opacity:0.6});
        var stalkGeom=new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0,0,0),
            new THREE.Vector3(0.5, 0.5, 0)
        ]);
        var stalk=new THREE.Line(stalkGeom, stalkMat);
        scaleGroup.add(stalk);
        // The cube itself - neutral white
        var scaleCubeMat=new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:0.85});
        var scaleCube=new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 0.08), scaleCubeMat);
        scaleCube.position.set(0.5, 0.5, 0);
        scaleCube.userData.axis="scale";
        scaleCube.userData.isHandle=true;
        scaleCube.userData.origColor=0xffffff;
        scaleGroup.add(scaleCube);
        // Wider invisible hitbox for easier picking
        var scaleHit=new THREE.Mesh(
            new THREE.BoxGeometry(0.18, 0.18, 0.18),
            new THREE.MeshBasicMaterial({visible:false})
        );
        scaleHit.position.set(0.5, 0.5, 0);
        scaleHit.userData.axis="scale";
        scaleHit.userData.isHandle=true;
        scaleGroup.add(scaleHit);
        scaleGroup.visible=false;  // hidden until R mode active
        gizmoGroup.add(scaleGroup);
        _lightGizmo._scaleGroup=scaleGroup;
        _lightGizmo._scaleCube=scaleCube;
        _lightGizmo._scaleStalk=stalk;
        _lightGizmo._scaleCubeMat=scaleCubeMat;
        _lightGizmo._handlesByMode.scale.push(scaleHit, scaleCube);

        _scene.add(gizmoGroup);
        _lightGizmo._group=gizmoGroup;

        // --- Target gizmo (cyan sphere + ring + axis handles)
        var targetGroup = new THREE.Group();
        var targetMat = new THREE.MeshBasicMaterial({color:0x40ffe0,transparent:true,opacity:0.95});
        var targetMesh = new THREE.Mesh(new THREE.SphereGeometry(0.06,20,20), targetMat);
        targetMesh.userData.axis = "tgt-free";
        targetMesh.userData.isHandle = true;
        targetGroup.add(targetMesh);
        var targetRing = new THREE.Mesh(
            new THREE.RingGeometry(0.085,0.105,32),
            new THREE.MeshBasicMaterial({color:0x40ffe0,side:THREE.DoubleSide,transparent:true,opacity:0.5})
        );
        targetGroup.add(targetRing);
        // Axis handles for the target gizmo (smaller than light's)
        var targetAxesGroup = new THREE.Group();
        var txH = makeHandle(0xff4040,"tgt-x",function(g){g.rotation.z=-Math.PI/2;});
        var tyH = makeHandle(0x40ff40,"tgt-y",null);
        var tzH = makeHandle(0x4080ff,"tgt-z",function(g){g.rotation.x=Math.PI/2;});
        targetAxesGroup.add(txH.group);
        targetAxesGroup.add(tyH.group);
        targetAxesGroup.add(tzH.group);
        targetAxesGroup.scale.set(0.6, 0.6, 0.6);
        targetGroup.add(targetAxesGroup);
        _scene.add(targetGroup);
        _lightGizmo._targetGroup = targetGroup;
        _lightGizmo._targetMesh = targetMesh;
        _lightGizmo._targetRing = targetRing;
        _lightGizmo._targetAxes = targetAxesGroup;
        _lightGizmo._targetHandleRefs = {x:txH, y:tyH, z:tzH};

        // --- Line connecting light to target
        var lineMat = new THREE.LineBasicMaterial({color:0xffd060, transparent:true, opacity:0.4, depthTest:false});
        var lineGeom = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,0)
        ]);
        var connectLine = new THREE.Line(lineGeom, lineMat);
        connectLine.renderOrder = 998;
        _scene.add(connectLine);
        _lightGizmo._connectLine = connectLine;

        // --- Spotlight cone wireframe
        var coneGeom = new THREE.ConeGeometry(1, 1, 32, 1, true);
        var coneMat = new THREE.MeshBasicMaterial({
            color: 0xffd060, wireframe: true, transparent: true, opacity: 0.03, depthTest: false
        });
        var coneMesh = new THREE.Mesh(coneGeom, coneMat);
        coneMesh.renderOrder = 997;
        _scene.add(coneMesh);
        _lightGizmo._coneMesh = coneMesh;

        // Target handles are MOVE-mode only. ROTATE/SCALE modes shouldn't allow
        // target dragging in the viewport (per user spec - sliders still work).
        // Track target hitboxes separately so visibility filter can strip them
        _lightGizmo._targetHandles = [targetMesh, txH.hitbox, tyH.hitbox, tzH.hitbox];
        _lightGizmo._handlesByMode.move.push(targetMesh, txH.hitbox, tyH.hitbox, tzH.hitbox);
        // Refresh the live pickable list to include the new target handles
        _lightGizmo._handles = _lightGizmo._handlesByMode.move.slice();

        // Ring always faces the camera (billboarded). Also handles target ring, line, cone.
        _lightGizmo._tick=function(){
            if(_camera){
                ring.lookAt(_camera.position);
                if(_lightGizmo._targetRing) _lightGizmo._targetRing.lookAt(_camera.position);
            }
            // Update connecting line and cone based on light + target positions
            if(_lightGizmo._connectLine && _lightGizmo._targetGroup && _lightGizmo._group){
                var lp = _lightGizmo._group.position;
                var tp = _lightGizmo._targetGroup.position;
                var positions = _lightGizmo._connectLine.geometry.attributes.position;
                if(positions){
                    positions.array[0] = lp.x; positions.array[1] = lp.y; positions.array[2] = lp.z;
                    positions.array[3] = tp.x; positions.array[4] = tp.y; positions.array[5] = tp.z;
                    positions.needsUpdate = true;
                }
                // Position cone: tip at light, base extends along (light->target) direction by cone_reach distance
                var cone = _lightGizmo._coneMesh;
                if(cone){
                    var dir = new THREE.Vector3().subVectors(tp, lp);
                    var dirLen = dir.length();
                    if(dirLen > 0.001){
                        // cone length = REACH slider (how far the light extends)
                        var coneLength = parseFloat((document.getElementById("sl-fal")||{value:5.0}).value);
                        var angDeg = parseFloat((document.getElementById("sl-cone")||{value:60}).value);
                        var halfRad = (angDeg * 0.5) * Math.PI / 180.0;
                        var baseRadius = Math.tan(halfRad) * coneLength;
                        cone.scale.set(baseRadius, coneLength, baseRadius);
                        // Cone default has tip at +Y, base at -Y. We want tip at light, base extending toward target.
                        var fwd = dir.clone().normalize();
                        // Cone center sits at light + fwd * coneLength * 0.5
                        var coneCenter = lp.clone().add(fwd.clone().multiplyScalar(coneLength * 0.5));
                        cone.position.copy(coneCenter);
                        // Local +Y must point from base (away from light) to tip (at light). So +Y = -fwd.
                        var quat = new THREE.Quaternion().setFromUnitVectors(
                            new THREE.Vector3(0,1,0),
                            fwd.clone().negate()
                        );
                        cone.quaternion.copy(quat);
                    }
                }
            }
        };

        // sync from sliders: position + color + handle length + target position + visibility
        function syncGizmoFromSliders(){
            if(!_lightGizmo||!_lightGizmo._group)return;
            var lx=parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value);
            var ly=parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value);
            var lz=parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value);
            _lightGizmo._group.position.set(lx,ly,lz);
            // Target gizmo position
            var tx=parseFloat((document.getElementById("sl-tx")||{value:0.0}).value);
            var ty=parseFloat((document.getElementById("sl-ty")||{value:0.0}).value);
            var tz=parseFloat((document.getElementById("sl-tz")||{value:0.0}).value);
            if(_lightGizmo._targetGroup) _lightGizmo._targetGroup.position.set(tx,ty,tz);
            // Color
            var lr=parseFloat((document.getElementById("sl-lr")||{value:1.0}).value);
            var lg=parseFloat((document.getElementById("sl-lg")||{value:0.95}).value);
            var lb=parseFloat((document.getElementById("sl-lb")||{value:0.85}).value);
            gizmoMat.color.setRGB(lr,lg,lb);
            ring.material.color.setRGB(lr,lg,lb);
            // Only the AXES group scales by GIZMO SZ - the actual light sphere uses SCALE
            var gs=parseFloat((document.getElementById("sl-gsz")||{value:1.0}).value);
            _gizmoScale=gs;
            axesGroup.scale.set(gs,gs,gs);
            // SCALE slider drives the light sphere visual size: 1x at scale=0 -> 4x at scale=1
            var sc=parseFloat((document.getElementById("sl-lsz")||{value:0.0}).value);
            var sphereVisualScale = 1.0 + sc * 3.0;  // [1, 4]
            _lightGizmo.scale.set(sphereVisualScale, sphereVisualScale, sphereVisualScale);
            ring.scale.set(sphereVisualScale, sphereVisualScale, sphereVisualScale);
            // Scale rotate ring group similarly to axes
            if(_lightGizmo._rotateGroup) _lightGizmo._rotateGroup.scale.set(gs,gs,gs);
            // === SCALE gizmo (cube on stalk) - position based on current SCALE value ===
            // The cube sits at distance 0.5 + sc * 0.5 along the diagonal (0.7, 0.7, 0)
            // So at SCALE=0 -> (0.35, 0.35, 0), at SCALE=1 -> (0.7, 0.7, 0)
            if(_lightGizmo._scaleGroup){
                _lightGizmo._scaleGroup.scale.set(gs,gs,gs);
                var dirN = new THREE.Vector3(0.7071, 0.7071, 0); // normalized diagonal
                var dist = 0.5 + sc * 0.5;
                var cubePos = dirN.clone().multiplyScalar(dist);
                if(_lightGizmo._scaleCube) _lightGizmo._scaleCube.position.copy(cubePos);
                // Cube hitbox is the second-to-last child after the stalk + cube
                var hitbox = _lightGizmo._scaleGroup.children[2]; // [stalk, cube, hitbox]
                if(hitbox) hitbox.position.copy(cubePos);
                // Re-build the stalk endpoints (Three.js needs explicit geometry update)
                if(_lightGizmo._scaleStalk){
                    _lightGizmo._scaleStalk.geometry.dispose();
                    _lightGizmo._scaleStalk.geometry = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(0,0,0), cubePos
                    ]);
                }
            }
            // Visibility logic
            var lt = (function(){var s=document.getElementById("sl-ltype");return s?parseInt(s.value):0;})();
            var gizmosOn = (function(){var b=document.getElementById("sl-gizmos");return !b||b.checked;})();
            var coneOn = (function(){var b=document.getElementById("sl-conevis");return !b||b.checked;})();
            // TARGET toggle - off by default, user opts in. When off, target gizmo is
            // both hidden and non-pickable in viewport. The TGT X/Y/Z sliders still work.
            var tgtOn = (function(){var b=document.getElementById("sl-tgtvis");return b&&b.checked;})();
            // Master GIZMOS toggle hides everything
            _lightGizmo._group.visible = gizmosOn;
            // === Mode discipline: target visible+pickable ONLY in MOVE mode (and only when light type allows it) ===
            var inMove = (_tmode === "t");
            var inRotate = (_tmode === "r");
            var inScale = (_tmode === "s");
            // Target sphere + ring + axes: requires user opt-in (tgtOn) AND MOVE mode AND non-POINT light AND GIZMOS on
            var showTarget = gizmosOn && tgtOn && inMove && (lt !== 0);
            if(_lightGizmo._targetGroup) _lightGizmo._targetGroup.visible = showTarget;
            if(_lightGizmo._targetAxes) _lightGizmo._targetAxes.visible = showTarget;
            // Connecting line + cone only in MOVE mode (so user can see relationship)
            // Note: cone *can* still be visible in ROTATE/SCALE if user wants, since
            // it doesn\'t imply interactivity. Let\'s allow it for spatial context.
            if(_lightGizmo._connectLine) _lightGizmo._connectLine.visible = gizmosOn && tgtOn && (lt !== 0);
            if(_lightGizmo._coneMesh) _lightGizmo._coneMesh.visible = gizmosOn && coneOn && (lt === 1);
            // Translate-axis arrows: MOVE only
            if(_lightAxes) _lightAxes.visible = gizmosOn && inMove;
            // Rotate rings: ROTATE only
            if(_lightGizmo._rotateGroup) _lightGizmo._rotateGroup.visible = gizmosOn && inRotate;
            // Scale gizmo: SCALE only
            if(_lightGizmo._scaleGroup) _lightGizmo._scaleGroup.visible = gizmosOn && inScale;

            // === Rebuild the live pickable handles list based on current mode ===
            // This is the key fix: hidden handles must also not be raycast-pickable.
            // Previous behavior left them in the list so misclicks could trigger
            // wrong-mode drags (e.g. translating while in rotate).
            if(_lightGizmo._handlesByMode){
                var modeKey = inMove ? "move" : (inRotate ? "rotate" : "scale");
                var live = gizmosOn ? _lightGizmo._handlesByMode[modeKey].slice() : [];
                // When target is hidden, also strip target hitboxes so a click in that
                // area can't trigger a target drag.
                if(!showTarget && _lightGizmo._targetHandles){
                    live = live.filter(function(h){
                        return _lightGizmo._targetHandles.indexOf(h) === -1;
                    });
                }
                _lightGizmo._handles = live;
            }

            // === Inactive light markers (v0.8.3) ===
            // For each light in _lights[] that isn't the active one, show a small
            // colored sphere at its position so user can tell where all their lights are.
            // Active light still has the full draggable gizmo.
            if(!window._slMarkers) window._slMarkers = [];
            // Ensure we have enough markers; create missing ones lazily
            while(window._slMarkers.length < _lights.length){
                var idx = window._slMarkers.length;
                // Neutral gray for all markers - per-light color stays in the outliner swatch
                var markerMat = new THREE.MeshBasicMaterial({color: 0xbbbbbb, transparent: true, opacity: 0.85});
                var marker = new THREE.Mesh(new THREE.SphereGeometry(0.05, 16, 16), markerMat);
                marker.renderOrder = 996;
                // Tag this mesh as a light-marker so raycast hit can identify it
                marker.userData.isLightMarker = true;
                marker.userData.lightIdx = idx;  // refreshed in the position loop below
                _scene.add(marker);
                window._slMarkers.push(marker);
            }
            // Position markers + visibility + refresh light index (in case lights array shifted)
            for(var mi = 0; mi < window._slMarkers.length; mi++){
                var m = window._slMarkers[mi];
                if(mi >= _lights.length){
                    m.visible = false;
                    continue;
                }
                var LL = _lights[mi];
                m.position.set(LL.light_x, LL.light_y, LL.light_z);
                m.userData.lightIdx = mi;  // keep mapping fresh after deletes/reorders
                // Hide the marker for the SELECTED light (it has the full gizmo)
                // and hide if disabled or master gizmos are off.
                m.visible = gizmosOn && LL.enabled && (mi !== _selectedLight);
            }
            // Re-apply gizmo opacity so newly-created handles/markers match current slider value
            if(window._slApplyGizmoOpacity){
                var gop = document.getElementById("sl-gop");
                window._slApplyGizmoOpacity(gop ? parseFloat(gop.value) : 0.5);
            }
        }
        window._slSyncGizmo=function(){syncGizmoFromSliders();requestRender();};
        syncGizmoFromSliders();
        new ResizeObserver(function(){
            var cw=vp.clientWidth,ch=vp.clientHeight;
            if(cw>0&&ch>0){_camera.aspect=cw/ch;_camera.updateProjectionMatrix();_renderer.setSize(cw,ch);requestRender();}
        }).observe(vp);
        // Dirty-flag rendering: only render when state changes
        // requestRender() schedules a single frame; after rendering, no more frames until next request.
        var _renderPending=false;
        window._slRequestRender=function(){
            if(_renderPending) return;
            _renderPending=true;
            _animId=requestAnimationFrame(function(){
                _renderPending=false;
                if(!_renderer||!_scene||!_camera) return;
                if(_lightGizmo&&_lightGizmo._tick)_lightGizmo._tick();
                _renderer.render(_scene,_camera);
            });
        };
        // Kick off the first frame
        window._slRequestRender();
    });

    document.getElementById("sl-gen").onclick=function(){
        var btn=document.getElementById("sl-gen");
        var st=document.getElementById("sl-status");
        var imageFile="";
        for(var i=0;i<app.graph._nodes.length;i++){
            var n=app.graph._nodes[i];
            if(n.type==="LoadImage"&&n.widgets){
                for(var j=0;j<n.widgets.length;j++){
                    if(n.widgets[j].type==="combo"||n.widgets[j].name==="image"){imageFile=n.widgets[j].value||"";break;}
                }
                if(imageFile) break;
            }
        }
        if(!imageFile){st.textContent="No image - add a LoadImage node first";st.style.color="#cc4444";return;}
        btn.disabled=true;btn.style.opacity="0.5";
        st.textContent="Generating depth...";st.style.color="#e8a020";
        var prompt={
            "1":{class_type:"LoadImage", inputs:{image:imageFile,upload:"image"}},
            "2":{class_type:"StagelightNode", inputs:{
                image:["1",0],
                light_x:  parseFloat((document.getElementById("sl-lx")||{value:-0.5}).value),
                light_y:  parseFloat((document.getElementById("sl-ly")||{value: 1.0}).value),
                light_z:  parseFloat((document.getElementById("sl-lz")||{value: 2.0}).value),
                intensity:parseFloat((document.getElementById("sl-int")||{value: 2.0}).value),
                cone_reach:       parseFloat((document.getElementById("sl-fal")||{value: 5.0}).value),
                falloff_curve:    parseFloat((document.getElementById("sl-fcv")||{value: 2.0}).value),
                // v0.8.9: ambient param is now neutral (1.0). The real grade chain
                // sends exposure/gain/lift/gamma below. Ambient is kept in the Python
                // signature for backward compat; it's no longer the brightness control.
                ambient:  1.0,
                gamma:    parseFloat((document.getElementById("sl-gam")||{value: 1.0}).value),
                light_r:  parseFloat((document.getElementById("sl-lr")||{value: 1.0}).value),
                light_g:  parseFloat((document.getElementById("sl-lg")||{value: 1.0}).value),
                light_b:  parseFloat((document.getElementById("sl-lb")||{value: 1.0}).value),
                // v0.8.9 grade chain (Nuke-style)
                exposure_stops: parseFloat((document.getElementById("sl-exp")||{value: 0.0}).value),
                grade_gain:     parseFloat((document.getElementById("sl-gain")||{value: 1.0}).value),
                grade_lift:     parseFloat((document.getElementById("sl-lift")||{value: 0.0}).value),
                shadow_strength: parseFloat((document.getElementById("sl-shd")||{value: 1.0}).value),
                // v0.8.13: per-channel grade tints (from color wheels in COMPOSITE panel)
                grade_gain_r:  (window._slGradeTints && window._slGradeTints.gain && window._slGradeTints.gain[0]) || 1.0,
                grade_gain_g:  (window._slGradeTints && window._slGradeTints.gain && window._slGradeTints.gain[1]) || 1.0,
                grade_gain_b:  (window._slGradeTints && window._slGradeTints.gain && window._slGradeTints.gain[2]) || 1.0,
                grade_lift_r:  (window._slGradeTints && window._slGradeTints.lift && window._slGradeTints.lift[0]) || 1.0,
                grade_lift_g:  (window._slGradeTints && window._slGradeTints.lift && window._slGradeTints.lift[1]) || 1.0,
                grade_lift_b:  (window._slGradeTints && window._slGradeTints.lift && window._slGradeTints.lift[2]) || 1.0,
                grade_gamma_r: (window._slGradeTints && window._slGradeTints.gamma && window._slGradeTints.gamma[0]) || 1.0,
                grade_gamma_g: (window._slGradeTints && window._slGradeTints.gamma && window._slGradeTints.gamma[1]) || 1.0,
                grade_gamma_b: (window._slGradeTints && window._slGradeTints.gamma && window._slGradeTints.gamma[2]) || 1.0,
                bg_depth_scale:  parseFloat((document.getElementById("sl-bgd")||{value: 0.24}).value),
                depth_cutoff:    parseFloat((document.getElementById("sl-cut")||{value: 1.0}).value),
                cutoff_feather:  parseFloat((document.getElementById("sl-fth")||{value: 0.1}).value),
                bg_visibility:   parseFloat((document.getElementById("sl-bgv")||{value: 0.0}).value),
                // v0.8.7: simple cutoff mask + invert. Old fields above are legacy
                // fallbacks for backward compat with the Python signature; their values
                // aren't used since the masking now reads mask_cutoff/mask_falloff/mask_invert.
                mask_cutoff:    parseFloat((document.getElementById("sl-mask")||{value: 0.05}).value),
                mask_falloff:   parseFloat((document.getElementById("sl-mfth")||{value: 0.05}).value),
                mask_invert:    ((document.getElementById("sl-minv")||{checked:false}).checked),
                weight_smart:     parseFloat((document.getElementById("sl-wsm")||{value: 0.7}).value),
                weight_bae:       parseFloat((document.getElementById("sl-wba")||{value: 0.2}).value),
                weight_depth:     parseFloat((document.getElementById("sl-wdp")||{value: 0.1}).value),
                smart_strength:   parseFloat((document.getElementById("sl-sst")||{value: 4.0}).value),
                smart_blur:       parseFloat((document.getElementById("sl-sbl")||{value: 1.0}).value),
                smart_detail:     parseFloat((document.getElementById("sl-sdt")||{value: 1.0}).value),
                normal_strength:  parseFloat((document.getElementById("sl-nrm")||{value: 1.5}).value),
                hf_detail:        parseFloat((document.getElementById("sl-nhf")||{value: 1.0}).value),
                bae_supersample:  parseFloat((document.getElementById("sl-nss")||{value: 2.0}).value),
                use_bae_normal:   ((document.getElementById("sl-bae")||{checked:true}).checked),
                smart_equalize:   false,
                spec_strength:    parseFloat((document.getElementById("sl-spc")||{value:0.3}).value),
                spec_shiny:       parseFloat((document.getElementById("sl-shn")||{value:16}).value),
                target_x:         parseFloat((document.getElementById("sl-tx")||{value:0}).value),
                target_y:         parseFloat((document.getElementById("sl-ty")||{value:0}).value),
                target_z:         parseFloat((document.getElementById("sl-tz")||{value:0}).value),
                spot_angle_deg:   parseFloat((document.getElementById("sl-cone")||{value:60}).value),
                spot_softness:    parseFloat((document.getElementById("sl-soft")||{value:0.7}).value),
                light_type:       (function(){var s=document.getElementById("sl-ltype");return s?parseInt(s.value):0;})(),
                fixed_intensity:  ((document.getElementById("sl-fixed")||{checked:true}).checked),
                disp_amount:      parseFloat((document.getElementById("sl-vdp")||{value:0.5}).value),
                light_size:       parseFloat((document.getElementById("sl-lsz")||{value:0.0}).value),
                lights_json:      (function(){
                    // Make sure live edits are committed to the active light first
                    if(window._slCommitSlidersToLight) window._slCommitSlidersToLight();
                    return JSON.stringify(_lights || []);
                })()
            }},
            "3":{class_type:"PreviewImage", inputs:{images:["2", (function(){var a=document.querySelector(".sl-vbtn-active");return a?parseInt(a.getAttribute("data-view")):3;})()]}},
            "4":{class_type:"PreviewImage", inputs:{images:["2", 1]}},
            "5":{class_type:"PreviewImage", inputs:{images:["2", 0]}},
            "6":{class_type:"PreviewImage", inputs:{images:["2", 2]}}
        };
        queuePrompt(prompt,function(err,result){
            if(err||!result||!result.prompt_id){
                st.textContent="Queue failed";st.style.color="#cc4444";
                btn.disabled=false;btn.style.opacity="1";return;
            }
            st.textContent="Processing...";
            pollHistory(result.prompt_id,90000,function(err2,outputs){
                btn.disabled=false;btn.style.opacity="1";
                if(err2||!outputs){st.textContent="Failed";st.style.color="#cc4444";return;}
                var o3=outputs["3"];
                if(!o3||!o3.images||!o3.images.length){st.textContent="No output";st.style.color="#cc4444";return;}
                // Slot 3 = view-toggled preview (legacy - the user-visible PreviewImage in graph)
                // Slot 4 = depth map (for vertex displacement + view modes)
                // Slot 5 = source color (always - for the displaced mesh texture)
                //
                // Critical: viewport mesh texture must ALWAYS be the source color image,
                // never a Python-relit one. The GPU shader does the lighting on top -
                // if we used a relit texture, the lighting would be applied twice.
                // (See v0.7.7 changelog - this fixed the post-render blowout bug.)
                var o4 = outputs["4"];
                var o5 = outputs["5"];
                var o6 = outputs["6"];
                var depthUrl = (o4&&o4.images&&o4.images.length) ?
                    viewUrl(o4.images[0].filename,o4.images[0].subfolder,o4.images[0].type) : null;
                // Always use source color from slot 5; fall back to slot 3 only if missing.
                var colorUrl = (o5&&o5.images&&o5.images.length)
                    ? viewUrl(o5.images[0].filename,o5.images[0].subfolder,o5.images[0].type)
                    : viewUrl(o3.images[0].filename,o3.images[0].subfolder,o3.images[0].type);
                var normalUrl = (o6&&o6.images&&o6.images.length) ?
                    viewUrl(o6.images[0].filename,o6.images[0].subfolder,o6.images[0].type) : null;
                loadMesh(colorUrl, depthUrl, normalUrl, st);
            });
        });
    };
}

app.registerExtension({
    name:"stagelight.v0905",
    beforeRegisterNodeDef(nodeType,nodeData,app){
        if(nodeData.name!=="StagelightNode") return;
        var _onAdded=nodeType.prototype.onAdded;
        nodeType.prototype.onAdded=function(){
            this.color="#2a1a04";this.bgcolor="#1a1004";this.title="STAGELIGHT";
            var self=this;
            this.onDblClick=function(){openPanel(self);return true;};
            openPanel(this);
            if(_onAdded) return _onAdded.apply(this,arguments);
        };
    }
});
"""

# === v0.9.0: AI cleanup workflow files ===
# These get written to ComfyUI/custom_nodes/STAGELIGHT/workflows/ so users
# have ready-to-use example workflows for Z Image Turbo cleanup and IC-Light.
PLUGIN_FILES["workflows/README.md"] = '# STAGELIGHT v0.9.5 — AI Cleanup Workflows\n\nTwo paths for the cleanup pass after STAGELIGHT relights your image. Pick based on what you need.\n\n---\n\n## Path 1 — Z Image Turbo (fast, generic cleanup)\n\n**File:** `stagelight_zimage_cleanup.json`\n\nGeneric img2img cleanup at denoise 0.20. Fast (2-3s), preserves your STAGELIGHT lighting decisions, adds detail and reduces AI artifacts.\n\n**Use this when:** you want the output to look very close to STAGELIGHT\'s viewport — just slightly cleaner and more polished.\n\n**Don\'t use when:** you want IC-Light\'s proper material response (skin sheen, fabric specularity, rim light wrap). Z Image Turbo can\'t do that — it just denoises.\n\n**Required models:**\n- `qwen_3_4b.safetensors` (CLIP)\n- `z_image_turbo_bf16.safetensors` (UNet)\n- `ae.safetensors` (VAE)\n\n---\n\n## Path 2 — IC-Light (proper material response and lighting wrap)\n\n**File:** `stagelight_iclight_relight.json`\n\nThis is the workflow you want if you couldn\'t get rim lighting from Path 1 and the lighting feels like "outfit tinting" instead of actual light wrapping the subject.\n\n### How it actually works\n\nIC-Light is an SD 1.5 model patched to **understand lighting**. Unlike Z Image Turbo (which is generic denoising), IC-Light is trained on relighting data — it knows that "bright on left + dark on right" means side lighting, and renders it with proper material specularity, color bleeding into shadows, etc.\n\nSTAGELIGHT outputs two things IC-Light needs:\n\n| STAGELIGHT output | Feeds into IC-Light as | Why |\n|---|---|---|\n| `image` (slot 0) | `ICLightConditioning(foreground)` (after VAE encode) | Preserves the subject\'s identity, features, costume |\n| `lighting_only` (slot 7) | `KSampler(latent_image)` (after VAE encode) | Biases the diffusion toward your light direction |\n\nThe text prompts describe the **style** of lighting and material quality you want — they don\'t define direction (the lighting_only image does that).\n\n### Required downloads\n\n1. **An SD 1.5 photoreal checkpoint** (~2GB). Recommended:\n   - Juggernaut Reborn — https://civitai.com/models/46422\n   - RealisticVision V5.1 — https://civitai.com/models/4201\n   - DreamShaper 8 — https://civitai.com/models/4384\n   - Drop into `ComfyUI/models/checkpoints/`\n\n2. **The IC-Light fc model** (~1.4GB):\n   - `iclight_sd15_fc.safetensors` from https://huggingface.co/lllyasviel/ic-light/tree/main\n   - Drop into `ComfyUI/models/unet/`\n\n3. **kijai\'s ComfyUI-IC-Light extension** (you already have this — visible in the log)\n\nTotal: ~3.4GB. Once downloaded, no further downloads needed for subsequent generations.\n\n### Setup steps\n\n1. Open `stagelight_iclight_relight.json` in ComfyUI\n2. In the **CheckpointLoaderSimple** node ("Load SD 1.5"), pick your downloaded SD 1.5 checkpoint\n3. In the **LoadAndApplyICLightUnet** node, verify it\'s pointing at `iclight_sd15_fc.safetensors`\n4. Place your image in the LoadImage node\n5. Light your subject in the STAGELIGHT panel — choose colors, positions, multiple lights as needed\n6. Hit ComfyUI\'s **Run** button (or STAGELIGHT\'s RENDER then Run)\n7. Wait ~5-15 seconds. Result appears in the "FINAL" PreviewImage at the right.\n\n### What you should see\n\n- **lighting_only preview** (PreviewImage labeled "STAGELIGHT lighting_only"): the gradient driving the lighting. Should clearly show where you placed lights.\n- **FINAL preview** (top-right): the photoreal relit output. Compare to the viewport. Lighting direction should match. Material response (skin glow, fabric specular, rim light) should be far more realistic than Path 1.\n\n### Tweaking\n\n- **Too much lighting change vs original?** Increase `multiplier` on ICLightConditioning above 0.18215 (try 0.25 or 0.30). This boosts subject preservation.\n- **Lighting not strong enough?** Add more lights or boost INTENSITY in STAGELIGHT. The `lighting_only` output is what IC-Light reads — brighter, more directional = stronger relight.\n- **Too AI-generic looking?** Tweak the positive prompt to describe specific material qualities ("sharp cotton fabric weave", "weathered leather", etc.). Avoid generic words like "high quality."\n- **Want sharper output?** Increase KSampler steps from 25 to 30-40. Slower but cleaner.\n\n### Licensing note\n\n- IC-Light v1 fc/fbc models (this workflow uses fc): Apache 2.0 — commercial use OK\n- IC-Light v2 models: non-commercial only — don\'t use those for paid work\n\n---\n\n## Troubleshooting\n\n**"Cannot find node ICLightConditioning"**\n→ Install kijai\'s ComfyUI-IC-Light via ComfyUI Manager → Custom Nodes Manager → search "IC-Light" → kijai\'s version.\n\n**"Cannot find checkpoint sd15_checkpoint.safetensors"**\n→ The workflow ships with a placeholder filename. Pick your real SD 1.5 file from the dropdown.\n\n**Error: "Return type mismatch... foreground IMAGE != LATENT"**\n→ This was a known issue in older IC-Light versions. Update the extension via ComfyUI Manager. In this workflow, all foreground/background inputs are pre-wired as LATENT correctly.\n\n**The output looks blown out / weirdly bright**\n→ Check the `lighting_only` preview. If it\'s mostly white, your STAGELIGHT intensity is too high. Lower it to 0.5–1.5 range so the lighting layer has clear shadows AND highlights.\n\n**The output ignores my lighting direction**\n→ The lighting_only image must be CLEAR about direction. If you have 5 lights all over the subject, lighting_only becomes a wash. Try with one strong key light first to verify the wiring works, then layer up.\n'

PLUGIN_FILES["workflows/stagelight_zimage_cleanup.json"] = '{\n  "last_node_id": 12,\n  "last_link_id": 14,\n  "nodes": [\n    {\n      "id": 1,\n      "type": "LoadImage",\n      "pos": [\n        80,\n        200\n      ],\n      "size": [\n        320,\n        314\n      ],\n      "flags": {},\n      "order": 0,\n      "mode": 0,\n      "outputs": [\n        {\n          "name": "IMAGE",\n          "type": "IMAGE",\n          "links": [\n            1\n          ]\n        },\n        {\n          "name": "MASK",\n          "type": "MASK",\n          "links": null\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "LoadImage"\n      },\n      "widgets_values": [\n        "example.png",\n        "image"\n      ]\n    },\n    {\n      "id": 2,\n      "type": "StagelightNode",\n      "pos": [\n        440,\n        200\n      ],\n      "size": [\n        320,\n        380\n      ],\n      "flags": {},\n      "order": 1,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "image",\n          "type": "IMAGE",\n          "link": 1\n        }\n      ],\n      "outputs": [\n        {\n          "name": "image",\n          "type": "IMAGE",\n          "links": null\n        },\n        {\n          "name": "depth_map",\n          "type": "IMAGE",\n          "links": null\n        },\n        {\n          "name": "normal_map",\n          "type": "IMAGE",\n          "links": null\n        },\n        {\n          "name": "relit_image",\n          "type": "IMAGE",\n          "links": null\n        },\n        {\n          "name": "subject_mask",\n          "type": "MASK",\n          "links": null\n        },\n        {\n          "name": "final_alpha",\n          "type": "MASK",\n          "links": null\n        },\n        {\n          "name": "relit_composite",\n          "type": "IMAGE",\n          "links": [\n            2,\n            13\n          ]\n        },\n        {\n          "name": "lighting_only",\n          "type": "IMAGE",\n          "links": null\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "STAGELIGHT"\n      },\n      "widgets_values": [\n        -0.5,\n        1.0,\n        2.0,\n        1.0,\n        5.0,\n        2.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        0.24,\n        1.0,\n        0.1,\n        0.0,\n        0.7,\n        0.2,\n        0.1,\n        2.9,\n        0.0,\n        0.3,\n        1.0,\n        1.0,\n        1.0,\n        true,\n        false,\n        0.3,\n        16.0,\n        0.0,\n        0.0,\n        0.0,\n        60.0,\n        0.7,\n        0,\n        true,\n        0.0,\n        0.5,\n        "[{\\"id\\": \\"L1\\", \\"name\\": \\"Key\\", \\"enabled\\": true, \\"light_x\\": -0.5, \\"light_y\\": 1.0, \\"light_z\\": 2.0, \\"target_x\\": 0.0, \\"target_y\\": 0.0, \\"target_z\\": 0.0, \\"intensity\\": 1.0, \\"falloff\\": 5.0, \\"fcv\\": 2.0, \\"light_r\\": 1.0, \\"light_g\\": 1.0, \\"light_b\\": 1.0, \\"spec_strength\\": 0.3, \\"spec_shiny\\": 16, \\"light_type\\": 0, \\"fixed_intensity\\": true, \\"light_size\\": 0.0, \\"cone_angle\\": 60, \\"cone_soft\\": 0.7, \\"target_visible\\": false}]",\n        0.05,\n        0.0,\n        false,\n        0.0,\n        1.0,\n        0.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0,\n        1.0\n      ]\n    },\n    {\n      "id": 3,\n      "type": "UNETLoader",\n      "pos": [\n        80,\n        580\n      ],\n      "size": [\n        320,\n        82\n      ],\n      "flags": {},\n      "order": 2,\n      "mode": 0,\n      "outputs": [\n        {\n          "name": "MODEL",\n          "type": "MODEL",\n          "links": [\n            3\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "UNETLoader"\n      },\n      "widgets_values": [\n        "z_image_turbo_bf16.safetensors",\n        "default"\n      ]\n    },\n    {\n      "id": 4,\n      "type": "CLIPLoader",\n      "pos": [\n        80,\n        700\n      ],\n      "size": [\n        320,\n        82\n      ],\n      "flags": {},\n      "order": 3,\n      "mode": 0,\n      "outputs": [\n        {\n          "name": "CLIP",\n          "type": "CLIP",\n          "links": [\n            4,\n            5\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "CLIPLoader"\n      },\n      "widgets_values": [\n        "qwen_3_4b.safetensors",\n        "stable_diffusion"\n      ]\n    },\n    {\n      "id": 5,\n      "type": "VAELoader",\n      "pos": [\n        80,\n        820\n      ],\n      "size": [\n        320,\n        58\n      ],\n      "flags": {},\n      "order": 4,\n      "mode": 0,\n      "outputs": [\n        {\n          "name": "VAE",\n          "type": "VAE",\n          "links": [\n            6,\n            11\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "VAELoader"\n      },\n      "widgets_values": [\n        "ae.safetensors"\n      ]\n    },\n    {\n      "id": 6,\n      "type": "CLIPTextEncode",\n      "pos": [\n        440,\n        580\n      ],\n      "size": [\n        320,\n        100\n      ],\n      "flags": {},\n      "order": 5,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "clip",\n          "type": "CLIP",\n          "link": 4\n        }\n      ],\n      "outputs": [\n        {\n          "name": "CONDITIONING",\n          "type": "CONDITIONING",\n          "links": [\n            7\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "CLIPTextEncode"\n      },\n      "widgets_values": [\n        "photorealistic portrait, cinematic lighting, sharp detail, high quality"\n      ]\n    },\n    {\n      "id": 7,\n      "type": "CLIPTextEncode",\n      "pos": [\n        440,\n        700\n      ],\n      "size": [\n        320,\n        100\n      ],\n      "flags": {},\n      "order": 6,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "clip",\n          "type": "CLIP",\n          "link": 5\n        }\n      ],\n      "outputs": [\n        {\n          "name": "CONDITIONING",\n          "type": "CONDITIONING",\n          "links": [\n            8\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "CLIPTextEncode"\n      },\n      "widgets_values": [\n        "low quality, blurry, distorted, artifacts, ugly"\n      ]\n    },\n    {\n      "id": 8,\n      "type": "VAEEncode",\n      "pos": [\n        800,\n        200\n      ],\n      "size": [\n        220,\n        46\n      ],\n      "flags": {},\n      "order": 7,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "pixels",\n          "type": "IMAGE",\n          "link": 2\n        },\n        {\n          "name": "vae",\n          "type": "VAE",\n          "link": 6\n        }\n      ],\n      "outputs": [\n        {\n          "name": "LATENT",\n          "type": "LATENT",\n          "links": [\n            9\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "VAEEncode"\n      }\n    },\n    {\n      "id": 9,\n      "type": "KSampler",\n      "pos": [\n        800,\n        320\n      ],\n      "size": [\n        320,\n        262\n      ],\n      "flags": {},\n      "order": 8,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "model",\n          "type": "MODEL",\n          "link": 3\n        },\n        {\n          "name": "positive",\n          "type": "CONDITIONING",\n          "link": 7\n        },\n        {\n          "name": "negative",\n          "type": "CONDITIONING",\n          "link": 8\n        },\n        {\n          "name": "latent_image",\n          "type": "LATENT",\n          "link": 9\n        }\n      ],\n      "outputs": [\n        {\n          "name": "LATENT",\n          "type": "LATENT",\n          "links": [\n            10\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "KSampler"\n      },\n      "widgets_values": [\n        42,\n        "randomize",\n        8,\n        1.0,\n        "euler",\n        "simple",\n        0.2\n      ]\n    },\n    {\n      "id": 10,\n      "type": "VAEDecode",\n      "pos": [\n        1160,\n        200\n      ],\n      "size": [\n        220,\n        46\n      ],\n      "flags": {},\n      "order": 9,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "samples",\n          "type": "LATENT",\n          "link": 10\n        },\n        {\n          "name": "vae",\n          "type": "VAE",\n          "link": 11\n        }\n      ],\n      "outputs": [\n        {\n          "name": "IMAGE",\n          "type": "IMAGE",\n          "links": [\n            12\n          ]\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "VAEDecode"\n      }\n    },\n    {\n      "id": 11,\n      "type": "PreviewImage",\n      "pos": [\n        1160,\n        320\n      ],\n      "size": [\n        320,\n        280\n      ],\n      "flags": {},\n      "order": 10,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "images",\n          "type": "IMAGE",\n          "link": 12\n        }\n      ],\n      "properties": {\n        "Node name for S&R": "PreviewImage"\n      }\n    },\n    {\n      "id": 12,\n      "type": "PreviewImage",\n      "pos": [\n        800,\n        640\n      ],\n      "size": [\n        320,\n        240\n      ],\n      "flags": {},\n      "order": 11,\n      "mode": 0,\n      "inputs": [\n        {\n          "name": "images",\n          "type": "IMAGE",\n          "link": 13\n        }\n      ],\n      "title": "STAGELIGHT relit_composite (input to cleanup)",\n      "properties": {\n        "Node name for S&R": "PreviewImage"\n      }\n    }\n  ],\n  "links": [\n    [\n      1,\n      1,\n      0,\n      2,\n      0,\n      "IMAGE"\n    ],\n    [\n      2,\n      2,\n      6,\n      8,\n      0,\n      "IMAGE"\n    ],\n    [\n      3,\n      3,\n      0,\n      9,\n      0,\n      "MODEL"\n    ],\n    [\n      4,\n      4,\n      0,\n      6,\n      0,\n      "CLIP"\n    ],\n    [\n      5,\n      4,\n      0,\n      7,\n      0,\n      "CLIP"\n    ],\n    [\n      6,\n      5,\n      0,\n      8,\n      1,\n      "VAE"\n    ],\n    [\n      7,\n      6,\n      0,\n      9,\n      1,\n      "CONDITIONING"\n    ],\n    [\n      8,\n      7,\n      0,\n      9,\n      2,\n      "CONDITIONING"\n    ],\n    [\n      9,\n      8,\n      0,\n      9,\n      3,\n      "LATENT"\n    ],\n    [\n      10,\n      9,\n      0,\n      10,\n      0,\n      "LATENT"\n    ],\n    [\n      11,\n      5,\n      0,\n      10,\n      1,\n      "VAE"\n    ],\n    [\n      12,\n      10,\n      0,\n      11,\n      0,\n      "IMAGE"\n    ],\n    [\n      13,\n      2,\n      6,\n      12,\n      0,\n      "IMAGE"\n    ]\n  ],\n  "groups": [\n    {\n      "title": "1. Load + STAGELIGHT relight",\n      "bounding": [\n        60,\n        160,\n        720,\n        460\n      ],\n      "color": "#3f789e",\n      "font_size": 20\n    },\n    {\n      "title": "2. Z Image Turbo models + prompts",\n      "bounding": [\n        60,\n        560,\n        720,\n        340\n      ],\n      "color": "#a1309b",\n      "font_size": 20\n    },\n    {\n      "title": "3. img2img cleanup (denoise 0.2)",\n      "bounding": [\n        780,\n        160,\n        720,\n        460\n      ],\n      "color": "#629e3f",\n      "font_size": 20\n    }\n  ],\n  "config": {},\n  "extra": {\n    "info": "STAGELIGHT v0.9.0b cleanup workflow. relit_composite is at output slot 6 (appended at end of v0.8.13 output list for backward compat). Adjust KSampler denoise to taste - lower preserves lighting more, higher cleans up artifacts."\n  },\n  "version": 0.4\n}'

PLUGIN_FILES["workflows/stagelight_iclight_relight.json"] = '{\n  "last_node_id": 14,\n  "last_link_id": 20,\n  "nodes": [\n    {\n      "id": 1,\n      "type": "LoadImage",\n      "pos": [40, 200],\n      "size": [320, 314],\n      "flags": {},\n      "order": 0,\n      "mode": 0,\n      "outputs": [\n        {"name": "IMAGE", "type": "IMAGE", "links": [1]},\n        {"name": "MASK", "type": "MASK", "links": null}\n      ],\n      "properties": {"Node name for S&R": "LoadImage"},\n      "widgets_values": ["example.png", "image"]\n    },\n    {\n      "id": 2,\n      "type": "StagelightNode",\n      "pos": [400, 200],\n      "size": [320, 380],\n      "flags": {},\n      "order": 1,\n      "mode": 0,\n      "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],\n      "outputs": [\n        {"name": "image",           "type": "IMAGE", "links": [2]},\n        {"name": "depth_map",       "type": "IMAGE", "links": null},\n        {"name": "normal_map",      "type": "IMAGE", "links": null},\n        {"name": "relit_image",     "type": "IMAGE", "links": null},\n        {"name": "subject_mask",    "type": "MASK",  "links": null},\n        {"name": "final_alpha",     "type": "MASK",  "links": null},\n        {"name": "relit_composite", "type": "IMAGE", "links": null},\n        {"name": "lighting_only",   "type": "IMAGE", "links": [3, 19]}\n      ],\n      "properties": {"Node name for S&R": "STAGELIGHT"},\n      "widgets_values": [-0.5, 1.0, 2.0, 1.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.24, 1.0, 0.1, 0.0, 0.7, 0.2, 0.1, 2.9, 0.0, 0.3, 1.0, 1.0, 1.0, true, false, 0.3, 16.0, 0.0, 0.0, 0.0, 60.0, 0.7, 0, true, 0.0, 0.5, "[{\\"id\\":\\"L1\\",\\"name\\":\\"Key\\",\\"enabled\\":true,\\"light_x\\":-0.5,\\"light_y\\":1.0,\\"light_z\\":2.0,\\"target_x\\":0,\\"target_y\\":0,\\"target_z\\":0,\\"intensity\\":1.0,\\"falloff\\":5.0,\\"fcv\\":2.0,\\"light_r\\":1,\\"light_g\\":1,\\"light_b\\":1,\\"spec_strength\\":0.3,\\"spec_shiny\\":16,\\"light_type\\":0,\\"fixed_intensity\\":true,\\"light_size\\":0,\\"cone_angle\\":60,\\"cone_soft\\":0.7}]", 0.05, 0.0, false, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]\n    },\n    {\n      "id": 3,\n      "type": "CheckpointLoaderSimple",\n      "pos": [40, 600],\n      "size": [320, 98],\n      "flags": {},\n      "order": 2,\n      "mode": 0,\n      "outputs": [\n        {"name": "MODEL", "type": "MODEL", "links": [4]},\n        {"name": "CLIP",  "type": "CLIP",  "links": [5, 6]},\n        {"name": "VAE",   "type": "VAE",   "links": [7, 8, 9, 10]}\n      ],\n      "properties": {"Node name for S&R": "CheckpointLoaderSimple"},\n      "widgets_values": ["juggernaut_reborn.safetensors"],\n      "title": "Load SD 1.5 (Juggernaut Reborn / RealisticVision / DreamShaper)"\n    },\n    {\n      "id": 4,\n      "type": "LoadAndApplyICLightUnet",\n      "pos": [400, 600],\n      "size": [320, 82],\n      "flags": {},\n      "order": 3,\n      "mode": 0,\n      "inputs": [{"name": "model", "type": "MODEL", "link": 4}],\n      "outputs": [{"name": "MODEL", "type": "MODEL", "links": [11]}],\n      "properties": {"Node name for S&R": "LoadAndApplyICLightUnet"},\n      "widgets_values": ["iclight_sd15_fc.safetensors"],\n      "title": "Apply IC-Light (fc = foreground-conditioned)"\n    },\n    {\n      "id": 5,\n      "type": "CLIPTextEncode",\n      "pos": [40, 720],\n      "size": [320, 100],\n      "flags": {},\n      "order": 4,\n      "mode": 0,\n      "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],\n      "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [12]}],\n      "properties": {"Node name for S&R": "CLIPTextEncode"},\n      "widgets_values": ["cinematic lighting, photorealistic, sharp focus, detailed skin, sharp fabric weave, high quality"],\n      "title": "POSITIVE - describe lighting style and material quality"\n    },\n    {\n      "id": 6,\n      "type": "CLIPTextEncode",\n      "pos": [40, 840],\n      "size": [320, 100],\n      "flags": {},\n      "order": 5,\n      "mode": 0,\n      "inputs": [{"name": "clip", "type": "CLIP", "link": 6}],\n      "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [13]}],\n      "properties": {"Node name for S&R": "CLIPTextEncode"},\n      "widgets_values": ["flat lighting, washed out, blurry, low quality, distorted features, plastic look"],\n      "title": "NEGATIVE"\n    },\n    {\n      "id": 7,\n      "type": "VAEEncode",\n      "pos": [760, 200],\n      "size": [220, 46],\n      "flags": {},\n      "order": 6,\n      "mode": 0,\n      "inputs": [\n        {"name": "pixels", "type": "IMAGE", "link": 2},\n        {"name": "vae",    "type": "VAE",   "link": 7}\n      ],\n      "outputs": [{"name": "LATENT", "type": "LATENT", "links": [14]}],\n      "properties": {"Node name for S&R": "VAEEncode"},\n      "title": "Encode SUBJECT (preserves identity)"\n    },\n    {\n      "id": 8,\n      "type": "VAEEncode",\n      "pos": [760, 280],\n      "size": [220, 46],\n      "flags": {},\n      "order": 7,\n      "mode": 0,\n      "inputs": [\n        {"name": "pixels", "type": "IMAGE", "link": 3},\n        {"name": "vae",    "type": "VAE",   "link": 8}\n      ],\n      "outputs": [{"name": "LATENT", "type": "LATENT", "links": [15]}],\n      "properties": {"Node name for S&R": "VAEEncode"},\n      "title": "Encode LIGHTING_ONLY (lighting direction bias)"\n    },\n    {\n      "id": 9,\n      "type": "ICLightConditioning",\n      "pos": [1020, 200],\n      "size": [340, 130],\n      "flags": {},\n      "order": 8,\n      "mode": 0,\n      "inputs": [\n        {"name": "positive",       "type": "CONDITIONING", "link": 12},\n        {"name": "negative",       "type": "CONDITIONING", "link": 13},\n        {"name": "vae",            "type": "VAE",          "link": 9},\n        {"name": "foreground",     "type": "LATENT",       "link": 14},\n        {"name": "opt_background", "type": "LATENT",       "link": null}\n      ],\n      "outputs": [\n        {"name": "positive",     "type": "CONDITIONING", "links": [16]},\n        {"name": "negative",     "type": "CONDITIONING", "links": [17]},\n        {"name": "empty_latent", "type": "LATENT",       "links": null}\n      ],\n      "properties": {"Node name for S&R": "ICLightConditioning"},\n      "widgets_values": [0.18215],\n      "title": "IC-Light Conditioning (multiplier 0.18215 - documented VAE scale)"\n    },\n    {\n      "id": 10,\n      "type": "KSampler",\n      "pos": [1400, 200],\n      "size": [320, 262],\n      "flags": {},\n      "order": 9,\n      "mode": 0,\n      "inputs": [\n        {"name": "model",        "type": "MODEL",        "link": 11},\n        {"name": "positive",     "type": "CONDITIONING", "link": 16},\n        {"name": "negative",     "type": "CONDITIONING", "link": 17},\n        {"name": "latent_image", "type": "LATENT",       "link": 15}\n      ],\n      "outputs": [{"name": "LATENT", "type": "LATENT", "links": [18]}],\n      "properties": {"Node name for S&R": "KSampler"},\n      "widgets_values": [42, "randomize", 25, 2.0, "dpmpp_2m", "karras", 0.5],\n      "title": "KSampler (latent_image = lighting bias)"\n    },\n    {\n      "id": 11,\n      "type": "VAEDecode",\n      "pos": [1760, 200],\n      "size": [220, 46],\n      "flags": {},\n      "order": 10,\n      "mode": 0,\n      "inputs": [\n        {"name": "samples", "type": "LATENT", "link": 18},\n        {"name": "vae",     "type": "VAE",    "link": 10}\n      ],\n      "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [20]}],\n      "properties": {"Node name for S&R": "VAEDecode"}\n    },\n    {\n      "id": 12,\n      "type": "PreviewImage",\n      "pos": [2000, 200],\n      "size": [320, 380],\n      "flags": {},\n      "order": 11,\n      "mode": 0,\n      "inputs": [{"name": "images", "type": "IMAGE", "link": 20}],\n      "properties": {"Node name for S&R": "PreviewImage"},\n      "title": "FINAL: IC-Light relit output"\n    },\n    {\n      "id": 13,\n      "type": "PreviewImage",\n      "pos": [760, 380],\n      "size": [240, 240],\n      "flags": {},\n      "order": 12,\n      "mode": 0,\n      "inputs": [{"name": "images", "type": "IMAGE", "link": 19}],\n      "properties": {"Node name for S&R": "PreviewImage"},\n      "title": "STAGELIGHT lighting_only (gradient driving IC-Light)"\n    }\n  ],\n  "links": [\n    [1,  1, 0, 2,  0, "IMAGE"],\n    [2,  2, 0, 7,  0, "IMAGE"],\n    [3,  2, 7, 8,  0, "IMAGE"],\n    [4,  3, 0, 4,  0, "MODEL"],\n    [5,  3, 1, 5,  0, "CLIP"],\n    [6,  3, 1, 6,  0, "CLIP"],\n    [7,  3, 2, 7,  1, "VAE"],\n    [8,  3, 2, 8,  1, "VAE"],\n    [9,  3, 2, 9,  2, "VAE"],\n    [10, 3, 2, 11, 1, "VAE"],\n    [11, 4, 0, 10, 0, "MODEL"],\n    [12, 5, 0, 9,  0, "CONDITIONING"],\n    [13, 6, 0, 9,  1, "CONDITIONING"],\n    [14, 7, 0, 9,  3, "LATENT"],\n    [15, 8, 0, 10, 3, "LATENT"],\n    [16, 9, 0, 10, 1, "CONDITIONING"],\n    [17, 9, 1, 10, 2, "CONDITIONING"],\n    [18, 10, 0, 11, 0, "LATENT"],\n    [19, 2, 7, 13, 0, "IMAGE"],\n    [20, 11, 0, 12, 0, "IMAGE"]\n  ],\n  "groups": [\n    {"title": "1. Load + STAGELIGHT", "bounding": [20, 160, 720, 460], "color": "#3f789e", "font_size": 20},\n    {"title": "2. SD 1.5 + IC-Light + prompts", "bounding": [20, 580, 720, 380], "color": "#a1309b", "font_size": 20},\n    {"title": "3. Encode + IC-Light conditioning", "bounding": [740, 160, 660, 480], "color": "#629e3f", "font_size": 20},\n    {"title": "4. Sample + decode", "bounding": [1380, 160, 960, 420], "color": "#9e6f3f", "font_size": 20}\n  ],\n  "config": {},\n  "extra": {\n    "info": "STAGELIGHT v0.9.5 IC-Light relight workflow. Key wiring: image -> foreground (preserves subject identity), lighting_only -> KSampler latent_image (directional lighting bias). Requires SD 1.5 checkpoint + iclight_sd15_fc.safetensors + kijai\'s ComfyUI-IC-Light extension. Multiplier 0.18215 is the documented VAE scale factor."\n  },\n  "version": 0.4\n}\n'

# ══════════════════════════════════════════════════════════════
#  CONFIG + MANIFEST
# ══════════════════════════════════════════════════════════════
def load_config():
    d = {"comfyui_path": ""}
    if CONFIG_FILE.exists():
        try: d.update(json.loads(CONFIG_FILE.read_text()))
        except: pass
    return d

def save_config(c):
    CONFIG_FILE.write_text(json.dumps(c, indent=2))

def load_manifest():
    d = {"plugin_installed": False, "plugin_path": ""}
    if MANIFEST_FILE.exists():
        try: d.update(json.loads(MANIFEST_FILE.read_text()))
        except: pass
    return d

def save_manifest(m):
    m["updated_at"] = datetime.now().isoformat()
    MANIFEST_FILE.write_text(json.dumps(m, indent=2))

def find_comfyui_python(cp):
    cp = Path(cp)
    for c in [
        cp / "python_embeded" / "python.exe",
        cp / "venv"  / "Scripts" / "python.exe",
        cp / ".venv" / "Scripts" / "python.exe",
    ]:
        if c.exists(): return str(c)
    return sys.executable

def scan(comfyui_path):
    r = {"valid": False, "plugin_installed": False, "plugin_version": ""}
    if not comfyui_path: return r
    cp = Path(comfyui_path)
    if not (cp / "main.py").exists(): return r
    r["valid"] = True
    dst = cp / "custom_nodes" / "stagelight"
    if dst.exists():
        r["plugin_installed"] = True
        vf = dst / "__init__.py"
        if vf.exists():
            # Read version from print statement
            for line in vf.read_text().splitlines():
                if "v0." in line:
                    import re
                    m = re.search(r"v(\d+\.\d+\.\d+)", line)
                    if m: r["plugin_version"] = m.group(1)
    return r

# ══════════════════════════════════════════════════════════════
#  WORKER
# ══════════════════════════════════════════════════════════════
class Worker(QThread):
    log      = Signal(str)
    progress = Signal(int)
    done     = Signal(bool)

    def __init__(self, mode, comfyui_path):
        super().__init__()
        self._mode = mode
        self._cp   = comfyui_path

    def run(self):
        if self._mode == "install":   self._install()
        elif self._mode == "uninstall": self._uninstall()

    # ── Helpers (v0.9.0d) ───────────────────────────────────────────────
    def _find_all_stagelight_installs(self, custom_nodes_dir):
        """Scan custom_nodes/ for ANY folder matching stagelight (case-insensitive).
        Returns list of paths. Catches old installs with different casing or names.
        """
        found = []
        if not custom_nodes_dir.exists():
            return found
        for entry in custom_nodes_dir.iterdir():
            if not entry.is_dir():
                continue
            name_lc = entry.name.lower()
            # Match common variants
            if name_lc in ("stagelight", "stage_light", "stage-light") or name_lc.startswith("stagelight"):
                found.append(entry)
        return found

    def _force_remove(self, path):
        """Remove a path with aggressive Windows file-lock handling.
        Returns (success: bool, error_message: str or None).
        """
        import stat
        import os as _os
        if not path.exists():
            return (True, None)

        # First pass: clear __pycache__ folders (these often hold locked .pyc files)
        for pycache in path.rglob("__pycache__"):
            try:
                shutil.rmtree(str(pycache), ignore_errors=True)
            except Exception:
                pass

        def on_rm_error(func, target, exc_info):
            # On Windows, files may be readonly or have permissions blocking deletion.
            # Try fixing permissions and retrying.
            try:
                _os.chmod(target, stat.S_IWRITE)
                func(target)
            except Exception:
                pass

        try:
            shutil.rmtree(str(path), onerror=on_rm_error)
        except Exception as e:
            return (False, str(e))

        # Verify it's actually gone (rmtree can silently fail on locks)
        if path.exists():
            return (False, f"Folder still exists after rmtree: {path}. "
                          "Likely cause: ComfyUI is still running and has Python files loaded. "
                          "Close ComfyUI completely (kill the server in terminal too) and try again.")
        return (True, None)

    def _verify_installed_file(self, dst):
        """Read back the installed node.py and confirm it has the v0.9.0d content.
        Returns (ok: bool, message: str).
        """
        node_py = dst / "node.py"
        if not node_py.exists():
            return (False, "node.py is missing after install")
        try:
            content = node_py.read_text(encoding="utf-8")
        except Exception as e:
            return (False, f"Could not read node.py: {e}")

        # Look for the critical mask_falloff line (with min=0.0)
        if '"mask_falloff":      f( 0.0,  0.0,' not in content:
            return (False, "mask_falloff fix not present in installed node.py. "
                          "The file on disk is older than v0.9.0c. Reinstall may not have written cleanly.")
        # Look for version marker
        if "STAGELIGHT v0.9.5" not in content and "v0.9.0" not in content:
            return (False, "Installed node.py does not contain v0.9.0 version marker.")
        return (True, "Verified: mask_falloff fix is present in installed node.py")

    def _install(self):
        try:
            cp  = Path(self._cp)
            custom_nodes = cp / "custom_nodes"
            dst = custom_nodes / "stagelight"

            # === Step 1: Aggressive cleanup ===
            # Scan for ANY stagelight install variant and remove them all.
            # This catches case-variant folders, leftover folders the manifest doesn't know about, etc.
            self.log.emit("=== Step 1: Scanning for existing STAGELIGHT installs ===")
            existing = self._find_all_stagelight_installs(custom_nodes)
            if existing:
                self.log.emit(f"Found {len(existing)} existing install(s):")
                for p in existing:
                    self.log.emit(f"  - {p}")
                    ok, err = self._force_remove(p)
                    if ok:
                        self.log.emit(f"    Removed.")
                    else:
                        self.log.emit(f"    [ERROR] {err}")
                        self.log.emit("\n*** INSTALL ABORTED ***")
                        self.log.emit("Close ComfyUI completely (including any background processes),")
                        self.log.emit("then try Install again.")
                        self.done.emit(False)
                        return
            else:
                self.log.emit("No existing install found - fresh install.")
            self.progress.emit(20)

            # === Step 2: Create fresh folder structure ===
            self.log.emit("\n=== Step 2: Creating folder structure ===")
            (dst / "web").mkdir(parents=True, exist_ok=True)
            (dst / "workflows").mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Created: {dst}")
            self.progress.emit(40)

            # === Step 3: Write every file ===
            self.log.emit("\n=== Step 3: Writing files ===")
            files_written = 0
            for rel, file_content in PLUGIN_FILES.items():
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(file_content, encoding="utf-8")
                files_written += 1
                self.log.emit(f"  wrote: {rel}")
            self.log.emit(f"Total files written: {files_written}")
            self.progress.emit(60)

            # === Step 4: Verify install ===
            self.log.emit("\n=== Step 4: Verifying installed content ===")
            ok, msg = self._verify_installed_file(dst)
            if ok:
                self.log.emit(f"  OK {msg}")
            else:
                self.log.emit(f"  [WARN] Verification: {msg}")
                self.log.emit("  Install will continue, but file may not be current.")
            self.progress.emit(70)

            # === Step 5: Install Python deps ===
            self.log.emit("\n=== Step 5: Installing Python dependencies ===")
            py = find_comfyui_python(str(cp))
            if py:
                self.log.emit(f"Using: {py}")
                deps = [
                    ("transformers",    "transformers"),
                    ("rembg",           "rembg"),
                    ("onnxruntime",     "onnxruntime"),
                    ("controlnet-aux",  "controlnet_aux"),
                    ("timm",            "timm"),
                    ("scipy",           "scipy"),
                ]
                for i, (pip_name, _import_name) in enumerate(deps):
                    self.log.emit(f"  pip install {pip_name}...")
                    try:
                        proc = subprocess.run(
                            [py, "-m", "pip", "install", pip_name, "--quiet"],
                            capture_output=True, text=True, timeout=300
                        )
                        if proc.returncode == 0:
                            self.log.emit(f"    OK {pip_name}")
                        else:
                            self.log.emit(f"    [WARN] {pip_name}: {proc.stderr[:200]}")
                    except Exception as pe:
                        self.log.emit(f"    [WARN] {pip_name}: {pe}")
                    self.progress.emit(70 + int(20 * (i+1) / len(deps)))
            else:
                self.log.emit("[WARN] Could not find ComfyUI Python - install deps manually:")
                self.log.emit("       pip install transformers rembg onnxruntime")
            self.progress.emit(90)

            # === Step 6: Save manifest ===
            m = load_manifest()
            m["plugin_installed"] = True
            m["plugin_path"]      = str(dst)
            m["installed_at"]     = datetime.now().isoformat()
            m["installed_version"] = "0.9.0d"
            save_manifest(m)
            self.progress.emit(100)

            self.log.emit("\n=== Install complete ===")
            self.log.emit(f"Installed to: {dst}")
            self.log.emit("\nIMPORTANT: Fully restart ComfyUI now (close the server, then restart).")
            self.log.emit("Then hard-refresh your browser (Ctrl+Shift+R).")
            self.log.emit("Confirm '[STAGELIGHT] v0.9.5 loaded.' appears in the ComfyUI console.")
            self.done.emit(True)

        except Exception as e:
            self.log.emit(f"\n[ERROR] {e}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.done.emit(False)

    def _uninstall(self):
        try:
            cp = Path(self._cp) if self._cp else None
            self.log.emit("=== Aggressive uninstall ===")

            # Try manifest first
            m = load_manifest()
            manifest_path = m.get("plugin_path", "")
            if manifest_path:
                self.log.emit(f"Manifest says: {manifest_path}")

            # Scan custom_nodes for ALL stagelight variants (don't trust manifest)
            removed_count = 0
            errors = []
            if cp:
                custom_nodes = cp / "custom_nodes"
                found = self._find_all_stagelight_installs(custom_nodes)
                if found:
                    self.log.emit(f"Found {len(found)} stagelight folder(s) to remove:")
                    for p in found:
                        self.log.emit(f"  - {p}")
                        ok, err = self._force_remove(p)
                        if ok:
                            self.log.emit(f"    Removed.")
                            removed_count += 1
                        else:
                            self.log.emit(f"    [ERROR] {err}")
                            errors.append(err)
                else:
                    self.log.emit("No stagelight folders found in custom_nodes/.")
            else:
                self.log.emit("[WARN] No ComfyUI path - skipping folder scan.")
                # Fall back to manifest path
                if manifest_path and Path(manifest_path).exists():
                    ok, err = self._force_remove(Path(manifest_path))
                    if ok:
                        self.log.emit(f"Removed: {manifest_path}")
                        removed_count += 1
                    else:
                        self.log.emit(f"[ERROR] {err}")
                        errors.append(err)

            # Wipe manifest + config
            for f in [MANIFEST_FILE, CONFIG_FILE]:
                if f.exists():
                    try:
                        f.unlink()
                        self.log.emit(f"Removed {f.name}")
                    except Exception as e:
                        self.log.emit(f"[WARN] Could not remove {f.name}: {e}")
            self.progress.emit(100)

            if errors:
                self.log.emit("\n*** UNINSTALL HAD ERRORS ***")
                self.log.emit("Some folders could not be removed. Most common cause:")
                self.log.emit("ComfyUI is still running. Close it completely and try again.")
                self.done.emit(False)
            else:
                self.log.emit(f"\nUninstall complete. Removed {removed_count} folder(s).")
                self.log.emit("Restart ComfyUI.")
                self.done.emit(True)
        except Exception as e:
            self.log.emit(f"\n[ERROR] {e}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.done.emit(False)

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
STYLE = """
QWidget { background:#111; color:#ccc; font-family:Consolas; }
QLabel#heading { color:#fff; font-size:18px; font-weight:bold; letter-spacing:4px; }
QLabel#sub     { color:#e8a020; font-size:9px; letter-spacing:4px; }
QLabel#section { color:#999; font-size:9px; letter-spacing:2px; }
QLabel#path_display {
    color:#aaa; font-size:9px; background:#0a0a0a;
    border:1px solid #1e1e1e; border-radius:3px; padding:5px 8px;
}
QLabel#st_ok      { color:#44dd88; font-size:10px; }
QLabel#st_missing { color:#cc4444; font-size:10px; }
QLabel#st_warn    { color:#e8a020; font-size:10px; }
QProgressBar {
    background:#0a0a0a; border:1px solid #1e1e1e;
    border-radius:2px; height:4px; color:transparent;
}
QProgressBar::chunk {
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #a05010, stop:1 #f0a030);
    border-radius:2px;
}
QTextEdit {
    background:#0a0a0a; color:#44bb77; font-size:10px;
    border:1px solid #1e1e1e; border-radius:3px; padding:6px;
}
QScrollBar:vertical { background:#0a0a0a; width:5px; border-radius:3px; }
QScrollBar::handle:vertical { background:#2a2a2a; border-radius:3px; min-height:16px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QPushButton#btn_install {
    background:#6b3aa8; color:#ffffff; font-size:11px; font-weight:bold;
    letter-spacing:2px; border:1px solid #8855cc; border-radius:3px; padding:9px 22px;
}
QPushButton#btn_install:hover    { background:#7e4cc4; border-color:#a070dd; }
QPushButton#btn_install:disabled { background:#2a2a2a; color:#999; border-color:#3a3a3a; }
QPushButton#btn_launch {
    background:#0c3820; color:#44dd88; font-size:11px; font-weight:bold;
    letter-spacing:2px; border:1px solid #1a9050; border-radius:3px; padding:9px 22px;
}
QPushButton#btn_launch:hover    { background:#14502c; }
QPushButton#btn_launch:disabled { background:#2a2a2a; color:#999; border-color:#3a3a3a; }
QPushButton#btn_uninstall {
    background:transparent; color:#bb6666; font-size:11px; letter-spacing:1px;
    border:1px solid #884444; border-radius:3px; padding:8px 18px;
}
QPushButton#btn_uninstall:hover    { border-color:#dd6666; color:#ff8888; }
QPushButton#btn_uninstall:disabled { color:#888; border-color:#444; }
QPushButton#btn_kill {
    background:#3a1414; color:#ee9966; font-size:11px; letter-spacing:1px; font-weight:bold;
    border:1px solid #663030; border-radius:3px; padding:8px 18px;
}
QPushButton#btn_kill:hover    { background:#5a2020; color:#ffbb88; border-color:#884040; }
QPushButton#btn_kill:disabled { background:#2a1818; color:#888; border-color:#443030; }
QPushButton#btn_browse {
    background:transparent; color:#ccc; font-size:10px; letter-spacing:1px;
    border:1px solid #555; border-radius:3px; padding:6px 14px;
}
QPushButton#btn_browse:hover { border-color:#999; color:#fff; }
"""

class Installer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.worker = None
        self._drag  = None
        self._scan  = {}
        self._build_ui()
        QTimer.singleShot(300, self._rescan)

    def _build_ui(self):
        self.setWindowTitle(f"STAGELIGHT v{APP_VER}")
        self.setFixedSize(580, 520)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setStyleSheet(STYLE)

        root = QWidget(self)
        root.setGeometry(0, 0, 580, 520)
        root.setStyleSheet("background:#111; border:1px solid #222;")
        self.setCentralWidget(root)

        lay = QVBoxLayout(root)
        lay.setContentsMargins(28, 24, 28, 20)
        lay.setSpacing(14)

        # Header
        lay.addWidget(self._lbl(APP_NAME, "heading"))
        lay.addWidget(self._lbl(f"COMFYUI PLUGIN  ·  v{APP_VER}", "sub"))
        lay.addWidget(self._div())

        # ComfyUI path
        lay.addWidget(self._lbl("COMFYUI FOLDER", "section"))
        pr = QHBoxLayout(); pr.setSpacing(8)
        self.path_lbl = QLabel(
            self.config.get("comfyui_path") or "Click BROWSE to locate your ComfyUI folder")
        self.path_lbl.setObjectName("path_display")
        self.path_lbl.setMinimumHeight(28)
        pr.addWidget(self.path_lbl, 1)
        b = QPushButton("BROWSE"); b.setObjectName("btn_browse"); b.setFixedWidth(80)
        b.setToolTip("Select your ComfyUI folder — the one that contains the custom_nodes directory.")
        b.clicked.connect(self._browse); pr.addWidget(b)
        lay.addLayout(pr)
        lay.addWidget(self._div())

        # Status
        lay.addWidget(self._lbl("STATUS", "section"))
        self.st_plugin = QLabel("● Plugin    checking...")
        self.st_plugin.setObjectName("st_missing")
        lay.addWidget(self.st_plugin)
        lay.addWidget(self._div())

        # Progress
        self.prog = QProgressBar()
        self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setTextVisible(False); self.prog.setFixedHeight(4)
        lay.addWidget(self.prog)

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        lay.addWidget(self.log, 1)

        # v0.9.5 release: seed the existing log widget with requirements info on startup.
        # Pure text into a widget that already exists - no layout or logic change.
        self.log.append(
            "STAGELIGHT v0.9.5  -  Created by DeJuan McCoy\n"
            "---------------------------------------------\n"
            "RECOMMENDED PATH: IC-Light (photoreal relighting)\n"
            "---------------------------------------------\n"
            "Required downloads (~3.7GB total, one time):\n"
            "\n"
            "1. SD 1.5 photoreal checkpoint (~2GB)\n"
            "   -> ComfyUI/models/checkpoints/\n"
            "   Juggernaut Reborn (recommended):\n"
            "   https://huggingface.co/HyperX-Sentience/Juggernaut-Reborn/\n"
            "       resolve/main/juggernaut_reborn.safetensors\n"
            "\n"
            "2. IC-Light fc model (~1.7GB)\n"
            "   -> ComfyUI/models/unet/\n"
            "   https://huggingface.co/lllyasviel/ic-light/\n"
            "       resolve/main/iclight_sd15_fc.safetensors\n"
            "\n"
            "3. kijai's ComfyUI-IC-Light extension\n"
            "   Install via ComfyUI Manager -> Custom Nodes Manager\n"
            "   -> search 'IC-Light' -> install kijai's version\n"
            "\n"
            "TIP: install_dependencies.bat (in this folder) can pip-install the\n"
            "Python packages STAGELIGHT needs. Model files still need manual download.\n"
            "\n"
            "AFTER INSTALL: restart ComfyUI fully, then hard-refresh your browser\n"
            "(Ctrl+Shift+R). If the panel looks stale or images don't load, it's\n"
            "almost always browser cache - hard-refresh fixes it.\n"
            "---------------------------------------------\n"
            "Set your ComfyUI folder above, then click INSTALL.\n"
        )

        # Buttons
        br = QHBoxLayout(); br.setSpacing(10)
        self.btn_uninstall = QPushButton("⊗  UNINSTALL")
        self.btn_uninstall.setObjectName("btn_uninstall")
        self.btn_uninstall.setFixedWidth(120)
        self.btn_uninstall.setEnabled(False)
        self.btn_uninstall.setToolTip("Remove STAGELIGHT from ComfyUI/custom_nodes. Safe to use — only affects STAGELIGHT files.")
        self.btn_uninstall.clicked.connect(self._uninstall)
        br.addWidget(self.btn_uninstall)
        # v0.9.0d: Kill ComfyUI button - finds and terminates running ComfyUI processes
        # so the install/uninstall doesn't fight with locked files
        self.btn_kill = QPushButton("⏻  KILL COMFYUI")
        self.btn_kill.setObjectName("btn_kill")
        self.btn_kill.setFixedWidth(140)
        self.btn_kill.setToolTip("Force-quit any running ComfyUI process. Use before INSTALL or UNINSTALL to avoid locked-file errors on Windows.")
        self.btn_kill.clicked.connect(self._kill_comfyui)
        br.addWidget(self.btn_kill)
        br.addStretch()
        self.btn_install = QPushButton("▶  INSTALL")
        self.btn_install.setObjectName("btn_install")
        self.btn_install.setFixedWidth(120)
        self.btn_install.setToolTip("Install STAGELIGHT into ComfyUI/custom_nodes. After install: restart ComfyUI and hard-refresh the browser (Ctrl+Shift+R).")
        self.btn_install.clicked.connect(self._install)
        br.addWidget(self.btn_install)
        self.btn_launch = QPushButton("◉  OPEN COMFYUI")
        self.btn_launch.setObjectName("btn_launch")
        self.btn_launch.setFixedWidth(160)
        self.btn_launch.setEnabled(False)
        self.btn_launch.setToolTip("Start ComfyUI and open it in your browser. If STAGELIGHT looks stale after a re-install, hard-refresh: Ctrl+Shift+R.")
        self.btn_launch.clicked.connect(self._launch)
        br.addWidget(self.btn_launch)
        lay.addLayout(br)

        # Window controls
        for sym, fn, x in [("✕", self.close, 552), ("—", self.showMinimized, 528)]:
            b = QPushButton(sym, root)
            b.setGeometry(x, 8, 22, 18)
            b.setStyleSheet(
                "QPushButton{background:transparent;color:#888;border:none;font-size:11px;}"
                "QPushButton:hover{color:#fff;}")
            b.clicked.connect(fn)

        # v8: Donate button - absolute positioning beside window controls, opens
        # Buy Me a Coffee in browser via webbrowser stdlib. No layout/logic change.
        self.btn_donate = QPushButton("☕  Buy me a coffee", root)
        self.btn_donate.setObjectName("btn_donate")
        self.btn_donate.setGeometry(380, 6, 140, 22)
        self.btn_donate.setToolTip("Support STAGELIGHT - opens Buy Me a Coffee in your browser")
        self.btn_donate.setStyleSheet(
            "QPushButton#btn_donate{"
            "background:#FFDD00;color:#000;font-size:10px;font-weight:bold;"
            "letter-spacing:1px;border:1px solid #c0a000;border-radius:11px;padding:0 10px;}"
            "QPushButton#btn_donate:hover{background:#FFE633;border-color:#806800;}")
        self.btn_donate.clicked.connect(
            lambda: __import__('webbrowser').open("https://buymeacoffee.com/pixlshift8d"))

    def _lbl(self, t, n):
        l = QLabel(t); l.setObjectName(n); return l

    def _div(self):
        f = QWidget(); f.setFixedHeight(1)
        f.setStyleSheet("background:#1e1e1e;"); return f

    def _rescan(self):
        self._scan = scan(self.config.get("comfyui_path", ""))
        self._render()

    def _render(self):
        r = self._scan
        if not r.get("valid"):
            self.st_plugin.setText("● Plugin    set ComfyUI path first")
            self.st_plugin.setObjectName("st_warn")
        elif r["plugin_installed"]:
            v = r["plugin_version"] or "?"
            self.st_plugin.setText(f"● Plugin    installed  v{v}")
            self.st_plugin.setObjectName("st_ok")
        else:
            self.st_plugin.setText("● Plugin    not installed")
            self.st_plugin.setObjectName("st_missing")
        self.st_plugin.style().unpolish(self.st_plugin)
        self.st_plugin.style().polish(self.st_plugin)

        valid     = r.get("valid", False)
        installed = r.get("plugin_installed", False)
        self.btn_launch.setEnabled(valid)
        self.btn_uninstall.setEnabled(installed)
        self.btn_install.setText("↺  REINSTALL" if installed else "▶  INSTALL")

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select ComfyUI Folder", str(Path.home()))
        if not path: return
        self.config["comfyui_path"] = path
        save_config(self.config)
        self.path_lbl.setText(path)
        self._rescan()

    def _install(self):
        if not self._scan.get("valid"):
            self._log("Set a valid ComfyUI path first."); return
        self._busy(True)
        self.worker = Worker("install", self.config.get("comfyui_path", ""))
        self.worker.log.connect(self._log)
        self.worker.progress.connect(self.prog.setValue)
        self.worker.done.connect(self._done)
        self.worker.start()

    def _uninstall(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Uninstall")
        msg.setText("Remove ComfyUI/custom_nodes/stagelight/ ?\n\nContinue?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setStyleSheet(
            "QMessageBox{background:#111;color:#ccc;font-family:Consolas;font-size:10px;}"
            "QPushButton{background:#1a1a1a;color:#ccc;border:1px solid #333;"
            "padding:6px 14px;border-radius:3px;min-width:60px;}"
            "QPushButton:hover{background:#222;}")
        if msg.exec() == QMessageBox.Yes:
            self._busy(True)
            self.worker = Worker("uninstall", self.config.get("comfyui_path", ""))
            self.worker.log.connect(self._log)
            self.worker.progress.connect(self.prog.setValue)
            self.worker.done.connect(self._done)
            self.worker.start()

    def _done(self, ok):
        self._busy(False)
        QTimer.singleShot(300, self._rescan)
        if ok: self.btn_launch.setEnabled(True)

    def _busy(self, busy):
        self.btn_install.setEnabled(not busy)
        self.btn_launch.setEnabled(not busy)
        self.btn_uninstall.setEnabled(not busy)
        self.btn_kill.setEnabled(not busy)
        if busy: self.prog.setValue(0)

    def _launch(self):
        cp = self.config.get("comfyui_path", "")
        if not cp: return
        py = find_comfyui_python(cp)
        self._log("Starting ComfyUI...")
        import subprocess, webbrowser, threading, time, socket
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        subprocess.Popen([py, "main.py", "--listen"], cwd=cp, creationflags=flags)
        def wait():
            for _ in range(30):
                time.sleep(1)
                try:
                    s = socket.socket(); s.settimeout(1)
                    s.connect(("127.0.0.1", COMFY_PORT)); s.close()
                    self._log("ComfyUI ready — opening browser...")
                    webbrowser.open(f"http://127.0.0.1:{COMFY_PORT}")
                    return
                except: pass
            self._log("Open http://127.0.0.1:8188 manually.")
        threading.Thread(target=wait, daemon=True).start()

    def _find_comfyui_processes(self):
        """Find all running ComfyUI processes. Returns list of (pid, cmdline) tuples.
        Uses wmic on Windows (built-in, no extra deps).
        """
        import subprocess
        procs = []
        try:
            # WMIC outputs process info as CSV. Filter for python.exe processes.
            result = subprocess.run(
                ["wmic", "process", "where", "name='python.exe'",
                 "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("Node"):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                # Format: Node,CommandLine,ProcessId
                cmdline = ",".join(parts[1:-1])  # CommandLine may contain commas
                pid_str = parts[-1].strip()
                if not pid_str.isdigit():
                    continue
                pid = int(pid_str)
                # Filter for ComfyUI processes
                cmd_lower = cmdline.lower()
                if "main.py" in cmd_lower and ("comfyui" in cmd_lower or "comfy" in cmd_lower):
                    procs.append((pid, cmdline))
                elif "comfyui" in cmd_lower:  # broader fallback
                    procs.append((pid, cmdline))
        except subprocess.TimeoutExpired:
            self._log("[WARN] Process scan timed out")
        except FileNotFoundError:
            self._log("[WARN] wmic not available (non-Windows system)")
        except Exception as e:
            self._log(f"[WARN] Process scan failed: {e}")
        return procs

    def _kill_comfyui(self):
        """Kill any running ComfyUI processes. This prevents file locks during install/uninstall."""
        import subprocess
        self._log("\n=== Searching for ComfyUI processes ===")
        procs = self._find_comfyui_processes()
        if not procs:
            self._log("No running ComfyUI processes found.")
            msg = QMessageBox(self)
            msg.setWindowTitle("Kill ComfyUI")
            msg.setText("No running ComfyUI processes were found.\n\n"
                       "If ComfyUI is running but not detected, close it manually "
                       "via its terminal window or Task Manager.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setStyleSheet(
                "QMessageBox{background:#111;color:#ccc;font-family:Consolas;font-size:10px;}"
                "QPushButton{background:#1a1a1a;color:#ccc;border:1px solid #333;"
                "padding:6px 14px;border-radius:3px;min-width:60px;}"
                "QPushButton:hover{background:#222;}")
            msg.exec()
            return

        # Show what we found and confirm
        self._log(f"Found {len(procs)} ComfyUI process(es):")
        proc_text = []
        for pid, cmd in procs:
            short = cmd[:80] + "..." if len(cmd) > 80 else cmd
            self._log(f"  PID {pid}: {short}")
            proc_text.append(f"PID {pid}: {short}")

        msg = QMessageBox(self)
        msg.setWindowTitle("Kill ComfyUI")
        msg.setText(f"Kill {len(procs)} ComfyUI process(es)?\n\n" +
                   "\n".join(proc_text) +
                   "\n\nThis is needed to safely uninstall/reinstall.")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        msg.setStyleSheet(
            "QMessageBox{background:#111;color:#ccc;font-family:Consolas;font-size:10px;}"
            "QPushButton{background:#1a1a1a;color:#ccc;border:1px solid #333;"
            "padding:6px 14px;border-radius:3px;min-width:60px;}"
            "QPushButton:hover{background:#222;}")
        if msg.exec() != QMessageBox.Yes:
            self._log("Kill cancelled.")
            return

        # Kill them with taskkill /F (force) /T (kill tree to get children too)
        killed = 0
        failed = 0
        for pid, _cmd in procs:
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    self._log(f"  Killed PID {pid}")
                    killed += 1
                else:
                    self._log(f"  [WARN] PID {pid}: {result.stderr.strip()[:120]}")
                    failed += 1
            except Exception as e:
                self._log(f"  [ERROR] PID {pid}: {e}")
                failed += 1

        self._log(f"\nKilled {killed} process(es), {failed} failed.")
        if killed > 0:
            self._log("ComfyUI is now stopped. Safe to uninstall/install.")
            # Disable launch button since ComfyUI is no longer running
            self.btn_launch.setEnabled(False)

    def _log(self, t):
        self.log.append(t)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and self._drag:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, e):
        self._drag = None


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,     QColor("#111111"))
    pal.setColor(QPalette.WindowText, QColor("#cccccc"))
    pal.setColor(QPalette.Base,       QColor("#0a0a0a"))
    pal.setColor(QPalette.Text,       QColor("#cccccc"))
    app.setPalette(pal)
    win = Installer(); win.show()
    sc = app.primaryScreen().geometry()
    win.move((sc.width() - win.width()) // 2, (sc.height() - win.height()) // 2)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
