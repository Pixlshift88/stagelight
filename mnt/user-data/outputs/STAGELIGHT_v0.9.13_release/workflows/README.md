# STAGELIGHT v0.9.5 — AI Cleanup Workflows

Two paths for the cleanup pass after STAGELIGHT relights your image. Pick based on what you need.

---

## Path 1 — Z Image Turbo (fast, generic cleanup)

**File:** `stagelight_zimage_cleanup.json`

Generic img2img cleanup at denoise 0.20. Fast (2-3s), preserves your STAGELIGHT lighting decisions, adds detail and reduces AI artifacts.

**Use this when:** you want the output to look very close to STAGELIGHT's viewport — just slightly cleaner and more polished.

**Don't use when:** you want IC-Light's proper material response (skin sheen, fabric specularity, rim light wrap). Z Image Turbo can't do that — it just denoises.

**Required models:**
- `qwen_3_4b.safetensors` (CLIP)
- `z_image_turbo_bf16.safetensors` (UNet)
- `ae.safetensors` (VAE)

---

## Path 2 — IC-Light (proper material response and lighting wrap)

**File:** `stagelight_iclight_relight.json`

This is the workflow you want if you couldn't get rim lighting from Path 1 and the lighting feels like "outfit tinting" instead of actual light wrapping the subject.

### How it actually works

IC-Light is an SD 1.5 model patched to **understand lighting**. Unlike Z Image Turbo (which is generic denoising), IC-Light is trained on relighting data — it knows that "bright on left + dark on right" means side lighting, and renders it with proper material specularity, color bleeding into shadows, etc.

STAGELIGHT outputs two things IC-Light needs:

| STAGELIGHT output | Feeds into IC-Light as | Why |
|---|---|---|
| `image` (slot 0) | `ICLightConditioning(foreground)` (after VAE encode) | Preserves the subject's identity, features, costume |
| `lighting_only` (slot 7) | `KSampler(latent_image)` (after VAE encode) | Biases the diffusion toward your light direction |

The text prompts describe the **style** of lighting and material quality you want — they don't define direction (the lighting_only image does that).

### Required downloads

1. **An SD 1.5 photoreal checkpoint** (~2GB). Recommended:
   - Juggernaut Reborn — https://civitai.com/models/46422
   - RealisticVision V5.1 — https://civitai.com/models/4201
   - DreamShaper 8 — https://civitai.com/models/4384
   - Drop into `ComfyUI/models/checkpoints/`

2. **The IC-Light fc model** (~1.4GB):
   - `iclight_sd15_fc.safetensors` from https://huggingface.co/lllyasviel/ic-light/tree/main
   - Drop into `ComfyUI/models/unet/`

3. **kijai's ComfyUI-IC-Light extension** (you already have this — visible in the log)

Total: ~3.4GB. Once downloaded, no further downloads needed for subsequent generations.

### Setup steps

1. Open `stagelight_iclight_relight.json` in ComfyUI
2. In the **CheckpointLoaderSimple** node ("Load SD 1.5"), pick your downloaded SD 1.5 checkpoint
3. In the **LoadAndApplyICLightUnet** node, verify it's pointing at `iclight_sd15_fc.safetensors`
4. Place your image in the LoadImage node
5. Light your subject in the STAGELIGHT panel — choose colors, positions, multiple lights as needed
6. Hit ComfyUI's **Run** button (or STAGELIGHT's RENDER then Run)
7. Wait ~5-15 seconds. Result appears in the "FINAL" PreviewImage at the right.

### What you should see

- **lighting_only preview** (PreviewImage labeled "STAGELIGHT lighting_only"): the gradient driving the lighting. Should clearly show where you placed lights.
- **FINAL preview** (top-right): the photoreal relit output. Compare to the viewport. Lighting direction should match. Material response (skin glow, fabric specular, rim light) should be far more realistic than Path 1.

### Tweaking

- **Too much lighting change vs original?** Increase `multiplier` on ICLightConditioning above 0.18215 (try 0.25 or 0.30). This boosts subject preservation.
- **Lighting not strong enough?** Add more lights or boost INTENSITY in STAGELIGHT. The `lighting_only` output is what IC-Light reads — brighter, more directional = stronger relight.
- **Too AI-generic looking?** Tweak the positive prompt to describe specific material qualities ("sharp cotton fabric weave", "weathered leather", etc.). Avoid generic words like "high quality."
- **Want sharper output?** Increase KSampler steps from 25 to 30-40. Slower but cleaner.

### Licensing note

- IC-Light v1 fc/fbc models (this workflow uses fc): Apache 2.0 — commercial use OK
- IC-Light v2 models: non-commercial only — don't use those for paid work

---

## Troubleshooting

**"Cannot find node ICLightConditioning"**
→ Install kijai's ComfyUI-IC-Light via ComfyUI Manager → Custom Nodes Manager → search "IC-Light" → kijai's version.

**"Cannot find checkpoint sd15_checkpoint.safetensors"**
→ The workflow ships with a placeholder filename. Pick your real SD 1.5 file from the dropdown.

**Error: "Return type mismatch... foreground IMAGE != LATENT"**
→ This was a known issue in older IC-Light versions. Update the extension via ComfyUI Manager. In this workflow, all foreground/background inputs are pre-wired as LATENT correctly.

**The output looks blown out / weirdly bright**
→ Check the `lighting_only` preview. If it's mostly white, your STAGELIGHT intensity is too high. Lower it to 0.5–1.5 range so the lighting layer has clear shadows AND highlights.

**The output ignores my lighting direction**
→ The lighting_only image must be CLEAR about direction. If you have 5 lights all over the subject, lighting_only becomes a wash. Try with one strong key light first to verify the wiring works, then layer up.
