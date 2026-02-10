# Standard library
import os
import random

# Third-party
import torch

# Local application
from klien_utils import (
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    get_value_at_index,
    import_custom_nodes,
    output_to_bytes,
)

# ---- ComfyUI bootstrap ----
add_comfyui_directory_to_sys_path()
add_extra_model_paths()
import_custom_nodes()

from nodes import NODE_CLASS_MAPPINGS


# load klien models globally

UNET = None
CLIP = None
VAE  = None


def load_klien_models():
    global UNET, CLIP, VAE

    if UNET is not None:
        return  # already loaded

    print("🔄 Loading Flux models once...", flush=True)

    with torch.inference_mode():
        UNET = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="flux-2-klein-9b-fp8.safetensors",
            weight_dtype="fp8_e4m3fn",
        )

        CLIP = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="qwen_3_8b_fp8mixed.safetensors",
            type="flux2",
            device="default",
        )

        VAE = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(
            vae_name="flux2-vae.safetensors"
        )

    print("✅ Models loaded and cached", flush=True)

load_klien_models()


class FluxKlienMaskedInpaint(object):
    """
    Clean ComfyUI → Python inpaint app
    Supports external mask input
    """

    def __init__(self):

        # ------------------------------------------------
        # Safe loader / utility nodes (OK to cache)
        # ------------------------------------------------
        with torch.inference_mode():
            self.load_image = NODE_CLASS_MAPPINGS["LoadImage"]()
            self.load_mask = NODE_CLASS_MAPPINGS["LoadImageMask"]()

            self.inpaint_crop = NODE_CLASS_MAPPINGS["InpaintCropImproved"]()

            self.unet_loader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            self.clip_loader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
            self.vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]()

            self.text_encode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            self.sampler_select = NODE_CLASS_MAPPINGS["KSamplerSelect"]()


    def run( self,image_path: str, mask_path: str, prompt: str, mask_invert=False):
        """
        image_path : input image
        mask_path  : white = edit, black = keep
        prompt     : inpaint instruction
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(mask_path)

        with torch.inference_mode():

            # ------------------------------------------------
            # Runtime-only nodes (MUST be created per run)
            # ------------------------------------------------
            random_noise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            reference_latent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
            inpaint_condition = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()

            cfg = NODE_CLASS_MAPPINGS["CFGGuider"]()
            scheduler = NODE_CLASS_MAPPINGS["Flux2Scheduler"]()
            sampler = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

            vae_decode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            stitch = NODE_CLASS_MAPPINGS["InpaintStitchImproved"]()

            #with torch.inference_mode():

            # ------------------------------------------------
            # Load image + mask
            # ------------------------------------------------
            image = self.load_image.load_image(image=image_path)

            mask = self.load_mask.load_image(
                image=mask_path,
                channel="red",
            )
            # ------------------------------------------------
            # Prompts
            # ------------------------------------------------
            positive = self.text_encode.encode(
                text=prompt,
                clip=get_value_at_index(CLIP, 0),
            )

            negative = self.text_encode.encode(
                text="",
                clip=get_value_at_index(CLIP, 0),
            )

            # ------------------------------------------------
            # Crop + mask processing
            # ------------------------------------------------
            crop = self.inpaint_crop.inpaint_crop(
                downscale_algorithm="bilinear",
                upscale_algorithm="bicubic",
                preresize=False,
                preresize_mode="ensure minimum and maximum resolution",
                preresize_min_width=1024,
                preresize_min_height=1024,
                preresize_max_width=1024,
                preresize_max_height=1024,
                mask_fill_holes=False,
                mask_expand_pixels=0,
                mask_invert=mask_invert,          # set True if mask is inverted
                mask_blend_pixels=64,
                mask_hipass_filter=0.1,
                extend_for_outpainting=False,
                extend_up_factor=1,
                extend_down_factor=1,
                extend_left_factor=1,
                extend_right_factor=1,
                context_from_mask_extend_factor=1.2,
                output_resize_to_target_size=True,
                output_target_width=1024,
                output_target_height=1024,
                output_padding="32",
                image=get_value_at_index(image, 0),
                mask=get_value_at_index(mask, 0),
                device_mode ="gpu (much faster)"
            )

            # ------------------------------------------------
            # Latents
            # ------------------------------------------------
            latent = NODE_CLASS_MAPPINGS["VAEEncode"]().encode(
                pixels=get_value_at_index(crop, 1),
                vae=get_value_at_index(VAE, 0),
            )

            pos_ref = reference_latent.EXECUTE_NORMALIZED(
                conditioning=get_value_at_index(positive, 0),
                latent=get_value_at_index(latent, 0),
            )

            neg_ref = reference_latent.EXECUTE_NORMALIZED(
                conditioning=get_value_at_index(negative, 0),
                latent=get_value_at_index(latent, 0),
            )

            conditioning = inpaint_condition.encode(
                noise_mask=True,
                positive=get_value_at_index(pos_ref, 0),
                negative=get_value_at_index(neg_ref, 0),
                vae=get_value_at_index(VAE, 0),
                pixels=get_value_at_index(crop, 1),
                mask=get_value_at_index(crop, 2),
            )

            # ------------------------------------------------
            # Sampling
            # ------------------------------------------------
            noise = random_noise.EXECUTE_NORMALIZED(
                noise_seed=36409988569184
            )

            guider = cfg.EXECUTE_NORMALIZED(
                cfg=1,
                model=get_value_at_index(UNET, 0),
                positive=get_value_at_index(conditioning, 0),
                negative=get_value_at_index(conditioning, 1),
            )


            # get image size fix
            pixels = get_value_at_index(crop, 1)

            # pixels = torch tensor [B, H, W, C]
            _, height, width, _ = pixels.shape

            sigmas = scheduler.EXECUTE_NORMALIZED(
                steps=4,
                width=width,
                height=height,
            )

            sample = self.sampler_select.EXECUTE_NORMALIZED(
                sampler_name="euler"
            )
            samples = sampler.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise, 0),
                guider=get_value_at_index(guider, 0),
                sampler=get_value_at_index(sample, 0),
                sigmas=get_value_at_index(sigmas, 0),
                latent_image=get_value_at_index(conditioning, 2),
            )

            decoded = vae_decode.decode(
                samples=get_value_at_index(samples, 0),
                vae=get_value_at_index(VAE, 0),
            )

            final = stitch.inpaint_stitch(
                stitcher=get_value_at_index(crop, 0),
                inpainted_image=get_value_at_index(decoded, 0),
            )

            image_bytes = output_to_bytes(
                get_value_at_index(final, 0)
            )

        return image_bytes

class FluxKlienGenFill(object):
    """
    Flux GenFill / Outpainting
    Args supported:
        top, bottom, left, right
    """

    def __init__(self):

        with torch.inference_mode():

            # -----------------------------
            # Loaders (safe to cache)
            # -----------------------------
            self.load_image = NODE_CLASS_MAPPINGS["LoadImage"]
            self.pad_outpaint = NODE_CLASS_MAPPINGS["ImagePadForOutpaint"]
            self.resize = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]

            self.text_encode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]
            self.sampler_select = NODE_CLASS_MAPPINGS["KSamplerSelect"]

            self.unet_loader = NODE_CLASS_MAPPINGS["UNETLoader"]
            self.clip_loader = NODE_CLASS_MAPPINGS["CLIPLoader"]
            self.vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]

            self.image_composite = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]
            self.outpaint_prompt = "去除图像边缘的深灰色区域，并参考非灰色区域中的元素来扩展和混合图像"


    def run(self, image_path: str, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0):

        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        with torch.inference_mode():

            # -----------------------------
            # Runtime-only nodes
            # -----------------------------
            random_noise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            reference_latent = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
            inpaint_condition = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()

            cfg = NODE_CLASS_MAPPINGS["CFGGuider"]()
            scheduler = NODE_CLASS_MAPPINGS["Flux2Scheduler"]()
            sampler = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            vae_decode = NODE_CLASS_MAPPINGS["VAEDecode"]()

            # -----------------------------
            # Load image
            # -----------------------------
            image = self.load_image().load_image(image=image_path)

            # -----------------------------
            # Pad (genfill geometry)
            # -----------------------------
            padded = self.pad_outpaint().expand_image(
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                feathering=200,
                image=get_value_at_index(image, 0),
            )

            # -----------------------------
            # Resize
            # -----------------------------
            resized = self.resize().resize(
                width=1024,
                height=1024,
                upscale_method="lanczos",
                keep_proportion="resize",
                pad_color="0,0,0",
                crop_position="center",
                divisible_by=8,
                device="cpu",
                image=get_value_at_index(padded, 0),
                mask=get_value_at_index(padded, 1),
                unique_id=random.randint(0, 2 ** 63 - 1)
            )

            # -----------------------------
            # Prompt
            # -----------------------------
            positive = self.text_encode().encode(
                text=self.outpaint_prompt,
                clip=get_value_at_index(CLIP, 0),
            )

            negative = self.text_encode().encode(
                text="",
                clip=get_value_at_index(CLIP, 0),
            )

            # -----------------------------
            # Latents
            # -----------------------------
            latent = NODE_CLASS_MAPPINGS["VAEEncode"]().encode(
                pixels=get_value_at_index(resized, 0),
                vae=get_value_at_index(VAE, 0),
            )

            pos_ref = reference_latent.EXECUTE_NORMALIZED(
                conditioning=get_value_at_index(positive, 0),
                latent=get_value_at_index(latent, 0),
            )

            neg_ref = reference_latent.EXECUTE_NORMALIZED(
                conditioning=get_value_at_index(negative, 0),
                latent=get_value_at_index(latent, 0),
            )

            conditioning = inpaint_condition.encode(
                noise_mask=False,
                positive=get_value_at_index(pos_ref, 0),
                negative=get_value_at_index(neg_ref, 0),
                vae=get_value_at_index(VAE, 0),
                pixels=get_value_at_index(resized, 0),
                mask=get_value_at_index(resized, 3),
            )

            # -----------------------------
            # Sampling
            # -----------------------------
            noise = random_noise.EXECUTE_NORMALIZED(
                noise_seed=random.randint(0, 2**63 - 1)
            )

            guider = cfg.EXECUTE_NORMALIZED(
                cfg=1,
                model=get_value_at_index(UNET, 0),
                positive=get_value_at_index(conditioning, 0),
                negative=get_value_at_index(conditioning, 1),
            )

            pixels = get_value_at_index(resized, 0)
            _, height, width, _ = pixels.shape

            sigmas = scheduler.EXECUTE_NORMALIZED(
                steps=5,
                width=width,
                height=height,
            )

            sampler_obj = self.sampler_select().EXECUTE_NORMALIZED(
                sampler_name="euler"
            )

            samples = sampler.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise, 0),
                guider=get_value_at_index(guider, 0),
                sampler=get_value_at_index(sampler_obj, 0),
                sigmas=get_value_at_index(sigmas, 0),
                latent_image=get_value_at_index(conditioning, 2),
            )

            decoded = vae_decode.decode(
                samples=get_value_at_index(samples, 0),
                vae=get_value_at_index(VAE, 0),
            )

            # -----------------------------
            # Composite back
            # -----------------------------
            final = self.image_composite().EXECUTE_NORMALIZED(
                x=0, y=0,
                resize_source=True,
                destination=get_value_at_index(padded, 0),
                source=get_value_at_index(decoded, 0),
                mask=get_value_at_index(padded, 1),
            )

            return output_to_bytes(
                get_value_at_index(final, 0)
            )

if __name__ == "__main__":
    image = os.path.abspath("test/image.png")
    mask = os.path.abspath("test/mask.png")
    prompt="Remove the wooden stand with books and object inside at the right side. Everything else in the image exactly same, including all other people, garment, background, lighting, poses and facial features."
    masked_inpainter = FluxKlienMaskedInpaint()
    output = masked_inpainter.run(image, mask, prompt)
    output_path = os.path.abspath("test/output.png")
    with open(output_path, "wb") as f:
        f.write(output)
