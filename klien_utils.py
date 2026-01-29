# Standard library
import base64
import gc
import logging
import os
import sys
from io import BytesIO
from time import sleep
from typing import Any, Mapping, Sequence, Union

# Third-party
import requests
import numpy as np
import torch
from PIL import Image,ImageOps

from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)



def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
            path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)

        manager_path = os.path.join(
            comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob"
        )

        if os.path.isdir(manager_path) and os.listdir(manager_path):
            sys.path.append(manager_path)
            global has_manager
            has_manager = True
        else:
            has_manager = False

        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from comfy.options import enable_args_parsing

    enable_args_parsing()
    from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                print("Patching manager_core.get_config to enforce offline mode.")
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    if has_manager:
        try:
            import manager_core as manager
        except ImportError:
            print("Could not import manager_core, proceeding without it.")
            return
        else:
            if hasattr(manager, "get_config"):
                print("Patching manager_core.get_config to enforce offline mode.")
                try:
                    get_config = manager.get_config

                    def _get_config(*args, **kwargs):
                        config = get_config(*args, **kwargs)
                        config["network_mode"] = "offline"
                        return config

                    manager.get_config = _get_config
                except Exception as e:
                    print("Failed to patch manager_core.get_config:", e)

    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def inner():
        # Creating an instance of PromptServer with the loop
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)

        # Initializing custom nodes
        await init_extra_nodes(init_custom_nodes=True)

    loop.run_until_complete(inner())


def output_to_bytes(processed_image):
    if isinstance(processed_image, torch.Tensor):
        processed_image = processed_image.detach().cpu().numpy()
    if isinstance(processed_image, np.ndarray):
        if processed_image.ndim == 4:
            processed_image = processed_image[0]
        if processed_image.shape[0] == 3:
            processed_image = np.transpose(processed_image, (1, 2, 0))
        if processed_image.shape[0] == 1:
            processed_image = np.repeat(processed_image, 3, axis=0)
        processed_image = (processed_image * 255).astype(np.uint8)
    processed_image = Image.fromarray(processed_image) if isinstance(processed_image, np.ndarray) else processed_image # pil already

    with BytesIO() as byte_stream:
        processed_image.save(byte_stream, format='JPEG')
        image_bytes = byte_stream.getvalue()
    return image_bytes

def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

# Convert bytes to base64 string
def bytes_to_base64(byte_data):
    return base64.b64encode(byte_data).decode('utf-8')

# Convert base64 string back to bytes
def base64_to_bytes(base64_string):
    return base64.b64decode(base64_string)

def fetch_image_data(image_url, attempts=5, timeout=30):
    for attempt in range(attempts):
        try:
            response = requests.get(image_url, timeout=timeout)
            response.raise_for_status()
            if response.ok:
                return response.content
        except Exception as e:
            logger.warning(f"attempt {attempt + 1} failed for {image_url}: {e}")
            if attempt < attempts - 1:
                sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
    logger.error(f"failed to fetch image from URL after {attempts} attempts: {image_url}")
    return None

def get_foreground_mask(image: bytes, mask_only='true') -> bytes:
    files = {'image': image}
    data = {'mask_only': mask_only}
    bgremoval_url = "https://twindragon.catalogix.ai/background-removal/rembg/model/candidate/cutout"

    try:
        response = requests.post(bgremoval_url, files=files, data=data)
        response.raise_for_status()
        logger.info("Background removal successful!")
        result = response.content
        return result
    except Exception as e:
        logger.exception(f"Background removal failed: {e}", exc_info=True)
        return None


def get_rgba_image(image_bytes: bytes, mask_bytes: bytes) -> bytes:
    """
    Accepts raw image bytes + mask bytes.
    Applies mask as alpha.
    Returns final RGBA PNG bytes.
    """
    # Load input image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Load mask
    mask = Image.open(BytesIO(mask_bytes)).convert("L")  # L = 8-bit mask

    # Ensure mask matches dimensions
    if mask.size != image.size:
        mask = mask.resize(image.size)

    # Add alpha channel from mask
    rgba_image = image.copy()
    rgba_image.putalpha(mask)

    # Convert to PNG bytes
    output = BytesIO()
    rgba_image.save(output, format="PNG")
    return output.getvalue()

def mask_to_region(
    mask: bytes,
    min_size: int = 500
) -> bytes:
    """
    Convert user drawn mask to rectangular region using scikit-image.
    Removes small noisy pixels <= min_size.
    """

    # convert binary mask
    mask = Image.open(BytesIO(mask)).convert("L")
    mask_np = np.array(mask.convert("L")) > 127
    h, w = mask_np.shape

    # remove noise
    cleaned = remove_small_objects(mask_np, min_size=min_size)

    # nothing valid → empty
    if cleaned.sum() == 0:
        empty = Image.new("L", mask.size, 0)
        output = BytesIO()
        empty.save(output, format="PNG")
        logger.warning("empty mask found...")
        return output.getvalue()

    regions = regionprops(label(cleaned))

    minr = min(r.bbox[0] for r in regions)
    minc = min(r.bbox[1] for r in regions)
    maxr = max(r.bbox[2] for r in regions)
    maxc = max(r.bbox[3] for r in regions)

    box_h = maxr - minr
    box_w = maxc - minc

    side = max(box_h, box_w)

    # square impossible → full image mask
    if side > min(h, w):
        full = Image.new("L", (w, h), 255)
        output = BytesIO()
        full.save(output, format="PNG")
        logger.warning("using full mask...")
        return output.getvalue()

    # square positioning
    cy = (minr + maxr) // 2
    cx = (minc + maxc) // 2
    half = side // 2

    y1 = max(0, min(cy - half, h - side))
    x1 = max(0, min(cx - half, w - side))
    y2 = y1 + side
    x2 = x1 + side

    square = np.zeros((h, w), dtype=np.uint8)
    square[y1:y2, x1:x2] = 255
    square = Image.fromarray(square).convert("L")

    # Convert to PNG bytes
    output = BytesIO()
    square.save(output, format="PNG")
    return output.getvalue()