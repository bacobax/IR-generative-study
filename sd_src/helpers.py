import numpy as np
from PIL import Image

def ir_to_3ch_with_stretch(pil_img: Image.Image) -> Image.Image:
    # Ensure 8-bit grayscale array
    arr = np.array(pil_img)
    if arr.ndim == 3:
        # if it is already 3ch, convert to luminance for safety
        arr = arr[..., 0]
    arr = arr.astype(np.uint8)

    mn = int(arr.min())
    mx = int(arr.max())

    # avoid divide-by-zero if flat
    if mx <= mn:
        stretched = np.zeros_like(arr, dtype=np.uint8)
    else:
        stretched = ((arr - mn) * 255.0 / (mx - mn)).clip(0, 255).astype(np.uint8)

    # SD expects 3 channels
    return Image.fromarray(stretched, mode="L").convert("RGB")

def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]



def generate_prompt(num_persons: int) -> str:
    base = "overhead infrared surveillance image, circular field of view"

    if num_persons == 0:
        return base
    elif num_persons == 1:
        return base + ", one person"
    else:
        return base + f", {num_persons} people"
    
P0001_PERCENTILE_RAW_IMAGES = 11667.0  # p0.001 percentile
P9999_PERCENTILE_RAW_IMAGES = 13944.0  # p99.999 percentile
A1 = P0001_PERCENTILE_RAW_IMAGES
B1 = P9999_PERCENTILE_RAW_IMAGES