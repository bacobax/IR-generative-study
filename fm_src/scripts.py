# Read npy files

import numpy as np
from PIL import Image

def load_image(path):
    im = np.load(path)
    # you can use other normalization strategies to make the image roughly in [0, 1].
    im = (im.max() - im) / 760.0

    im *= 255.0
    # make it 3 channels
    if len(im.shape) == 2:
        im = np.stack((im,) * 3, axis=-1)
    
    return Image.fromarray(im.astype(np.uint8))

# Load annotation from annotations.json (there is one file per split)
# box formats [x, y, width, height]
import json

def load_annotations(path):
    with open(path, 'r') as f:
        annotations = json.load(f)
    
    categories = annotations['categories']

    annot = {}

    for image in annotations['images']:
        image_id = image['id']
        annot[image_id] = {
            'image_path': image['file_name'],
            'boxes': [],
            'labels': [],
            'iscrowd': [],
            'area': [],
        }
        # filter annotations with this image id
        image_annotations = [a for a in annotations['annotations'] if a['image_id'] == image_id]
        for a in image_annotations:
            annot[image_id]['boxes'].append(a['bbox'])
            annot[image_id]['labels'].append(categories[a['category_id']]['id'])
            annot[image_id]['area'].append(a['area'])
            annot[image_id]['iscrowd'].append(a.get('iscrowd', 0))


    return annot