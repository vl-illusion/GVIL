import os
import json
from PIL import Image
from typing import Optional, Union, Dict
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def load_all_imgs(img_dir: str) -> Dict[str, np.ndarray]:
    # load all images in the img_dir into memory
    img_fn_to_img: Dict[str, np.ndarray] = {}
    print("Loading images...")
    for img_fn in os.listdir(img_dir):
        img_file = os.path.join(img_dir, img_fn)
        img_fn_to_img[img_fn] = np.array(Image.open(img_file))

    print(f"{len(img_fn_to_img)} images loaded.")
    return img_fn_to_img


def load_json(json_file: str) -> Union[dict, list]:
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def compute_iou(boxA: list, boxB: list):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max((xB - xA), 0) * max((yB - yA), 0)
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def box_contains(boxA: list, boxB: list):
    # check if boxA contains boxB, return True if so
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max((xB - xA), 0) * max((yB - yA), 0)
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    if interArea == boxBArea:
        return True
    return False


def eval_bbox(bbox_gen: list, bbox_gt: list, threshold: float = 0.5) -> bool:
    # check if bbox_gen is within bbox_gt or has a high enough iou
    if box_contains(bbox_gt, bbox_gen) or compute_iou(bbox_gen, bbox_gt) > threshold:
        return True
    return False


def fuzzy_match(generated: str, oracle: str) -> bool:
    """Try to improve the matching robustness by adding some simple rules."""
    oracle = oracle.lower()
    generated = generated.lower()
    if generated.endswith("."):
        generated = generated[:-1]

    if generated == oracle:
        return True
    if oracle == "yes" and generated[:3] == "yes":
        return True
    if oracle == "no" and generated[:2] == "no":
        return True
    if generated == "more " + oracle:
        # more blue -> blue
        return True
    if generated == "larger" and oracle == "bigger":
        return True
    if generated == "top " + oracle or oracle == "top " + generated:
        return True
    return False


def print_results(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_results(value, indent + 4)
        else:
            print(" " * indent + f"{key:14s}: {value:.2%}")


def plot_image_with_bbox(
    image: Image,
    bbox: list,
    label: Optional[str],
    save_file: Optional[str],
    show: bool = False,
):
    x1, y1, x2, y2 = bbox
    fig, ax = plt.subplots()
    w = x2 - x1
    h = y2 - y1
    rect = patches.Rectangle(
        (x1, y1), w, h, linewidth=2, edgecolor="g", facecolor="none"
    )
    plt.text(x1, y1, label, color="r", fontsize=7)
    ax.add_patch(rect)
    ax.imshow(image)
    plt.axis("off")
    if save_file:
        plt.savefig(save_file, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close()
