import os
import argparse
from typing import Any, List, Dict
from collections import Counter
from utils import load_json, eval_bbox, fuzzy_match, print_results


illusion_categories = [
    "assimilation",
    "contrast",
    "constancy",
    "perspective",
    "relativity",
]

id_to_illusion_category = {
    1: "assimilation",
    2: "assimilation",
    3: "contrast",
    4: "contrast",
    5: "contrast",
    6: "constancy",
    7: "assimilation",
    8: "assimilation",
    9: "perspective",
    10: "relativity",
    11: "relativity",
    12: "relativity",
    13: "perspective",
    14: "assimilation",
}


def samediff_qa_answer_match(pred1: str, pred2: str, gt1: str, gt2: str) -> str:
    """Check if the model answers match the ground truth answers.

    :param pred1: the model answer for the no-illusion image
    :param pred2: the model answer for the illusion image
    :param gt1: the humanlike answer for the no-illusion image
    :param gt2: the humanlike answer for the illusion image
    :return: "n/a" if the model answer do not match the humanlike answer on
        the no-illusion image, otherwise, "humanlike" if the model answers
        match the humanlike answers and "no_illusion" if the model answers
        do not match the humanlike answers on the illusion image.
    """

    if fuzzy_match(pred1, gt1):
        return "humanlike" if fuzzy_match(pred2, gt2) else "no_illusion"
    else:
        return "n/a"


def ref_attr_qa_answer_match(pred1: str, pred2: str, gt1: str, gt2: str) -> str:
    """Check if the model answers match the ground truth answers.

    :param pred1: the model answer for the original illusion image
    :param pred2: the model answer for the flipped illusion image
    :param gt1: the humanlike answer for the original illusion image
    :param gt2: the humanlike answer for the flipped illusion image
    :return: "humanlike" if the model answers match the humanlike answers,
        "unlike" if either of the model answers do not match the humanlike answer.
    """
    if fuzzy_match(pred1, gt1) and fuzzy_match(pred2, gt2):
        return "humanlike"
    else:
        return "unlike"


def refloc_bbox_match(
    pred1: List[float], pred2: List[float], gt1: List[float], gt2: List[float]
) -> str:
    """Check if the model bounding boxes match the humanlike bounding boxes.

    :param pred1: the model bounding box for the original illusion image
    :param pred2: the model bounding box for the flipped illusion image
    :param gt1: the humanlike bounding box for the original illusion image
    :param gt2: the humanlike bounding box for the flipped illusion image
    :return: "humanlike" if the predicted bboxs match the gt (humanlike) bboxs,
        "unlike" if either of the predicted bboxs do not match the gt bboxs.
    """
    if eval_bbox(pred1, gt1) and eval_bbox(pred2, gt2):
        return "humanlike"
    else:
        return "unlike"


def eval_vqa(
    predictions: Dict[str, str],
    pair_info: Dict[str, List[str]],
    vqa_annotation: Dict[str, Any],
) -> Dict[str, Counter]:
    """Evaluate the VQA results.

    :param predictions: the model answers for the questions. The keys are the
        data IDs and the values are the answers.
    :param pair_info: the pair info for the data points. The keys are the
        task type and the values are list of two data IDs.
    :param vqa_annotation: the ground truth answers for the questions
    :return: the evaluation results
    """

    all_results = {
        "SameDiffQA": Counter(),
        "RefQA": Counter(),
        "AttrQA": Counter(),
    }
    for k in list(all_results.keys()):
        all_results[k + "_per_category"] = {c: Counter() for c in illusion_categories}

    for task_type in ["samediff_qa", "subj_qa", "desc_qa"]:
        if task_type == "samediff_qa":
            eval_func = samediff_qa_answer_match
            eval_name = "SameDiffQA"
        else:
            eval_func = ref_attr_qa_answer_match
            eval_name = "RefQA" if task_type == "subj_qa" else "AttrQA"

        for eval_id1, eval_id2 in pair_info["samediff_qa"]:
            img1_pred = predictions[eval_id1]
            img2_pred = predictions[eval_id2]
            img1_gt = vqa_annotation[eval_id1]["answer_match"]
            img2_gt = vqa_annotation[eval_id2]["answer_mismatch"]
            match = eval_func(img1_pred, img2_pred, img1_gt, img2_gt)

            category = id_to_illusion_category[int(eval_id1.split("_")[0])]
            all_results[eval_name][match] += 1
            all_results[f"{eval_name}_per_category"][category][match] += 1

    return all_results


def eval_vg(
    predictions: Dict[str, List[float]],
    pair_info: Dict[str, List[str]],
    vg_annotation: Dict[str, Any],
) -> Dict[str, Counter]:
    """Evaluate the visual grounding results.

    :param predictions: the model answers for the questions. The keys are the
        data IDs and the values are the answers.
    :param pair_info: the pair info for the data points. The keys are the
        task type and the values are list of two data IDs.
    :param vg_annotation: the ground truth grounding bbox for the queries

    :return: the evaluation results
    """

    all_results = {
        "RefLoc": Counter(),
        "RefLoc_per_category": {c: Counter() for c in illusion_categories},
    }

    for eval_id1, eval_id2 in pair_info["localization"]:
        img1_pred = predictions[eval_id1]
        img2_pred = predictions[eval_id2]
        img1_gt = vg_annotation[eval_id1]["bbox_match"]
        img2_gt = vg_annotation[eval_id2]["bbox_match"]
        match = refloc_bbox_match(img1_pred, img2_pred, img1_gt, img2_gt)

        category = id_to_illusion_category[int(eval_id1.split("_")[0])]
        all_results["RefLoc"][match] += 1
        all_results["RefLoc_per_category"][category][match] += 1

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa_predictions", type=str, default=None)
    parser.add_argument("--vg_predictions", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--save_dir", type=str, default="./results")
    args = parser.parse_args()

    assert (
        args.vqa_predictions is not None or args.vg_predictions is not None
    ), "Please specify at least one result file to evaluate."

    pair_info = load_json(os.path.join(args.data_dir, "pair_info.json"))
    all_results = {}
    if args.vqa_predictions is not None:
        vqa_predictions = load_json(args.vqa_predictions)
        vqa_annotation = load_json(os.path.join(args.data_dir, "vqa_annotation.json"))
        vqa_results = eval_vqa(vqa_predictions, pair_info, vqa_annotation)
        all_results.update(vqa_results)
    if args.vg_predictions is not None:
        vg_predictions = load_json(args.vg_predictions)
        vg_annotation = load_json(os.path.join(args.data_dir, "vg_annotation.json"))
        vg_results = eval_vg(vg_predictions, pair_info, vg_annotation)
        all_results.update(vg_results)

    # compute scores
    rates = {k: {} for k in all_results if not k.endswith("_per_category")}
    rates_percat = {k: {} for k in all_results if k.endswith("_per_category")}
    for k, v in all_results.items():
        if k.endswith("_per_category"):
            rates_percat[k] = {c: {} for c in all_results[k]}
            for c, v2 in v.items():
                denominator = sum(v2.values())
                for r in v2:
                    rates_percat[k][c][r] = v2[r] / denominator
        else:
            denominator = sum(v.values())
            for r in v:
                rates[k][r] = v[r] / denominator

    # print results
    print(
        f"Prediction files - VQA: {args.vqa_predictions} | VG: {args.vg_predictions})"
    )
    print("-"*30, "Evaluation Results", "-"*30)
    print_results(rates)
    print("-"*30, "Per-category Results", "-"*30)
    print_results(rates_percat)
