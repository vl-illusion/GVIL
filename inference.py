import os
import json
import argparse
from tqdm import tqdm
from utils import load_all_imgs, load_json
from model import CustomModel
from functools import partial

model_build_func = {
    "custom": CustomModel.build,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["vqa", "vg"], default="vqa")
    parser.add_argument("--model", type=str, default="custom")
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    args = parser.parse_args()

    # load data
    df = "vqa_annotation.json" if args.task == "vqa" else "vg_annotation.json"
    data = load_json(os.path.join(args.data_dir, df))

    # pre-load all the imgs cuz it's not big and each img is used multiple times
    img_dir = os.path.join(args.data_dir, "images")
    img_file_to_img = load_all_imgs(img_dir)

    # load model
    model = model_build_func[args.model]()

    # run inference
    print("Running inference...")
    results = {}
    for k, v in tqdm(data.items()):
        img = img_file_to_img[v["img"]]
        if args.task == "vqa":
            question = v["question"]
            candidates = [v["answer_match"], v["answer_mismatch"]]
            response = model.get_answer(img, question, candidates)
        else:
            question = v["query"]
            response = model.get_box(img, question)
        results[k] = response

    # save predictions
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = f"{args.task}__{args.model}.json"
    save_file = os.path.join(args.save_dir, save_name)
    with open(save_file, "w") as f:
        json.dump(results, f, indent=2)
