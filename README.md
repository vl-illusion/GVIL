# illusion-dataset
- [Check out project homepage](https://vl-illusion.github.io/)
- [Check out the full paper](https://arxiv.org/abs/2311.00047)

# Setup
1. Unzip the data.
```
unzip dataset.zip
```
PS: The dataset is also available on Huggingface: [link](https://huggingface.co/datasets/sled-umich/VL-Illusion).

2. Install some utilities.
```
pip install matplotlib tqdm
```

# Inference
1. Add your model in `model.py`.
2. Add the build function in `inference.py`.
3. Run inference.
```
python inference.py --task vqa --model YOUR_MODEL_NAME
``` 

# Evaluation
We provide the model predictions from OFA and Unified-IO for reference. For example, you can run evaluation on the prediction of the OFA-Large model after unzipping `unzip predictions.zip`.
```
python eval.py \
    --vqa_predictions predictions/vqa__ofa_large.json \
    --vg_predictions predictions/vg__ofa_large.json
```

# Citation
```bibtex
@inproceedings{zhang2023grounding,
    title={Grounding Visual Illusions in Language: Do Vision-Language Models Perceive Illusions Like Humans?},
    author={Zhang, Yichi and Pan, Jiayi and Zhou, Yuchen and Pan, Rui and Chai, Joyce},
    booktitle={Proceedings of Conference of Empirical Methods in Natural Language Processing},
    year={2023},
    organization={EMNLP 2023}
}
```