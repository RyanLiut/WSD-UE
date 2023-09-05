## Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation

This repository contains the code for the findings of ACL 2023: [Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation](https://aclanthology.org/2023.findings-acl.245/).

## Abstract

Word sense disambiguation (WSD) is a crucial task in natural language understanding, aiming to determine the appropriate sense for a target word given its context. Existing supervised methods treat WSD as a classification task and have achieved remarkable performance. However, these methods often overlook uncertainty estimation (UE) in real-world scenarios where the data is noisy and out of distribution. In this paper, we extensively study UE on a benchmark designed for WSD. Firstly, we compare four uncertainty scores for a state-of-the-art WSD model and find that conventional predictive probabilities obtained at the end of the model are insufficient for quantifying uncertainty. Next, we examine the model's capability to capture data and model uncertainties through selected UE score using well-designed test scenarios. Our analysis shows that the model adequately reflects data uncertainty but underestimates model uncertainty. Furthermore, we explore various lexical properties that intrinsically affect data uncertainty and provide a detailed analysis of four critical aspects: syntactic category, morphology, sense granularity, and semantic relations.

## Code Execution

1. Install dependencies.

   ```shell
   conda env create -f environment.yml
   conda activate XXX
   ```

2. Download the data and model from the cloud.

   Dataset: https://cloud.tsinghua.edu.cn/d/feaec5147ef6458cb0f7/

   Model (Checkpoint): https://cloud.tsinghua.edu.cn/d/cbc10bf9881b4450bb4d/

   Final results: https://cloud.tsinghua.edu.cn/d/7abe2bdc29a44978abf4/

3. Run the following code to evaluate uncertainty:

   ```shell
   python scripts/mycode/mypredict.py \
      --model bert-large/best_checkpoint_val_f1=0.7626_epoch=018.ckpt \
      --processor bert-large/processor_config.json \
      --model_input data/preprocessed/semeval2007/semeval2007.json \
      --evaluation_input data/original/semeval2007/semeval2007.gold.key.txt \
      --n_Tsamples 2 \
      --rand_seed 10 \
      --mark K07
   ```

4. Run the following code to explore lexical factors:

   ```shell
   python scripts/mycode/effect_UE.py
   ```

   After execution, the results will be saved in the specified output path (default: `results4/`).

## Citation
```
@inproceedings{liu-liu-2023-ambiguity,
    title = "Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation",
    author = "Liu, Zhu  and
      Liu, Ying",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.245",
    doi = "10.18653/v1/2023.findings-acl.245",
    pages = "3963--3977",
```
## Acknowledgement

The code is based on [multilabel-wsd](https://github.com/SapienzaNLP/multilabel-wsd). We would like to express our gratitude to the authors for their work and code.

If you have any problem or comment, feel free to contact [me](liuzhu22@mails.tsinghua.edu.cn) or pose an issue:)
