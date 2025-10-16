#  Semi-Supervised Model Evaluation (SSME)

![SSME Diagram](./explainer.png)
---

This repository contains the code for "Evaluating Multiple Models using Labeled and Unlabeled Data" (NeurIPS 2025). The repository implements working examples of Semi-Supervised model Evaluation (SSME), a new framework designed to estimate evaluation metrics (e.g., accuracy, calibration error, AUC, etc.) for predictive models in settings with limited labeled data.
SSME takes advantage of unlabeled data, multiple models, and continuous probabilistic predictions to deliver more accurate estimates of performance
than standard approaches to classifier evaluation. For additional details and results, do check out the associated [paper](https://arxiv.org/abs/2501.11866).

## Reproducing results: CivilComments

We provide code to reproduce results on CivilComments, along with pre-computed model predictions in the `inputs` folder. To replicate a comparison of SSME to labeled data alone and ground truth performance estimates: 

1. Clone this repository
2. Install requirements 
3. Run `demo_notebook.ipynb`

## Applying SSME to your own task 

SSME accepts three inputs: labeled data, unlabeled data, and a set of classifiers. You can apply SSME in your setting by:

1. Creating arrays of model predictions across all examples, as demonstrated in demo notebook.
2. Modifying the DATASET_INFO dictionary to reflect properties of the new data.
3. Running python run_ssme.py --dataset <your_dataset> --model_set <your_classifiers> 

## Contact

For questions, bug reports, or collaborations, please reach out to Divya Shanmugam at [divyas@cornell.edu](mailto:divyas@cornell.edu) and Shuvom Sadhuka at [ssadhuka@mit.edu](mailto:ssadhuka@mit.edu). We pronounce SSME as "Sesame", but you're welcome to your favorite pronunciation :) 
