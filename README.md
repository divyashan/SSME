#  Semi-Supervised Model Evaluation (SSME)

![SSME Diagram](./explainer.png)
---

This repository contains the code for "Evaluating Multiple Models using Labeled and Unlabeled Data" (NeurIPS 2025). The repository implements working examples of Semi-Supervised model Evaluation (SSME), a new framework designed to estimate evaluation metrics (e.g., accuracy or calibration error) for models in settings with limited labeled data.
SSME takes advantage of unlabeled data, multiple models, and continuous probabilistic predictions to deliver performance estimates that are far more accurate 
than standard approaches to model evaluation. For additional details and results, do check out the associated [paper](https://arxiv.org/abs/2501.11866).

## 👩🏾‍💻 Reproducing results on CivilComments.

We provide code to reproduce results on CivilComments, a dataset containing comments annotated for toxicity. Here, we have access to seven models; for ease of reproducibility, we provide the pre-computed predictions from each model in the `inputs` folder. To replicate a comparison of SSME to labeled data alone and ground truth performance estimates: 

1. Clone this repository by running `git clone git@github.com:divyashan/SSME.git`.
2. Install requirements via `conda env create -f environment.yml`.
3. Estimate performance metrics for each model using labeled data compared to SSME in `demo_notebook.ipynb`.

## 🌎 Applying SSME to a new task.

SSME accepts three inputs: labeled data, unlabeled data, and a set of models. You can apply SSME in your setting by following these steps.

1. Apply each available model to all examples, creating an array of model predictions of shape (n_examples, n_classes). 
2. Save each model's predictions to a distinct folder under `inputs`.
3. Modify `DATASET_INFO`, in `utils.py`, to include a new entry for the desired task. Refer to `utils.py` for detailed instructions on what parameters must be specified and the format SSME expects them in. 
4. Run the following command to estimate model performance using SSME: `python run_ssme.py -d <your_dataset> -nl <n_labeled> -nu <n_unlabeled>`. Results will be saved to the `outputs` folder. 

Our goal is to provide an easily extensible implementation of SSME, for ease of use and research. The provided implementation currently supports subgroup-specific performance estimation and evaluation of multi-class outputs, and supports estimation of accuracy, ECE, AUC, and AUPRC. If there are additional features that you would find particularly useful, do let us know! 

## Contact

Please reach out to Divya Shanmugam at [divyas@cornell.edu](mailto:divyas@cornell.edu) and Shuvom Sadhuka at [ssadhuka@mit.edu](mailto:ssadhuka@mit.edu) with any questions or interest in applying SSME to your setting. We pronounce SSME as "Sesame", but you're welcome to your favorite pronunciation :) 
