import numpy as np 
import pandas as pd 

from sklearn.metrics import roc_auc_score, average_precision_score 
from sklearn.calibration import calibration_curve

DATASET_TO_PRIOR = {'civilcomments': [0.887, 0.113]}

DATASET_TO_N_CLASSES = {'imagenet': 1000, 'mimic_hospitalization': 2, 'mimic_critical': 2, 'mimic_ed_revisit_3d': 2,
                        'civilcomments': 2, 'ogb-molpcba_task_94': 2, 'ogb-molpcba_task_93': 2, 'ogb-molpcba_task_0': 2,
                        'ogb-molpcba_task_47': 2, 'ogb-molpcba_task_60': 2,'ogb-molpcba': 2, 'simulated': 2}

DATASET_TO_THRESHOLD = {'mimic_hospitalization': 0.5, 
                        'mimic_critical': 0.5,
                        'mimic_ed_revisit_3d': 0.5,
                        'civilcomments': 0.5,
                        'ogb-molpcba_task_60': 0.5,
                        'ogb-molpcba_task_94': 0.5,
                        'ogb-molpcba_task_0': 0.5,
                        'ogb-molpcba_task_47': 0.5,
                        'ogb-molpcba_task_93': 0.5,
                        'simulated_independence': 0.5}



def ALR_transform_data(data):
    data[data == 0] = 1e-4
    data[data == 1] = 1-1e-4
    transformed_data = np.log(data/(1-data))
    return transformed_data    

def inverse_logit_transform(scores : np.ndarray) -> np.ndarray:
    # Designed for binary settings
    # TODO add dimensionality for inputs
    exp_scores = np.exp(scores)
    return exp_scores/(1+exp_scores)

    
def concatenate_demographic_data(group_1_demographics, group_2_demographics):
    # group_1_demographics is a list of lists, where each list contains the demographic data for a particular data field (e.g. race)
    n_fields = len(group_1_demographics)
    merged_demographics = []
    for i in range(n_fields):
        merged_demographics.append(np.concatenate([group_1_demographics[i], group_2_demographics[i]]))
    return merged_demographics

def calculate_bandwidth(data, rule='silverman'):
    import statsmodels.api as sm

    n, d = data.shape
    if rule == 'scott':
        bandwidth = np.power(n, -1.0/(d+4))
    elif rule == 'silverman':
        sigma_hat = np.std(data, axis=0).mean()  # Mean std deviation across dimensions
        bandwidth = np.power(4.0*sigma_hat**5 / (3*n), 1.0/5)
    elif rule == 'sheather-jones':
        # Could estimate the bandwidth per model and then average? 
        min_bandwidth = np.inf
        for i in range(d):
            kde = sm.nonparametric.KDEUnivariate(data[:, i])
            kde.fit()
            bandwidth = kde.bw / 4.0
            min_bandwidth = min(min_bandwidth, bandwidth)
            # Use the bandwidth for further calculations
        bandwidth = min_bandwidth
    else:
        return None
    return bandwidth

def create_metrics_df(labels: np.ndarray, predictions: np.ndarray, 
                      demographics_list, dataset=None) -> pd.DataFrame:
    """
    Create a pandas DataFrame containing evaluation metrics for each model.

    Parameters:
    - labels (np.ndarray): Array of true labels.
    - predictions (np.ndarray): Array of predicted probabilities for each model.
    - group (str, optional): Group name for the metrics. Default is 'global'.

    Returns:
    - metrics_df (pd.DataFrame): DataFrame containing evaluation metrics for each model.
      Columns: 'auc', 'auprc', 'ece', 'acc', 'group', 'model_idx'.
    """
    # Should we instead make demographics a dictionary, mapping demographic names to lists of demographic values for each example?
    n_models = predictions.shape[1]
    metrics_df = []
    for demographics in demographics_list:
        unique_groups = sorted(list(set(demographics)))
        for model_idx in range(n_models):
            for group in unique_groups:
                group_idxs = np.where(demographics == group)[0]
                labels_group = labels[group_idxs]
                predictions_group = predictions[group_idxs, model_idx]

                if len(set(labels_group)) == 1:
                    auc = None
                    auprc = None
                else:
                    auc = roc_auc_score(labels_group, predictions_group)
                    auprc = average_precision_score(labels_group, predictions_group)
                
                # Not enough points to compute ECE 
                n_bins = 10
                if len(labels_group) < n_bins:
                    ece = None
                else:
                    prob_true, prob_pred = calibration_curve(labels_group, predictions_group, 
                                                            n_bins=n_bins, strategy='quantile')

                    ece = np.mean(np.abs(prob_pred - prob_true))

                binarized_predictions_group = (predictions_group > DATASET_TO_THRESHOLD.get(dataset, 0.5)).astype(int)
                class_1_acc = np.mean(binarized_predictions_group[labels_group == 1])
                class_0_acc = 1 - np.mean(binarized_predictions_group[labels_group == 0])
                acc = np.mean((predictions_group > DATASET_TO_THRESHOLD.get(dataset,0.5)).astype(int) == labels_group)
                # acc  = (class_0_acc + class_1_acc)/2
                metrics_df.append({'auc': auc, 'auprc': auprc, 'ece': ece, 'demographic': group, 'acc': acc,
                                'model_idx': model_idx})
    
    return pd.DataFrame(metrics_df)