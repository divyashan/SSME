import numpy as np 
import pandas as pd 

from sklearn.metrics import roc_auc_score, average_precision_score 
from sklearn.calibration import calibration_curve
from paths import create_predictions_path
import pdb 
from composition_stats import alr, alr_inv

N_DRAWS = 5
DATASET_INFO = {'CivilComments': {'n_classes': 2,
                                  
                                  # Directories where each classifier's predictions are stored.
                                  # Each directory should contain:
                                  # 1. preds.npy, shape=(n_examples x n_classes).
                                  # 2. metadata.npy, shape=(n_examples x n_metadata_cols), if available, 
                                  # 3. labels.npy, shape=(n_examples,), where label = -1 if unknown.
                                  'model_names': ['alg_CORAL', 'alg_ERM', 'alg_IRM', 
                                                  'alg_ERM_seed1', 'alg_ERM_seed2', 
                                                  'alg_IRM_seed1',  'alg_IRM_seed2'],
                                  
                                  # [Optional] A list of strings, where each string corresponds to a column in the metadata file.
                                  # These columns can be used to customize performance estimates to subgroups.
                                  'metadata_cols': ['sex:male', 'sex:female', 'orientation:LGBTQ', 
                                                    'religion:christian', 'religion:muslim', 'religion:other_religions', 
                                                    'race:black', 'race:white', 'race:identity_any', 'severe_toxicity', 
                                                    'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'],
                                  
                                  # [Optional] The prevalences of each class. It can be set to None when not known. 
                                  # SSME does not use this information by default. To try SSME with known priors, 
                                  # set 'prior_type' to 'fixed_oracle' in the method config.
                                  'prior': [0.887, 0.113],
                                  
                                  # [Optional] Probability threshold above which a prediction is considered positive. Used when computing accuracy; 0.5 by default.
                                  'threshold': 0.5
                                 },
                
                'MultiNLI': {'n_classes': 3,
                             'model_names': ['alg_ReWeight','alg_SqrtReWeight',
                                             'alg_IRM','alg_ReSample'],
                             'metadata_cols': [],
                             'prior': None,
                             'threshold': 0.5
                            },
}
### Data loading functions ###

def get_model_values_df(dataset, model_names):
    n_classes = DATASET_INFO[dataset]['n_classes']
    prob_predictions_matrix = []
    metadata_matrix = []
    for model_name in model_names:
        
        # predictions = np.load(create_predictions_path(preds_config) + "/preds.npy", allow_pickle=True)
        # metadata = np.load(create_predictions_path(preds_config) + "/metadata.npy", allow_pickle=True)
        predictions = np.load(create_predictions_path(dataset, model_name) + "/preds.npy", allow_pickle=True)
        metadata = np.load(create_predictions_path(dataset, model_name) + "/metadata.npy", allow_pickle=True)
        
        prob_predictions_matrix.append(predictions)
        metadata_matrix.append(metadata)

        # labels = np.load(create_predictions_path(preds_config) + "/labels.npy")
        labels = np.load(create_predictions_path(dataset, model_name) + "/labels.npy")

    prob_predictions_matrix = np.stack(prob_predictions_matrix, axis=0) 
    metadata_matrix = np.stack(metadata_matrix, axis=0)[0]

    results_df = []
    n_examples = prob_predictions_matrix.shape[1]
    for i in range(n_examples):
        result = {'index': i, 'metadata': metadata_matrix[i], 'label': labels[i]}
        for j, model_name in enumerate(model_names):
            if n_classes == 2:
                result[model_name] = prob_predictions_matrix[j, i, 1]
            else:
                result[model_name] = prob_predictions_matrix[j, i]
        results_df.append(result)
    
    results_df = pd.DataFrame(results_df)
    # Note: if your dataset has metadata you wish to break down performance by, 
    # modify the line below to run add_metadata_columns for your dataset as well.
    # Values in the "metadata" column must be a list of length equal to the number of metadata columns
    # specified in DATASET_INFO[dataset]['metadata_cols'].
    results_df = add_metadata_columns(dataset, results_df)
    return results_df

def sample_data(train_dataset_df, n_labeled, n_unlabeled, model_names, run, n_classes):

    sampled_labeled_dataset_df = train_dataset_df.sample(n=n_labeled, random_state=run)
    train_dataset_df = train_dataset_df[~train_dataset_df.index.isin(sampled_labeled_dataset_df.index)]
    sampled_unlabeled_dataset_df = train_dataset_df.sample(n=n_unlabeled, random_state=run)
    train_dataset_df = train_dataset_df[~train_dataset_df.index.isin(sampled_unlabeled_dataset_df.index)]
    
    # Sample at least 1 positive and negative example
    sampled_one_positive_df = train_dataset_df[train_dataset_df['label'] == 1].sample(n=1, random_state=run)
    sampled_one_negative_df = train_dataset_df[train_dataset_df['label'] == 0].sample(n=1, random_state=run)

    sampled_dataset_df = pd.concat([sampled_labeled_dataset_df, sampled_unlabeled_dataset_df])
    sampled_dataset_df = pd.concat([sampled_dataset_df.head(n_labeled + n_unlabeled - 2), sampled_one_positive_df, sampled_one_negative_df])
                                   
    sampled_data = sampled_dataset_df[model_names].values
    sampled_true_labels = sampled_dataset_df['label'].values

    
    np.random.seed(run)  
    # Sample at least 1 example per class
    labeled_idxs = []
    if  n_labeled > 0:
        for c in range(n_classes) :
            class_idxs = np.where(sampled_true_labels == c)[0]
            labeled_idxs.extend(np.random.choice(class_idxs, 1))

        # Sample remaining labeled examples
        remaining_idxs = [x for x in np.arange(len(sampled_true_labels)) if x not in labeled_idxs]
        labeled_idxs.extend(np.random.choice(remaining_idxs, n_labeled - n_classes, replace=False))
    
    sampled_labels = np.ones(sampled_true_labels.shape) * -1
    sampled_labels[labeled_idxs] = sampled_true_labels[labeled_idxs]

    return sampled_data, sampled_labels, sampled_true_labels, sampled_dataset_df

def add_metadata_columns(dataset, results_df):
    metadata_cols = DATASET_INFO[dataset].get('metadata_cols', [])
    for i,metadata_col_name in enumerate(metadata_cols):
        results_df[metadata_col_name] = results_df['metadata'].apply(lambda x: x[i]) 
    return results_df

### Data processing functions ###

def collapse_multiclass_predictions(predictions, n_models, n_classes):
    flattened_preds = [item for sublist in predictions for item in sublist]
    concatenated_preds = np.concatenate(flattened_preds)
    return concatenated_preds.reshape(-1, n_models, n_classes)


def ALR_transform_data(data):
    data[data == 0] = 1e-4
    data[data == 1] = 1-1e-4
    transformed_data = np.log(data/(1-data))
    return transformed_data    


def concatenate_demographic_data(group_1_demographics, group_2_demographics):
    # group_1_demographics and group_2_demographics are lists of lists, where each list contains the demographic data for a particular data field (e.g. race)
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
        min_bandwidth = np.inf
        for i in range(d):
            kde = sm.nonparametric.KDEUnivariate(data[:, i])
            kde.fit()
            bandwidth = kde.bw / 4.0
            min_bandwidth = min(min_bandwidth, bandwidth)
        bandwidth = min_bandwidth
    else:
        return None
    return bandwidth

### Metric estimation functions ###

def create_metrics_df(labels: np.ndarray, predictions: np.ndarray, 
                      demographics_list, dataset=None) -> pd.DataFrame:
    """
    Create a pandas DataFrame containing evaluation metrics for each model, for all groups reflected in demographics.

    Parameters:
    - labels (np.ndarray): Array of true labels.
    - predictions (np.ndarray): Array of predicted probabilities for each model.
    - group (str, optional): Group name for the metrics. Default is 'global'.

    Returns:
    - metrics_df (pd.DataFrame): DataFrame containing evaluation metrics for each model.
      Columns: 'auc', 'auprc', 'ece', 'acc', 'group', 'model_idx'.
    """
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

                threshold = DATASET_INFO[dataset].get('threshold', 0.5)
                binarized_predictions_group = (predictions_group > threshold).astype(int)
                acc = np.mean((predictions_group > threshold).astype(int) == labels_group)
                metrics_df.append({'auc': auc, 'auprc': auprc, 'ece': ece, 'demographic': group, 'acc': acc,
                                'model_idx': model_idx})
    return pd.DataFrame(metrics_df)


## Merge this into the above function
def create_multiclass_metrics_df(labels: np.ndarray, predictions: np.ndarray, demographics_list):
    """
    Estimate multiclass metrics (accuracy and ECE) for each model for a single set of sampled (or argmax) labels.
    Intended to be called once per sampled label set (multiple draws can be handled externally).

    Parameters:
    - labels (np.ndarray): Array of true labels [Unused].
    - predictions (np.ndarray): Array of predicted probabilities for each model (shape: [n_samples, n_models, n_classes]).
    - demographics_list: List of demographic arrays.
    - sample_labels (bool, optional): Whether to sample labels from predictions or use argmax.

    Returns:
    - results_df (pd.DataFrame): DataFrame with accuracy and ECE per demographic group and model.
    """

    import torch
    from torchmetrics.classification import MulticlassCalibrationError

    _, n_models, n_classes = predictions.shape

    results = []

    for demographics in demographics_list:
        unique_groups = np.unique(demographics)
        for group in unique_groups:
            group_indices = np.where(demographics == group)[0]
            if len(group_indices) == 0:
                continue

            group_labels = labels[group_indices]
            group_target = torch.tensor(group_labels)

            for model_idx in range(n_models):
                group_predictions = predictions[group_indices, model_idx, :]  # (group_size, n_classes)
                model_argmax_indices_group = np.argmax(group_predictions, axis=1)
                acc = np.mean(model_argmax_indices_group == group_labels)
                torch_preds_one_model = torch.tensor(group_predictions)
                metric = MulticlassCalibrationError(num_classes=n_classes)
                ece = metric(torch_preds_one_model, group_target).item()
                results.append({
                    "model_idx": model_idx,
                    "demographic": group,
                    "acc": acc,
                    "ece": ece
                })

    results_df = pd.DataFrame(results)

    return results_df

def sample_from_rows(prob_matrix):
    # Number of rows in the matrix
    n_rows = prob_matrix.shape[0]
    
    # Output array of sampled indices
    sampled_indices = np.zeros(n_rows, dtype=int)

    # Sample from each row
    for i in range(n_rows):
        sampled_indices[i] = np.random.choice(prob_matrix.shape[1], p=prob_matrix[i])
        
    return sampled_indices

def inverse_softmax(probs):
    # Ensure the probabilities sum to 1
    if not np.isclose(np.sum(probs), 1):
        raise ValueError("Probabilities do not sum to 1")

    # Compute the logits
    logits = np.log(probs)
    
    # Adjust by subtracting the log of the sum of exponentiated logits to ensure correct scale
    C = np.log(np.sum(np.exp(logits)))
    adjusted_logits = logits - C
    
    return adjusted_logits