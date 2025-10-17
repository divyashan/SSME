import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from scipy.special import softmax
from composition_stats import alr, alr_inv

import pdb 

from utils import (
    ALR_transform_data,
    calculate_bandwidth,
    concatenate_demographic_data,
    create_metrics_df,
    sample_from_rows, create_multiclass_metrics_df,
    DATASET_INFO, N_DRAWS
)

def SSME_KDE(train_labeled_data, train_unlabeled_data, method_config):
    """
    Applies SSME (parameterized by a KDE) to a new dataset.
    Args:
        train_labeled_data: tuple of (X_labeled_preds, demographics_labeled, y_labeled)
        train_unlabeled_data: tuple of (X_unlabeled_preds, demographics_unlabeled, y_unlabeled)
        method_config: dict with required fields (see below)
    Returns:
        metrics_df: pd.DataFrame with estimated metric(s)
    """
    # -- Prepare data and parameters --    
    X_labeled_preds, demographics_labeled, y_labeled = train_labeled_data
    X_unlabeled_preds, demographics_unlabeled, y_unlabeled = train_unlabeled_data
    dataset = method_config.get('dataset', None)
    n_clusters = DATASET_INFO[dataset]['n_classes']
    n_labeled, n_unlabeled = X_labeled_preds.shape[0], X_unlabeled_preds.shape[0]
    init = method_config.get('init', 'draw')
    n_epochs = method_config.get('epochs', 20)
    ldw = method_config.get('labeled_data_weight', 10.0)
    prior_type = method_config.get('prior_type', 'learned')
    use_sample_weights = method_config.get('use_sample_weights', True)
    simulate_predictions = method_config.get('simulate_predictions', True)
    binary = n_clusters == 2
    
    X_preds = np.concatenate([X_labeled_preds, X_unlabeled_preds], axis=0)
    if binary:
        X = ALR_transform_data(X_preds)
    else:
        X = np.array([alr(X_preds[:, i, :]) for i in range(X_preds.shape[1])]).transpose(1,0,2)
        X = X.reshape(n_labeled + n_unlabeled, -1)
    demographics = concatenate_demographic_data(demographics_labeled, demographics_unlabeled)
    y = np.concatenate([y_labeled, y_unlabeled], axis=0)

    
    # -- Initialize cluster assignments --
    cluster_assignments = np.random.randint(0, n_clusters, size=X.shape[0])
    labeled_idxs = np.arange(n_labeled)
    if n_unlabeled:
        if init == 'KNN':
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X[:n_labeled], y[:n_labeled])
            y_unlabeled_pred = knn.predict(X[n_labeled:])
            cluster_assignments[n_labeled:] = y_unlabeled_pred
        elif init == 'draw':
            if binary: 
                est_p = np.mean(X_preds, axis=1)
                cluster_assignments = (np.random.random(est_p.shape) < est_p).astype(int)
            else:
                list_of_lists = [[list(cell) for cell in row] for row in X_preds]
                array_3d = np.array(list_of_lists, dtype=np.float32)
                est_p = np.mean(array_3d, axis=1)
                cluster_assignments = sample_from_rows(est_p)
                
    cluster_assignments[:n_labeled] = y[:n_labeled]

    # --- Calculate bandwidth(s), set kernel ---
    kernel = 'gaussian' # Note: we experimented with other kernels but found no meaningful differences.
    bandwidths = np.zeros(n_clusters)
    for k in range(n_clusters):
        bandwidths[k] = calculate_bandwidth(X, rule='sheather-jones')

    for epoch in tqdm(range(n_epochs)):
        kdes = []
        for j in range(n_clusters):
            # sample weights for cluster j
            if use_sample_weights and epoch > 0:
                sample_weight = cluster_preds[:, j]
                sample_weight[:n_labeled] = ldw * sample_weight[:n_labeled]
                thresh = 0 # Threshold for filtering samples to fit KDE.
                idxs_to_fit = np.where(sample_weight > thresh)[0]
                kde = KernelDensity(
                    bandwidth=bandwidths[j], kernel=kernel
                ).fit(X[idxs_to_fit], sample_weight=sample_weight[idxs_to_fit])
            else:
                idxs_to_fit = np.where(cluster_assignments == j)[0]
                kde = KernelDensity(
                    bandwidth=bandwidths[j], kernel=kernel
                ).fit(X[idxs_to_fit])
            kdes.append(kde)
        # Estimate priors
        if prior_type == 'fixed_oracle':
            priors = DATASET_INFO[dataset]['prior']
        elif prior_type == 'learned':
            priors = [np.mean(cluster_assignments == j) for j in range(n_clusters)]

        # KDE scoring
        log_likelihoods = np.array([kde.score_samples(X) for kde in kdes]).T
        scores = log_likelihoods
        for j in range(n_clusters):
            scores[:, j] += np.log(priors[j] + 1e-12)
        cluster_preds = softmax(scores, axis=1)

        # Sample cluster assignments for next epoch
        if binary:
            random_sample = np.random.random(cluster_preds[:, 1].shape)
            cluster_assignments = (random_sample < cluster_preds[:, 1]).astype(int)
            cluster_assignments[:n_labeled] = y[:n_labeled]
        else:
            n_examples = cluster_preds.shape[0]
            random_sample = np.random.rand(n_examples)
            cumulative_sum = np.cumsum(cluster_preds, axis=1)
            for i in range(n_examples):
                cluster_assignments[i] = np.searchsorted(cumulative_sum[i], random_sample[i])
            cluster_assignments[:n_labeled] = y[:n_labeled]
    print("Estimated priors: ", [x.round(3) for x in priors])

    # Output metrics based on cluster_assignments, X_preds, and demographics
    if binary:
        metrics_df = create_metrics_df(cluster_assignments, X_preds, demographics, dataset=dataset)
    else:
        metrics_df = create_multiclass_metrics_df(cluster_assignments, X_preds, demographics)

    # Optionally simulate predictions (averaging metrics over multiple draws from soft assignments)
    if simulate_predictions:
        all_metrics_dfs = [] 
        for _ in range(N_DRAWS):
            if binary:
                random_sample = np.random.random(cluster_preds[:, 1].shape)
                new_assignments = (random_sample < cluster_preds[:, 1]).astype(int)
                new_assignments[:n_labeled] = y[:n_labeled]
                drawn_metrics_df = create_metrics_df(new_assignments, X_preds, demographics, dataset=dataset)
            else:
                n_examples = cluster_preds.shape[0]
                random_sample = np.random.rand(n_examples)
                new_assignments = np.zeros(n_examples, dtype=int)
                cumulative_sum = np.cumsum(cluster_preds, axis=1)
                for i in range(n_examples):
                    new_assignments[i] = np.searchsorted(cumulative_sum[i], random_sample[i])
                new_assignments[:n_labeled] = y[:n_labeled]
                drawn_metrics_df = create_multiclass_metrics_df(new_assignments, X_preds, demographics)
            all_metrics_dfs.append(drawn_metrics_df)
        metrics_df = pd.concat(all_metrics_dfs).groupby(['demographic', 'model_idx']).mean().reset_index()
    metrics_df['model_idx'] = metrics_df['model_idx'].astype(int)
    return  metrics_df
