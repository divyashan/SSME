import numpy as np 
import pandas as pd 
from tqdm import tqdm

from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from scipy.special import softmax

from utils import ALR_transform_data, inverse_logit_transform, calculate_bandwidth
from utils import concatenate_demographic_data, create_metrics_df
from utils import DATASET_TO_PRIOR

### TODO
# Figure out package installation issues
# Write out notebook such that it runs 
# Implement labeled thing

N_DRAWS = 500

def SSME_KDE_binary(train_labeled_data, train_unlabeled_data, method_config): 
    ## Concatenate labeled and unlabeled data
    X_preds = np.concatenate([train_labeled_data[0], train_unlabeled_data[0]], axis=0)
    X = ALR_transform_data(X_preds)
    y = np.concatenate([train_labeled_data[2], train_unlabeled_data[2]], axis=0)
    demographics = concatenate_demographic_data(train_labeled_data[1], train_unlabeled_data[1])

    ## Initialize parameters
    n_clusters = 2
    dataset = method_config['dataset']    
    n_epochs = method_config['epochs']
    ldw = method_config['labeled_data_weight']
    simulate_predictions = method_config['simulate_predictions']
    prior_type = method_config['prior_type']
    use_sample_weights = method_config['use_sample_weights']

    n_labeled = len(train_labeled_data[0])
    n_unlabeled = len(train_unlabeled_data[0])
    ## Initialize clusters
    cluster_assignments = np.random.randint(0, n_clusters, size=X.shape[0])
    cluster_assignments[:n_labeled] = y[:n_labeled]
    labeled_idxs = list(range(n_labeled))
    # Smart initialization for unlabeled examples
    if n_unlabeled:
        if method_config['init'] == 'KNN':
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X[:n_labeled], y[:n_labeled])  
            y_unlabeled = knn.predict(X[n_labeled:])
            cluster_assignments[n_labeled:] = y_unlabeled
        elif method_config['init'] == 'draw':
            est_p = np.mean(X_preds, axis=1)
            cluster_assignments = (np.random.random(est_p.shape) < est_p).astype(int)
            cluster_assignments[:n_labeled] = y[:n_labeled]

    ## Sheather-Jones applied to p(x) to identify optimal bandwidth
    ## Note: we experimented with different bandwidths and found this to produce the most accurate estimates of p(y|s)
    bandwidths = np.zeros(n_clusters)    
    for i in range(n_clusters):
        bandwidths[i] = calculate_bandwidth(X, rule='sheather-jones')
    
    kernel = 'gaussian'
    for i in tqdm(range(n_epochs)):
        kdes = []
        for j in range(n_clusters):
            # sample_weights 
            if use_sample_weights and i > 0:
                sample_weight = (cluster_assignments == j).astype(int)
                sample_weight = cluster_preds[:,j]
                sample_weight[:n_labeled] = ldw*sample_weight[:n_labeled]
                
                thresh = 0 # Threshold on sample weight to qualify for fitting; higher thresholds ignore more points.
                idxs_to_fit = np.where(sample_weight > thresh)[0]
                
                kde = KernelDensity(bandwidth=bandwidths[j], kernel=kernel).fit(X[idxs_to_fit], sample_weight=sample_weight[idxs_to_fit])

            else:
                idxs_to_fit = np.where(cluster_assignments == j)[0]
                kde = KernelDensity(bandwidth=bandwidths[j], kernel=kernel).fit(X[idxs_to_fit])
            kdes.append(kde)

        priors = [np.mean(cluster_assignments[labeled_idxs] == j) for j in range(n_clusters)]
        if prior_type == 'fixed_oracle':
            priors = [np.mean(cluster_assignments[labeled_idxs] == j) for j in range(n_clusters)]
            priors = DATASET_TO_PRIOR[dataset]
        elif prior_type == 'learned' or np.sum(train_labeled_data[2]) <= 15:
            priors = [np.mean(cluster_assignments == j) for j in range(n_clusters)]
        
        log_likelihoods = np.array([kde.score_samples(X) for kde in kdes]).T
        scores = log_likelihoods

        for j in range(n_clusters):
            scores[:, j] += np.log(priors[j])
        
        if i % 10 == 0:
            print(np.mean(scores))
        cluster_preds = softmax(scores, axis=1)
        random_sample = np.random.random(cluster_preds[:,1].shape)
        cluster_assignments = (random_sample < cluster_preds[:,1]).astype(int)
        cluster_assignments[:n_labeled] = y[:n_labeled]
    print("Estimated priors: ", priors)
    metrics_df = create_metrics_df(cluster_assignments, X_preds, demographics, dataset=dataset)

    if simulate_predictions:
        all_metrics_dfs = []
        for _ in range(N_DRAWS):
            random_sample = np.random.random(cluster_preds[:,1].shape)
            cluster_assignments = (random_sample < cluster_preds[:,1]).astype(int)
            cluster_assignments[:n_labeled] = y[:n_labeled]
            all_metrics_dfs.append(create_metrics_df(cluster_assignments, X_preds, demographics, dataset=dataset))

        metrics_df = pd.concat(all_metrics_dfs).groupby(level=0).agg(lambda x: x.mean() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
        metrics_df['model_idx'] = metrics_df['model_idx'].astype(int)

    n_samples = 10000
    class_samples = []
    class_labels = []
    for j, kde in enumerate(kdes):
        # Sample from the KDE using the prior
        samples = inverse_logit_transform(kde.sample(int(priors[j] * n_samples)))

        class_samples.append(samples)
        class_labels.append(j*np.ones(samples.shape[0]))
  
    # return X_preds, cluster_assignments, metrics_df
    return class_samples, class_labels, metrics_df