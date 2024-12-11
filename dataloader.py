import pandas as pd
import numpy as np 

from paths import create_predictions_path
from utils import DATASET_TO_N_CLASSES

def sample_data(train_dataset_df, n_labeled, n_unlabeled, model_alg_list, run, n_classes):

    sampled_labeled_dataset_df = train_dataset_df.sample(n=n_labeled, random_state=run)
    train_dataset_df = train_dataset_df[~train_dataset_df.index.isin(sampled_labeled_dataset_df.index)]
    sampled_unlabeled_dataset_df = train_dataset_df.sample(n=n_unlabeled, random_state=run)
    train_dataset_df = train_dataset_df[~train_dataset_df.index.isin(sampled_unlabeled_dataset_df.index)]
    
    # Sample at least 1 positive and negative example
    sampled_one_positive_df = train_dataset_df[train_dataset_df['label'] == 1].sample(n=1, random_state=run)
    sampled_one_negative_df = train_dataset_df[train_dataset_df['label'] == 0].sample(n=1, random_state=run)

    sampled_dataset_df = pd.concat([sampled_labeled_dataset_df, sampled_unlabeled_dataset_df])
    sampled_dataset_df = pd.concat([sampled_dataset_df.head(n_labeled + n_unlabeled - 2), sampled_one_positive_df, sampled_one_negative_df])
                                   
    sampled_data = sampled_dataset_df[model_alg_list].values
    sampled_true_labels = sampled_dataset_df['label'].values

    
    np.random.seed(run)  
    # Sample at least 1 example per class
    labeled_idxs = []
    if  n_labeled > 0:
        for c in range(n_classes) :
            class_idxs = np.where(sampled_true_labels == c)[0]
            labeled_idxs.extend(np.random.choice(class_idxs, 1))

        # Sample remaining labeled  examples
        remaining_idxs = [x for x in np.arange(len(sampled_true_labels)) if x not in labeled_idxs]
        labeled_idxs.extend(np.random.choice(remaining_idxs, n_labeled - n_classes, replace=False))
    
    sampled_labels = np.ones(sampled_true_labels.shape) * -1
    sampled_labels[labeled_idxs] = sampled_true_labels[labeled_idxs]

    return sampled_data, sampled_labels, sampled_true_labels, sampled_dataset_df


def get_model_values_df(dataset, model_algs, value='accuracy', task=None):
    
    prob_predictions_matrix = []
    discrete_prediction_matrix = []
    accuracy_matrix = []
    metadata_matrix = []
    split = 'test'
    for model_alg in model_algs:
        model= model_alg.split('_')[0]
        alg = '_'.join(model_alg.split('_')[1:])
        if dataset == 'civilcomments':
            model='distilbert-base-uncased'

        preds_config = {
            "dataset": dataset,
            "algorithm": alg,
            "split": split,
            "model": model,
        }
        predictions = np.load(create_predictions_path(preds_config) + "/preds.npy", allow_pickle=True)
        metadata = np.load(create_predictions_path(preds_config) + "/metadata.npy", allow_pickle=True)
        
        discretized_preds = np.argmax(predictions, axis=1)
        discrete_prediction_matrix.append(discretized_preds)
        prob_predictions_matrix.append(predictions)
        metadata_matrix.append(metadata)

        if split != 'unlabeled':
            labels = np.load(create_predictions_path(preds_config) + "/labels.npy")
            accuracy = discretized_preds == labels
            accuracy_matrix.append(accuracy)

    discrete_prediction_matrix = np.stack(discrete_prediction_matrix)   
    prob_predictions_matrix = np.stack(prob_predictions_matrix, axis=0) 
    metadata_matrix = np.stack(metadata_matrix, axis=0)[0]
    if split != 'unlabeled':
        accuracy_matrix = np.stack(accuracy_matrix, axis=0).astype(int)

    #TODO: FIX THIS
    results_df = []
    n_examples = discrete_prediction_matrix.shape[1]
    for i in range(n_examples):
        result = {'index': i, 'metadata': metadata_matrix[i]}
        if split != 'unlabeled':
            result['label'] = labels[i]
        for j, model_alg in enumerate(model_algs):
            result[model_alg] = prob_predictions_matrix[j, i, 1]
               
        results_df.append(result)
    
    results_df = pd.DataFrame(results_df)

    # Specific to civilcomments
    results_df = add_metadata_columns(results_df)
    
    return results_df


def add_metadata_columns(results_df):
    civilcomments_metadata = ['sex:male', 'sex:female', 'orientation:LGBTQ', 'religion:christian', 'religion:muslim', 'religion:other_religions', 
                              'race:black', 'race:white', 'race:identity_any', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']
            
    for i,metadata_col_name in enumerate(civilcomments_metadata):
        results_df[metadata_col_name] = results_df['metadata'].apply(lambda x: x[i])
    return results_df
