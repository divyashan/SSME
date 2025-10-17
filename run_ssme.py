import os
import pandas as pd
import numpy as np
import argparse
import warnings

import pdb 

warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import get_model_values_df, DATASET_INFO, sample_data, collapse_multiclass_predictions
from model import SSME_KDE
from baselines import labeled_data_alone

pd.options.mode.chained_assignment = None


def run_experiment(dataset,  n_labeled, n_unlabeled, run, subgroups=None, simulation=True):
    np.random.seed(run)
    subgroup_list = subgroups.split(',') if subgroups is not None else None
    n_classes = DATASET_INFO[dataset]['n_classes']
    model_names = DATASET_INFO[dataset]['model_names'] 
    n_models = len(model_names)
    print(f"Loading {dataset} dataset...")
    dataset_df = get_model_values_df(dataset, model_names)

    print(f"Number of examples: {len(dataset_df)}")
    train_dataset_df = dataset_df
    test_dataset_df = []
    if simulation:
        # Simulating n_labeled and n_unlabeled based on a labeled dataset.
        train_dataset_df = dataset_df.sample(frac=0.5, random_state=run)
        test_dataset_df = dataset_df[~dataset_df.index.isin(train_dataset_df.index)]
        
        sampled_data, sampled_labels, _, sampled_data_df = sample_data(
            train_dataset_df, n_labeled, n_unlabeled, model_names, run, n_classes)
        labeled_idxs = np.where(sampled_labels != -1)[0]
        unlabeled_idxs = np.where(sampled_labels == -1)[0]
        
        test_data = test_dataset_df[model_names].values
        if n_classes > 2:
            test_data = collapse_multiclass_predictions(test_data, n_models, n_classes)
    else:
        # Using real labeled and unlabeled data.
        sampled_data = train_dataset_df[model_names].values
        sampled_labels = train_dataset_df['label'].values
        labeled_idxs = np.where(sampled_labels != -1)[0]
        unlabeled_idxs = np.where(sampled_labels == -1)[0]
    
    # collapse for multiclass
    if n_classes > 2:
        sampled_data = collapse_multiclass_predictions(sampled_data, n_models, n_classes)

    print("No. of Labeled Examples: ", len(labeled_idxs))
    print("No. of Unlabeled Examples: ", len(unlabeled_idxs))
    print("Subgroups: ", subgroup_list)
    assert len(labeled_idxs) == n_labeled
    assert len(unlabeled_idxs) == n_unlabeled
    
    sampled_groups = [np.array(['global'] * (n_labeled + n_unlabeled))]
    test_groups = [np.array(['global'] * len(test_dataset_df))]
    if subgroup_list:
        for subgroup in subgroup_list:
            subgroup_values = sampled_data_df[subgroup].values
            subgroup_values = [f"{subgroup}_{v}" for v in subgroup_values]
            sampled_groups.append(np.array(subgroup_values))
            if simulation: 
                test_subgroup_values = test_dataset_df[subgroup].values
                test_subgroup_values = [f"{subgroup}_{v}" for v in test_subgroup_values]
                test_groups.append(np.array(test_subgroup_values))
            

    estimation_labeled_data = (
        sampled_data[labeled_idxs],
        [d[labeled_idxs] for d in sampled_groups],
        sampled_labels[labeled_idxs]
    )
        
    estimation_unlabeled_data = (
        sampled_data[unlabeled_idxs],
        [d[unlabeled_idxs] for d in sampled_groups],
        sampled_labels[unlabeled_idxs]
    )

    print("Running SSME...")
    method_config = {'dataset': dataset, 'subgroup': subgroups}
    metrics_df = SSME_KDE(estimation_labeled_data, estimation_unlabeled_data, method_config)
    metrics_df['model'] = metrics_df['model_idx'].apply(lambda x: model_names[x])
    
    # Compare to ground truth metrics.
    if simulation:
        test_labels = test_dataset_df['label'].values
        gt_metrics_df = labeled_data_alone((test_data, test_groups, test_labels), method_config)
        gt_metrics_df['model'] = gt_metrics_df['model_idx'].apply(lambda x: model_names[x])
        metrics_df = pd.merge(metrics_df, gt_metrics_df, on=['model', 'demographic', 'model_idx'], suffixes=('', '_gt'))
    
    # Log and print results.
    metric_cols = ['demographic', 'model', 'acc', 'acc_gt', 'ece', 'ece_gt']
    if n_classes == 2:
        metric_cols.extend(['auc', 'auc_gt', 'auprc', 'auprc_gt'])
    print(metrics_df[metric_cols])
    
    config_vars_to_log = [
            ('dataset', dataset),
            ('model_names', ','.join(model_names)),
            ('subgroups', subgroups),
            ('method', 'labeled'),
            ('run', run),
            ('n_labeled', n_labeled),
            ('n_unlabeled', n_unlabeled), 
            ('subgroups', subgroups)
        ]
    metrics_df = metrics_df.assign(**dict(config_vars_to_log))

    os.makedirs('./outputs', exist_ok=True)
    fname = f"{dataset}__{'-'.join(model_names)}__{subgroups}__labeled__{run}__{n_labeled}_{n_unlabeled}.csv"
    print(f"Saving results to {os.path.join('./outputs', fname)}")
    metrics_df.to_csv(os.path.join('./outputs', fname), index=False)
    return metrics_df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SSME (labeled baseline only)")
    parser.add_argument("-d", "--dataset", help="Dataset to run on", required=True)
    parser.add_argument("-nl", "--n_labeled", help="No. of labeled points. Do not set if ground truth is unavailable.", default=-1, type=int)
    parser.add_argument("-nu", "--n_unlabeled", help="No. of unlabeled points. Not necessary if ground truth is unavailable.", default=-1, type=int)
    parser.add_argument("-r", "--run", help="Run", default=0, type=int)
    parser.add_argument("-sbgrp", "--subgroup", help="Subgroup variable to report performance over (e.g. race)", default=None)
    parser.add_argument("-sim", "--simulation", help="Flag for whether to run in simulation mode (i.e. synthetically generate labeled/unlabeled data).", default=False)

    args = parser.parse_args()
    run_experiment(
        args.dataset,
        args.n_labeled,
        args.n_unlabeled,
        args.run,
        args.subgroup,
        args.simulation
    )