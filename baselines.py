from utils import create_metrics_df, create_multiclass_metrics_df
from utils import DATASET_INFO

def labeled_data_alone(train_labeled_data, method_cfg):
    preds = train_labeled_data[0]
    demographics = train_labeled_data[1]
    labels = train_labeled_data[2]

    n_classes = DATASET_INFO[method_cfg['dataset']]['n_classes']
    if n_classes > 2:
        metrics_df = create_multiclass_metrics_df(labels, preds, demographics)
    else:
        metrics_df = create_metrics_df(labels, preds, demographics, dataset=method_cfg['dataset'])
    return metrics_df