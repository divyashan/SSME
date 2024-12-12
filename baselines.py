from utils import create_metrics_df

def labeled_binary(train_labeled_data, method_cfg):
    preds = train_labeled_data[0]
    demographics = train_labeled_data[1]
    labels = train_labeled_data[2]

    metrics_df = create_metrics_df(labels, preds, demographics, dataset=method_cfg['dataset'])
    return metrics_df

