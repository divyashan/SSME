PREDICTIONS_PATH = './inputs/'
RESULTS_PATH = './outputs/'

def create_predictions_path(dataset, model_dir):
    return PREDICTIONS_PATH + f'/{dataset}/' + model_dir + '/'