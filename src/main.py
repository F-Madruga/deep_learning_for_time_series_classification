import os
import numpy as np
import pandas as pd
from utils import split_train_dataset_to_val
from constants import DATASET_TYPES, RESULTS
from train_resnet import train_resnet50, train_resnet101, train_resnet152
from train_inceptiontime import train_inceptiontime
from train_vit import train_vit

def main():
    datasets_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='datasets_parameters')
    results = pd.DataFrame(datasets_parameters['name'])
    if os.path.exists(RESULTS):
        results = pd.read_csv(RESULTS)
    for data_type in DATASET_TYPES:
        split_train_dataset_to_val(datasets_parameters, data_type)
        # results = train_resnet50(datasets_parameters, data_type, results)
        # results = train_resnet101(datasets_parameters, data_type, results)
        # results = train_resnet152(datasets_parameters, data_type, results)
        # results = train_vit(datasets_parameters, data_type, results)
        results = train_vit(datasets_parameters, data_type, results)
        # results = train_inceptiontime(datasets_parameters, data_type, results)

if __name__ == '__main__':
    main()