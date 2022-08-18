import os
import numpy as np
import pandas as pd
from utils import split_train_dataset_to_val, undo_all_split_train_dataset, kfold_split_train_dataset
from constants import DATASET_TYPES, RESULTS
from train_resnet import train_resnet, train_resnet_with_kfold
from train_inceptiontime import train_inceptiontime, train_inceptiontime_with_kfold
from train_vit import train_vit, train_vit_with_kfold

def main():
    datasets_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='datasets_parameters')
    results = pd.DataFrame(datasets_parameters['name'])
    if os.path.exists(RESULTS):
        results = pd.read_csv(RESULTS)
    for data_type in DATASET_TYPES:
        undo_all_split_train_dataset(datasets_parameters, data_type)
        train_resnet_with_kfold(datasets_parameters, data_type, results, resnet_type='resnet50')
        train_resnet_with_kfold(datasets_parameters, data_type, results, resnet_type='resnet101')
        train_vit_with_kfold(datasets_parameters, data_type, results)
        split_train_dataset_to_val(datasets_parameters, data_type)
        results = train_resnet(datasets_parameters, data_type, results, resnet_type='resnet50')
        results = train_resnet(datasets_parameters, data_type, results, resnet_type='resnet101')
        results = train_vit(datasets_parameters, data_type, results)
        results = train_inceptiontime(datasets_parameters, data_type, results)
        results = train_resnet(datasets_parameters, data_type, results, resnet_type='resnet152')
        undo_all_split_train_dataset(datasets_parameters, data_type)
        train_inceptiontime_with_kfold(datasets_parameters, data_type, results)
        train_resnet_with_kfold(datasets_parameters, data_type, results, resnet_type='resnet152')

if __name__ == '__main__':
    main()