import os
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from torchvision import transforms
from resnet import ResNet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import load_image_datasets, get_dataloaders, plotting_time_normalization, black_and_white_image_to_binary
from constants import TRAIN_NEW_MODELS, DATASETS_WITH_BAD_IMAGE_RESOLUTIONS, DATASETS_PATH, GPUS, RESULTS, FAST_DEV_RUN


def train_resnet50(datasets_parameters, data_type, results):
    # transformers
    train_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    val_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    test_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    if data_type == 'line_plots':
        train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
        val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
        test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
    # add column for model results
    if 'resnet50_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        results['resnet50_train_accuracy'] = np.full((len(results)), -1).tolist()
        results['resnet50_val_accuracy'] = np.full((len(results)), -1).tolist()
        results['resnet50_test_accuracy'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='resnet_parameters')
    # training model
    for i, dataset in datasets_parameters.iterrows():
        # model parameters
        num_epochs = int(model_parameters.iloc[i]['resnet50_epochs'])
        batch_size = int(model_parameters.iloc[i]['resnet50_batch_size'])
        learning_rate = float(model_parameters.iloc[i]['resnet50_learning_rate'])
        early_stop_patience = int(model_parameters.iloc[i]['resnet50_early_stop_patience'])
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=.0, patience=early_stop_patience, verbose=True, mode='min')
        print(f'[{i + 1}/{len(datasets_parameters)}] RESNET50 {dataset["name"]}: num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, early_stop_patience = {early_stop_patience}')
        if dataset["name"] not in DATASETS_WITH_BAD_IMAGE_RESOLUTIONS:
            dataset_path = os.path.join(DATASETS_PATH, dataset['name'], data_type)
            train_dataset = None
            test_dataset = None
            val_dataset = None
            train_loader = None
            test_loader = None
            val_loader = None
            has_val_dataset = os.path.isdir(os.path.join(dataset_path, 'val'))
            if has_val_dataset:
                train_dataset, val_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform, val_transform)
                train_loader, val_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset, val_dataset)
            else:
                train_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform)
                train_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset)
            model = None
            trainer = Trainer(callbacks=[early_stop_callback], gpus=GPUS, max_epochs=num_epochs, fast_dev_run=FAST_DEV_RUN)
            if results.iloc[i]['resnet50_train_accuracy'] == -1:
                print(dataset['num_classes'])
                model = ResNet.ResNet50(img_channels=1, num_classes=int(dataset['num_classes']), learning_rate=learning_rate)
                if has_val_dataset:
                    trainer.fit(model, train_loader, val_loader)
                else:
                    trainer.fit(model, train_loader)
                train_accuracy = trainer.test(model, train_loader)[0]['test_accuracy']
                test_accuracy = trainer.test(model, train_loader)[0]['test_accuracy']
                results.loc[i,'resnet50_train_accuracy'] = train_accuracy
                results.loc[i,'resnet50_test_accuracy'] = test_accuracy
                if has_val_dataset:
                    val_accuracy = trainer.test(model, val_loader)[0]['test_accuracy']
                    results.loc[i,'resnet50_val_accuracy'] = val_accuracy
                    print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}, val_accuracy = {val_accuracy}')
                else:
                    print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}')
                results.to_csv(RESULTS, index=False)
            else:
                print('Model already trained')
        else:
            print('Bad image dimension')
    results.to_csv(RESULTS, index=False)
    return results