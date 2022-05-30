import os
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from torchvision import transforms
from resnet import ResNet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import load_image_datasets, get_dataloaders, plotting_time_normalization


RESULTS = 'results.csv'
TRAIN_NEW_MODELS = False
DATASETS_WITH_BAD_IMAGE_RESOLUTIONS = ['Computers', 'Earthquakes', 'ElectricDevices', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Haptics', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Phoneme', 'RefrigerationDevices', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShapeletSim']

def train_resnet50(datasets_parameters, results):
    if 'resnet50_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained ResNet50')
        results['resnet50_train_accuracy'] = np.full((len(results)), -1).tolist()
        results['resnet50_val_accuracy'] = np.full((len(results)), -1).tolist()
        results['resnet50_test_accuracy'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='resnet_parameters')
    train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
    val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
    test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
    for i, dataset in datasets_parameters.iterrows():
        print(f'[{i + 1}/{len(datasets_parameters)}] training ResNet50 for dataset {dataset["name"]}')
        num_epochs = int(model_parameters.iloc[i]['resnet50_epochs'])
        batch_size = int(model_parameters.iloc[i]['resnet50_batch_size'])
        learning_rate = float(model_parameters.iloc[i]['resnet50_learning_rate'])
        early_stop_patience = int(model_parameters.iloc[i]['resnet50_early_stop_patience'])
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=.0, patience=early_stop_patience, verbose=True, mode='min')
        print(f'num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, early_stop_patience = {early_stop_patience}')
        if dataset["name"] not in DATASETS_WITH_BAD_IMAGE_RESOLUTIONS:
            dataset_path = '######################################################################'
            train_dataset, val_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform, val_transform)
            train_loader, val_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset, val_dataset)
            model = None
            trainer = Trainer(callbacks=[early_stop_callback], gpus=1, max_epochs=num_epochs, fast_dev_run=False)
            if results.iloc[i]['resnet50_train_accuracy'] == -1:
                model = ResNet.ResNet50(img_channels=1, num_classes=dataset['num_classes'], learning_rate=learning_rate)
                trainer.fit(model, train_loader, val_loader)
                results.loc[i,'resnet50_train_accuracy'] = trainer.test(model, train_loader)[0]['test_accuracy']
                results.loc[i,'resnet50_val_accuracy'] = trainer.test(model, val_loader)[0]['test_accuracy']
                results.loc[i,'resnet50_test_accuracy'] = trainer.test(model, test_loader)[0]['test_accuracy']
                results.to_csv(RESULTS, index=False)
            else:
                print('Model already trained')
        else:
            print('Bad image dimension')
    results.to_csv(RESULTS, index=False)
    return results

def main():
    datasets_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='datasets_parameters')
    results = pd.DataFrame(datasets_parameters['name'])
    if os.path.exists(RESULTS):
        results = pd.read_csv(RESULTS)
    train_resnet50(datasets_parameters, results)

if __name__ == '__main__':
    main()