import os
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from torchvision import transforms
from inceptiontime import InceptionTime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from utils import load_image_datasets, get_dataloaders, load_image_datasets_from_paths, plotting_time_normalization, black_and_white_image_to_binary, undo_all_split_train_dataset, undo_split_train_dataset
from constants import TRAIN_NEW_MODELS, DATASETS_WITH_BAD_IMAGE_RESOLUTIONS, DATASETS_PATH, GPUS, RESULTS, FAST_DEV_RUN


def train_inceptiontime_with_kfold(datasets_parameters, data_type, results, n_splits=5, shuffle=True, random_state=None):
     # transformers
    train_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    val_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    test_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    if data_type == 'line_plots':
        train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
        val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
        test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])    # add column for model results
    if f'inceptiontime_kfold_0_{data_type}_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        for i in range(n_splits):
            results[f'inceptiontime_kfold_{i}_{data_type}_train_accuracy'] = np.full((len(results)), -1).tolist()
            results[f'inceptiontime_kfold_{i}_{data_type}_val_accuracy'] = np.full((len(results)), -1).tolist()
            results[f'inceptiontime_kfold_{i}_{data_type}_test_accuracy'] = np.full((len(results)), -1).tolist()
            results[f'inceptiontime_kfold_{i}_{data_type}_train_loss'] = np.full((len(results)), -1).tolist()
            results[f'inceptiontime_kfold_{i}_{data_type}_val_loss'] = np.full((len(results)), -1).tolist()
            results[f'inceptiontime_kfold_{i}_{data_type}_test_loss'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='inceptiontime_parameters')
    for i, dataset in datasets_parameters.iterrows():
        undo_split_train_dataset(dataset, data_type)
        # model parameters
        num_epochs = int(model_parameters.iloc[i]['epochs'])
        batch_size = int(model_parameters.iloc[i]['batch_size'])
        learning_rate = float(model_parameters.iloc[i]['learning_rate'])
        num_inception_blocks = int(model_parameters.iloc[i]['num_inception_blocks'])
        n_filters = []
        for i in range(num_inception_blocks):
            n_filters.append(int(model_parameters.iloc[i]['num_filters_' + str(i + 1)]))
        use_residual = bool(model_parameters.iloc[i]['use_residual'])
        print(f'[{i + 1}/{len(datasets_parameters)}] INCEPTIONTIME {dataset["name"]}: num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, num_inception_blocks = {num_inception_blocks}, n_filters = {n_filters}, use_residual = {use_residual}')
        if dataset["name"] not in DATASETS_WITH_BAD_IMAGE_RESOLUTIONS:
            dataset_path = os.path.join(DATASETS_PATH, dataset['name'], data_type)
            test_dataset = None
            test_loader = None
            has_val_dataset = os.path.isdir(os.path.join(dataset_path, 'val'))
            if has_val_dataset:
                train_dataset, val_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform, val_transform)
                train_loader, val_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset, val_dataset)
            else:
                train_dataset, test_dataset = load_image_datasets(dataset_path, train_transform, test_transform)
                train_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset)
            train_path = os.path.join(DATASETS_PATH, dataset['name'], data_type, 'train')
            if os.path.isdir(train_path):
                imgs = []
                for dataset_class in os.listdir(train_path):
                    train_class_path = os.path.join(train_path, dataset_class)
                    for img in os.listdir(train_class_path):
                        img_path = os.path.join(train_class_path, img)
                        imgs.append(img_path)
                imgs = np.array(imgs)
                kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                num_fold = 0
                for train_index, val_index in kfold.split(imgs):
                    if results.iloc[i][f'inceptiontime_kfold_{num_fold}_{data_type}_train_accuracy'] == -1:
                        undo_split_train_dataset(dataset, data_type)
                        X_train, X_val = imgs[train_index], imgs[val_index]
                        for img in X_train:
                            path_splits = img.split('/')
                            path_splits[3] = path_splits[3].replace('train', f'train_kfold_{num_fold}')
                            split_path = '/'.join(path_splits[:4])
                            if not os.path.isdir(split_path):
                                os.mkdir(split_path)
                            class_path = '/'.join(path_splits[:5])
                            if not os.path.isdir(class_path):
                                os.mkdir(class_path)
                            new_path = '/'.join(path_splits)
                            os.rename(img, new_path)
                        for img in X_val:
                            path_splits = img.split('/')
                            path_splits[3] = path_splits[3].replace('train', f'val_kfold_{num_fold}')
                            split_path = '/'.join(path_splits[:4])
                            if not os.path.isdir(split_path):
                                os.mkdir(split_path)
                            class_path = '/'.join(path_splits[:5])
                            if not os.path.isdir(class_path):
                                os.mkdir(class_path)
                            new_path = '/'.join(path_splits)
                            os.rename(img, new_path)
                        model = InceptionTime(1, int(dataset['num_classes']), learning_rate, n_filters=[32, 32, 32, 32, 32], use_residual=use_residual)
                        checkpoints = ModelCheckpoint(dirpath='models/inceptiontime/' + dataset["name"] + '/' + data_type + '/' + 'kfold_' + str(num_fold), filename="{epoch}--{step}--{val_loss:.2f}--{val_accuracy:.2f}--{train_accuracy:.2f}", monitor='val_loss', mode='min', save_top_k=3, save_last=True)
                        trainer = Trainer(gpus=GPUS, max_epochs=num_epochs, fast_dev_run=FAST_DEV_RUN, callbacks=[checkpoints])
                        train_dataset, val_dataset, _ = load_image_datasets_from_paths(os.path.join(dataset_path, f'train_kfold_{num_fold}'), os.path.join(dataset_path, 'test'), train_transform, test_transform, os.path.join(dataset_path, f'val_kfold_{num_fold}'), val_transform)
                        train_loader, val_loader, _ = get_dataloaders(batch_size, train_dataset, _, val_dataset)
                        trainer.fit(model, train_loader, val_loader)
                        train_scores = trainer.test(model, train_loader)
                        train_accuracy = train_scores[0]['test_accuracy']
                        train_loss = train_scores[0]['test_loss']
                        val_scores = trainer.test(model, val_loader)
                        val_accuracy = val_scores[0]['test_accuracy']
                        val_loss = val_scores[0]['test_loss']
                        test_scores = trainer.test(model, test_loader)
                        test_accuracy = test_scores[0]['test_accuracy']
                        test_loss = test_scores[0]['test_loss']
                        results.loc[i,f'inceptiontime_kfold_{num_fold}_{data_type}_train_accuracy'] = train_accuracy
                        results.loc[i,f'inceptiontime_kfold_{num_fold}_{data_type}_val_accuracy'] = val_accuracy
                        results.loc[i,f'inceptiontime_kfold_{num_fold}_{data_type}_test_accuracy'] = test_accuracy
                        results.loc[i,f'inceptiontime_kfold_{num_fold}_{data_type}_train_loss'] = train_loss
                        results.loc[i,f'inceptiontime_kfold_{num_fold}_{data_type}_val_loss'] = val_loss
                        results.loc[i,f'inceptiontime_kfold_{num_fold}_{data_type}_test_loss'] = test_loss
                        results.to_csv(RESULTS, index=False)
                        print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}, val_accuracy = {val_accuracy}')
                        print(f'train_loss = {train_loss}, test_loss = {test_loss}, val_loss = {val_loss}')
                    num_fold += 1
                undo_split_train_dataset(dataset, data_type)
    return results


def train_inceptiontime(datasets_parameters, data_type, results):
    # transformers
    train_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    val_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    test_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    if data_type == 'line_plots':
        train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
        val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
        test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor(), lambda x: torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))])
    # add column for model results
    if f'inceptiontime_{data_type}_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        results[f'inceptiontime_{data_type}_train_accuracy'] = np.full((len(results)), -1).tolist()
        results[f'inceptiontime_{data_type}_val_accuracy'] = np.full((len(results)), -1).tolist()
        results[f'inceptiontime_{data_type}_test_accuracy'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='inceptiontime_parameters')
    # training model
    for i, dataset in datasets_parameters.iterrows():
        # model parameters
        num_epochs = int(model_parameters.iloc[i]['epochs'])
        batch_size = int(model_parameters.iloc[i]['batch_size'])
        learning_rate = float(model_parameters.iloc[i]['learning_rate'])
        num_inception_blocks = int(model_parameters.iloc[i]['num_inception_blocks'])
        n_filters = []
        for i in range(num_inception_blocks):
            n_filters.append(int(model_parameters.iloc[i]['num_filters_' + str(i + 1)]))
        use_residual = bool(model_parameters.iloc[i]['use_residual'])
        print(f'[{i + 1}/{len(datasets_parameters)}] INCEPTIONTIME {dataset["name"]}: num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, num_inception_blocks = {num_inception_blocks}, n_filters = {n_filters}, use_residual = {use_residual}')
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
            checkpoints = ModelCheckpoint(dirpath='models/inceptiontime/' + dataset["name"] + '/' + data_type + '/no_kfold', filename="{epoch}--{step}--{val_loss:.2f}--{val_accuracy:.2f}--{train_accuracy:.2f}", monitor='val_loss', mode='min', save_top_k=3, save_last=True)
            trainer = Trainer(gpus=GPUS, max_epochs=num_epochs, fast_dev_run=FAST_DEV_RUN, callbacks=checkpoints)
            if results.iloc[i][f'inceptiontime_{data_type}_train_accuracy'] == -1:
                model = InceptionTime(1, int(dataset['num_classes']), learning_rate, n_filters=[32, 32, 32, 32, 32], use_residual=use_residual)
                if has_val_dataset:
                    trainer.fit(model, train_loader, val_loader)
                else:
                    trainer.fit(model, train_loader)
                train_accuracy = trainer.test(model, train_loader)[0]['test_accuracy']
                test_accuracy = trainer.test(model, test_loader)[0]['test_accuracy']
                results.loc[i,f'inceptiontime_{data_type}_train_accuracy'] = train_accuracy
                results.loc[i,f'inceptiontime_{data_type}_test_accuracy'] = test_accuracy
                if has_val_dataset:
                    val_accuracy = trainer.test(model, val_loader)[0]['test_accuracy']
                    results.loc[i,f'inceptiontime_{data_type}_val_accuracy'] = val_accuracy
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
