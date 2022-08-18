import os
import pandas as pd
import numpy as np
from vit import Visual_Transformer
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from utils import black_and_white_image_to_binary, load_image_datasets_from_paths, plotting_time_normalization, load_image_datasets, get_dataloaders, undo_split_train_dataset
from constants import FAST_DEV_RUN, GPUS, TRAIN_NEW_MODELS, RESULTS, DATASETS_WITH_BAD_IMAGE_RESOLUTIONS, DATASETS_PATH

def train_vit_with_kfold(datasets_parameters, data_type, results, n_splits=5, shuffle=True, random_state=None):
     # transformers
    train_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    val_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    test_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    if data_type == 'line_plots':
        train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
        val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
        test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
    # add column for model results
    if f'vit_kfold_0_{data_type}_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        for i in range(n_splits):
            results[f'vit_kfold_{i}_{data_type}_train_accuracy'] = np.full((len(results)), -1).tolist()
            results[f'vit_kfold_{i}_{data_type}_val_accuracy'] = np.full((len(results)), -1).tolist()
            results[f'vit_kfold_{i}_{data_type}_test_accuracy'] = np.full((len(results)), -1).tolist()
            results[f'vit_kfold_{i}_{data_type}_train_loss'] = np.full((len(results)), -1).tolist()
            results[f'vit_kfold_{i}_{data_type}_val_loss'] = np.full((len(results)), -1).tolist()
            results[f'vit_kfold_{i}_{data_type}_test_loss'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='vit_parameters')
    for i, dataset in datasets_parameters.iterrows():
        undo_split_train_dataset(dataset, data_type)
        # model parameters
        num_epochs = int(model_parameters.iloc[i]['epochs'])
        batch_size = int(model_parameters.iloc[i]['batch_size'])
        learning_rate = float(model_parameters.iloc[i]['learning_rate'])
        gamma = float(model_parameters.iloc[i]['gamma'])
        dim = int(model_parameters.iloc[i]['dim'])
        depth = int(model_parameters.iloc[i]['depth'])
        heads = int(model_parameters.iloc[i]['heads'])
        mlp_dim = int(model_parameters.iloc[i]['mlp_dim'])
        dropout = float(model_parameters.iloc[i]['dropout'])
        dim_head = int(model_parameters.iloc[i]['dim_head'])
        emb_dropout = float(model_parameters.iloc[i]['emb_dropout'])
        pool = str(model_parameters.iloc[i]['pool'])
        print(f'[{i + 1}/{len(datasets_parameters)}] VIT {dataset["name"]}: num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, gamma = {gamma}, dim = {dim}, depth = {depth}, heads = {heads}, mlp_dim = {mlp_dim}, dropout = {dropout}, dim_head = {dim_head}, emb_dropout = {emb_dropout}, pool = {pool}')
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
                    if results.iloc[i][f'vit_kfold_{num_fold}_{data_type}_train_accuracy'] == -1:
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
                        checkpoints = ModelCheckpoint(dirpath='models/vit/' + dataset["name"] + '/' + data_type + '/' + 'kfold_' + str(num_fold), filename="{epoch}--{step}--{val_loss:.2f}--{val_accuracy:.2f}--{train_accuracy:.2f}", monitor='val_loss', mode='min', save_top_k=3, save_last=True)
                        trainer = Trainer(gpus=GPUS, max_epochs=num_epochs, fast_dev_run=FAST_DEV_RUN, callbacks=[checkpoints])
                        train_dataset, val_dataset, _ = load_image_datasets_from_paths(os.path.join(dataset_path, f'train_kfold_{num_fold}'), os.path.join(dataset_path, 'test'), train_transform, test_transform, os.path.join(dataset_path, f'val_kfold_{num_fold}'), val_transform)
                        train_loader, val_loader, _ = get_dataloaders(batch_size, train_dataset, _, val_dataset)
                        if train_dataset[0][0].shape[1] == 288 and train_dataset[0][0].shape[2] == 432:
                            patch_size = 0
                            image_size = train_dataset[0][0].shape[1] if train_dataset[0][0].shape[1] > train_dataset[0][0].shape[2] else train_dataset[0][0].shape[2]
                            for patch_size in range(32, image_size):
                                if image_size % patch_size == 0:
                                    break
                            model = Visual_Transformer(
                                image_size,
                                patch_size,
                                int(dataset['num_classes']),
                                dim,
                                depth,
                                heads,
                                mlp_dim,
                                pool,
                                1,
                                dim_head,
                                dropout,
                                emb_dropout,
                                learning_rate
                            )
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
                            results.loc[i,f'vit{num_fold}_{data_type}_train_accuracy'] = train_accuracy
                            results.loc[i,f'vit_kfold_{num_fold}_{data_type}_val_accuracy'] = val_accuracy
                            results.loc[i,f'vit_kfold_{num_fold}_{data_type}_test_accuracy'] = test_accuracy
                            results.loc[i,f'vit_kfold_{num_fold}_{data_type}_train_loss'] = train_loss
                            results.loc[i,f'vit_kfold_{num_fold}_{data_type}_val_loss'] = val_loss
                            results.loc[i,f'vit_kfold_{num_fold}_{data_type}_test_loss'] = test_loss
                            results.to_csv(RESULTS, index=False)
                            print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}, val_accuracy = {val_accuracy}')
                            print(f'train_loss = {train_loss}, test_loss = {test_loss}, val_loss = {val_loss}')
                    num_fold += 1
                undo_split_train_dataset(dataset, data_type)
    return results


def train_vit(datasets_parameters, data_type, results):
    # transformers
    train_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    val_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    test_transform = transforms.Compose([lambda x: black_and_white_image_to_binary(x), transforms.ToTensor()])
    if data_type == 'line_plots':
        train_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
        val_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
        test_transform = transforms.Compose([lambda x: plotting_time_normalization(x), transforms.ToTensor()])
    # add column for model results
    if f'vit_{data_type}_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        results[f'vit_{data_type}_train_accuracy'] = np.full((len(results)), -1).tolist()
        results[f'vit_{data_type}_val_accuracy'] = np.full((len(results)), -1).tolist()
        results[f'vit_{data_type}_test_accuracy'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='vit_parameters')
    # training model
    for i, dataset in datasets_parameters.iterrows():
        # model parameters
        num_epochs = int(model_parameters.iloc[i]['epochs'])
        batch_size = int(model_parameters.iloc[i]['batch_size'])
        learning_rate = float(model_parameters.iloc[i]['learning_rate'])
        gamma = float(model_parameters.iloc[i]['gamma'])
        dim = int(model_parameters.iloc[i]['dim'])
        depth = int(model_parameters.iloc[i]['depth'])
        heads = int(model_parameters.iloc[i]['heads'])
        mlp_dim = int(model_parameters.iloc[i]['mlp_dim'])
        dropout = float(model_parameters.iloc[i]['dropout'])
        dim_head = int(model_parameters.iloc[i]['dim_head'])
        emb_dropout = float(model_parameters.iloc[i]['emb_dropout'])
        pool = str(model_parameters.iloc[i]['pool'])
        print(f'[{i + 1}/{len(datasets_parameters)}] VIT {dataset["name"]}: num_epochs = {num_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate}, gamma = {gamma}, dim = {dim}, depth = {depth}, heads = {heads}, mlp_dim = {mlp_dim}, dropout = {dropout}, dim_head = {dim_head}, emb_dropout = {emb_dropout}, pool = {pool}')
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
            if train_dataset[0][0].shape[1] == 288 and train_dataset[0][0].shape[2] == 432:
                patch_size = 0
                image_size = train_dataset[0][0].shape[1] if train_dataset[0][0].shape[1] > train_dataset[0][0].shape[2] else train_dataset[0][0].shape[2]
                for patch_size in range(32, image_size):
                    if image_size % patch_size == 0:
                        break
                print(f'image_size = {image_size}, patch_size = {patch_size}, num_classes = {dataset["num_classes"]}, dim = {dim}, depth = {depth}, heads = {heads}, mlp_dim = {mlp_dim}, pool = {pool}, dim_head = {dim_head}, dropout = {dropout}, emb_dropout = {emb_dropout}')
                model = None
                checkpoints = ModelCheckpoint(dirpath='models/vit/' + dataset["name"] + '/' + data_type + '/no_kfold', filename="{epoch}--{step}--{val_loss:.2f}--{val_accuracy:.2f}--{train_accuracy:.2f}", monitor='val_loss', mode='min', save_top_k=3, save_last=True)
                trainer = Trainer(gpus=GPUS, max_epochs=num_epochs, fast_dev_run=FAST_DEV_RUN, callbacks=checkpoints)
                if results.iloc[i][f'vit_{data_type}_train_accuracy'] == -1:
                    model = Visual_Transformer(
                        image_size,
                        patch_size,
                        int(dataset['num_classes']),
                        dim,
                        depth,
                        heads,
                        mlp_dim,
                        pool,
                        1,
                        dim_head,
                        dropout,
                        emb_dropout,
                        learning_rate
                    )
                    if has_val_dataset:
                        trainer.fit(model, train_loader, val_loader)
                    else:
                        trainer.fit(model, train_loader)
                    train_accuracy = trainer.test(model, train_loader)[0]['test_accuracy']
                    test_accuracy = trainer.test(model, test_loader)[0]['test_accuracy']
                    results.loc[i,f'vit_{data_type}_train_accuracy'] = train_accuracy
                    results.loc[i,f'vit_{data_type}_test_accuracy'] = test_accuracy
                    if has_val_dataset:
                        val_accuracy = trainer.test(model, val_loader)[0]['test_accuracy']
                        results.loc[i,f'vit_{data_type}_val_accuracy'] = val_accuracy
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