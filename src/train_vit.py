import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from vit_pytorch import ViT
from torch.optim.lr_scheduler import StepLR
from utils import black_and_white_image_to_binary, plotting_time_normalization, load_image_datasets, get_dataloaders
from constants import TRAIN_NEW_MODELS, RESULTS, DATASETS_WITH_BAD_IMAGE_RESOLUTIONS, DATASETS_PATH


def test_model(model, data_loader, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            n_samples += label.size(0)
            n_correct += (predicted == label).sum().item()
        train_accuracy = n_correct / n_samples
        return train_accuracy

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
    if 'vit_train_accuracy' not in results.columns or TRAIN_NEW_MODELS:
        print('Never trained this model')
        results['vit_train_accuracy'] = np.full((len(results)), -1).tolist()
        results['vit_val_accuracy'] = np.full((len(results)), -1).tolist()
        results['vit_test_accuracy'] = np.full((len(results)), -1).tolist()
        results.to_csv(RESULTS, index=False)
    model_parameters = pd.read_excel(open('datasets_and_model_parameters.xlsx', 'rb'), sheet_name='vit_parameters')
    # training model
    for i, dataset in datasets_parameters.iterrows():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                # model_file_path = os.path.join(TRAINED_MODELS, 'vit_' + dataset['name'])
                model = None
                if results.iloc[i]['vit_train_accuracy'] == -1:
                    model = ViT(
                        image_size = image_size,
                        patch_size = patch_size,
                        num_classes = int(dataset['num_classes']),
                        dim = dim,
                        depth = depth,
                        heads = heads,
                        mlp_dim = mlp_dim,
                        pool = pool,
                        channels = 1,
                        dim_head = dim_head,
                        dropout = dropout,
                        emb_dropout = emb_dropout).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
                    for epoch in range(2):
                        epoch_loss = 0
                        epoch_accuracy = 0
                        for data, label in train_loader:
                            data = data.to(device)
                            label = label.to(device)
                            output = model(data)
                            loss = criterion(output, label)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            acc = (output.argmax(dim=1) == label).float().mean()
                            epoch_accuracy += acc / len(train_loader)
                            epoch_loss += loss / len(train_loader)
                        with torch.no_grad():
                            epoch_val_accuracy = 0
                            epoch_val_loss = 0
                            for data, label in val_loader:
                                data = data.to(device)
                                label = label.to(device)
                                val_output = model(data)
                                val_loss = criterion(val_output, label)
                                acc = (val_output.argmax(dim=1) == label).float().mean()
                                epoch_val_accuracy += acc / len(val_loader)
                                epoch_val_loss += val_loss / len(val_loader)
                        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}", end='\r')
                    train_accuracy = test_model(model, train_loader, device)
                    test_accuracy = test_model(model, test_loader, device)
                    results.loc[i,'vit_train_accuracy'] = train_accuracy
                    results.loc[i,'vit_test_accuracy'] = test_accuracy
                    if has_val_dataset:
                        val_accuracy = test_model(model, val_loader, device)
                        results.loc[i,'vit_val_accuracy'] = val_accuracy
                        print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}, val_accuracy = {val_accuracy}')
                    else:
                        print(f'train_accuracy = {train_accuracy}, test_accuracy = {test_accuracy}')
                    results.to_csv(RESULTS, index=False)
                else:
                    print('Model already trained')
            else:
                print('Wrong shape')
        else:
            print('Bad image resolution')
    results.to_csv(RESULTS, index=False)
    return results