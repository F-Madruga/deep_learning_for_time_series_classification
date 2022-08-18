import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets
from sklearn.model_selection import train_test_split
from constants import DATASETS_PATH
from sklearn.model_selection import KFold


def plotting_time_normalization(img, treshold=170):
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    thresh, img_array = cv2.threshold(img_array, treshold, 255, cv2.THRESH_BINARY)
    img_array[img_array == 0] = [1]
    img_array[img_array == 255] = [0]
    img = Image.fromarray(img_array)
    return img

def black_and_white_image_to_binary(img):
    img_array = np.array(img)
    img_array[img_array == 0] = [0]
    img_array[img_array == 255] = [1]
    img = Image.fromarray(img_array)
    return img

def load_image_datasets(dataset_path, train_transform, test_transform, val_transform=None):
    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), test_transform)
    if val_transform == None:
        return train_dataset, test_dataset
    val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), val_transform)
    return train_dataset, val_dataset, test_dataset

def load_image_datasets_from_paths(train_path, test_path, train_transform, test_transform, val_path=None, val_transform=None):
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    test_dataset = datasets.ImageFolder(test_path, test_transform)
    if val_transform == None:
        return train_dataset, test_dataset
    val_dataset = datasets.ImageFolder(val_path, val_transform)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(batch_size, train_dataset, test_dataset, val_dataset=None):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    if val_dataset == None:
        return train_loader, test_loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def undo_all_split_train_dataset(datasets_parameters, data_type):
    for i, dataset in datasets_parameters.iterrows():
        undo_split_train_dataset(dataset, data_type)

def undo_split_train_dataset(dataset, data_type):
    dataset_path = os.path.join(DATASETS_PATH, dataset['name'], data_type)
    train_path = os.path.join(dataset_path, 'train')
    if os.path.isdir(train_path):
        for split in os.listdir(dataset_path):
            if split != 'train' and split != 'test' and split != 'encoder.pkl':
                split_path = os.path.join(dataset_path, split)
                for dataset_class in os.listdir(split_path):
                    val_class_path = os.path.join(split_path, dataset_class)
                    train_class_path = os.path.join(train_path, dataset_class)
                    if not os.path.isdir(train_class_path):
                        os.mkdir(train_class_path)
                    for img in os.listdir(val_class_path):
                        img_path = os.path.join(val_class_path, img)
                        new_img_path = os.path.join(train_class_path, img)
                        os.rename(img_path, new_img_path)
                    os.rmdir(val_class_path)
                os.rmdir(split_path)

def kfold_split_train_dataset(datasets_parameters, data_type, n_splits=5, shuffle=True, random_state=None):
    undo_all_split_train_dataset(datasets_parameters, data_type)
    for i, dataset in datasets_parameters.iterrows():
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
            for train_index, test_index in kfold.split(imgs):
                num_fold += 1
                X_train, X_test = imgs[train_index], imgs[test_index]
                # print('Train index', train_index.tolist())
                # print('Test index', test_index)
                print('Train', X_train)
                print('Test', X_test)



def split_train_dataset_to_val(datasets_parameters, data_type):
    for i, dataset in datasets_parameters.iterrows():
        val_path = os.path.join(DATASETS_PATH, dataset['name'], data_type, 'val')
        train_path = os.path.join(DATASETS_PATH, dataset['name'], data_type, 'train')
        if os.path.isdir(val_path):
            for dataset_class in os.listdir(val_path):
                val_class_path = os.path.join(val_path, dataset_class)
                for img in os.listdir(val_class_path):
                    img_path = os.path.join(val_class_path, img)
                    path_splits = img_path.split('/')
                    path_splits[3] = path_splits[3].replace('val', 'train')
                    class_path = '/'.join(path_splits[:5])
                    if not os.path.isdir(class_path):
                        os.mkdir(class_path)
                    new_path = '/'.join(path_splits)
                    os.rename(img_path, new_path)
                os.rmdir(val_class_path)
        else:                    
            os.mkdir(val_path)
        imgs = []
        for dataset_class in os.listdir(train_path):
                val_class_path = os.path.join(val_path, dataset_class)
                train_class_path = os.path.join(train_path, dataset_class)
                for img in os.listdir(train_class_path):
                    img_path = os.path.join(train_class_path, img)
                    imgs.append(img_path)
        _, val_dataset = train_test_split(imgs, test_size=int(dataset['val_size']) * 0.01)
        for img_path in val_dataset:
            path_splits = img_path.split('/')
            path_splits[3] = path_splits[3].replace('train', 'val')
            class_path = '/'.join(path_splits[:5])
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            new_path = '/'.join(path_splits)
            os.rename(img_path, new_path)
            path_splits = img_path.split('/')
            class_path = '/'.join(path_splits[:5])
            if len(os.listdir(class_path)) == 0:
                os.rmdir(class_path)