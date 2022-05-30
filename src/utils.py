import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets


def plotting_time_normalization(img, treshold=170):
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    thresh, img_array = cv2.threshold(img_array, treshold, 255, cv2.THRESH_BINARY)
    img_array[img_array == 0] = [1]
    img_array[img_array == 255] = [0]
    img = Image.fromarray(img_array)
    return img

def load_image_datasets(dataset_path, train_transform, test_transform, val_transform=None):
    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), test_transform)
    if val_transform == None:
        return train_dataset, test_dataset
    val_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'val'), val_transform)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(batch_size, train_dataset, test_dataset, val_dataset=None):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    if val_dataset == None:
        return train_loader, test_loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader