import torch
import torch.nn as nn
from residual_block import ResidualBlock
import pytorch_lightning as pl
from torchmetrics import Accuracy


class ResNet(pl.LightningModule):
    def __init__(self, block, layers, image_channels, num_classes, learning_rate):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.train_accuracy(outputs, labels)
        self.log('train_loss', loss)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.val_accuracy(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)
    
    def test_step(self, test_batch, batch_idx):
        images, labels = test_batch
        outputs = self(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        self.test_accuracy(outputs, labels)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy)


    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels * 4,
                                                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(intermediate_channels * 4),)
        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)

    @classmethod
    def ResNet50(cls, img_channels, num_classes, learning_rate):
        return ResNet(ResidualBlock, [3, 4, 6, 3], img_channels, num_classes, learning_rate)

    @classmethod
    def ResNet101(cls, img_channels, num_classes, learning_rate):
        return ResNet(ResidualBlock, [3, 4, 23, 3], img_channels, num_classes, learning_rate)

    @classmethod
    def ResNet152(cls, img_channels, num_classes, learning_rate):
        return ResNet(ResidualBlock, [3, 8, 36, 3], img_channels, num_classes, learning_rate)
