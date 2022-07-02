import torch
import torch.nn as nn
from vit_pytorch import ViT
import pytorch_lightning as pl
from torchmetrics import Accuracy

class Visual_Transformer(pl.LightningModule):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool, channels, dim_head, dropout, emb_dropout, learning_rate):
        super(Visual_Transformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.channels = channels
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.learning_rate = learning_rate
        self.model = ViT(
            image_size = self.image_size,
            patch_size = self.patch_size,
            num_classes = self.num_classes,
            dim = self.dim,
            depth = self.depth,
            heads = self.heads,
            mlp_dim = self.mlp_dim,
            pool = self.pool,
            channels = self.channels,
            dim_head = self.dim_head,
            dropout = self.dropout,
            emb_dropout = self.emb_dropout
        )
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
    
    def forward(self, x):
        x = self.model(x)
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