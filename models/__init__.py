# models/__init__.py

from .cnn import new_ResNet18, MNISTCNN
from .fully_connected import FullyConnectedMNIST, SimpleFCN, two_layer_fc

__all__ = [
    'new_ResNet18',
    'MNISTCNN',
    'FullyConnectedMNIST',
    'SimpleFCN',
    'two_layer_fc'
]
