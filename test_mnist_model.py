import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import pytest
from mnist_cnn import TinyCNN, HyperParameters, load_and_evaluate_model
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = TinyCNN(dropout_rate=0.25)
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_model_architecture():
    model = TinyCNN(dropout_rate=0.25)
    
    # Check for BatchNorm
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"
    
    # Check for Dropout
    has_dropout = any(isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules())
    assert has_dropout, "Model should use Dropout"
    
    # Check for Fully Connected layer
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_fc, "Model should have at least one Fully Connected layer"

def test_model_accuracy():
    params = HyperParameters()
    device = torch.device('cpu')  # Explicitly use CPU for testing
    model, accuracy = load_and_evaluate_model('best_mnist_model.pth', params)
    assert accuracy >= 99.4, f"Model accuracy {accuracy:.2f}% is less than required 99.4%"

def test_epoch_count():
    params = HyperParameters()
    assert params.num_epochs <= 20, f"Number of epochs {params.num_epochs} should be less than or equal to 20" 