# MNIST TinyCNN Classifier

[![Model Tests](https://github.com/rshekhar93/MNIST_TinyCNN_Optimizer/actions/workflows/model_tests.yml/badge.svg)](https://github.com/rshekhar93/MNIST_TinyCNN_Optimizer/actions/workflows/model_tests.yml)

A lightweight CNN model for MNIST digit classification that achieves >99.4% accuracy with less than 20k parameters.

## Model Architecture

- 8 Convolutional layers organized in 4 blocks
- Uses Batch Normalization and Dropout for regularization
- Single Fully Connected layer at the end
- Total parameters: 19,354

### Architecture Details:


## Model Features

1. **Convolutional Blocks**:
   - First Block: Basic convolutions for initial feature extraction
   - Second Block: Introduces BatchNorm and first MaxPool
   - Third Block: Full regularization with Dropout2d
   - Fourth Block: Final feature refinement with 1x1 convolution

2. **Regularization**:
   - BatchNorm in later blocks
   - Strategic Dropout2d placement
   - Dropout before final layer

3. **Receptive Field**:
   - Gradually increases through the network
   - Final RF: 28x28 (covers entire input)

4. **Parameter Efficiency**:
   - Strategic channel expansion/reduction
   - 1x1 convolutions for channel reduction

## Training Configuration

- Epochs: 20
- Batch Size: 64
- Learning Rate: 0.0009
- Dropout Rate: 0.25
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

## Performance Metrics

Training Logs:
```
Model Summary:
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
              ReLU-2            [-1, 4, 28, 28]               0
            Conv2d-3            [-1, 8, 28, 28]             296
              ReLU-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 12, 28, 28]             876
       BatchNorm2d-6           [-1, 12, 28, 28]              24
              ReLU-7           [-1, 12, 28, 28]               0
            Conv2d-8           [-1, 16, 28, 28]           1,744
       BatchNorm2d-9           [-1, 16, 28, 28]              32
             ReLU-10           [-1, 16, 28, 28]               0
        MaxPool2d-11           [-1, 16, 14, 14]               0
           Conv2d-12           [-1, 20, 14, 14]           2,900
      BatchNorm2d-13           [-1, 20, 14, 14]              40
             ReLU-14           [-1, 20, 14, 14]               0
        Dropout2d-15           [-1, 20, 14, 14]               0
           Conv2d-16           [-1, 16, 14, 14]           2,896
      BatchNorm2d-17           [-1, 16, 14, 14]              32
             ReLU-18           [-1, 16, 14, 14]               0
        MaxPool2d-19             [-1, 16, 7, 7]               0
           Conv2d-20             [-1, 16, 7, 7]           2,320
      BatchNorm2d-21             [-1, 16, 7, 7]              32
             ReLU-22             [-1, 16, 7, 7]               0
        Dropout2d-23             [-1, 16, 7, 7]               0
           Conv2d-24             [-1, 16, 7, 7]             272
             ReLU-25             [-1, 16, 7, 7]               0
          Dropout-26                  [-1, 784]               0
           Linear-27                   [-1, 10]           7,850
================================================================
Total params: 19,354
Trainable params: 19,354
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.91
Params size (MB): 0.07
Estimated Total Size (MB): 0.99
----------------------------------------------------------------

Starting Training...

Epoch 1/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:44<00:00, 21.17it/s, loss=0.268, acc=92.49%] 
Epoch 1/20 - Loss: 0.2683, Train Acc: 92.49%, Test Acc: 98.13%
Epoch 2/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:44<00:00, 20.95it/s, loss=0.082, acc=97.51%]
Epoch 2/20 - Loss: 0.0824, Train Acc: 97.51%, Test Acc: 98.53%
Epoch 3/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.61it/s, loss=0.064, acc=98.04%]
Epoch 3/20 - Loss: 0.0643, Train Acc: 98.04%, Test Acc: 98.49%
Epoch 4/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:46<00:00, 20.07it/s, loss=0.056, acc=98.30%]
Epoch 4/20 - Loss: 0.0560, Train Acc: 98.30%, Test Acc: 99.01%
Epoch 5/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:48<00:00, 19.30it/s, loss=0.050, acc=98.48%]
Epoch 5/20 - Loss: 0.0504, Train Acc: 98.48%, Test Acc: 99.01%
Epoch 6/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:47<00:00, 19.57it/s, loss=0.048, acc=98.53%]
Epoch 6/20 - Loss: 0.0477, Train Acc: 98.53%, Test Acc: 99.05%
Epoch 7/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.64it/s, loss=0.044, acc=98.66%]
Epoch 7/20 - Loss: 0.0439, Train Acc: 98.66%, Test Acc: 99.11%
Epoch 8/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.45it/s, loss=0.040, acc=98.75%]
Epoch 8/20 - Loss: 0.0403, Train Acc: 98.75%, Test Acc: 99.20%
Epoch 9/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.68it/s, loss=0.039, acc=98.79%]
Epoch 9/20 - Loss: 0.0394, Train Acc: 98.79%, Test Acc: 99.08%
Epoch 10/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:44<00:00, 20.93it/s, loss=0.038, acc=98.83%]
Epoch 10/20 - Loss: 0.0382, Train Acc: 98.83%, Test Acc: 99.27%
Epoch 11/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.56it/s, loss=0.036, acc=98.86%]
Epoch 11/20 - Loss: 0.0362, Train Acc: 98.86%, Test Acc: 99.23%
Epoch 12/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.70it/s, loss=0.035, acc=98.93%]
Epoch 12/20 - Loss: 0.0355, Train Acc: 98.93%, Test Acc: 99.25%
Epoch 12/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.70it/s, loss=0.035, acc=98.93%] 
Epoch 12/20 - Loss: 0.0355, Train Acc: 98.93%, Test Acc: 99.25%
Epoch 13/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.54it/s, loss=0.031, acc=99.01%] 
Epoch 13/20 - Loss: 0.0310, Train Acc: 99.01%, Test Acc: 99.24%
Epoch 13/20 - Loss: 0.0310, Train Acc: 99.01%, Test Acc: 99.24%
Epoch 14/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:46<00:00, 20.22it/s, loss=0.031, acc=99.00%] 
Epoch 14/20 - Loss: 0.0310, Train Acc: 99.00%, Test Acc: 99.42%
Epoch 15/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:44<00:00, 20.86it/s, loss=0.031, acc=99.03%] 
Epoch 15/20 - Loss: 0.0308, Train Acc: 99.03%, Test Acc: 99.36%
Epoch 16/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:43<00:00, 21.81it/s, loss=0.028, acc=99.11%] 
Epoch 16/20 - Loss: 0.0285, Train Acc: 99.11%, Test Acc: 99.32%
Epoch 17/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:43<00:00, 21.58it/s, loss=0.027, acc=99.15%] 
Epoch 17/20 - Loss: 0.0272, Train Acc: 99.15%, Test Acc: 99.31%
Epoch 18/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:43<00:00, 21.57it/s, loss=0.027, acc=99.11%] 
Epoch 18/20 - Loss: 0.0272, Train Acc: 99.11%, Test Acc: 99.35%
Epoch 19/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:42<00:00, 22.05it/s, loss=0.027, acc=99.17%] 
Epoch 19/20 - Loss: 0.0269, Train Acc: 99.17%, Test Acc: 99.30%
Epoch 20/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:45<00:00, 20.83it/s, loss=0.025, acc=99.17%] 
Epoch 20/20 - Loss: 0.0255, Train Acc: 99.17%, Test Acc: 99.38%

Loaded best model from epoch 14 with test accuracy: 99.42%

Evaluating saved best model:
Current test accuracy: 99.42%
```

## Model Validation

The model is automatically tested for:
- Parameter count (<20k)
- Use of Batch Normalization
- Use of Dropout
- Presence of Fully Connected layer
- Accuracy threshold (>99.4%)
- Epoch limit (≤20)

## Usage

To evaluate the saved model:
```python
from mnist_cnn import HyperParameters, load_and_evaluate_model

params = HyperParameters()
model, accuracy = load_and_evaluate_model('best_mnist_model.pth', params)
print(f'Model accuracy: {accuracy:.2f}%')
```

## Requirements

- PyTorch
- torchvision
- tqdm
- matplotlib
- torchsummary
- pytest (for testing)

## Testing

Run the automated tests:
```bash
pytest test_mnist_model.py -v
```

## GitHub Actions

Continuous Integration is set up to automatically test:
- Model architecture requirements
- Parameter count limits
- Performance thresholds
- Training configuration

The workflow runs on every push and pull request to the main branch.