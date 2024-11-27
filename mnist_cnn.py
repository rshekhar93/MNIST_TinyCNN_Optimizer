import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

# Hyperparameters class for easy tuning
class HyperParameters:
    def __init__(self):
        self.num_epochs = 20
        self.batch_size = 512  # Changed from list to single value
        self.learning_rate = 0.001
        self.dropout_rate = 0.25
        self.num_classes = 10
        self.momentum = 0.9
        self.random_seed = 42

# CNN Model Definition
# class TinyCNN(nn.Module):
#     def __init__(self, dropout_rate):
#         super(TinyCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 8x28x28
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16x14x14
#         self.fc1 = nn.Linear(32 * 7 * 7, 10)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)  # 8x14x14
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)  # 16x7x7
#         x = x.view(-1, 32 * 7 * 7)
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)
    
class TinyCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(TinyCNN, self).__init__()
        
        # First Convolutional Block - No BatchNorm or MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),  # RF: 3x3
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),  # RF: 5x5
            nn.ReLU()
        )
        
        # Second Convolutional Block - Add BatchNorm and MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, padding=1),  # RF: 7x7
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=3, padding=1),  # RF: 9x9
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # RF: 10x10, Size: 14x14
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=3, padding=1),  # RF: 14x14
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(20, 16, kernel_size=3, padding=1),  # RF: 18x18
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # RF: 20x20, Size: 7x7
        )
        
        # Fourth Convolutional Block - Reduced channels from 20 to 18
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 18, kernel_size=3, padding=1),  # RF: 28x28 (reduced from 20 to 18 channels)
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(18, 16, kernel_size=1),  # RF: 28x28 (1x1 conv)
            nn.ReLU()
        )
        
        # Single Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(16 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(params):
    # Set random seed for reproducibility
    torch.manual_seed(params.random_seed)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyCNN(params.dropout_rate).to(device)
    
    # Print model summary
    print("\nModel Summary:")
    summary(model, input_size=(1, 28, 28))
    print("\nStarting Training...\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Track best model
    best_test_accuracy = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(params.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{params.num_epochs}')
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{running_loss/len(progress_bar):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        # Save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            print(f'New best model saved with test accuracy: {test_accuracy:.2f}%')
        
        # Modified metrics printing
        print(f'Epoch {epoch+1}/{params.num_epochs} - Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
    
    # Load best model before returning
    model.load_state_dict(best_model_state)
    print(f'\nLoaded best model with test accuracy: {best_test_accuracy:.2f}%')
    
    # Save the best model to disk (modified)
    torch.save(
        best_model_state,  # Save only the model state dict
        'best_mnist_model.pth'
    )
    
    return model, train_losses, train_accuracies, test_accuracies

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def plot_metrics(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def batch_size_analysis(params):
    results = {}
    
    for batch_size in params.batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        params.batch_size = batch_size
        
        # Train model with current batch size
        model, train_losses, train_accuracies, test_accuracies = train_model(params)
        
        # Store results
        results[batch_size] = {
            'final_train_acc': train_accuracies[-1],
            'final_test_acc': test_accuracies[-1],
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Plot training curves for different batch sizes
    plt.subplot(1, 2, 1)
    for batch_size in params.batch_sizes:
        plt.plot(results[batch_size]['train_accuracies'], 
                label=f'Batch {batch_size}')
    plt.title('Training Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot final accuracies vs batch size
    plt.subplot(1, 2, 2)
    batch_sizes = list(results.keys())
    train_accs = [results[bs]['final_train_acc'] for bs in batch_sizes]
    test_accs = [results[bs]['final_test_acc'] for bs in batch_sizes]
    
    plt.plot(batch_sizes, train_accs, 'o-', label='Train')
    plt.plot(batch_sizes, test_accs, 'o-', label='Test')
    plt.title('Final Accuracy vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nSummary of results:")
    print("Batch Size | Train Acc | Test Acc")
    print("-" * 35)
    for batch_size in params.batch_sizes:
        print(f"{batch_size:^10d} | {results[batch_size]['final_train_acc']:^9.2f} | {results[batch_size]['final_test_acc']:^8.2f}")

# Add function to load and evaluate saved model
def load_and_evaluate_model(model_path, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = TinyCNN(params.dropout_rate).to(device)
    
    # Load saved model (modified)
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    print(f"Loaded model weights from {model_path}")
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    
    # Evaluate
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'Current test accuracy: {test_accuracy:.2f}%')
    
    return model, test_accuracy

if __name__ == '__main__':
    # Initialize hyperparameters
    params = HyperParameters()
    
    # Train the model
    model, train_losses, train_accuracies, test_accuracies = train_model(params)
    
    # Plot training metrics
    plot_metrics(train_losses, train_accuracies, test_accuracies)
    
    # Load and evaluate best saved model
    print("\nEvaluating saved best model:")
    best_model, final_accuracy = load_and_evaluate_model('best_mnist_model.pth', params) 