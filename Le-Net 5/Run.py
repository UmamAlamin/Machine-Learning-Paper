import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import the LeNet5 model from LeNet5.py
from LeNet5 import LeNet5

# Function to load MNIST data
def load_mnist_data(batch_size=64):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # LeNet-5 expects 32x32 images
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Load test data
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Function to train the model
def train_model(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return correct / total


# Function to validate the model
def validate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            # Sum up batch loss
            test_loss += criterion(outputs, target).item()
            
            # Get predictions
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {100.*accuracy:.2f}%')
    
    return accuracy, test_loss, all_preds, all_targets


# Function to plot training results
def plot_results(tanh_history, relu_history, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training accuracy
    ax1.plot(range(1, epochs+1), tanh_history['train_acc'], 'b-', label='Tanh - Train')
    ax1.plot(range(1, epochs+1), relu_history['train_acc'], 'r-', label='ReLU - Train')
    ax1.plot(range(1, epochs+1), tanh_history['val_acc'], 'b--', label='Tanh - Validation')
    ax1.plot(range(1, epochs+1), relu_history['val_acc'], 'r--', label='ReLU - Validation')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Tanh vs ReLU: Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training loss
    ax2.plot(range(1, epochs+1), tanh_history['train_loss'], 'b-', label='Tanh - Train')
    ax2.plot(range(1, epochs+1), relu_history['train_loss'], 'r-', label='ReLU - Train')
    ax2.plot(range(1, epochs+1), tanh_history['val_loss'], 'b--', label='Tanh - Validation')
    ax2.plot(range(1, epochs+1), relu_history['val_loss'], 'r--', label='ReLU - Validation')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Tanh vs ReLU: Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt.gcf()


# Main function to run the experiment
def run_experiment(epochs=10, batch_size=64, learning_rate=0.001):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize models
    tanh_model = LeNet5(activation='tanh').to(device)
    relu_model = LeNet5(activation='relu').to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizers
    tanh_optimizer = optim.Adam(tanh_model.parameters(), lr=learning_rate)
    relu_optimizer = optim.Adam(relu_model.parameters(), lr=learning_rate)
    
    # Training history
    tanh_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    relu_history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    print("Starting training with Tanh activation...")
    tanh_train_time = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        tanh_train_acc = train_model(tanh_model, train_loader, tanh_optimizer, criterion, device, epoch)
        tanh_val_acc, tanh_val_loss, _, _ = validate_model(tanh_model, test_loader, criterion, device)
        end_time = time.time()
        tanh_train_time += (end_time - start_time)
        
        # Calculate training loss
        tanh_model.eval()
        tanh_train_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                outputs = tanh_model(data)
                tanh_train_loss += criterion(outputs, target).item()
        tanh_train_loss /= len(train_loader)
        
        # Save history
        tanh_history['train_acc'].append(tanh_train_acc)
        tanh_history['val_acc'].append(tanh_val_acc)
        tanh_history['train_loss'].append(tanh_train_loss)
        tanh_history['val_loss'].append(tanh_val_loss)
    
    print("\nStarting training with ReLU activation...")
    relu_train_time = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        relu_train_acc = train_model(relu_model, train_loader, relu_optimizer, criterion, device, epoch)
        relu_val_acc, relu_val_loss, _, _ = validate_model(relu_model, test_loader, criterion, device)
        end_time = time.time()
        relu_train_time += (end_time - start_time)
        
        # Calculate training loss
        relu_model.eval()
        relu_train_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                outputs = relu_model(data)
                relu_train_loss += criterion(outputs, target).item()
        relu_train_loss /= len(train_loader)
        
        # Save history
        relu_history['train_acc'].append(relu_train_acc)
        relu_history['val_acc'].append(relu_val_acc)
        relu_history['train_loss'].append(relu_train_loss)
        relu_history['val_loss'].append(relu_val_loss)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("Tanh model:")
    _, _, tanh_preds, tanh_targets = validate_model(tanh_model, test_loader, criterion, device)
    
    print("\nReLU model:")
    _, _, relu_preds, relu_targets = validate_model(relu_model, test_loader, criterion, device)
    
    # Plot results
    results_fig = plot_results(tanh_history, relu_history, epochs)
    
    # Plot confusion matrices
    tanh_cm_fig = plot_confusion_matrix(tanh_targets, tanh_preds, 'Confusion Matrix - Tanh Activation')
    relu_cm_fig = plot_confusion_matrix(relu_targets, relu_preds, 'Confusion Matrix - ReLU Activation')
    
    # Print training time comparison
    print(f"\nTraining time comparison:")
    print(f"Tanh model: {tanh_train_time:.2f} seconds")
    print(f"ReLU model: {relu_train_time:.2f} seconds")
    print(f"ReLU is {tanh_train_time/relu_train_time:.2f}x faster than Tanh")
    
    return {
        'tanh_model': tanh_model,
        'relu_model': relu_model,
        'tanh_history': tanh_history,
        'relu_history': relu_history,
        'results_fig': results_fig,
        'tanh_cm_fig': tanh_cm_fig,
        'relu_cm_fig': relu_cm_fig
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the experiment
    results = run_experiment(epochs=5)
    
    # Save figures
    results['results_fig'].savefig('tanh_vs_relu_performance.png')
    results['tanh_cm_fig'].savefig('tanh_confusion_matrix.png')
    results['relu_cm_fig'].savefig('relu_confusion_matrix.png')
    
    print("\nExperiment completed. Results saved to disk.")