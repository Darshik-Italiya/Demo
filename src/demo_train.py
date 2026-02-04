"""
Demo Training Script for RiverGen Custom Model
This script demonstrates a PyTorch training function compatible with RiverGen's Custom Model Studio.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import sys

def train(model, dataloader, epochs=10, learning_rate=0.001, 
          batch_size=32, optimizer='adam', loss_function='cross_entropy'):
    """
    Training function for custom model
    
    This function will be called by RiverGen's training infrastructure.
    All logs and metrics printed in specific format will be captured automatically.
    
    Args:
        model: PyTorch model to train
        dataloader: DataLoader with training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size (informational, dataloader should already have this)
        optimizer: Optimizer type ('adam' or 'sgd')
        loss_function: Loss function type
    
    Returns:
        Trained model
    """
    
    # Initialize optimizer based on parameter
    if optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Initialize loss function
    if loss_function.lower() == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function.lower() == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training: {epochs} epochs, LR={learning_rate}, Optimizer={optimizer}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        # Print metrics in format that RiverGen can capture
        print(f'[EPOCH {epoch}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Also output as JSON for automatic metric capture
        metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': learning_rate
        }
        print(f'METRICS: {json.dumps(metrics)}')
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'CHECKPOINT_SAVED: {checkpoint_path}')
    
    # Save final model
    final_model_path = 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'MODEL_SAVED: {final_model_path}')
    
    return model


def main():
    """
    Example usage (for local testing)
    RiverGen will call the train() function directly
    """
    # Create dummy model and data for testing
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Dummy dataset
    dummy_data = torch.randn(1000, 10)
    dummy_labels = torch.randint(0, 10, (1000,))
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train with default parameters
    trained_model = train(model, dataloader, epochs=5)
    print("Training completed!")


if __name__ == "__main__":
    main()
