#!/usr/bin/env python3
"""
Create a dummy emotion recognition model for testing purposes.
This script creates a simple PyTorch model that can be used for testing the emotion agent.
"""

import torch
import torch.nn as nn
import os
from pathlib import Path

class SimpleEmotionModel(nn.Module):
    """A simple neural network for emotion recognition."""
    
    def __init__(self, input_size=26, hidden_size=128, num_emotions=7):
        super(SimpleEmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_emotions)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_dummy_model():
    """Create and save a dummy emotion model."""
    
    # Create model
    model = SimpleEmotionModel()
    
    # Initialize with random weights (you would normally train this)
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.1, 0.1)
    
    # Set to evaluation mode
    model.eval()
    
    return model

if __name__ == "__main__":
    # Create model directory if it doesn't exist
    model_dir = Path(__file__).parent
    model_dir.mkdir(exist_ok=True)
    
    # Create and save dummy model
    model = create_dummy_model()
    model_path = model_dir / "emotion_model.pt"
    
    torch.save(model, model_path)
    print(f"Dummy emotion model saved to: {model_path}")
    print("This is a placeholder model for testing. Replace with your trained model.")
    
    # Test loading the model
    loaded_model = torch.load(model_path, map_location='cpu')
    print("Model loading test: SUCCESS")
