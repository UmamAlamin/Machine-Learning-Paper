import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    class ConvLayer(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, activation='tanh'):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
            
        def forward(self, x):
            return self.activation(self.conv(x))
    
    class SubsamplingLayer(nn.Module):
        def __init__(self, kernel_size, activation='tanh'):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size)
            self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
            
        def forward(self, x):
            return self.activation(self.pool(x))
    
    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features, out_features, activation='tanh', output_layer=False):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)
            self.output_layer = output_layer
            if not output_layer:
                self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
            
        def forward(self, x):
            x = self.fc(x)
            if not self.output_layer:
                x = self.activation(x)
            return x
    
    def __init__(self, num_classes=10, activation='tanh'):
        super(LeNet5, self).__init__()
        self.activation = activation
        
        # Layer C1: Convolutional layer
        self.c1 = self.ConvLayer(1, 6, 5, activation)
        
        # Layer S2: Subsampling layer
        self.s2 = self.SubsamplingLayer(2, activation)
        
        # Layer C3: Convolutional layer
        self.c3 = self.ConvLayer(6, 16, 5, activation)
        
        # Layer S4: Subsampling layer
        self.s4 = self.SubsamplingLayer(2, activation)
        
        # Layer C5: Convolutional layer
        self.c5 = self.ConvLayer(16, 120, 5, activation)
        
        # Layer F6: Fully connected layer
        self.f6 = self.FullyConnectedLayer(120, 84, activation)
        
        # Output layer
        self.output = self.FullyConnectedLayer(84, num_classes, activation, output_layer=True)
        
    def forward(self, x):
        # Feature extraction
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)
        
        # Flatten
        x = x.view(-1, 120)
        
        # Classification
        x = self.f6(x)
        logits = self.output(x)
        
        return logits