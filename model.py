import torch.nn as nn
import torch
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class BaseModel:
    """Base class for all anomaly detection models"""
    def __init__(self, input_dims):
        self.input_dims = input_dims
        self.model_type = 'base'
    
    def fit(self, X_train):
        """Train the model"""
        pass
    
    def predict(self, X):
        """Make predictions"""
        pass
    
    def get_anomaly_score(self, X):
        """Calculate anomaly scores"""
        pass


class DeepAutoencoder(nn.Module, BaseModel):
    """Deep autoencoder with optional convolutional layers and attention mechanism"""
    def __init__(self, input_dims, use_conv=1, use_attention=1, n_heads=4):
        nn.Module.__init__(self)
        BaseModel.__init__(self, input_dims)
        self.model_type = 'deep_autoencoder'
        self.use_conv = use_conv
        self.use_attention = use_attention
        self.n_heads = n_heads
        print("input_dims:", input_dims)
        if self.use_conv == 1:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(input_dims, 128)
        self.Linear2 = nn.Linear(128, 64)
        self.Linear3 = nn.Linear(64, 64)
        self.Linear4 = nn.Linear(64, 32)
        if self.use_attention == 1:
            self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=n_heads)

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        if self.use_conv == 1:
            x_sequence = x.unsqueeze(1)
            # 卷积层
            conv1_output = self.conv1(x_sequence)
            conv2_output = self.conv2(conv1_output)
            conv3_output = self.conv3(conv2_output)
            conv1_output = conv1_output.squeeze(1)
            conv2_output = conv2_output.squeeze(1)
            conv3_output = conv3_output.squeeze(1)

            x = self.Linear1(x)
            x = x + conv1_output
            x = self.Linear2(x)
            x = x + conv2_output
            x = self.Linear3(x)
            x = self.Linear4(x)
            x = x + conv3_output

        else:
            x = self.Linear1(x)
            x = self.Linear2(x)
            x = self.Linear3(x)
            x = self.Linear4(x)

        if self.use_attention == 1:
            x = x.unsqueeze(0)
            # 多头注意力机制
            attn_output, _ = self.attention(x, x, x)
            attn_output = attn_output.squeeze(0)
            # 融合
            x = x.squeeze(0) + attn_output
            x = self.decoder(x)
        else:
            x = self.decoder(x)
        return x
    
    def fit(self, X_train, criterion=None, optimizer=None, epochs=40, batch_size=512, device='cpu'):
        """Train the autoencoder"""
        if criterion is None:
            criterion = nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            
        self.to(device)
        self.train()
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    
    def get_anomaly_score(self, X):
        """Calculate reconstruction error as anomaly score"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructions = self(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
        return mse.numpy()


class SimpleAutoencoder(nn.Module, BaseModel):
    """Simple autoencoder without convolutional layers or attention"""
    def __init__(self, input_dims, hidden_dims=32):
        nn.Module.__init__(self)
        BaseModel.__init__(self, input_dims)
        self.model_type = 'simple_autoencoder'
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dims)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dims)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, X_train, criterion=None, optimizer=None, epochs=40, batch_size=512, device='cpu'):
        """Train the autoencoder"""
        if criterion is None:
            criterion = nn.MSELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            
        self.to(device)
        self.train()
        
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    
    def get_anomaly_score(self, X):
        """Calculate reconstruction error as anomaly score"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructions = self(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1)
        return mse.numpy()


class IsolationForestModel(BaseModel):
    """Isolation Forest model for anomaly detection"""
    def __init__(self, input_dims, n_estimators=100, contamination='auto', random_state=42):
        super().__init__(input_dims)
        self.model_type = 'isolation_forest'
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
    
    def fit(self, X_train):
        """Train the Isolation Forest model"""
        self.model.fit(X_train)
        return self
    
    def predict(self, X):
        """Predict anomalies (1 for anomalies, 0 for normal)"""
        # IsolationForest returns -1 for anomalies and 1 for normal data
        # We convert to 1 for anomalies and 0 for normal to match our convention
        return (self.model.predict(X) == -1).astype(int)
    
    def get_anomaly_score(self, X):
        """Calculate anomaly scores"""
        # Convert decision_function output to positive anomaly scores
        # Higher score = more anomalous
        return -self.model.decision_function(X)


class LOFModel(BaseModel):
    """Local Outlier Factor model for anomaly detection"""
    def __init__(self, input_dims, n_neighbors=20, contamination='auto'):
        super().__init__(input_dims)
        self.model_type = 'lof'
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True  # Enable predict and decision_function methods
        )
    
    def fit(self, X_train):
        """Train the LOF model"""
        self.model.fit(X_train)
        return self
    
    def predict(self, X):
        """Predict anomalies (1 for anomalies, 0 for normal)"""
        # LOF returns -1 for anomalies and 1 for normal data
        # We convert to 1 for anomalies and 0 for normal to match our convention
        return (self.model.predict(X) == -1).astype(int)
    
    def get_anomaly_score(self, X):
        """Calculate anomaly scores"""
        # Convert decision_function output to positive anomaly scores
        # Higher score = more anomalous
        return -self.model.decision_function(X)


# For backward compatibility
Autoencoder = DeepAutoencoder
