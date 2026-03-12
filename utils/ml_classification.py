"""
Paderborn Bearing Dataset - ML Classification
===============================================
Classification pipeline:
  1. Traditional ML (CART, RF, GBT, kNN, XGB) with hand-crafted features
  2. 1D-CNN on raw vibration segments (requires GPU for practical training)
  3. 2D-CNN on STFT/CWT images (architecture defined; not yet wired into pipeline)
  4. Evaluation and comparison

Author: [Your Name]
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                              accuracy_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. TRADITIONAL ML PIPELINE
# ============================================================

class TraditionalMLPipeline:
    """
    Traditional ML classification using hand-crafted features.
    Replicates and extends the approach in the Lessmeier et al. paper.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = self._init_models()
        self.results = {}
    
    def _init_models(self) -> Dict:
        """Initialize all ML classifiers (matching the paper + extras)."""
        return {
            'CART': DecisionTreeClassifier(random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'GBT': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'kNN': KNeighborsClassifier(n_neighbors=5),
            'XGB': XGBClassifier(n_estimators=100, learning_rate=0.1,
                                  max_depth=6, random_state=42,
                                  eval_metric='mlogloss', verbosity=0),
        }
    
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> Dict:
        """
        Train all models and evaluate on test set.
        
        Args:
            X_train, y_train: Training data and labels
            X_test, y_test: Test data and labels
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of results per model
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...", end=' ')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': acc,
                'f1_score': f1,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'report': classification_report(y_test, y_pred, output_dict=True),
            }
            
            print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Soft-voting ensemble over the three strongest models.
        # Hard voting with equal weight across all classifiers dragged the aggregate
        # below the best individual model in early experiments.
        # Soft voting uses predicted probabilities so well-calibrated models dominate.
        _top_names = [n for n in ['GBT', 'RF', 'XGB'] if n in self.models]
        ensemble = VotingClassifier(
            estimators=[(n, self.models[n]) for n in _top_names],
            voting='soft',
        )
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ens = ensemble.predict(X_test_scaled)
        acc_ens = accuracy_score(y_test, y_pred_ens)
        f1_ens = f1_score(y_test, y_pred_ens, average='weighted')

        results['Ensemble'] = {
            'accuracy': acc_ens,
            'f1_score': f1_ens,
            'confusion_matrix': confusion_matrix(y_test, y_pred_ens),
            'y_pred': y_pred_ens,
            'report': classification_report(y_test, y_pred_ens, output_dict=True),
        }
        print(f"  Ensemble (soft, {'+'.join(_top_names)}): "
              f"Accuracy: {acc_ens:.4f}, F1: {f1_ens:.4f}")
        
        self.results = results
        # Store fitted individual models so callers (e.g. MLflow) can log the best one
        self.fitted_pipelines = {name: model for name, model in self.models.items()}
        self.fitted_pipelines['Ensemble'] = ensemble
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       n_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation for all models.
        
        Returns:
            Dictionary with mean and std accuracy per model
        """
        X_scaled = self.scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'all_scores': scores,
            }
            print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return results


# ============================================================
# 2. 1D-CNN FOR RAW SIGNAL CLASSIFICATION
# ============================================================

def build_1d_cnn_model(input_length: int, n_classes: int = 3):
    """
    Build a 1D-CNN model for raw signal classification.
    
    Architecture:
    - Conv1D blocks with increasing filters
    - BatchNorm + ReLU + MaxPool
    - Global Average Pooling
    - Dense layers with dropout
    
    Args:
        input_length: Length of input signal segment
        n_classes: Number of output classes
        
    Returns:
        PyTorch model (or TensorFlow/Keras if preferred)
    """
    try:
        import torch
        import torch.nn as nn
        
        class BearingCNN1D(nn.Module):
            def __init__(self):
                super().__init__()
                
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=30),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    
                    # Block 2
                    nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=15),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    
                    # Block 3
                    nn.Conv1d(64, 128, kernel_size=16, stride=1, padding=7),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    
                    # Block 4
                    nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=3),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, n_classes),
                )
            
            def forward(self, x):
                # x shape: (batch, 1, signal_length)
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = BearingCNN1D()
        print(f"1D-CNN model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        return None


# ============================================================
# 3. 2D-CNN FOR SPECTROGRAM CLASSIFICATION
# ============================================================

def build_2d_cnn_model(input_size: Tuple[int, int] = (128, 128), 
                        n_classes: int = 3):
    """
    Build a 2D-CNN model for STFT/CWT image classification.
    
    Args:
        input_size: (height, width) of input spectrogram
        n_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    try:
        import torch
        import torch.nn as nn
        
        class BearingCNN2D(nn.Module):
            def __init__(self):
                super().__init__()
                
                self.features = nn.Sequential(
                    # Block 1
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    # Block 2
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    # Block 3
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    # Block 4
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, n_classes),
                )
            
            def forward(self, x):
                # x shape: (batch, 1, H, W)
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = BearingCNN2D()
        print(f"2D-CNN model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except ImportError:
        print("PyTorch not installed.")
        return None


# ============================================================
# 4. TRAINING UTILITIES
# ============================================================

def train_pytorch_model(model, X_train, y_train, X_val, y_val,
                         epochs: int = 50, batch_size: int = 32, 
                         lr: float = 1e-3):
    """
    Train a PyTorch model with training and validation monitoring.
    
    Args:
        model: PyTorch model
        X_train, y_train: Training data (numpy arrays)
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
        
        train_acc = correct / total
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            _, val_pred = val_outputs.max(1)
            val_acc = val_pred.eq(y_val_t).sum().item() / len(y_val_t)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
    
    return model, history


# ============================================================
# 5. DATA PREPARATION UTILITIES
# ============================================================

def prepare_segments(signals: np.ndarray, labels: np.ndarray,
                     segment_length: int = 8192, 
                     overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split long signals into shorter overlapping segments for training.
    
    Args:
        signals: Array of signals, shape (n_samples, signal_length)
        labels: Array of labels, shape (n_samples,)
        segment_length: Length of each segment
        overlap: Overlap ratio between segments
        
    Returns:
        (segments, segment_labels)
    """
    step = int(segment_length * (1 - overlap))
    segments = []
    segment_labels = []
    
    for i in range(len(signals)):
        sig = signals[i]
        n_segments = (len(sig) - segment_length) // step + 1
        
        for j in range(n_segments):
            start = j * step
            end = start + segment_length
            segments.append(sig[start:end])
            segment_labels.append(labels[i])
    
    return np.array(segments), np.array(segment_labels)


def prepare_1d_cnn_data(segments: np.ndarray) -> np.ndarray:
    """Reshape segments for 1D-CNN input: (N, 1, L)."""
    return segments[:, np.newaxis, :]


def prepare_2d_cnn_data(segments: np.ndarray, fs: int,
                         method: str = 'stft',
                         image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Convert signal segments to 2D images for CNN input.
    
    Args:
        segments: Signal segments, shape (N, L)
        fs: Sampling frequency
        method: 'stft' or 'cwt'
        image_size: Target image size
        
    Returns:
        Array of images, shape (N, 1, H, W)
    """
    from tqdm import tqdm
    
    # Import the image generation functions
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dsp_features import signal_to_stft_image, signal_to_cwt_image
    
    images = []
    for i in tqdm(range(len(segments)), desc=f"Generating {method} images"):
        if method == 'stft':
            img = signal_to_stft_image(segments[i], fs, target_size=image_size)
        elif method == 'cwt':
            img = signal_to_cwt_image(segments[i], fs, target_size=image_size)
        else:
            raise ValueError(f"Unknown method: {method}")
        images.append(img)
    
    images = np.array(images)[:, np.newaxis, :, :]  # Add channel dim
    return images


# ============================================================
# 6. EXPERIMENT CONFIGURATIONS (matching the paper)
# ============================================================

# Experiment 7.2: Train on artificial, test on real
EXPERIMENT_ARTIFICIAL_TO_REAL = {
    'name': 'Artificial -> Real (Paper Section 7.2)',
    'train': {
        'healthy': ['K002'],
        'OR': ['KA01', 'KA05', 'KA07'],
        'IR': ['KI01', 'KI05', 'KI07'],
    },
    'test': {
        'healthy': ['K001'],
        'OR': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
        'IR': ['KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21'],
    },
}

# Experiment 7.3: Train and test on real (5-fold CV)
EXPERIMENT_REAL_CV = {
    'name': 'Real Damage 5-Fold CV (Paper Section 7.3)',
    'data': {
        'healthy': ['K001', 'K002', 'K003', 'K004', 'K005'],
        'OR': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
        'IR': ['KI04', 'KI14', 'KI16', 'KI18', 'KI21'],
    },
}

# Experiment 7.4: Including multiple damages
EXPERIMENT_MULTIPLE = {
    'name': 'Including Multiple Damages (Paper Section 7.4)',
    'data': {
        'healthy': ['K001', 'K002', 'K003', 'K004', 'K005'],
        'OR': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB27'],
        'IR': ['KI04', 'KI14', 'KI16', 'KI18', 'KI21', 'KB23', 'KB24'],
    },
}


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    print("=== Testing ML Pipeline ===\n")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 300
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Make classes somewhat separable
    y = np.random.choice([0, 1, 2], size=n_samples)
    X[y == 0] += 0.5
    X[y == 1] -= 0.5
    X[y == 2] += np.array([0.3, -0.3] * (n_features // 2))
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Test traditional ML
    print("Traditional ML (synthetic data):")
    pipeline = TraditionalMLPipeline()
    results = pipeline.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print(f"\nBest model: {max(results, key=lambda k: results[k]['accuracy'])}")
    
    # Test CNN model creation
    print("\n1D-CNN:")
    model_1d = build_1d_cnn_model(input_length=8192, n_classes=3)
    
    print("\n2D-CNN:")
    model_2d = build_2d_cnn_model(input_size=(128, 128), n_classes=3)
    
    print("\nAll tests passed!")
