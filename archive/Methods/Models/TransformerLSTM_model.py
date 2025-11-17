import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import os
import json
import importlib.util
import pathlib
UnifiedModelEvaluator = None
try:
    from Methods.unified_model_evaluation import UnifiedModelEvaluator as _UME
    UnifiedModelEvaluator = _UME
except Exception:
    try:
        current_dir = pathlib.Path(__file__).resolve().parent
        ume_path = current_dir.parent / 'unified_model_evaluation.py'
        spec = importlib.util.spec_from_file_location('unified_model_evaluation', str(ume_path))
        ume = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ume)
        UnifiedModelEvaluator = ume.UnifiedModelEvaluator
    except Exception:
        UnifiedModelEvaluator = None

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.device)

class TransformerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_heads=4, n_layers=2, lstm_hidden=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pe = PositionalEncoding(hidden_dim, max_len=1024)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out[:, -1, :]
        logits = self.classifier(pooled)
        return logits

def train_transformer_lstm(X_seq, y, X_static=None, test_size=0.2, random_state=42, epochs=10, lr=1e-3, batch_size=64, base_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=test_size, random_state=random_state, stratify=y)
    input_dim = X_train.shape[-1]
    model = TransformerLSTM(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0], device=device))

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    def to_label(arr):
        return torch.tensor(arr, dtype=torch.long)

    def batches(X, y, bs):
        for i in range(0, len(X), bs):
            yield X[i:i+bs], y[i:i+bs]

    for epoch in range(epochs):
        model.train()
        for xb, yb in batches(X_train, y_train, batch_size):
            xb = to_tensor(xb).to(device)
            yb = to_label(yb).to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(to_tensor(X_test).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs[:, 1] >= 0.5
    metrics = {
        'test_accuracy': float(accuracy_score(y_test, preds)),
        'test_precision': float(precision_score(y_test, preds)),
        'test_recall': float(recall_score(y_test, preds)),
        'test_f1': float(f1_score(y_test, preds)),
        'test_roc_auc': float(roc_auc_score(y_test, probs[:,1]))
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_path or './Results', f'TransformerLSTM_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), os.path.join(results_dir, 'model.pt'))
    return {'results_dir': results_dir, 'metrics': metrics, 'model': model}

class TorchSklearnWrapper:
    def __init__(self, model, threshold=0.5, device=None):
        self.model = model
        self.threshold = float(threshold)
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32).to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= self.threshold).astype(int)

def evaluate_with_unified(trained_model, X_test, y_test, model_name='TransformerLSTM', save_dir='./evaluation_results', threshold=0.5):
    wrapper = TorchSklearnWrapper(trained_model, threshold=threshold)
    if UnifiedModelEvaluator is not None:
        evaluator = UnifiedModelEvaluator(task_type='classification', save_dir=save_dir)
        return evaluator.evaluate_model(model=wrapper, X_test=X_test, y_test=y_test, model_name=model_name)
    else:
        probs = wrapper.predict_proba(X_test)
        preds = (probs[:,1] >= threshold).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds)),
            'recall': float(recall_score(y_test, preds)),
            'f1_score': float(f1_score(y_test, preds)),
            'roc_auc': float(roc_auc_score(y_test, probs[:,1]))
        }
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(save_dir, f'{model_name}_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump({'model_name': model_name, 'metrics': metrics}, f, indent=2)
        return metrics