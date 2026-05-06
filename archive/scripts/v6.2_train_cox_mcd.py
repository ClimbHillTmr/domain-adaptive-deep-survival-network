"""
v6.2 Training Script - Simplified & Corrected
===============================================
Major fixes from v6.0/v6.1:
1. Replace DeepHit with Cox PH (simpler, more robust for IDH prediction)
2. Fix MCD training with proper parameter freezing
3. Remove useless features (history_HBP_rate with nunique=1)
4. Add proper validation with clinical metrics

Author: Clinical Data Analytics Specialist
Date: 2026-05-02
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    "source_data": "data_preprocessing/data/深医_final_data.csv",
    "target_data": "data_preprocessing/data/福鼎_final_data.csv",
    
    # Clean feature list (remove NaN-only columns)
    "static_features": [
        "透析年龄", "性别", "首次透析年龄", "透析龄",
        "透前收缩压", "透前舒张压", "透前动脉压", "透前体重",
        "干体重", "高血压诊断", "透析方式", "实际透析时长",
        "超滤量MAX", "超滤率_mean", "超滤率_std",
    ],
    
    "event_col": "透中低血压_计算",
    "duration_col": "et_min",
    
    # Model parameters
    "d_model": 64,
    "dropout": 0.1,
    
    # Training parameters
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 50,
    "patience": 10,
    "mcd_weight": 0.3,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}


# ============================================================
# Data Loading
# ============================================================
def load_and_preprocess_data(config, is_training=True, imputer=None, scaler=None, label_encoders=None):
    data_path = config["source_data"] if is_training else config["target_data"]
    
    if not os.path.exists(data_path):
        return None, imputer, scaler, label_encoders
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    static_features = config["static_features"]
    available_features = [f for f in static_features if f in df.columns]
    
    X_df = df[available_features].copy()
    
    # Handle categorical columns
    categorical_cols = ["性别", "高血压诊断", "透析方式"]
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if col in X_df.columns:
            if is_training:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                label_encoders[col] = le
            else:
                if col in label_encoders:
                    le = label_encoders[col]
                    X_df[col] = X_df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
    
    X = X_df.values.astype(float)
    
    if is_training:
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if imputer is None or scaler is None:
            raise RuntimeError("Imputer/scaler not fitted.")
        X = imputer.transform(X)
        X = scaler.transform(X)
    
    event_col = config["event_col"]
    duration_col = config["duration_col"]
    
    event = df[event_col].fillna(0).values.astype(int)
    duration = df[duration_col].fillna(0).values.astype(float)
    
    return {
        "X": torch.FloatTensor(X),
        "event": torch.LongTensor(event),
        "duration": torch.FloatTensor(duration),
        "df": df,
    }, imputer, scaler, label_encoders


# ============================================================
# Cox PH Neural Network
# ============================================================
class CoxPHNetwork(nn.Module):
    """
    Cox Proportional Hazards Neural Network.
    
    Simpler than DeepHit, more robust for IDH prediction.
    Outputs a risk score h(x) = exp(f(x)), where f(x) is the neural network output.
    """
    
    def __init__(self, num_features, d_model=64, dropout=0.1):
        super(CoxPHNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),  # Risk score
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)  # (B,)


class MCDCoxNetwork(nn.Module):
    """
    Cox PH + MCD Domain Adaptation.
    
    Uses two classifiers with different decision boundaries.
    The discrepancy between their predictions on target domain samples
    is used to detect distribution shift.
    """
    
    def __init__(self, num_features, d_model=64, dropout=0.1):
        super(MCDCoxNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Two classifiers (for MCD)
        self.classifier1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        self.classifier2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        risk1 = self.classifier1(features).squeeze(-1)
        risk2 = self.classifier2(features).squeeze(-1)
        return {
            "features": features,
            "risk1": risk1,
            "risk2": risk2,
            "risk": (risk1 + risk2) / 2,  # Average for prediction
        }


# ============================================================
# Loss Functions
# ============================================================
def cox_ph_loss(risk_scores, event, duration):
    """
    Cox PH partial likelihood loss.
    
    L = -sum_{i: event_i=1} [risk_i - log(sum_{j in R(t_i)} exp(risk_j))]
    
    where R(t_i) is the risk set at time t_i (all patients with t_j >= t_i).
    """
    # Sort by duration (descending) for efficient risk set computation
    sorted_idx = torch.argsort(duration, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    event_sorted = event[sorted_idx]
    
    # Compute log-sum-exp for risk sets
    # For patient i, risk set includes all patients j with t_j >= t_i
    # Since sorted descending, risk set for i is [0, 1, ..., i]
    cumsum_risk = torch.logcumsumexp(risk_sorted, dim=0)
    
    # Loss for events only
    event_mask = event_sorted.float()
    loss = -(risk_sorted - cumsum_risk) * event_mask
    
    return loss.sum() / (event_mask.sum() + 1e-8)


def mcd_discrepancy_loss(risk1_tgt, risk2_tgt):
    """
    MCD discrepancy loss.
    
    Measures the difference between two classifiers on target domain.
    """
    return torch.mean(torch.abs(risk1_tgt - risk2_tgt))


# ============================================================
# Training Functions
# ============================================================
def train_epoch_mcd_cox(model, source_loader, target_loader, optimizer, config):
    """
    Train with MCD domain adaptation.
    
    Step 1: Train on source domain (minimize Cox PH loss)
    Step 2: Maximize discrepancy on target (train classifiers)
    Step 3: Minimize discrepancy on target (train feature extractor)
    """
    model.train()
    total_loss = 0
    cox_loss_sum = 0
    mcd_loss_sum = 0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    n_batches = len(source_loader)
    
    for batch_idx in range(n_batches):
        # Source batch
        try:
            X_src, event_src, duration_src = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            X_src, event_src, duration_src = next(source_iter)
        
        X_src = X_src.to(config["device"])
        event_src = event_src.to(config["device"])
        duration_src = duration_src.to(config["device"])
        
        # ===== Step 1: Train on source domain =====
        outputs_src = model(X_src)
        cox_loss = cox_ph_loss(outputs_src["risk"], event_src, duration_src)
        
        optimizer.zero_grad()
        cox_loss.backward()
        optimizer.step()
        
        total_loss += cox_loss.item()
        cox_loss_sum += cox_loss.item()
        
        # ===== Step 2 & 3: MCD on target domain =====
        try:
            X_tgt, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            try:
                X_tgt, _, _ = next(target_iter)
            except StopIteration:
                continue
        
        X_tgt = X_tgt.to(config["device"])
        
        # Step 2: Maximize discrepancy (train classifiers only)
        # Freeze feature extractor
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        
        # Forward through target (step 2)
        outputs_tgt_2 = model(X_tgt)
        mcd_loss_2 = mcd_discrepancy_loss(outputs_tgt_2["risk1"], outputs_tgt_2["risk2"])
        
        optimizer.zero_grad()
        mcd_loss_max = -mcd_loss_2  # Negative to maximize
        mcd_loss_max.backward()
        optimizer.step()
        
        # Unfreeze feature extractor
        for param in model.feature_extractor.parameters():
            param.requires_grad = True
        
        # Step 3: Minimize discrepancy (train feature extractor only)
        # Freeze classifiers
        for param in model.classifier1.parameters():
            param.requires_grad = False
        for param in model.classifier2.parameters():
            param.requires_grad = False
        
        # Forward through target (step 3) - recompute
        outputs_tgt_3 = model(X_tgt)
        mcd_loss_3 = mcd_discrepancy_loss(outputs_tgt_3["risk1"], outputs_tgt_3["risk2"])
        
        optimizer.zero_grad()
        mcd_loss_min = config["mcd_weight"] * mcd_loss_3
        mcd_loss_min.backward()
        optimizer.step()
        
        # Unfreeze classifiers
        for param in model.classifier1.parameters():
            param.requires_grad = True
        for param in model.classifier2.parameters():
            param.requires_grad = True
        
        mcd_loss_sum += mcd_loss_2.item()
    
    return {
        "total_loss": total_loss / n_batches,
        "cox_loss": cox_loss_sum / n_batches,
        "mcd_loss": mcd_loss_sum / n_batches,
    }


def evaluate_cox(model, dataloader, config):
    """Evaluate Cox PH model."""
    model.eval()
    all_risks = []
    all_events = []
    all_durations = []
    
    with torch.no_grad():
        for batch in dataloader:
            X, event, duration = batch
            
            X = X.to(config["device"])
            outputs = model(X)
            
            all_risks.append(outputs["risk"].cpu().numpy())
            all_events.append(event.cpu().numpy())
            all_durations.append(duration.cpu().numpy())
    
    all_risks = np.concatenate(all_risks, axis=0)
    all_events = np.concatenate(all_events, axis=0)
    all_durations = np.concatenate(all_durations, axis=0)
    
    # Compute C-index using lifelines
    c_index = concordance_index(all_durations, -all_risks, all_events)
    
    return {
        "c_index": c_index,
        "risks": all_risks,
        "events": all_events,
        "durations": all_durations,
    }


# ============================================================
# Main Training Loop
# ============================================================
def run_v6_2_experiment(config):
    """Run v6.2 experiment with Cox PH + MCD."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    source_data, imputer, scaler, label_encoders = load_and_preprocess_data(config, is_training=True)
    target_data, _, _, _ = load_and_preprocess_data(config, is_training=False, imputer=imputer, scaler=scaler, label_encoders=label_encoders)
    
    if source_data is None:
        print("Error: Source data not found.")
        return
    
    X = source_data["X"]
    event = source_data["event"]
    duration = source_data["duration"]
    
    X_train, X_val, event_train, event_val, duration_train, duration_val = train_test_split(
        X, event, duration, test_size=0.2, random_state=config["seed"], stratify=event
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Source event rate: {event.float().mean():.2%}")
    if target_data is not None:
        print(f"Target event rate: {target_data['event'].float().mean():.2%}")
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, event_train, duration_train)
    val_dataset = TensorDataset(X_val, event_val, duration_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Target dataloader (for MCD)
    target_loader = None
    if target_data is not None:
        X_tgt = target_data["X"]
        event_tgt = target_data["event"]
        duration_tgt = target_data["duration"]
        tgt_dataset = TensorDataset(X_tgt, event_tgt, duration_tgt)
        target_loader = DataLoader(tgt_dataset, batch_size=config["batch_size"], shuffle=True)
    
    if target_loader is None:
        target_loader = train_loader  # Fallback
    
    # Initialize model
    num_features = X_train.shape[1]
    model = MCDCoxNetwork(
        num_features=num_features,
        d_model=config["d_model"],
        dropout=config["dropout"],
    ).to(config["device"])
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    best_val_c_index = 0
    best_model_state = None
    patience_counter = 0
    history = []
    
    for epoch in range(config["epochs"]):
        # Train with MCD
        train_metrics = train_epoch_mcd_cox(
            model, train_loader, target_loader, optimizer, config
        )
        
        # Evaluate
        val_metrics = evaluate_cox(model, val_loader, config)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f} (cox: {train_metrics['cox_loss']:.4f}, mcd: {train_metrics['mcd_loss']:.4f})")
            print(f"  Val C-index: {val_metrics['c_index']:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["total_loss"],
            "train_cox_loss": train_metrics["cox_loss"],
            "train_mcd_loss": train_metrics["mcd_loss"],
            "val_c_index": val_metrics["c_index"],
        })
        
        if val_metrics["c_index"] > best_val_c_index:
            best_val_c_index = val_metrics["c_index"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation...")
    print("="*60)
    
    final_val_metrics = evaluate_cox(model, val_loader, config)
    print(f"Best Val C-index: {best_val_c_index:.4f}")
    
    if target_data is not None:
        X_tgt = target_data["X"]
        event_tgt = target_data["event"]
        duration_tgt = target_data["duration"]
        tgt_dataset = TensorDataset(X_tgt, event_tgt, duration_tgt)
        tgt_loader = DataLoader(tgt_dataset, batch_size=config["batch_size"], shuffle=False)
        tgt_metrics = evaluate_cox(model, tgt_loader, config)
        print(f"Target C-index: {tgt_metrics['c_index']:.4f}")
    
    # Save results
    save_dir = "runs/v6.2_cox_mcd_fixed"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    
    return {
        "best_val_c_index": best_val_c_index,
        "history": history,
        "model": model,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v6.2 Cox PH + MCD Training (Fixed)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--mcd_weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model,
        "mcd_weight": args.mcd_weight,
        "seed": args.seed,
    })
    
    results = run_v6_2_experiment(config)
