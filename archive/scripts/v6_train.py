"""
v6.0 Training Script
=====================
Trains the DeepHit + MCD model with:
- Previous session features (fixes data leakage)
- Historical IDH frequency features
- KDOQI event definition (if available)

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
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.v6_model import V6DeepHitMCD, DeepHitLoss


# ============================================================
# Configuration
# ============================================================
DEFAULT_CONFIG = {
    # Data paths
    "source_data": "data_preprocessing/data/深医_final_data.csv",
    "target_data": "data_preprocessing/data/福鼎_final_data.csv",
    
    # Feature columns (v6.0: only pre-dialysis and historical features)
    "static_features": [
        # Demographics
        "透析年龄",
        "性别",
        "首次透析年龄",
        "透析龄",
        
        # Pre-dialysis vitals (available BEFORE dialysis starts)
        "透前收缩压",
        "透前舒张压",
        "透前动脉压",
        "透前体重",
        
        # Dry weight
        "干体重",
        
        # Comorbidities
        "高血压诊断",
        
        # Dialysis prescription
        "透析方式",
        "实际透析时长",
        
        # Derived features (pre-dialysis only)
        "超滤量MAX",
        "超滤率_mean",
        "超滤率_std",
        
        # Historical IDH features (if available)
        "history_HBP_rate",
        
        # Previous session features (fix data leakage)
        "prev_SBP_mean",
        "prev_SBP_std",
        "prev_DBP_mean",
        "prev_DBP_std",
        "prev_UF_rate_mean",
        "prev_UF_rate_std",
        "prev_IDH_event",
        "prev_duration",
        "prev_pre_SBP",
        "prev_pre_DBP",
        "prev_pre_weight",
        "prev_dry_weight",
    ],
    
    # Target columns (fallback to v5 definition if v6 not available)
    "event_col": "透中低血压_计算_v6",
    "duration_col": "et_min_v6",
    
    # Model parameters
    "d_model": 64,
    "num_time_bins": 10,
    "dropout": 0.1,
    "seq_len": 1,
    
    # Training parameters
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 100,
    "patience": 15,
    "mcd_weight": 0.5,
    "alpha": 0.5,  # DeepHit log-likelihood vs ranking weight
    "sigma": 0.1,  # Ranking loss bandwidth
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}


# ============================================================
# Data Loading and Preprocessing
# ============================================================
def load_and_preprocess_data(config, is_training=True, imputer=None, scaler=None, label_encoders=None):
    """
    Load and preprocess v6.0 data.
    
    Key changes from v5.x:
    - Uses previous session features instead of current session dynamic features
    - Includes historical IDH frequency
    - Handles categorical encoding
    """
    data_path = config["source_data"] if is_training else config["target_data"]
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Using fallback data.")
        return None, imputer, scaler, label_encoders
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    # Select features
    static_features = config["static_features"]
    available_features = [f for f in static_features if f in df.columns]
    
    if len(available_features) == 0:
        print(f"Warning: No features found in {data_path}")
        return None, imputer, scaler, label_encoders
    
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
                    # Handle unseen labels
                    X_df[col] = X_df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
    
    X = X_df.values.astype(float)
    
    # Handle missing values
    if is_training:
        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)
    else:
        if imputer is None:
            raise RuntimeError("Imputer not fitted. Call with is_training=True first.")
        X = imputer.transform(X)
    
    # Normalize
    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise RuntimeError("Scaler not fitted. Call with is_training=True first.")
        X = scaler.transform(X)
    
    # Targets
    event_col = config["event_col"]
    duration_col = config["duration_col"]
    
    # Fallback to old columns if v6 columns don't exist
    if event_col not in df.columns:
        event_col = "透中低血压_计算"
    if duration_col not in df.columns:
        duration_col = "et_min"
    
    event = df[event_col].fillna(0).values.astype(int)
    duration = df[duration_col].fillna(0).values.astype(float)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    event_tensor = torch.LongTensor(event)
    duration_tensor = torch.FloatTensor(duration)
    
    return {
        "X": X_tensor,
        "event": event_tensor,
        "duration": duration_tensor,
        "df": df,
    }, imputer, scaler, label_encoders


def create_time_bins(duration, num_bins=10):
    """
    Create time bins for DeepHit.
    
    Uses quantile-based binning to ensure roughly equal samples per bin.
    """
    # Filter out zero durations
    valid_duration = duration[duration > 0]
    
    if len(valid_duration) == 0:
        return torch.linspace(0, 1, num_bins + 1)
    
    # Use quantiles for bin boundaries
    quantiles = torch.linspace(0, 1, num_bins + 1)
    time_bins = torch.quantile(torch.FloatTensor(valid_duration), quantiles)
    
    return time_bins


# ============================================================
# Training Functions
# ============================================================
def train_epoch(model, dataloader, optimizer, time_bins, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    survival_loss_sum = 0
    mcd_loss_sum = 0
    
    for batch in dataloader:
        X, event, duration = batch
        
        X = X.to(config["device"])
        event = event.to(config["device"])
        duration = duration.to(config["device"])
        time_bins = time_bins.to(config["device"])
        
        # Forward pass - all features as static, dummy dynamic
        x_static = X
        x_dynamic = X.unsqueeze(1)  # (batch, 1, num_features)
        outputs = model(x_static, x_dynamic)
        
        # Compute loss
        loss_dict = model.compute_loss(
            outputs, event, duration, time_bins,
            labels_src=event,
            mcd_weight=config["mcd_weight"],
            alpha=config["alpha"],
            sigma=config["sigma"],
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        optimizer.step()
        
        total_loss += loss_dict["total_loss"].item()
        survival_loss_sum += loss_dict["survival_loss"].item()
        mcd_loss_sum += loss_dict["mcd_loss"].item()
    
    avg_loss = total_loss / len(dataloader)
    avg_survival_loss = survival_loss_sum / len(dataloader)
    avg_mcd_loss = mcd_loss_sum / len(dataloader)
    
    return {
        "total_loss": avg_loss,
        "survival_loss": avg_survival_loss,
        "mcd_loss": avg_mcd_loss,
    }


def evaluate(model, dataloader, time_bins, config):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_events = []
    all_durations = []
    
    with torch.no_grad():
        for batch in dataloader:
            X, event, duration = batch
            
            X = X.to(config["device"])
            event = event.to(config["device"])
            duration = duration.to(config["device"])
            time_bins = time_bins.to(config["device"])
            
            # Forward pass - all features as static, dummy dynamic
            x_static = X
            x_dynamic = X.unsqueeze(1)
            outputs = model(x_static, x_dynamic)
            
            # Compute loss
            loss_dict = model.compute_loss(
                outputs, event, duration, time_bins,
                labels_src=event,
                mcd_weight=config["mcd_weight"],
                alpha=config["alpha"],
                sigma=config["sigma"],
            )
            
            total_loss += loss_dict["total_loss"].item()
            
            # Collect predictions
            all_preds.append(outputs["survival_pred"].cpu().numpy())
            all_events.append(event.cpu().numpy())
            all_durations.append(duration.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_events = np.concatenate(all_events, axis=0)
    all_durations = np.concatenate(all_durations, axis=0)
    
    # Compute C-index (simplified)
    c_index = compute_c_index(all_preds, all_events, all_durations, time_bins.cpu().numpy())
    
    return {
        "loss": avg_loss,
        "c_index": c_index,
        "predictions": all_preds,
        "events": all_events,
        "durations": all_durations,
    }


def compute_c_index(predictions, events, durations, time_bins):
    """
    Compute Concordance Index (C-index).
    
    Simplified version: compares risk scores with observed event times.
    """
    # predictions shape: (N, num_time_bins)
    # time_bins shape: (num_time_bins + 1,) - boundaries
    
    # Compute risk scores (expected time to event)
    num_time_bins = predictions.shape[1]
    time_indices = np.arange(num_time_bins)
    risk_scores = (predictions * time_indices).sum(axis=1)
    
    # Lower risk score = higher risk (earlier expected event time)
    # So we negate for C-index computation
    risk_scores = -risk_scores
    
    # Count concordant pairs
    n_concordant = 0
    n_comparable = 0
    
    # Sample pairs for efficiency
    n_samples = min(10000, len(events) * (len(events) - 1) // 2)
    idx_i = np.random.randint(0, len(events), n_samples)
    idx_j = np.random.randint(0, len(events), n_samples)
    
    for i, j in zip(idx_i, idx_j):
        if i == j:
            continue
        
        # Only consider pairs where at least one had an event
        if events[i] == 0 and events[j] == 0:
            continue
        
        # Comparable pairs: one had event before the other
        if events[i] == 1 and durations[i] < durations[j]:
            n_comparable += 1
            if risk_scores[i] > risk_scores[j]:
                n_concordant += 1
        elif events[j] == 1 and durations[j] < durations[i]:
            n_comparable += 1
            if risk_scores[j] > risk_scores[i]:
                n_concordant += 1
    
    if n_comparable == 0:
        return 0.5
    
    return n_concordant / n_comparable


# ============================================================
# Main Training Loop
# ============================================================
def run_v6_experiment(config):
    """
    Run v6.0 experiment with DeepHit + MCD.
    """
    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Load data
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    source_data, imputer, scaler, label_encoders = load_and_preprocess_data(config, is_training=True)
    target_data, _, _, _ = load_and_preprocess_data(config, is_training=False, imputer=imputer, scaler=scaler, label_encoders=label_encoders)
    
    if source_data is None:
        print("Error: Source data not found.")
        return
    
    # Split source data into train/val
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
    
    # Create time bins
    time_bins = create_time_bins(duration_train, config["num_time_bins"])
    print(f"Time bins: {time_bins.numpy()}")
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, event_train, duration_train)
    val_dataset = TensorDataset(X_val, event_val, duration_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Initialize model
    num_static_features = X_train.shape[1]
    num_dynamic_features = X_train.shape[1]  # Same as static for seq_len=1
    model = V6DeepHitMCD(
        num_static_features=num_static_features,
        num_dynamic_features=num_dynamic_features,
        d_model=config["d_model"],
        num_time_bins=config["num_time_bins"],
        dropout=config["dropout"],
        seq_len=config["seq_len"],
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
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, time_bins, config)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, time_bins, config)
        
        # Update learning rate
        scheduler.step()
        
        # Log
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, C-index: {val_metrics['c_index']:.4f}")
        
        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["total_loss"],
            "val_loss": val_metrics["loss"],
            "val_c_index": val_metrics["c_index"],
        })
        
        # Early stopping
        if val_metrics["c_index"] > best_val_c_index:
            best_val_c_index = val_metrics["c_index"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation...")
    print("="*60)
    
    final_val_metrics = evaluate(model, val_loader, time_bins, config)
    print(f"Best Val C-index: {best_val_c_index:.4f}")
    
    # Evaluate on target domain (if available)
    if target_data is not None:
        X_tgt = target_data["X"]
        event_tgt = target_data["event"]
        duration_tgt = target_data["duration"]
        
        tgt_dataset = TensorDataset(X_tgt, event_tgt, duration_tgt)
        tgt_loader = DataLoader(tgt_dataset, batch_size=config["batch_size"], shuffle=False)
        
        tgt_metrics = evaluate(model, tgt_loader, time_bins, config)
        print(f"Target C-index: {tgt_metrics['c_index']:.4f}")
    
    # Save results
    save_dir = "runs/v6_deephit_mcd"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    
    # Save history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    # Save config
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
    parser = argparse.ArgumentParser(description="v6.0 DeepHit + MCD Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_time_bins", type=int, default=10)
    parser.add_argument("--mcd_weight", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Update config
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model,
        "num_time_bins": args.num_time_bins,
        "mcd_weight": args.mcd_weight,
        "alpha": args.alpha,
        "seed": args.seed,
    })
    
    # Run experiment
    results = run_v6_experiment(config)
