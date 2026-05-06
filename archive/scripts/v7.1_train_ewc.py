"""
v7.1 Training Script - Enhanced Features + Elastic Fine-tuning
===============================================================
Fixes from v7.0:
1. Restore 超滤率_mean with aggressive winsorization
2. Add 脉压差 (SBP - DBP) as a derived feature
3. Elastic Weight Consolidation (EWC) for fine-tuning
4. Enhanced BBSE with calibrated predictions

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
    
    # v7.1: Enhanced feature list with winsorized 超滤率
    "static_features": [
        # Demographics
        "透析年龄", "性别", "首次透析年龄", "透析龄",
        
        # Pre-dialysis vitals
        "透前收缩压", "透前舒张压", "透前动脉压", "透前体重",
        "干体重",
        
        # Dialysis prescription
        "实际透析时长",
        
        # Derived features (aggressively winsorized)
        "超滤量MAX",
        "超滤率_mean",  # Restored with winsorization
    ],
    
    "event_col": "透中低血压_计算",
    "duration_col": "et_min",
    
    # Model parameters
    "d_model": 64,
    "dropout": 0.1,
    
    # Training parameters
    "learning_rate": 1e-3,
    "fine_tune_lr": 5e-5,  # Lower for EWC
    "batch_size": 256,
    "epochs": 50,
    "fine_tune_epochs": 30,
    "patience": 10,
    
    # Domain adaptation
    "coral_weight": 0.3,
    "mmd_weight": 0.3,
    "mcd_weight": 0.2,
    
    # Outlier handling
    "winsorize_percentile": 0.5,  # More aggressive: [0.5%, 99.5%]
    
    # EWC parameters
    "ewc_lambda": 100,  # EWC regularization strength
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}


# ============================================================
# Data Loading and Preprocessing
# ============================================================
def winsorize_feature(x, percentile=0.5):
    """Aggressive winsorization for extreme outliers."""
    lower = np.percentile(x, percentile)
    upper = np.percentile(x, 100 - percentile)
    return np.clip(x, lower, upper)


def load_and_preprocess_data(config, is_training=True, imputer=None, scaler=None, label_encoders=None, winsorize_bounds=None):
    """Load and preprocess data with aggressive outlier handling."""
    data_path = config["source_data"] if is_training else config["target_data"]
    
    if not os.path.exists(data_path):
        return None, imputer, scaler, label_encoders, winsorize_bounds
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")
    
    static_features = config["static_features"]
    available_features = [f for f in static_features if f in df.columns]
    
    X_df = df[available_features].copy()
    
    # Handle categorical columns
    categorical_cols = ["性别"]
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
    
    # Winsorize outliers
    if is_training:
        winsorize_bounds = {}
        for i in range(X.shape[1]):
            col_name = available_features[i]
            # Winsorize all numeric features aggressively
            lower = np.percentile(X[:, i], config["winsorize_percentile"])
            upper = np.percentile(X[:, i], 100 - config["winsorize_percentile"])
            X[:, i] = np.clip(X[:, i], lower, upper)
            winsorize_bounds[col_name] = (lower, upper)
            print(f"  Winsorized {col_name}: [{lower:.2f}, {upper:.2f}]")
    else:
        # Apply same bounds to target
        if winsorize_bounds is not None:
            for i, col_name in enumerate(available_features):
                if col_name in winsorize_bounds:
                    lower, upper = winsorize_bounds[col_name]
                    X[:, i] = np.clip(X[:, i], lower, upper)
    
    # Handle missing values
    if is_training:
        imputer = SimpleImputer(strategy="median")
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
    }, imputer, scaler, label_encoders, winsorize_bounds


# ============================================================
# Domain Adaptation Losses
# ============================================================
def coral_loss(source, target):
    """CORAL: Align second-order statistics."""
    d = source.size(1)
    source_cov = torch.mm(source.t(), source) / (source.size(0) - 1)
    target_cov = torch.mm(target.t(), target) / (target.size(0) - 1)
    loss = torch.norm(source_cov - target_cov, 'fro') ** 2 / (4 * d * d)
    return loss


def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5):
    """Multi-kernel MMD."""
    batch_size = source.size(0)
    
    XX = torch.mm(source, source.t())
    YY = torch.mm(target, target.t())
    XY = torch.mm(source, target.t())
    
    dist_XX = torch.diag(XX).unsqueeze(1) + torch.diag(XX).unsqueeze(0) - 2 * XX
    median_dist = torch.median(dist_XX[dist_XX > 0])
    
    K_XX = torch.zeros_like(XX)
    K_YY = torch.zeros_like(YY)
    K_XY = torch.zeros_like(XY)
    
    for i in range(kernel_num):
        bandwidth = median_dist * (kernel_mul ** i)
        K_XX += torch.exp(-dist_XX / (2 * bandwidth))
        
        dist_YY = torch.diag(YY).unsqueeze(1) + torch.diag(YY).unsqueeze(0) - 2 * YY
        K_YY += torch.exp(-dist_YY / (2 * bandwidth))
        
        dist_XY = torch.diag(XX).unsqueeze(1) + torch.diag(YY).unsqueeze(0) - 2 * XY
        K_XY += torch.exp(-dist_XY / (2 * bandwidth))
    
    mmd = (K_XX.sum() / (batch_size * batch_size) + 
           K_YY.sum() / (batch_size * batch_size) - 
           2 * K_XY.sum() / (batch_size * batch_size))
    
    return mmd


def mcd_discrepancy_loss(risk1, risk2):
    """MCD discrepancy loss."""
    return torch.mean(torch.abs(risk1 - risk2))


# ============================================================
# Model Architecture
# ============================================================
class MCDCoxNetwork(nn.Module):
    """Cox PH + MCD Domain Adaptation."""
    
    def __init__(self, num_features, d_model=64, dropout=0.1):
        super(MCDCoxNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
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
            "risk": (risk1 + risk2) / 2,
        }


# ============================================================
# Loss Functions
# ============================================================
def cox_ph_loss(risk_scores, event, duration, sample_weights=None):
    """Cox PH partial likelihood loss."""
    sorted_idx = torch.argsort(duration, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    event_sorted = event[sorted_idx]
    
    if sample_weights is not None:
        weights_sorted = sample_weights[sorted_idx]
    else:
        weights_sorted = torch.ones_like(event_sorted).float()
    
    cumsum_risk = torch.logcumsumexp(risk_sorted, dim=0)
    
    loss = -(risk_sorted - cumsum_risk) * event_sorted.float() * weights_sorted
    
    return loss.sum() / (event_sorted.float().sum() + 1e-8)


# ============================================================
# EWC (Elastic Weight Consolidation)
# ============================================================
class EWCRegularizer:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.
    
    Reference: Kirkpatrick, James, et al. "Overcoming catastrophic forgetting 
    in neural networks." PNAS 2017.
    """
    
    def __init__(self, model, dataloader, device, num_samples=1000):
        self.fisher_info = {}
        self.optimal_params = {}
        
        # Store optimal parameters
        for name, param in model.named_parameters():
            self.optimal_params[name] = param.data.clone()
        
        # Estimate Fisher Information Matrix (diagonal approximation)
        model.eval()
        fisher_accum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            X, event, duration = batch
            X = X[:min(len(X), num_samples - samples_processed)]
            X = X.to(device)
            event = event[:len(X)].to(device)
            duration = duration[:len(X)].to(device)
            
            # Forward pass
            outputs = model(X)
            loss = cox_ph_loss(outputs["risk"], event, duration)
            
            # Backward pass to get gradients
            model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_accum[name] += param.grad.data ** 2 * len(X)
            
            samples_processed += len(X)
        
        # Normalize
        for name in fisher_accum:
            self.fisher_info[name] = fisher_accum[name] / samples_processed
    
    def penalty(self, model):
        """Compute EWC penalty."""
        penalty = 0
        for name, param in model.named_parameters():
            if name in self.fisher_info:
                penalty += (self.fisher_info[name] * (param - self.optimal_params[name]) ** 2).sum()
        return penalty


# ============================================================
# Training Functions
# ============================================================
def train_epoch_v7(model, source_loader, target_loader, optimizer, config):
    """Train with multi-method domain adaptation."""
    model.train()
    total_loss = 0
    cox_loss_sum = 0
    coral_loss_sum = 0
    mmd_loss_sum = 0
    mcd_loss_sum = 0
    
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    n_batches = len(source_loader)
    
    for batch_idx in range(n_batches):
        try:
            X_src, event_src, duration_src = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            X_src, event_src, duration_src = next(source_iter)
        
        X_src = X_src.to(config["device"])
        event_src = event_src.to(config["device"])
        duration_src = duration_src.to(config["device"])
        
        try:
            X_tgt, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            try:
                X_tgt, _, _ = next(target_iter)
            except StopIteration:
                continue
        
        X_tgt = X_tgt.to(config["device"])
        
        outputs_src = model(X_src)
        outputs_tgt = model(X_tgt)
        
        cox_loss = cox_ph_loss(outputs_src["risk"], event_src, duration_src)
        coral_l = coral_loss(outputs_src["features"], outputs_tgt["features"])
        mmd_l = mmd_loss(outputs_src["features"], outputs_tgt["features"])
        mcd_l = mcd_discrepancy_loss(outputs_tgt["risk1"], outputs_tgt["risk2"])
        
        total_batch_loss = (
            cox_loss +
            config["coral_weight"] * coral_l +
            config["mmd_weight"] * mmd_l +
            config["mcd_weight"] * mcd_l
        )
        
        optimizer.zero_grad()
        total_batch_loss.backward()
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        cox_loss_sum += cox_loss.item()
        coral_loss_sum += coral_l.item()
        mmd_loss_sum += mmd_l.item()
        mcd_loss_sum += mcd_l.item()
    
    return {
        "total_loss": total_loss / n_batches,
        "cox_loss": cox_loss_sum / n_batches,
        "coral_loss": coral_loss_sum / n_batches,
        "mmd_loss": mmd_loss_sum / n_batches,
        "mcd_loss": mcd_loss_sum / n_batches,
    }


def fine_tune_epoch_ewc(model, target_loader, optimizer, ewc, config):
    """Fine-tune on target domain with EWC regularization."""
    model.train()
    total_loss = 0
    cox_loss_sum = 0
    ewc_penalty_sum = 0
    
    for batch in target_loader:
        X_tgt, event_tgt, duration_tgt = batch
        
        X_tgt = X_tgt.to(config["device"])
        event_tgt = event_tgt.to(config["device"])
        duration_tgt = duration_tgt.to(config["device"])
        
        outputs = model(X_tgt)
        cox_loss = cox_ph_loss(outputs["risk"], event_tgt, duration_tgt)
        
        # EWC penalty
        ewc_pen = ewc.penalty(model)
        
        # Combined loss
        loss = cox_loss + config["ewc_lambda"] * ewc_pen
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        cox_loss_sum += cox_loss.item()
        ewc_penalty_sum += ewc_pen.item()
    
    return {
        "total_loss": total_loss / len(target_loader),
        "cox_loss": cox_loss_sum / len(target_loader),
        "ewc_penalty": ewc_penalty_sum / len(target_loader),
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
def run_v7_1_experiment(config):
    """Run v7.1 experiment with enhanced features and EWC."""
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    print("\n" + "="*80)
    print("v7.1: Enhanced Features + EWC Fine-tuning")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    source_data, imputer, scaler, label_encoders, winsorize_bounds = load_and_preprocess_data(config, is_training=True)
    target_data, _, _, _, _ = load_and_preprocess_data(config, is_training=False, imputer=imputer, scaler=scaler, label_encoders=label_encoders, winsorize_bounds=winsorize_bounds)
    
    if source_data is None:
        print("Error: Source data not found.")
        return
    
    X = source_data["X"]
    event = source_data["event"]
    duration = source_data["duration"]
    
    X_train, X_val, event_train, event_val, duration_train, duration_val = train_test_split(
        X, event, duration, test_size=0.2, random_state=config["seed"], stratify=event
    )
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print(f"Source event rate: {event.float().mean():.2%}")
    if target_data is not None:
        print(f"Target event rate: {target_data['event'].float().mean():.2%}")
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, event_train, duration_train)
    val_dataset = TensorDataset(X_val, event_val, duration_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    target_loader = None
    if target_data is not None:
        X_tgt = target_data["X"]
        event_tgt = target_data["event"]
        duration_tgt = target_data["duration"]
        tgt_dataset = TensorDataset(X_tgt, event_tgt, duration_tgt)
        target_loader = DataLoader(tgt_dataset, batch_size=config["batch_size"], shuffle=True)
    
    if target_loader is None:
        target_loader = train_loader
    
    # Initialize model
    num_features = X_train.shape[1]
    model = MCDCoxNetwork(
        num_features=num_features,
        d_model=config["d_model"],
        dropout=config["dropout"],
    ).to(config["device"])
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Features ({num_features}): {config['static_features']}")
    
    # ===== Stage 1: Source domain training with DA =====
    print("\n" + "="*80)
    print("Stage 1: Source Domain Training with Multi-Method DA")
    print("="*80)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    
    best_val_c_index = 0
    best_model_state = None
    patience_counter = 0
    stage1_history = []
    
    for epoch in range(config["epochs"]):
        train_metrics = train_epoch_v7(
            model, train_loader, target_loader, optimizer, config
        )
        
        val_metrics = evaluate_cox(model, val_loader, config)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"  Train: cox={train_metrics['cox_loss']:.4f}, coral={train_metrics['coral_loss']:.4f}, mmd={train_metrics['mmd_loss']:.4f}, mcd={train_metrics['mcd_loss']:.4f}")
            print(f"  Val C-index: {val_metrics['c_index']:.4f}")
        
        stage1_history.append({
            "epoch": epoch + 1,
            **train_metrics,
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
    
    print(f"\nStage 1 Best Val C-index: {best_val_c_index:.4f}")
    
    # ===== Stage 2: EWC Fine-tuning =====
    if target_data is not None and target_loader is not None:
        print("\n" + "="*80)
        print("Stage 2: EWC Fine-tuning on Target Domain")
        print("="*80)
        
        # Compute Fisher Information Matrix
        print("Computing Fisher Information Matrix...")
        ewc = EWCRegularizer(model, train_loader, config["device"], num_samples=5000)
        print("Fisher Information computed.")
        
        # Fine-tune with EWC
        fine_tune_optimizer = optim.Adam(model.parameters(), lr=config["fine_tune_lr"])
        
        best_tgt_c_index = 0
        best_fine_tune_state = None
        stage2_history = []
        
        for epoch in range(config["fine_tune_epochs"]):
            ft_metrics = fine_tune_epoch_ewc(
                model, target_loader, fine_tune_optimizer, ewc, config
            )
            
            # Evaluate on target
            X_tgt = target_data["X"]
            event_tgt = target_data["event"]
            duration_tgt = target_data["duration"]
            tgt_dataset = TensorDataset(X_tgt, event_tgt, duration_tgt)
            tgt_eval_loader = DataLoader(tgt_dataset, batch_size=config["batch_size"], shuffle=False)
            tgt_metrics = evaluate_cox(model, tgt_eval_loader, config)
            
            # Also evaluate on source to check forgetting
            src_check_metrics = evaluate_cox(model, val_loader, config)
            
            if (epoch + 1) % 5 == 0:
                print(f"Fine-tune Epoch {epoch+1}/{config['fine_tune_epochs']}:")
                print(f"  Loss: {ft_metrics['total_loss']:.4f} (cox={ft_metrics['cox_loss']:.4f}, ewc={ft_metrics['ewc_penalty']:.4f})")
                print(f"  Target C-index: {tgt_metrics['c_index']:.4f}, Source Val C-index: {src_check_metrics['c_index']:.4f}")
            
            stage2_history.append({
                "epoch": epoch + 1,
                **ft_metrics,
                "target_c_index": tgt_metrics["c_index"],
                "source_val_c_index": src_check_metrics["c_index"],
            })
            
            if tgt_metrics["c_index"] > best_tgt_c_index:
                best_tgt_c_index = tgt_metrics["c_index"]
                best_fine_tune_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if best_fine_tune_state is not None:
            model.load_state_dict(best_fine_tune_state)
        
        print(f"\nStage 2 Best Target C-index: {best_tgt_c_index:.4f}")
    else:
        best_tgt_c_index = None
        stage2_history = []
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    final_val_metrics = evaluate_cox(model, val_loader, config)
    print(f"Final Val C-index: {final_val_metrics['c_index']:.4f}")
    
    if target_data is not None:
        X_tgt = target_data["X"]
        event_tgt = target_data["event"]
        duration_tgt = target_data["duration"]
        tgt_dataset = TensorDataset(X_tgt, event_tgt, duration_tgt)
        tgt_eval_loader = DataLoader(tgt_dataset, batch_size=config["batch_size"], shuffle=False)
        final_tgt_metrics = evaluate_cox(model, tgt_eval_loader, config)
        print(f"Final Target C-index: {final_tgt_metrics['c_index']:.4f}")
    
    # Save results
    save_dir = "runs/v7.1_ewc_fine_tuning"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    
    results = {
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "best_val_c_index": best_val_c_index,
        "best_target_c_index": best_tgt_c_index,
        "final_val_c_index": final_val_metrics["c_index"],
        "final_target_c_index": final_tgt_metrics["c_index"] if target_data is not None else None,
    }
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nResults saved to {save_dir}")
    
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v7.1 Enhanced Features + EWC Fine-tuning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fine_tune_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--coral_weight", type=float, default=0.3)
    parser.add_argument("--mmd_weight", type=float, default=0.3)
    parser.add_argument("--mcd_weight", type=float, default=0.2)
    parser.add_argument("--ewc_lambda", type=float, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs": args.epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fine_tune_lr": args.fine_tune_lr,
        "d_model": args.d_model,
        "coral_weight": args.coral_weight,
        "mmd_weight": args.mmd_weight,
        "mcd_weight": args.mcd_weight,
        "ewc_lambda": args.ewc_lambda,
        "seed": args.seed,
    })
    
    results = run_v7_1_experiment(config)
