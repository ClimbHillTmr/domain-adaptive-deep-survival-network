import copy
import torch
import torch.optim as optim
from src.evaluate.metrics import weighted_cox_loss, domain_loss_from_logits, evaluate_survival_metrics

def set_trainable_state(model, phase):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazard_head.parameters():
        param.requires_grad = True
    for param in model.domain_head.parameters():
        param.requires_grad = True
    if phase in {"partial_unfreeze", "full_finetune"}:
        for param in model.transformer.layers[-1].parameters():
            param.requires_grad = True
    if phase == "full_finetune":
        if model.tokenizer_type == "kan":
            for param in model.tokenizer.parameters():
                param.requires_grad = True
        else:
            model.feature_weight.requires_grad = True
            model.feature_bias.requires_grad = True
        model.feature_pos.requires_grad = True
        model.cls_token.requires_grad = True
        model.feature_mask.requires_grad = True
        for param in model.transformer.parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True

def build_optimizer(model, phase, lr, top_encoder_lr=None):
    if top_encoder_lr is None:
        top_encoder_lr = lr
        
    if phase == "head_only":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": lr},
            {"params": model.domain_head.parameters(), "lr": lr},
        ]
        return optim.AdamW(groups, weight_decay=1e-3)
    if phase == "partial_unfreeze":
        groups = [
            {"params": model.hazard_head.parameters(), "lr": lr},
            {"params": model.domain_head.parameters(), "lr": lr},
            {"params": model.transformer.layers[-1].parameters(), "lr": top_encoder_lr},
        ]
        return optim.AdamW(groups, weight_decay=1e-3)
    if phase == "full_finetune":
        params_to_update = [
            {"params": model.hazard_head.parameters(), "lr": lr},
            {"params": model.domain_head.parameters(), "lr": lr},
            {"params": model.transformer.parameters(), "lr": lr},
            {"params": model.norm.parameters(), "lr": lr},
        ]
        base_params = [model.feature_pos, model.cls_token, model.feature_mask]
        if model.tokenizer_type == "kan":
            base_params.extend(list(model.tokenizer.parameters()))
        else:
            base_params.extend([model.feature_weight, model.feature_bias])
        params_to_update.append({"params": base_params, "lr": lr})
        
        return optim.AdamW(params_to_update, weight_decay=1e-4)
    raise ValueError(phase)

def run_source_pretrain(model, source_loader, x_val, e_val, t_val, lr, device, max_epochs=18, patience=4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val, best_epoch = -float('inf'), 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = patience
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_s, e_s, t_s, w_s in source_loader:
            x_s = x_s.to(device)
            e_s = e_s.to(device)
            t_s = t_s.to(device)
            w_s = w_s.to(device)
            
            optimizer.zero_grad()
            _, hazard, _, _ = model(x_s, grl_coeff=None)
            
            loss = weighted_cox_loss(hazard, e_s, t_s, w_s)
            loss.backward()
            optimizer.step()
            
        metrics = evaluate_survival_metrics(model, x_val, e_val, t_val, device=device)
        val_cindex = metrics["C-index"]
        
        if val_cindex > best_val:
            best_val, best_epoch = val_cindex, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
                
    model.load_state_dict(best_state)
    return best_val, best_epoch, best_state

def run_domain_stratified_da(
    model, source_loader, target_loader, x_val, e_val, t_val, 
    adv_weight, source_replay_weight, mask_l1_weight, lr, device,
    phases_config=None
):
    if phases_config is None:
        phases_config = [
            ("head_only", 4, 2),
            ("partial_unfreeze", 6, 2),
            ("full_finetune", 20, 4),
        ]
        
    phase_results = []
    
    for phase, max_epochs, patience in phases_config:
        set_trainable_state(model, phase)
        optimizer = build_optimizer(model, phase, lr)
        
        metrics = evaluate_survival_metrics(model, x_val, e_val, t_val, device=device)
        best_val, best_epoch = metrics["C-index"], 0
        best_state = copy.deepcopy(model.state_dict())
        patience_left = patience
        
        for epoch in range(1, max_epochs + 1):
            model.train()
            coeff = min(1.0, epoch / max_epochs)
            source_iter = iter(source_loader)
            
            for x_t, e_t, t_t, w_t in target_loader:
                try:
                    x_s, e_s, t_s, w_s = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_s, e_s, t_s, w_s = next(source_iter)
                
                x_t = x_t.to(device)
                e_t = e_t.to(device)
                t_t = t_t.to(device)
                w_t = w_t.to(device)
                
                x_s = x_s.to(device)
                e_s = e_s.to(device)
                t_s = t_s.to(device)
                w_s = w_s.to(device)
                
                optimizer.zero_grad()
                
                _, hazard_t, dom_t, mask_l1_t = model(x_t, grl_coeff=coeff)
                _, hazard_s, dom_s, mask_l1_s = model(x_s, grl_coeff=coeff)
                
                target_cox = weighted_cox_loss(hazard_t, e_t, t_t, w_t)
                source_cox = weighted_cox_loss(hazard_s, e_s, t_s, w_s)
                
                adv_loss = domain_loss_from_logits(dom_s, dom_t)
                l1_loss = (mask_l1_t + mask_l1_s) / 2.0
                
                loss = target_cox + source_replay_weight * source_cox + adv_weight * adv_loss + mask_l1_weight * l1_loss
                loss.backward()
                optimizer.step()
                
            metrics = evaluate_survival_metrics(model, x_val, e_val, t_val, device=device)
            val_cindex = metrics["C-index"]
            
            if val_cindex > best_val:
                best_val, best_epoch = val_cindex, epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
                    
        model.load_state_dict(best_state)
        phase_results.append({"phase": phase, "best_adapt_val_cindex": best_val, "best_epoch": best_epoch})
        
    return phase_results
