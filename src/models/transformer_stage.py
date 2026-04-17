import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HybridTransformer(nn.Module):
    def __init__(self, config, num_static_features, num_dynamic_features, num_classes=4):
        super(HybridTransformer, self).__init__()
        
        d_model = config.transformer.d_model
        nhead = config.transformer.nhead
        num_layers = config.transformer.num_layers
        dropout = config.transformer.dropout
        
        # 1. Dynamic Branch (Transformer)
        self.embedding = nn.Linear(num_dynamic_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 2. Static Branch (MLP)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 3. Fusion & Head
        # We concat [Dynamic_CLS, Static_Emb] -> 2 * d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes) # Output logits for classes
        )
        
        # Survival Head (Optional, if risk score needed)
        self.survival_head = nn.Linear(d_model * 2, 1)

    def forward(self, x_static, x_dynamic):
        # x_dynamic: (Batch, Seq, Feat) -> (Seq, Batch, Feat) for PyTorch Transformer
        x_dyn = x_dynamic.permute(1, 0, 2)
        
        # Dynamic embedding
        x_dyn = self.embedding(x_dyn)
        x_dyn = self.pos_encoder(x_dyn)
        
        # Transformer Encoder
        # output: (Seq, Batch, d_model)
        out_dyn = self.transformer_encoder(x_dyn)
        
        # Pooling (Global Average Pooling or use last token)
        # Here we take Mean Pooling across sequence dimension
        # (Batch, d_model)
        emb_dyn = out_dyn.mean(dim=0) 
        
        # Static embedding
        # (Batch, d_model)
        emb_static = self.static_mlp(x_static)
        
        # Fusion
        combined = torch.cat((emb_dyn, emb_static), dim=1)
        
        # Output
        logits = self.classifier(combined)
        risk_score = self.survival_head(combined)
        
        return logits, risk_score, combined # Return combined for CORAL loss
