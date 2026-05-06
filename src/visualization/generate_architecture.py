import os
from pathlib import Path

def generate_architecture_md(out_path):
    mermaid_code = """
```mermaid
graph TD
    %% Data Sources
    subgraph Data Sources
        S[Source Domain: Shenyi Hospital<br/>N=210,000+ sessions] --> |Pre-dialysis & History| FS[Source Features: 95 Dims]
        T[Target Domain: Fuding Hospital<br/>N=60,000+ sessions] --> |Pre-dialysis & History| FT[Target Features: 95 Dims]
    end

    %% Preprocessing
    subgraph Epidemiological Correction
        FS --> KM_S[Kaplan-Meier Estimator<br/>Source Censoring Dist]
        FT --> KM_T[Kaplan-Meier Estimator<br/>Target Censoring Dist]
        KM_S --> IPCW_S[IPCW Weights Source]
        KM_T --> IPCW_T[IPCW Weights Target]
    end

    %% Model Architecture
    subgraph CDAN-GSN Architecture
        FS & FT --> KAN[Kolmogorov-Arnold Network Tokenizer<br/>Non-linear physiological mappings]
        
        KAN --> CLS[Learnable CLS Token]
        KAN --> Physio[Physiological Tokens]
        
        CLS & Physio --> Transformer[Transformer Encoder<br/>Self-Attention]
        
        Transformer --> |CLS Token| Hazard[Hazard Head<br/>Risk Prediction]
        Transformer --> |Physio Tokens| Mask[Learnable Gated Sparsity Mask<br/>Prevents Negative Transfer]
        
        Mask --> GRL[Gradient Reversal Layer]
    end

    %% Domain Adaptation
    subgraph Treatment-Conditioned Domain Adversarial Network
        GRL --> Concat[Concat]
        Hazard -.-> |Risk Condition| Concat
        FS & FT -.-> |Treatment Condition<br/>e.g. UFR| Concat
        
        Concat --> Discriminator[Domain Discriminator]
    end

    %% Losses
    Hazard --> |Source + IPCW| CoxS[Domain-Stratified Cox Loss Source]
    Hazard --> |Target + IPCW| CoxT[Domain-Stratified Cox Loss Target]
    Discriminator --> AdvLoss[Adversarial Loss]

    %% Styling
    classDef source fill:#d4e6f1,stroke:#0072b2,stroke-width:2px;
    classDef target fill:#f5cba7,stroke:#d55e00,stroke-width:2px;
    classDef model fill:#e8daef,stroke:#5dade2,stroke-width:2px;
    classDef loss fill:#f9e79f,stroke:#f39c12,stroke-width:2px;

    class S,FS,KM_S,IPCW_S,CoxS source;
    class T,FT,KM_T,IPCW_T,CoxT target;
    class KAN,CLS,Physio,Transformer,Hazard,Mask,GRL,Concat,Discriminator model;
    class CoxS,CoxT,AdvLoss loss;
```
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Model Architecture & Study Design\n\n")
        f.write("This diagram illustrates the data flow and architectural design of the CDAN-GSN model.\n")
        f.write(mermaid_code)
    print(f"Architecture diagram (Mermaid) saved to {out_path}")

if __name__ == "__main__":
    generate_architecture_md("figures/fig1_architecture.md")
