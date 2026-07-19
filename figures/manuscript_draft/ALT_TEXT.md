# Draft Figure Alt Text

## Figure 1

Three panels compare the two centers and target split. The source cohort is larger, but IDH prevalence is much higher at the target center while IH prevalence is lower. The 430 target patients are separated into 219 update, 39 validation, and 172 held-out test patients without overlap.

## Figure 2

Two parallel model diagrams show the prespecified outcome-specific update. For IDH, the 20-feature physiology/history branch receives CORAL alignment and the four-feature treatment branch is retained. For IH, the treatment branch receives CORAL and the physiology/history branch is retained. Both branches feed a shared event-risk head followed by target-validation Platt calibration.

## Figure 3

The seed-2024 forest plot shows a large IDH AUC increase from source to updated MLP and a smaller IH increase. Updated MLP and target-local logistic intervals overlap for both outcomes. Five-seed panels show substantial source-model variation but nearly identical updated AUCs. Precision-recall AUC also improves after updating, especially for IDH.

## Figure 4

Four slope charts compare representation diagnostics before and after updating. IDH RBF MMD decreases and physiology/history SHAP share rises markedly, while IH changes little. Domain-classifier AUC remains high, especially for IDH, and cross-center SHAP rank correlation increases for IDH but slightly decreases for IH. The panels use different prespecified seeds and are exploratory.
