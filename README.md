# Sherpa II

Sherpa II is a PyTorch-based framework for explainable EEG classification.  
The accompanying paper is in preparation: 

**Holthe, J., Sylvester, S., Haug, M., Sagehorn, M., Kisker, J., Gruber, T., and Schöne, B. (2025). SHERPA II: Polarity-Sensitive Explainable AI for decoding facilitative and suppressive contributions in perceptual inference. (*in prep*)**

A link to the preprint or published version will be added here once available.

SHERPA II is an extension of the first SHERPA paper (Paper: https://link.springer.com/article/10.3758/s13428-023-02335-7, Github: https://github.com/sophiasylvester/sherpa)

---

## Overview

The code provides the workflow for:

1. Loading preprocessed EEG data
2. Training and cross-validating the model with Optuna hyperparameter tuning
3. Evaluating performance on a holdout test set
4. Computing SHAP values for model interpretation

Only the **SherpaModel** was used in the final experiments.  
Other models (EEGNet, MidCNN, DeeperTemporalCNN) were used in preliminary experiments and are included for comparison.

---

## Further analyses

Further analyses were carried out in Matlab. SHAP values were averaged over trials and squared to provide SHAP Energy values. With those values, electrode clusters of interest were identified and plots were created
splitting the SHAP Energy values for positive and negative contributions.
