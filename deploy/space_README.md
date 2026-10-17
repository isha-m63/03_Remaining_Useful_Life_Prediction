---
title: Turbofan RUL Prediction
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Engine RUL prediction with conformal intervals
---

# Turbofan Engine — Remaining Useful Life

Interactive demo for predicting the Remaining Useful Life of turbofan engines (NASA C-MAPSS FD001)
with **split-conformal prediction intervals**.

Pipeline: 14 sensors → StandardScaler → PCA(10) → XGBoost → RUL ± conformal quantile.

Source code and training pipeline: https://github.com/isha-m63/03_Remaining_Useful_Life_Prediction
