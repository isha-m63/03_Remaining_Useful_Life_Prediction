## 03. Remaining Useful Life Prediction
```
├── artifacts
├── configs
│   ├── data_constants_config.py
├── data
│   ├── processed
│   │   └── pca
│   └── raw
├── logs
├── notebooks
│   └── pca
│       ├── 01_eda.ipynb
│       ├── 02_initial_model_training.ipynb
│       └── 03_conformal_results.ipynb
├── README.md
├── setup.py
└── src
    ├── components
    │   ├── base_model_training.py
    │   ├── conformal_calibration.py
    │   ├── conformal_plots.py
    │   ├── data_ingestion_ideal_case.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── experiment_splitter.py
    │   ├── gp_training.py
    │   └── __init__.py
    ├── exception.py
    ├── __init__.py
    ├── logger.py
    ├── pipeline
    │   ├── __init__.py
    │   └── predict_pipeline.py
    └── utils.py
