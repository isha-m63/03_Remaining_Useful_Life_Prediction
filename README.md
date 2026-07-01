## 03. Remaining Useful Life Prediction

── root
├── artifacts
├── configs
│   ├── data_constants_config.py
├── data
│   ├── processed
│   │   ├── test_set_001.csv
│   │   └── train_set_001.csv
│   └── raw
├── logs
├── notebooks
│   └── pca
│       ├── 01_eda.ipynb
│       └── 02_initial_model_training.ipynb
├── README.md
├── requirements.txt
├── setup.py
└── src
    ├── components
    │   ├── base_model_training.py
    │   ├── conformal_calibration.py
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
    │   └── __init__.py
    └── utils.py
