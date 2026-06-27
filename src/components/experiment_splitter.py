"""
ExperimentSplitter generates a calibration set out of the processed training data.

Why this is a seaparate module (not part of DataTransformation):

  DataTransformation's job is to load raw data, apply PCA/scaling, save artifacts.
  It knows nothing about how many models you're training or whether you need
  conformal prediction.

  ExperimentSplitter's job is given the already-transformed training data,
  split it by engine_id into train_proper and calibration.

  The split only exists because of conformal prediction. If you drop
  conformal from the project, ExperimentSplitter disappears entirely
  DataTransformation stays unchanged. That's the test for whether a module
  belongs here, can you remove it without touching anything else?

Why split by engine ID (not by row index)?

  Each engine is one independent degradation trajectory. The training data
  has ~20,000 rows from 100 engines. If you split by row index (e.g. first
  80% of rows = train, last 20% = cal), you might put cycles 1-150 of engine
  42 in train and cycles 151-200 of the same engine in calibration.

  That leaks: the calibration residuals would be computed on data the model
  has partially seen (the early life of that engine). Conformal's exchangeability
  assumption requires calibration points to be truly held-out.

  Engine-level split is the only correct approach for time series with
  multiple independent entities.

Where this step is taken in the pipeline:

  DataIngestion → DataTransformation → [ExperimentSplitter] → ModelTrainer
                                             ↑
                                   runs here, after PCA is applied,
                                   before any model sees the data
"""