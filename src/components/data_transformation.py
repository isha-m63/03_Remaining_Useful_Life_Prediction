import sys
import os
import pickle
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
from src.exception import CustomException
from src.logger import logging
from configs.data_constants_config import RUL_CAP, PCA_N_COMPONENETS

SENSOR_COLS  = [f"s{i}" for i in range(1, 22)]
FEATURE_COLS = [c for c in SENSOR_COLS]  


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str  = os.path.join('artifacts', 'preprocessor.pkl')
    train_data_raw_csv_path: str = os.path.join('data/processed', 'train_set_001.csv')
    test_data_raw_csv_path: str = os.path.join('data/processed', 'test_set_001.csv')
    rul_path: str = os.path.join('data/raw', 'RUL_FD001.txt')

    X_train_processed_path: str = os.path.join("artifacts", "X_train_processed.csv")
    y_train_path: str = os.path.join("artifacts", "y_train.csv")
    #X_cal_processed_path: str = os.path.join("artifacts", "X_cal_processed.csv")
    #y_cal_processed_path: str = os.path.join("artifacts", "y_cal_processed.csv")
    X_test_processed_path:  str = os.path.join("artifacts", "X_test_processed.csv")
    y_test_path:  str = os.path.join("artifacts", "y_test.csv")



class DataTransformation:
    """
    Steps:
      1. Add piecewise-linear RUL labels to the training set.
      2. Fit StandardScaler + PCA on training sensor rows only.
      3. Transform train and test sets using those fitted objects.
      4. Save the fitted preprocessor to artifacts/ for inference reuse.
 
    What this class does NOT do:
      - Split train into train_processed / calibration. That split only exists
        because of conformal prediction — it lives in ExperimentSplitter.
      - Choose PCA n_components. That belongs in config.
      - Train any model.

    Essentially, 
    X_train_processed - scaled + PCA transformed   (model trains on this)
    y_train_processed - raw RUL values, no scaling (scaling RUL causes no issues but adds no benefit), , clipped to 125
    (X_train_processed)
    
    X_cal_processed - scaled + PCA transformed   (same pipeline, .transform() only)
    y_cal_processed - raw RUL values, clipped to 125

    X_test_processed - scaled + PCA transformed   (same pipeline, .transform() only)
    y_test_processed - raw RUL values, clipped to 125
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.pca_n_components = PCA_N_COMPONENETS


    @staticmethod
    def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        max_cycles = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
        df = df.join(max_cycles, on="engine_id")
        df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CAP)
        df.drop(columns=["max_cycle"], inplace=True)
        return df


    def get_data_transformer_object(self):
        try: 
            transform_pipeline = Pipeline(
                steps = [
                    ('scaler', StandardScaler()), 
                    ('pca', PCA(n_components=self.pca_n_components))
                ]
            )
            logging.info('Preprocessing complete - sensor columns are scaled and PCA transformed')
            return transform_pipeline
        except Exception as e: 
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_data_raw_csv_path: str, test_data_raw_csv_path: str, rul_path: str):
        """
        Full transformation pipeline: load - label - fit - transform - save.
 
        Args:
            train_path: Path to train_set_001.csv (contains header, but does not have capped RUL values = y_train)
            test_path: Path to train_set_001.csv (contains header, also does not contain capped RUL_values appeneded)
            rul_path: Path to RUL_FD001.txt  (ground-truth test RUL = y_train)
 
        Returns:
            X_train, y_train, X_test, y_test
            X_* are DataFrames with PC1..PCn columns.
            y_* are Series with RUL values.
        """

        try: 
            train_df = pd.read_csv(train_data_raw_csv_path)
            test_df = pd.read_csv(test_data_raw_csv_path)
            y_test_raw = pd.read_csv(rul_path, sep=r'\s+', header = None, names = ['RUL'])
            logging.info('Reading training and testing data is complete')

            logging.info("Loaded: %d train rows (%d engines), %d test engines", len(train_df),
                         train_df["engine_id"].nunique(), test_df["engine_id"].nunique(),)
            

            #Add RUL labels to raw training data
            train_df = self.add_rul_labels(train_df)
            logging.info( "RUL labels added: range [%.0f, %.0f], cap=%d",
                         train_df["RUL"].min(), train_df["RUL"].max(), RUL_CAP)


            #Extract features from raw training data
            X_train_raw = train_df[FEATURE_COLS]
            y_train = train_df["RUL"].reset_index(drop=True)

            #For test set, take the last observed cycle per engine (CMAPSS convention).
            #ground-truth RUL is from RUL_FD001.txt, clipped to same cap.
            test_last  = test_df.groupby("engine_id").last().reset_index()
            X_test_raw = test_last[FEATURE_COLS]
            y_test = y_test_raw["RUL"].clip(upper=RUL_CAP).reset_index(drop=True)


            logging.info("Unprocessed feature shapes: train: %s, test: %s", X_train_raw.shape, X_test_raw.shape)

            #Fitting the pipeline on training data, transforming training and test data
            logging.info('Obtaining preprocessing object')
            pipeline = self.get_data_transformer_object()
            X_train_arr = pipeline.fit_transform(X_train_raw)  #fit + transform
            X_test_arr  = pipeline.transform(X_test_raw)        #transform only
 
            explained = pipeline.named_steps["pca"].explained_variance_ratio_.cumsum()[-1]
            logging.info("PCA: %d components explain %.1f%% variance", self.pca_n_components, explained * 100)

            #Convert to dataframes
            pc_cols = [f"PC{i+1}" for i in range(self.pca_n_components)]
            X_train_processed = pd.DataFrame(X_train_arr, columns=pc_cols)
            X_test_processed  = pd.DataFrame(X_test_arr,  columns=pc_cols)
            os.makedirs("artifacts", exist_ok=True)
 
            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(pipeline, f)
            logging.info("Preprocessor saved - %s", self.data_transformation_config.preprocessor_obj_file_path)

            X_train_processed.to_csv(self.data_transformation_config.X_train_processed_path, index = False)
            y_train.to_csv(self.data_transformation_config.y_train_path, index = False, header= True)
            X_test_processed.to_csv(self.data_transformation_config.X_test_processed_path,  index = False)
            y_test.to_csv(self.data_transformation_config.y_test_path, index = False, header = True)
            logging.info("Processed CSVs saved to artifacts")
 
            return X_train_processed, y_train, X_test_processed, y_test
 

        except Exception as e: 
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataTransformation()

    obj.initiate_data_transformation(
        train_data_raw_csv_path=obj.data_transformation_config.train_data_raw_csv_path,
        test_data_raw_csv_path=obj.data_transformation_config.test_data_raw_csv_path,
        rul_path=obj.data_transformation_config.rul_path,
    )