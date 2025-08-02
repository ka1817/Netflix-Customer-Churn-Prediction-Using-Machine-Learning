import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

class DataPreprocessor:
    def __init__(self, numerical_features, categorical_features, target_column='churned'):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.pipeline = None

    def build_pipeline(self):
        """
        Create a preprocessing pipeline.
        """
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
        ])
        self.pipeline = preprocessor
        logging.info("Preprocessing pipeline built.")

    def transform_and_save(self, df: pd.DataFrame, output_path: str = "data/netflix_customer_churn.csv") -> None:
        if self.pipeline is None:
            raise ValueError("Pipeline is not built. Call build_pipeline() .")

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].reset_index(drop=True)

        X_transformed = self.pipeline.fit_transform(X)

        num_features = self.numerical_features
        cat_features = self.pipeline.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        all_features = list(num_features) + list(cat_features)

        X_transformed_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed,
                                        columns=all_features)
        X_transformed_df[self.target_column] = y

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        X_transformed_df.to_csv(output_path, index=False)
        logging.info(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion

    df = DataIngestion().load_data()

    numerical_features = ['watch_hours', 'last_login_days', 'number_of_profiles', 'avg_watch_time_per_day']
    categorical_features = ['subscription_type', 'payment_method']

    preprocessor = DataPreprocessor(numerical_features, categorical_features)
    preprocessor.build_pipeline()
    preprocessor.transform_and_save(df, output_path="data/netflix_customer_churn.csv")
