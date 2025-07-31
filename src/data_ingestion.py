import os
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)

class DataIngestion:
    def __init__(self, file_name: str = "netflix_customer_churn.csv", data_dir: str = "data"):
        root_dir = os.path.dirname(os.path.dirname(__file__))  
        self.data_path = os.path.join(root_dir, data_dir, file_name)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            logging.error(f"File not found: {self.data_path}")
            raise FileNotFoundError(f"File not found: {self.data_path}")

        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Data loaded successfully with shape {self.df.shape}")
            return self.df
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing CSV: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise

    def get_summary(self):
        if self.df is None:
            logging.warning("No data loaded. Call load_data() first.")
            return
        logging.info("Displaying DataFrame Info and Summary Statistics:")
        print(self.df.info())
        print("\nSummary Statistics:")
        print(self.df.describe(include='all'))


if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    ingestion.get_summary()

