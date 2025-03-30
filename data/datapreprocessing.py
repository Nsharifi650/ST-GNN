import os
import pickle

import yfinance as yf
import datetime as dt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.config import configuration
from utils.logging_config import get_logger

logger = get_logger(__name__)


class StockDataException(Exception):
    pass


def download_stock_data(config: configuration) -> pd.DataFrame:
    start_date = dt.datetime.strptime(config.stock_config.start_date, "%Y-%m-%d")
    end_date = dt.datetime.strptime(config.stock_config.end_date, "%Y-%m-%d")
    companies_list = config.stock_config.company_list

    open_prices = {}
    try:
        for company in companies_list:
            logger.info(f"Downloading stocks for company: {company}")
            company_stock = yf.download(company, start_date, end_date)
            open_prices[company] = company_stock["Open"]
    except Exception as e:
        raise StockDataException(f"Error Downloading company stocks: {e}")

    return pd.DataFrame(open_prices)


def scale_data(data: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    logger.info("Finished scaling data")
    return data_scaled, scaler


def get_dataset(config: configuration) -> None:
    dataset = download_stock_data(config)
    scaled_dataset, scaler = scale_data(dataset)
    os.makedirs(config.stock_config.dataset_path, exist_ok=True)
    results_path = os.path.join(
        config.stock_config.dataset_path, config.stock_config.scaled_data_file_name
    )
    scaled_dataset.to_csv(results_path)

    with open(
        os.path.join(config.stock_config.dataset_path, "scaler.pkl"), "wb"
    ) as file:
        pickle.dump(scaler, file)

    logger.info("completed downloading, scaling and saving data")
