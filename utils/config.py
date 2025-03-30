# Configuration
from datetime import date
from pydantic import BaseModel


class finance_config(BaseModel):
    start_data: date
    end_date: date
    company_list: list[str]
    dataset_path: str
    scaled_data_file_name: str


class training_hyperparameters(BaseModel):
    BATCH_SIZE = int
    EPOCHS = int
    WEIGHT_DECAY = float
    LEARNING_RATE = float
    CHECKPOINT_DIR: str
    N_PRED: int
    N_HIST: int
    DROPOUT: float
    training_percent: float
    validation_percent: float
    test_percent: float


class Model_Parameters(BaseModel):
    in_channels: int
    out_channels: int
    attention_heads: int
    gru_l1_hidden_size: int
    gru_l1_layers: int
    gru_l2_hidden_size: int
    gru_l2_layers: int


class configuration(BaseModel):
    stock_config: finance_config
    training: training_hyperparameters
