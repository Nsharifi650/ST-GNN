# Configuration
from pydantic import BaseModel


class finance_config(BaseModel):
    start_date: str
    end_date: str
    company_list: list[str]
    dataset_path: str
    scaled_data_file_name: str


class training_hyperparameters(BaseModel):
    BATCH_SIZE: int
    EPOCHS: int
    WEIGHT_DECAY: float
    LEARNING_RATE: float
    N_PRED: int
    N_HIST: int
    DROPOUT: float
    training_percent: float
    validation_percent: float
    test_percent: float
    shuffle_data: bool
    checkpoint_dir: str
    do_training: bool


class Model_Parameters(BaseModel):
    attention_heads: int
    gru_l1_hidden_size: int
    gru_l1_layers: int
    gru_l2_hidden_size: int
    gru_l2_layers: int
    gru_decoder_hidden_size: int
    gru_decoder_layers: int


class Inference(BaseModel):
    output_dir: str
    node: str
    step: int


class configuration(BaseModel):
    stock_config: finance_config
    training: training_hyperparameters
    model: Model_Parameters
    inference: Inference
