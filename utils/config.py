#Configuration 
from datetime import date
from pydantic import BaseModel, Field

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
    
class configuration(BaseModel):
    stock_config: finance_config
    training: training_hyperparameters

