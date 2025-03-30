import os

import torch
from tqdm import tqdm

from src.model import Spatio_Temporal_GAT
from utils.config import configuration
from utils.logging_config import get_logger

logger = get_logger(__name__)


def save_checkpoint(state: dict, filename: str = "checkpoint.pth") -> None:
    torch.save(state, filename)


def load_checkpoint(
    model: Spatio_Temporal_GAT,
    optimiser,
    config: configuration,
    filename: str = "checkpoint.pth",
):
    checkpoint_dir = os.path.join(config.training.checkpoint_dir, filename)
    if os.path.isfile(checkpoint_dir):
        logger.info(f"Loading Check point: {filename}")
        checkpoint = torch.load(checkpoint_dir)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        logger.info(f"Loss at last checkpoint: {checkpoint['loss']}")
        return model, optimiser

    else:
        logger.info("No saved Checkpoints found, starting from untrained model")
        return model, optimiser


def train_model(
    model: Spatio_Temporal_GAT,
    config: configuration,
    optimiser,
    train_dataloader,
    val_dataloader,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, optimiser = load_checkpoint(model, optimiser, config)

    criterion = torch.nn.MSELoss()
    model.to(device)

    for epoch in range(config.training.EPOCHS):
        model.train()
        Training_loss = 0
        for _, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            optimiser.zero_grad()
            batch = batch.to(device)
            y_pred = torch.squeeze(model(batch, device))
            loss = criterion(y_pred.float(), torch.squeeze(batch.y).float())
            Training_loss += loss.item()
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            val_error = 0
            model.eval()
            for batch in val_dataloader:
                batch = batch.to(device)
                y_pred = torch.squeeze(model(batch, device))
                loss = criterion(y_pred.float(), torch.squeeze(batch.y).float())
                val_error += loss.item()

        print(
            f"Epoch: {epoch}, Traiining Loss: {Training_loss / len(train_dataloader):.7f}, validation_loss: {val_error / len(val_dataloader):.7f}"
        )
        state = {
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "loss": Training_loss,
            "epoch": epoch,
        }

        save_checkpoint(state)
