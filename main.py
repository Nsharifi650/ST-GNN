import os
import yaml

import torch
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from data.datapreprocessing import get_dataset
from src.model import Spatio_Temporal_GAT
from src.build_dataset import StockDataset
from src.model_training import train_model
from utils.config import configuration
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_configuration(file_path: str = "config/config.yaml") -> configuration:
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)

    return configuration(**config_dict)


def main():
    config = load_configuration()

    output_dir = config.inference.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.training.do_training:
        logger.info("running model training pipeline")
        get_dataset(config=config)

        dataset_builder = StockDataset(config)
        train_dataloader, val_dataloader, test_dataloader = dataset_builder.process()

        model = Spatio_Temporal_GAT(config)

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=config.training.LEARNING_RATE,
            weight_decay=config.training.WEIGHT_DECAY,
        )

        # Train Le Model:
        train_model(model, config, optimiser, train_dataloader, val_dataloader)
    else:
        dataset_builder = StockDataset(config)
        _, _, test_dataloader = dataset_builder.process()
        model = Spatio_Temporal_GAT(config)

    checkpoint_path = os.path.join(config.training.checkpoint_dir, "checkpoint.pth")
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f" loaded model parameters from checkpoint from {checkpoint_path}")
    else:
        print("no checkpoint found (!!), using untrained model.")

    # Inference:
    rmse, mae, mape, y_pred_list, y_truth_list = eval_model(
        model, device, test_dataloader, eval_type="Test"
    )
    y_pred = y_pred_list[0]
    y_truth = y_truth_list[0]

    batch_size = config.training.BATCH_SIZE
    num_nodes = len(config.stock_config.company_list)
    N_PRED = config.training.N_PRED

    # Reshape to [batch_size, num_nodes, N_PRED]
    y_pred = y_pred.view(batch_size, num_nodes, N_PRED)
    y_truth = y_truth.view(batch_size, num_nodes, N_PRED)

    # Select a specific node and prediction time step for analysis
    node = config.stock_config.company_list.index(config.inference.node)
    time_step = config.inference.step
    y_pred_selected = y_pred[:, node, time_step].cpu().numpy().flatten()
    y_truth_selected = y_truth[:, node, time_step].cpu().numpy().flatten()

    # Generate and save plots
    plot_and_save_line(
        y_truth_selected, y_pred_selected, output_dir, filename="line_plot.html"
    )
    plot_and_save_scatter(
        y_truth_selected, y_pred_selected, output_dir, filename="scatter_plot.html"
    )


@torch.no_grad()
def eval_model(model, device, dataloader, eval_type=""):
    model.eval()
    model.to(device)
    rmse, mae, mape = 0, 0, 0
    n = 0
    y_pred_list = []
    y_truth_list = []

    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch, device)
        truth = batch.y.view(pred.shape)
        rmse += torch.sqrt(torch.mean((truth - pred) ** 2))
        mae += torch.mean(torch.abs(truth - pred))
        mape += torch.mean(torch.abs((pred - truth)) / (truth + 1e-15) * 100)
        y_pred_list.append(pred.cpu())
        y_truth_list.append(truth.cpu())
        n += 1

    rmse, mae, mape = rmse / n, mae / n, mape / n
    print(f"{eval_type} | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
    return rmse, mae, mape, y_pred_list, y_truth_list


def plot_and_save_line(actual, predicted, output_path, filename="line_plot.html"):
    t = np.arange(len(predicted))
    trace_pred = go.Scatter(x=t, y=predicted, mode="lines", name="Predicted")
    trace_actual = go.Scatter(x=t, y=actual, mode="lines", name="Actual")
    layout = go.Layout(
        title="Predicted vs Actual",
        xaxis=dict(title="Time Steps"),
        yaxis=dict(title="Value"),
    )
    fig = go.Figure(data=[trace_pred, trace_actual], layout=layout)
    output_file = os.path.join(output_path, filename)
    pio.write_html(fig, file=output_file, auto_open=False)
    logger.info(f"Line plot saved to {output_file}")


def plot_and_save_scatter(actual, predicted, output_path, filename="scatter_plot.html"):
    trace = go.Scatter(x=actual, y=predicted, mode="markers", name="Predictions")
    line_trace = go.Scatter(
        x=[min(actual), max(actual)],
        y=[min(actual), max(actual)],
        mode="lines",
        name="Perfect Prediction",
    )
    layout = go.Layout(
        title="Scatter Plot: Actual vs Predicted",
        xaxis=dict(title="Actual"),
        yaxis=dict(title="Predicted"),
        width=1000,
        height=600,
    )
    fig = go.Figure(data=[trace, line_trace], layout=layout)
    output_file = os.path.join(output_path, filename)
    pio.write_html(fig, file=output_file, auto_open=False)
    logger.info(f"Scatter plot saved to {output_file}")


if __name__ == "__main__":
    main()
