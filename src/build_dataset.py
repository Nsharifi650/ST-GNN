import os

import random
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from utils.config import configuration
from utils.logging_config import get_logger


logger = get_logger(__name__)


def AdjacencyMatrix(n: int):
    AdjM = np.ones((n, n))
    # np.fill_diagonal(AdjM, 1)
    return AdjM


class StockDataset:
    """
    Simplified Dataset for Graph Neural Networks.
    """

    def __init__(self, config: configuration):
        self.config = config
        self.n_node = len(config.stock_config.company_list)
        self.W = AdjacencyMatrix(self.n_node)

    def process(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        data_dir = os.path.join(
            self.config.stock_config.dataset_path,
            self.config.stock_config.scaled_data_file_name,
        )
        data = pd.read_csv(data_dir, index_col=0).values

        sequences = self.generate_graphs(data)

        train, val, test = self.get_splits(sequences)

        logger.info(
            f"Loaded {len(train)} train, {len(val)} val, {len(test)} test samples."
        )

        train_dataloader = DataLoader(
            train,
            batch_size=self.config.training.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val,
            batch_size=self.config.training.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
        test_dataloader = DataLoader(
            test,
            batch_size=self.config.training.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )

        return train_dataloader, val_dataloader, test_dataloader

    def generate_graphs(self, data: np.ndarray):
        n_window = self.config.training.N_PRED + self.config.training.N_HIST
        edge_index, edge_attr = self._create_edges(self.n_node)
        sequences = self._create_sequences(
            data, self.n_node, n_window, edge_index, edge_attr
        )
        return sequences

    def _create_edges(self, n_node: int) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = torch.zeros((2, n_node**2), dtype=torch.long)
        edge_attr = torch.zeros((n_node**2, 1))
        num_edges = 0
        for i in range(n_node):
            for j in range(n_node):
                if self.W[i, j] != 0:
                    edge_index[:, num_edges] = torch.tensor([i, j], dtype=torch.long)
                    edge_attr[num_edges, 0] = self.W[i, j]
                    num_edges += 1
        edge_index = edge_index[:, :num_edges]
        edge_attr = edge_attr[:num_edges]
        return edge_index, edge_attr

    def _create_sequences(
        self,
        data: np.ndarray,
        n_node: int,
        n_window: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> list[Data]:
        sequences = []
        num_days, _ = data.shape

        for i in range(num_days - n_window + 1):
            sta = i
            end = i + n_window
            full_window = np.swapaxes(data[sta:end, :], 0, 1)

            g = Data(
                x=torch.FloatTensor(full_window[:, : self.config.training.N_HIST]),
                y=torch.FloatTensor(full_window[:, self.config.training.N_HIST :]),
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=n_node,
            )
            sequences.append(g)
        return sequences

    def get_splits(
        self, sequences: list[Data]
    ) -> tuple[list[Data], list[Data], list[Data]]:
        total = len(sequences)
        split_train = self.config.training.training_percent
        split_val = self.config.training.validation_percent
        # split_test = self.config.training.test_percent

        if self.config.training.shuffle_data:
            random.shuffle(sequences)

        # Calculate split indices
        idx_train = int(total * split_train)
        idx_val = int(total * (split_train + split_val))

        # Split the dataset
        train = sequences[:idx_train]
        val = sequences[idx_train:idx_val]
        test = sequences[idx_val:]

        return train, val, test
