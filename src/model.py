import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from utils.config import configuration


class ST_GAT(torch.nn.Module):
    def __init__(self, config: configuration):
        super(ST_GAT, self).__init__()

        self.n_pred = config.training.N_PRED
        self.n_nodes = len(config.stock_config.company_list)

        # Graph Attention step
        self.gat = GATConv(
            in_channels=config.training.N_HIST,
            out_channels=config.training.N_HIST,
            heads=config.model.attention_heads,
            dropout=config.training.DROPOUT,
            concat=False,
        )

        # enconder GRU layers
        self.encoder_gru_l1 = torch.nn.GRU(
            input_size=self.n_nodes,
            hidden_size=config.model.gru_l1_hidden_size,
            num_layers=config.model.gru_l1_layers,
            bias=True,
        )

        for name, param in self.encoder_gru_l1.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)

        self.encoder_gru_l2 = torch.nn.GRU(
            input_size=config.model.gru_l1_hidden_size,
            hidden_size=config.model.gru_l2_hidden_size,
            num_layers=config.model.gru_l2_layers,
            bias=True,
        )

        for name, param in self.encoder_gru_l2.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)

        # gru decoders
        self.GRU_decoder = torch.nn.GRU(
            input_size=config.model.gru_l2_hidden_size,
            hidden_size=config.model.gru_decoder_hidden_size,
            num_layers=config.model.gru_decoder_layers,
            bias=True,
            dropout=config.training.DROPOUT,
        )

        self.prediction_layer = torch.nn.Linear(
            config.model.gru_decoder_hidden_size, self.n_nodes * self.n_pred, bias=True
        )

        torch.nn.init.xavier_uniform_(self.prediction_layer.weight)
        torch.nn.init.constant_(self.prediction_layer.bias, 0)

    def forward(self, data, device):
        x, edge_index = data.x, data.edge_index

        if device == "cpu":
            x = torch.FloatTensor(x)
        else:
            x = torch.cuda.FloatTensor(x)

        x = self.gat(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        batch_size = data.num_graphs
        n_node = int(data.num_nodes / batch_size)
        x = torch.reshape(x, (batch_size, n_node, data.num_features))
        x = torch.movedim(x, 2, 0)
        encoderl1_outputs, _ = self.encoder_gru_l1(x)
        x = F.relu(encoderl1_outputs)

        encoderl2_outputs, h2 = self.encoder_gru_l2(x)
        x = F.relu(encoderl2_outputs)

        # decoder
        x, _ = self.GRU_decoder(x, h2)

        x = torch.squeeze(x[-1, :, :])
        # print("lin",x.shape)
        x = self.prediction_layer(x)
        # print("linO",x.shape)
        # [batch_size, nodes*num of future steps]

        # Now reshape into final output
        x = torch.reshape(x, (batch_size, self.n_nodes, self.n_pred))
        x = torch.reshape(x, (batch_size * self.n_nodes, self.n_pred))

        return x
