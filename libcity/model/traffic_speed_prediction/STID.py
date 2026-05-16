from logging import getLogger

import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class STID(AbstractTrafficStateModel):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get('num_nodes')
        self.input_window = config.get('input_window')
        self.output_window = config.get('output_window')
        self.feature_dim = data_feature.get('feature_dim', 2)
        self.output_dim = self.data_feature.get('output_dim', 1)
        # STID only uses flow data for time series, so use 1 channel
        self.model_output_dim = config.get('model_output_dim', 1)
        self.time_intervals = config.get('time_intervals')
        self._scaler = self.data_feature.get('scaler')

        self.num_block = config.get('num_block')
        self.time_series_emb_dim = config.get('time_series_emb_dim')
        self.spatial_emb_dim = config.get('spatial_emb_dim')
        self.temp_dim_tid = config.get('temp_dim_tid')
        self.temp_dim_diw = config.get('temp_dim_diw')
        self.if_spatial = config.get('if_spatial')
        self.if_time_in_day = config.get('if_TiD')
        self.if_day_in_week = config.get('if_DiW')

        self.device = config.get('device', torch.device('cpu'))

        assert (24 * 60 * 60) % self.time_intervals == 0, "time_of_day_size should be Int"
        self.time_of_day_size = int((24 * 60 * 60) / self.time_intervals)
        self.day_of_week_size = 7

        self._logger = getLogger()

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.model_output_dim * self.input_window, out_channels=self.time_series_emb_dim, kernel_size=(1, 1),
            bias=True)

        # encoding
        self.hidden_dim = self.time_series_emb_dim + self.spatial_emb_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_block)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.model_output_dim * self.output_window, kernel_size=(1, 1), bias=True)

    def forward(self, batch):
        # prepare data
        input_data = batch['X']  # [B, L, N, C]
        # STID only uses flow data (first channel) for time series embedding
        time_series = input_data[..., :self.model_output_dim]

        if self.if_time_in_day:
            tid_data = input_data[..., 1]
            time_in_day_emb = self.time_in_day_emb[(tid_data[:, -1, :] * self.time_of_day_size).long()]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            # input_data[..., 2:9] is one-hot encoded day_in_week with 7 classes
            # Clamp to ensure valid indices [0, 6]
            diw_data = torch.argmax(input_data[..., 2:9], dim=-1)
            day_in_week_emb = self.day_in_week_emb[diw_data[:, -1, :].long()]
        else:
            day_in_week_emb = None

        # time series embedding: [B, L, N, C] -> [B, C*L, N, 1]
        batch_size, seq_len, num_nodes, num_features = time_series.shape
        time_series = time_series.permute(0, 2, 1, 3).contiguous()
        time_series = time_series.view(batch_size, num_nodes, -1)
        time_series = time_series.transpose(1, 2)
        time_series = time_series.unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(time_series)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)  # concat all embeddings

        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)

        return prediction

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        # y_predicted shape: [B, output_window, N, model_output_dim]
        # y_true shape: [B, output_window, N, output_dim]
        # Expand y_predicted from [B, L, N, 1] to [B, L, N, output_dim] for loss computation
        if self.model_output_dim == 1 and self.output_dim > 1:
            y_predicted = y_predicted.expand(-1, -1, -1, self.output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        output = self.forward(batch)
        # Expand output from [B, L, N, model_output_dim] to [B, L, N, output_dim]
        # for evaluation compatibility. Repeat the flow prediction for all output dims.
        if self.model_output_dim == 1 and self.output_dim > 1:
            output = output.repeat(1, 1, 1, self.output_dim)
        return output
