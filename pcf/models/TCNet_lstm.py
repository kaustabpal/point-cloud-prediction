#!/usr/bin/env python3
# @brief:     Pytorch Module for range image-based point cloud prediction
# @author     Benedikt Mersch    [mersch@igg.uni-bonn.de]
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from pcf.models.base_lstm import BasePredictionModel

class TCNet_lstm(BasePredictionModel):
    def __init__(self, cfg):
        """Init all layers needed for range image-based point cloud prediction"""
        print("Import done")
        super().__init__(cfg)
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.circular_padding = self.cfg["MODEL"]["CIRCULAR_PADDING"]
        self.feature_vector = self.cfg["MODEL"]["FEATURE_VECTOR"]

        self.input_layer = nn.Conv2d(
            self.n_inputs,
            self.channels[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            padding_mode='circular',
        )

        self.DownLayers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.DownLayers.append(
                DownBlock(
                    self.cfg,
                    self.channels[i],
                    self.channels[i + 1],
                )
            )
        
        self.feature_conv_down = nn.Conv2d(
                            self.channels[-1],
                            self.feature_vector,
                            kernel_size=(2, 2),
                            stride=(1, 1),
                            padding=(0, 0),
                            bias=True,
                            )
        
        self.lstm_layer = nn.LSTM(
                                input_size=self.feature_vector,
                                hidden_size=self.feature_vector,
                                num_layers=2,
                                batch_first=True
                                )

        self.feature_conv_up = nn.ConvTranspose2d(
                            self.feature_vector,
                            self.channels[-1],
                            kernel_size=(2, 2),
                            stride=(1, 1),
                            padding=(0, 0),
                            bias=True,
                            )

        self.UpLayers = nn.ModuleList()
        for i in reversed(range(len(self.channels) - 1)):
            self.UpLayers.append(
                UpBlock(
                    self.cfg,
                    self.channels[i + 1],
                    self.channels[i],
                )
            )

        self.n_outputs = 2
        self.output_layer = nn.Conv2d(
            self.channels[0],
            self.n_outputs,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            padding_mode='circular',
        )

    def forward(self, x):
        """Forward range image-based point cloud prediction

        Args:
            x (torch.tensor): Input tensor of concatenated, unnormalize range images

        Returns:
            dict: Containing the predicted range tensor and mask logits
        """
        # Only select inputs specified in base model
        x = x[:, self.inputs, :, :, :]
        batch_size, n_inputs, n_past_steps, H, W = x.size()
        assert n_inputs == self.n_inputs

        # Get mask of valid points
        past_mask = x != -1.0

        # Standardization and set invalid points to zero
        mean = self.mean[None, self.inputs, None, None, None]

        std = self.std[None, self.inputs, None, None, None]
        x = torch.true_divide(x - mean, std)
        x = x * past_mask

        features = torch.zeros((batch_size, n_past_steps, self.feature_vector)).to(self.device)
        y = torch.zeros((batch_size, self.n_outputs, n_past_steps, H, W)).to(self.device)

        x_in = x.view(batch_size, n_inputs, n_past_steps, H, W)
        for i in range(n_past_steps):
            x = x_in[:, :, i, :, :]
            x = self.input_layer(x)
            for layer in self.DownLayers:
                x = layer(x)
            x = self.feature_conv_down(x)
            features[:, i, :] = x.view(batch_size, self.feature_vector)
        
        x_lstm, decoder_hidden = self.lstm_layer(features)

        x = self.feature_conv_up(x_lstm[:, -1, :].view(batch_size, self.feature_vector, 1, 1))
        for layer in self.UpLayers:
            x = layer(x)
        x = self.output_layer(x)
        y[:, :, 0, :, :] = x

        for i in range(1, self.n_future_steps):
            x_lstm, decoder_hidden = self.lstm_layer(x_lstm[:, -1, :].view(batch_size, 1, self.feature_vector), decoder_hidden)
            x = self.feature_conv_up(x_lstm.view(batch_size, self.feature_vector, 1, 1))
            for layer in self.UpLayers:
                x = layer(x)
            x = self.output_layer(x)
            y[:, :, i, :, :] = x

        output = {}
        output["rv"] = self.min_range + nn.Sigmoid()(y[:, 0, :, :, :]) * (
            self.max_range - self.min_range
        )
        # output["rv"] = y[:, 0, :, :, :]
        output["mask_logits"] = y[:, 1, :, :, :]

        return output


class Normalization(nn.Module):
    """Custom Normalization layer to enable different normalization strategies"""

    def __init__(self, cfg, n_channels):
        """Init custom normalization layer"""
        super(Normalization, self).__init__()
        self.cfg = cfg
        self.norm_type = self.cfg["MODEL"]["NORM"]
        n_channels_per_group = self.cfg["MODEL"]["N_CHANNELS_PER_GROUP"]

        if self.norm_type == "batch":
            self.norm = nn.BatchNorm2d(n_channels)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(n_channels // n_channels_per_group, n_channels)
        elif self.norm_type == "instance":
            self.norm = nn.InstanceNorm3d(n_channels)
        elif self.norm_type == "none":
            self.norm = nn.Identity()

    def forward(self, x):
        """Forward normalization pass

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        x = self.norm(x)
        return x


class DownBlock(nn.Module):
    """Downsamples the input tensor"""

    def __init__(
        self, cfg, in_channels, out_channels
    ):
        """Init module"""
        super(DownBlock, self).__init__()
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.conv0 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            padding_mode='circular',
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2, 4),
            stride=(2, 4),
            bias=False,
            padding_mode='circular',
        )
        self.norm1 = Normalization(cfg, out_channels)

    def forward(self, x):
        """Forward pass for downsampling

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Downsampled output tensor
        """
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    """Upsamples the input tensor using transposed convolutions"""

    def __init__(
        self, cfg, in_channels, out_channels
    ):
        """Init module"""
        super(UpBlock, self).__init__()
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.conv0 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=(2, 4),
            stride=(2, 4),
            bias=False,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            padding_mode='circular',
        )
        self.norm1 = Normalization(cfg, out_channels)

    def forward(self, x):
        """Forward pass for upsampling

        Args:
            x (torch.tensor): Input tensor
            skip (bool, optional): Use skip connection. Defaults to None.

        Returns:
            torch.tensor: Upsampled output tensor
        """
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x
