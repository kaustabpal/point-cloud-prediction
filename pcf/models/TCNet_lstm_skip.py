#!/usr/bin/env python3
# @brief:     Pytorch Module for range image-based point cloud prediction
# @author     Benedikt Mersch    [mersch@igg.uni-bonn.de]
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from pcf.models.base_lstm import BasePredictionModel

class CustomConv2d(nn.Module):
    """Custom 2D Convolution that enables circular padding along the width dimension only"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        bias=False,
        circular_padding=False,
    ):
        """Init custom 2D Conv with circular padding"""
        super().__init__()
        self.circular_padding = circular_padding
        self.padding = padding

        if self.circular_padding:
            # Only apply zero padding in time and height
            zero_padding = (self.padding[0], 0)
        else:
            zero_padding = padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=zero_padding,
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        """Forward custom 2D convolution

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        if self.circular_padding:
            x = F.pad(
                x, (self.padding[1], self.padding[1], 0, 0), mode="circular"
            )
        x = self.conv(x)
        return x


class TCNet_lstm_skip(BasePredictionModel):
    def __init__(self, cfg):
        """Init all layers needed for range image-based point cloud prediction"""
        print("Import done")
        super().__init__(cfg)
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_CHANNEL_SIZE"]
        self.circular_padding = self.cfg["MODEL"]["CIRCULAR_PADDING"]
        self.feature_vector = self.cfg["MODEL"]["FEATURE_VECTOR"]

        self.input_layer = CustomConv2d(
            self.n_inputs,
            self.channels[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=True,
            circular_padding=self.circular_padding,
        )

        self.DownLayers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.DownLayers.append(
                    DownBlock(
                        self.cfg,
                        self.channels[i],
                        self.channels[i + 1],
                        skip=True,
                    )
                )
            else:
                self.DownLayers.append(
                    DownBlock(
                        self.cfg,
                        self.channels[i],
                        self.channels[i + 1],
                        skip=False,
                    )
                )
            
        # self.feature_conv_down = nn.Conv2d(
        #                     self.channels[-1],
        #                     self.feature_vector,
        #                     kernel_size=(2, 2),
        #                     stride=(1, 1),
        #                     padding=(0, 0),
        #                     bias=True,
        #                     )
        
        self.feature_linear_down = nn.Linear(
                            self.channels[-1]*2*2,
                            self.feature_vector,
                            bias=True,
                            )

        self.lstm_layer_encoder = nn.LSTM(
                                input_size=self.feature_vector,
                                hidden_size=self.feature_vector,
                                num_layers=2,
                                batch_first=True
                                )
        
        self.lstm_layer_decoder = nn.LSTM(
                                input_size=self.feature_vector,
                                hidden_size=self.feature_vector,
                                num_layers=2,
                                batch_first=True
                                )

        # self.feature_conv_up = nn.ConvTranspose2d(
        #                     self.feature_vector,
        #                     self.channels[-1],
        #                     kernel_size=(2, 2),
        #                     stride=(1, 1),
        #                     padding=(0, 0),
        #                     bias=True,
        #                     )

        self.feature_linear_up = nn.Linear(
                            self.feature_vector,
                            self.channels[-1]*2*2,
                            bias=True,
                            )

        self.unflatten = nn.Unflatten(1, (512, 2, 2))

        self.UpLayers = nn.ModuleList()
        for i in reversed(range(len(self.channels) - 1)):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.UpLayers.append(
                    UpBlock(
                        self.cfg,
                        self.channels[i + 1],
                        self.channels[i],
                        skip=True,
                    )
                )
            else:
                self.UpLayers.append(
                    UpBlock(
                        self.cfg,
                        self.channels[i + 1],
                        self.channels[i],
                        skip=False,
                    )
                )

        self.n_outputs = 2
        self.output_layer = CustomConv2d(
            self.channels[0],
            self.n_outputs,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=True,
            circular_padding=self.circular_padding,
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

        skip_list = []

        # features = torch.zeros((batch_size, n_past_steps, self.feature_vector)).to(self.device)
        # y_rv = torch.zeros((batch_size, self.n_future_steps, H, W)).to(self.device)
        # y_mask = torch.zeros((batch_size, self.n_future_steps, H, W)).to(self.device)

        decoder_input = torch.zeros((batch_size, self.n_future_steps, self.feature_vector)).to(self.device)

        x = x.view(batch_size*n_past_steps, n_inputs, H, W)
        x = self.input_layer(x)
        for layer in self.DownLayers:
            x = layer(x)
            if layer.skip:
                skip_list.append(x.clone())
        # x = self.feature_conv_down(x)
        x = torch.flatten(x, start_dim=1)
        x = self.feature_linear_down(x)
        
        x = x.view(batch_size, n_past_steps, self.feature_vector)

        x, h_c = self.lstm_layer_encoder(x)
        # x, (z, c) = self.lstm_layer_encoder(x) # added by KP
        # print("x shape: ", x.shape)
        # print("z shape: ", z.shape)
        # exit()

        for i in range(self.n_future_steps):
            x, h_c = self.lstm_layer_decoder(x[:, -1, :].view(batch_size, 1, self.feature_vector), h_c)
            decoder_input[:, i, :] = x[:, -1, :]
        
        x = torch.flatten(decoder_input.view(batch_size*self.n_future_steps, self.feature_vector, 1, 1), start_dim=1)
        # x = self.feature_conv_up(x)
        x = self.feature_linear_up(x)
        x = self.unflatten(x)
        for layer in self.UpLayers:
            if layer.skip:
                x = layer(x, skip_list.pop())
            else:
                x = layer(x)
        x = self.output_layer(x).view(batch_size, self.n_future_steps, self.n_outputs, H, W)
        
        output = {}
        output["rv"] = self.min_range + nn.Sigmoid()(x[:, :, 0, :, :]) * (
            self.max_range - self.min_range
        )
        output["mask_logits"] = x[:, :, 1, :, :]

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
            self.norm = nn.InstanceNorm2d(n_channels)
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
        self, cfg, in_channels, out_channels, skip=False
    ):
        """Init module"""
        super(DownBlock, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.conv0 = CustomConv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv2d(
            in_channels,
            out_channels,
            kernel_size=(2, 4),
            stride=(2, 4),
            bias=False,
            circular_padding=self.circular_padding,
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
        self, cfg, in_channels, out_channels, skip=False
    ):
        """Init module"""
        super(UpBlock, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        if self.skip:
            self.conv_skip = CustomConv2d(
                2 * in_channels,
                in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
                circular_padding=self.circular_padding,
            )
            self.norm_skip = Normalization(cfg, in_channels)
        self.conv0 = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=(2, 4),
            stride=(2, 4),
            bias=False,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization(cfg, out_channels)

    def forward(self, x, skip=None):
        """Forward pass for upsampling

        Args:
            x (torch.tensor): Input tensor
            skip (bool, optional): Use skip connection. Defaults to None.

        Returns:
            torch.tensor: Upsampled output tensor
        """
        if self.skip:
            x = torch.cat((x, skip), dim=1)
            x = self.conv_skip(x)
            x = self.norm_skip(x)
            x = self.relu(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        return x
