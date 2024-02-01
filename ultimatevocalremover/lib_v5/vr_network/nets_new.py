import torch
from torch import nn
import torch.nn.functional as F
from . import layers_new as layers


class BaseNet(nn.Module):
    """
    BaseNet Class:
    This class defines the base network architecture for vocal removal. It includes a series of encoders for feature extraction,
    an ASPP module for capturing multi-scale context, and a series of decoders for reconstructing the output. Additionally,
    it incorporates an LSTM module for capturing temporal dependencies.
    """

    def __init__(self, input_channels, output_channels, input_channels_lstm, output_channels_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        # Initialize the encoder layers with increasing output channels for hierarchical feature extraction.
        self.encoder1 = layers.Conv2DBNActiv(input_channels, output_channels, 3, 1, 1)
        self.encoder2 = layers.Encoder(output_channels, output_channels * 2, 3, 2, 1)
        self.encoder3 = layers.Encoder(output_channels * 2, output_channels * 4, 3, 2, 1)
        self.encoder4 = layers.Encoder(output_channels * 4, output_channels * 6, 3, 2, 1)
        self.encoder5 = layers.Encoder(output_channels * 6, output_channels * 8, 3, 2, 1)

        # ASPP module for capturing multi-scale features with different dilation rates.
        self.aspp_module = layers.ASPPModule(output_channels * 8, output_channels * 8, dilations, dropout=True)

        # Decoder layers for upscaling and merging features from different levels of the encoder and ASPP module.
        self.decoder4 = layers.Decoder(output_channels * (6 + 8), output_channels * 6, 3, 1, 1)
        self.decoder3 = layers.Decoder(output_channels * (4 + 6), output_channels * 4, 3, 1, 1)
        self.decoder2 = layers.Decoder(output_channels * (2 + 4), output_channels * 2, 3, 1, 1)
        # LSTM module for capturing temporal dependencies in the sequence of features.
        self.lstm_decoder2 = layers.LSTMModule(output_channels * 2, input_channels_lstm, output_channels_lstm)
        self.decoder1 = layers.Decoder(output_channels * (1 + 2) + 1, output_channels * 1, 3, 1, 1)

    def __call__(self, input_tensor):
        # Sequentially pass the input through the encoder layers.
        encoded1 = self.encoder1(input_tensor)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.encoder3(encoded2)
        encoded4 = self.encoder4(encoded3)
        encoded5 = self.encoder5(encoded4)

        # Pass the deepest encoder output through the ASPP module.
        bottleneck = self.aspp_module(encoded5)

        # Sequentially upscale and merge the features using the decoder layers.
        bottleneck = self.decoder4(bottleneck, encoded4)
        bottleneck = self.decoder3(bottleneck, encoded3)
        bottleneck = self.decoder2(bottleneck, encoded2)
        # Concatenate the LSTM module output for temporal feature enhancement.
        bottleneck = torch.cat([bottleneck, self.lstm_decoder2(bottleneck)], dim=1)
        bottleneck = self.decoder1(bottleneck, encoded1)

        return bottleneck


class CascadedNet(nn.Module):
    """
    CascadedNet Class:
    This class defines a cascaded network architecture that processes input in multiple stages, each stage focusing on different frequency bands.
    It utilizes the BaseNet for processing, and combines outputs from different stages to produce the final mask for vocal removal.
    """

    def __init__(self, fft_size, nn_architecture_size=51000, output_channels=32, output_channels_lstm=128):
        super(CascadedNet, self).__init__()
        # Calculate frequency bins based on FFT size.
        self.max_frequency_bin = fft_size // 2
        self.output_frequency_bin = fft_size // 2 + 1
        self.input_channels_lstm = self.max_frequency_bin // 2
        self.offset = 64
        # Adjust output channels based on the architecture size.
        output_channels = 64 if nn_architecture_size == 218409 else output_channels

        # Initialize the network stages, each focusing on different frequency bands and progressively refining the output.
        self.stage1_low_band_net = nn.Sequential(
            BaseNet(2, output_channels // 2, self.input_channels_lstm // 2, output_channels_lstm), layers.Conv2DBNActiv(output_channels // 2, output_channels // 4, 1, 1, 0)
        )
        self.stage1_high_band_net = BaseNet(2, output_channels // 4, self.input_channels_lstm // 2, output_channels_lstm // 2)

        self.stage2_low_band_net = nn.Sequential(
            BaseNet(output_channels // 4 + 2, output_channels, self.input_channels_lstm // 2, output_channels_lstm), layers.Conv2DBNActiv(output_channels, output_channels // 2, 1, 1, 0)
        )
        self.stage2_high_band_net = BaseNet(output_channels // 4 + 2, output_channels // 2, self.input_channels_lstm // 2, output_channels_lstm // 2)

        self.stage3_full_band_net = BaseNet(3 * output_channels // 4 + 2, output_channels, self.input_channels_lstm, output_channels_lstm)

        # Output layer for generating the final mask.
        self.output_layer = nn.Conv2d(output_channels, 2, 1, bias=False)
        # Auxiliary output layer for intermediate supervision during training.
        self.auxiliary_output_layer = nn.Conv2d(3 * output_channels // 4, 2, 1, bias=False)

    def forward(self, input_tensor):
        # Preprocess input tensor to match the maximum frequency bin.
        input_tensor = input_tensor[:, :, : self.max_frequency_bin]

        # Split the input into low and high frequency bands.
        bandwidth = input_tensor.size()[2] // 2
        low_band_input = input_tensor[:, :, :bandwidth]
        high_band_input = input_tensor[:, :, bandwidth:]
        # Process each band through the first stage networks.
        low_band_stage1 = self.stage1_low_band_net(low_band_input)
        high_band_stage1 = self.stage1_high_band_net(high_band_input)
        # Combine the outputs for auxiliary supervision.
        auxiliary_output1 = torch.cat([low_band_stage1, high_band_stage1], dim=2)

        # Prepare inputs for the second stage by concatenating the original and processed bands.
        low_band_stage2_input = torch.cat([low_band_input, low_band_stage1], dim=1)
        high_band_stage2_input = torch.cat([high_band_input, high_band_stage1], dim=1)
        # Process through the second stage networks.
        low_band_stage2 = self.stage2_low_band_net(low_band_stage2_input)
        high_band_stage2 = self.stage2_high_band_net(high_band_stage2_input)
        # Combine the outputs for auxiliary supervision.
        auxiliary_output2 = torch.cat([low_band_stage2, high_band_stage2], dim=2)

        # Prepare input for the third stage by concatenating all previous outputs with the original input.
        full_band_stage3_input = torch.cat([input_tensor, auxiliary_output1, auxiliary_output2], dim=1)
        # Process through the third stage network.
        full_band_stage3 = self.stage3_full_band_net(full_band_stage3_input)

        # Apply the output layer to generate the final mask and apply sigmoid for normalization.
        mask = torch.sigmoid(self.output_layer(full_band_stage3))
        # Pad the mask to match the output frequency bin size.
        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_frequency_bin - mask.size()[2]), mode="replicate")

        # During training, generate and pad the auxiliary output for additional supervision.
        if self.training:
            auxiliary_output = torch.cat([auxiliary_output1, auxiliary_output2], dim=1)
            auxiliary_output = torch.sigmoid(self.auxiliary_output_layer(auxiliary_output))
            auxiliary_output = F.pad(input=auxiliary_output, pad=(0, 0, 0, self.output_frequency_bin - auxiliary_output.size()[2]), mode="replicate")
            return mask, auxiliary_output
        else:
            return mask

    # Method for predicting the mask given an input tensor.
    def predict_mask(self, input_tensor):
        mask = self.forward(input_tensor)

        # If an offset is specified, crop the mask to remove edge artifacts.
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert mask.size()[3] > 0

        return mask

    # Method for applying the predicted mask to the input tensor to obtain the predicted magnitude.
    def predict(self, input_tensor):
        mask = self.forward(input_tensor)
        predicted_magnitude = input_tensor * mask

        # If an offset is specified, crop the predicted magnitude to remove edge artifacts.
        if self.offset > 0:
            predicted_magnitude = predicted_magnitude[:, :, :, self.offset : -self.offset]
            assert predicted_magnitude.size()[3] > 0

        return predicted_magnitude
