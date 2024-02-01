import torch
from torch import nn
import torch.nn.functional as F

from . import layers


class BaseASPPNet(nn.Module):
    """
    BaseASPPNet Class:
    This class defines the base architecture for an Atrous Spatial Pyramid Pooling (ASPP) network.
    It is designed to extract features from input data at multiple scales by using dilated convolutions.
    This is particularly useful for tasks that benefit from understanding context at different resolutions,
    such as semantic segmentation. The network consists of a series of encoder layers for downsampling and feature extraction,
    followed by an ASPP module for multi-scale feature extraction, and finally a series of decoder layers for upsampling.
    """

    def __init__(self, network_architecture, input_channels, channel_multiplier, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.network_architecture = network_architecture
        # Encoder layers progressively increase the number of channels while reducing spatial dimensions.
        self.encoder_layer1 = layers.Encoder(input_channels, channel_multiplier, 3, 2, 1)
        self.encoder_layer2 = layers.Encoder(channel_multiplier, channel_multiplier * 2, 3, 2, 1)
        self.encoder_layer3 = layers.Encoder(channel_multiplier * 2, channel_multiplier * 4, 3, 2, 1)
        self.encoder_layer4 = layers.Encoder(channel_multiplier * 4, channel_multiplier * 8, 3, 2, 1)

        # Depending on the network architecture, an additional encoder layer and a specific ASPP module are initialized.
        if self.network_architecture == 129605:
            self.encoder_layer5 = layers.Encoder(channel_multiplier * 8, channel_multiplier * 16, 3, 2, 1)
            self.aspp_module = layers.ASPPModule(network_architecture, channel_multiplier * 16, channel_multiplier * 32, dilations)
            self.decoder_layer5 = layers.Decoder(channel_multiplier * (16 + 32), channel_multiplier * 16, 3, 1, 1)
        else:
            self.aspp_module = layers.ASPPModule(network_architecture, channel_multiplier * 8, channel_multiplier * 16, dilations)

        # Decoder layers progressively decrease the number of channels while increasing spatial dimensions.
        self.decoder_layer4 = layers.Decoder(channel_multiplier * (8 + 16), channel_multiplier * 8, 3, 1, 1)
        self.decoder_layer3 = layers.Decoder(channel_multiplier * (4 + 8), channel_multiplier * 4, 3, 1, 1)
        self.decoder_layer2 = layers.Decoder(channel_multiplier * (2 + 4), channel_multiplier * 2, 3, 1, 1)
        self.decoder_layer1 = layers.Decoder(channel_multiplier * (1 + 2), channel_multiplier, 3, 1, 1)

    def __call__(self, input_tensor):
        # The input tensor is passed through a series of encoder layers.
        hidden_state, encoder_output1 = self.encoder_layer1(input_tensor)
        hidden_state, encoder_output2 = self.encoder_layer2(hidden_state)
        hidden_state, encoder_output3 = self.encoder_layer3(hidden_state)
        hidden_state, encoder_output4 = self.encoder_layer4(hidden_state)

        # Depending on the network architecture, the hidden state is processed by an additional encoder layer and the ASPP module.
        if self.network_architecture == 129605:
            hidden_state, encoder_output5 = self.encoder_layer5(hidden_state)
            hidden_state = self.aspp_module(hidden_state)
            # The decoder layers use skip connections from the encoder layers for better feature integration.
            hidden_state = self.decoder_layer5(hidden_state, encoder_output5)
        else:
            hidden_state = self.aspp_module(hidden_state)

        # The hidden state is further processed by the decoder layers, using skip connections for feature integration.
        hidden_state = self.decoder_layer4(hidden_state, encoder_output4)
        hidden_state = self.decoder_layer3(hidden_state, encoder_output3)
        hidden_state = self.decoder_layer2(hidden_state, encoder_output2)
        hidden_state = self.decoder_layer1(hidden_state, encoder_output1)

        return hidden_state


# The determine_model_capacity function is designed to select the appropriate model configuration
# based on the frequency bins and network architecture. It maps specific architectures to predefined
# model capacities, which dictate the structure and parameters of the CascadedASPPNet model.
def determine_model_capacity(frequency_bins, network_architecture):
    # Predefined model architectures categorized by their precision level.
    single_precision_model_architectures = [31191, 33966, 129605]
    high_precision_model_architectures = [123821, 123812]
    high_precision_2_model_architectures = [537238, 537227]

    # Mapping network architectures to their corresponding model capacity data.
    if network_architecture in single_precision_model_architectures:
        model_capacity_data = [(2, 16), (2, 16), (18, 8, 1, 1, 0), (8, 16), (34, 16, 1, 1, 0), (16, 32), (32, 2, 1), (16, 2, 1), (16, 2, 1)]

    if network_architecture in high_precision_model_architectures:
        model_capacity_data = [(2, 32), (2, 32), (34, 16, 1, 1, 0), (16, 32), (66, 32, 1, 1, 0), (32, 64), (64, 2, 1), (32, 2, 1), (32, 2, 1)]

    if network_architecture in high_precision_2_model_architectures:
        model_capacity_data = [(2, 64), (2, 64), (66, 32, 1, 1, 0), (32, 64), (130, 64, 1, 1, 0), (64, 128), (128, 2, 1), (64, 2, 1), (64, 2, 1)]

    # Initializing the CascadedASPPNet model with the selected model capacity data.
    cascaded_network = CascadedASPPNet
    model = cascaded_network(frequency_bins, model_capacity_data, network_architecture)

    return model


class CascadedASPPNet(nn.Module):
    """
    CascadedASPPNet Class:
    This class implements a cascaded version of the ASPP network, designed for processing audio signals
    for tasks such as vocal removal. It consists of multiple stages, each with its own ASPP network,
    to process different frequency bands of the input signal. This allows the model to effectively
    handle the full spectrum of audio frequencies by focusing on different frequency bands separately.
    """

    def __init__(self, fft_bins, model_capacity_data, network_architecture):
        super(CascadedASPPNet, self).__init__()
        # The first stage processes the low and high frequency bands separately.
        self.stage1_low_band_network = BaseASPPNet(network_architecture, *model_capacity_data[0])
        self.stage1_high_band_network = BaseASPPNet(network_architecture, *model_capacity_data[1])

        # Bridge layers connect different stages of the network.
        self.stage2_bridge = layers.Conv2DBNActiv(*model_capacity_data[2])
        self.stage2_full_band_network = BaseASPPNet(network_architecture, *model_capacity_data[3])

        self.stage3_bridge = layers.Conv2DBNActiv(*model_capacity_data[4])
        self.stage3_full_band_network = BaseASPPNet(network_architecture, *model_capacity_data[5])

        # Output layers for the final mask prediction and auxiliary outputs.
        self.output_layer = nn.Conv2d(*model_capacity_data[6], bias=False)
        self.auxiliary_output1 = nn.Conv2d(*model_capacity_data[7], bias=False)
        self.auxiliary_output2 = nn.Conv2d(*model_capacity_data[8], bias=False)

        # Parameters for handling the frequency bins of the input signal.
        self.maximum_frequency_bin = fft_bins // 2
        self.output_frequency_bin = fft_bins // 2 + 1

        self.padding_offset = 128

    def forward(self, input_tensor):
        # The forward pass processes the input tensor through each stage of the network,
        # combining the outputs of different frequency bands and stages to produce the final mask.
        mixed_signal = input_tensor.detach()
        input_tensor_clone = input_tensor.clone()

        # Preparing the input tensor by selecting the maximum frequency bin.
        input_tensor_clone = input_tensor_clone[:, :, : self.maximum_frequency_bin]

        # Processing the low and high frequency bands separately in the first stage.
        bandwidth = input_tensor_clone.size()[2] // 2
        auxiliary_output1 = torch.cat([self.stage1_low_band_network(input_tensor_clone[:, :, :bandwidth]), self.stage1_high_band_network(input_tensor_clone[:, :, bandwidth:])], dim=2)

        # Combining the outputs of the first stage and passing through the second stage.
        hidden_state = torch.cat([input_tensor_clone, auxiliary_output1], dim=1)
        auxiliary_output2 = self.stage2_full_band_network(self.stage2_bridge(hidden_state))

        # Further processing the combined outputs through the third stage.
        hidden_state = torch.cat([input_tensor_clone, auxiliary_output1, auxiliary_output2], dim=1)
        hidden_state = self.stage3_full_band_network(self.stage3_bridge(hidden_state))

        # Applying the final output layer to produce the mask.
        mask = torch.sigmoid(self.output_layer(hidden_state))
        # Padding the mask to match the output frequency bin size.
        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_frequency_bin - mask.size()[2]), mode="replicate")

        # During training, auxiliary outputs are also produced and padded accordingly.
        if self.training:
            auxiliary_output1 = torch.sigmoid(self.auxiliary_output1(auxiliary_output1))
            auxiliary_output1 = F.pad(input=auxiliary_output1, pad=(0, 0, 0, self.output_frequency_bin - auxiliary_output1.size()[2]), mode="replicate")
            auxiliary_output2 = torch.sigmoid(self.auxiliary_output2(auxiliary_output2))
            auxiliary_output2 = F.pad(input=auxiliary_output2, pad=(0, 0, 0, self.output_frequency_bin - auxiliary_output2.size()[2]), mode="replicate")
            return mask * mixed_signal, auxiliary_output1 * mixed_signal, auxiliary_output2 * mixed_signal
        else:
            return mask  # * mixed_signal

    def predict_mask(self, input_tensor):
        # This method predicts the mask for the input tensor by calling the forward method
        # and applying any necessary padding adjustments.
        mask = self.forward(input_tensor)

        # Adjusting the mask by removing padding offsets if present.
        if self.padding_offset > 0:
            mask = mask[:, :, :, self.padding_offset : -self.padding_offset]

        return mask
