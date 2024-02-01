import torch
from torch import nn
import torch.nn.functional as F

from ultimatevocalremover.lib_v5 import spec_utils


class Conv2DBNActiv(nn.Module):
    """
    Conv2DBNActiv Class:
    This class implements a convolutional layer followed by batch normalization and an activation function.
    It is a fundamental building block for constructing neural networks, especially useful in image and audio processing tasks.
    The class encapsulates the pattern of applying a convolution, normalizing the output, and then applying a non-linear activation.
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1, dilation=1, activation_function=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        # Sequential model combining Conv2D, BatchNorm, and activation function into a single module
        self.convolution = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(num_output_channels),
            activation_function(),
        )

    def __call__(self, input_tensor):
        # Forward pass through the sequential model
        return self.convolution(input_tensor)


class Encoder(nn.Module):
    """
    Encoder Class:
    This class defines an encoder module typically used in autoencoder architectures.
    It consists of two convolutional layers, each followed by batch normalization and an activation function.
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1, activation_function=nn.LeakyReLU):
        super(Encoder, self).__init__()
        # First convolutional layer of the encoder
        self.first_convolution = Conv2DBNActiv(num_input_channels, num_output_channels, kernel_size, stride, padding, activation_function=activation_function)
        # Second convolutional layer of the encoder
        self.second_convolution = Conv2DBNActiv(num_output_channels, num_output_channels, kernel_size, 1, padding, activation_function=activation_function)

    def __call__(self, input_tensor):
        # Applying the first and then the second convolutional layers
        hidden = self.first_convolution(input_tensor)
        hidden = self.second_convolution(hidden)
        return hidden


class Decoder(nn.Module):
    """
    Decoder Class:
    This class defines a decoder module, which is the counterpart of the Encoder class in autoencoder architectures.
    It applies a convolutional layer followed by batch normalization and an activation function, with an optional dropout layer for regularization.
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1, activation_function=nn.ReLU, include_dropout=False):
        super(Decoder, self).__init__()
        # Convolutional layer with optional dropout for regularization
        self.convolution_with_activation = nn.Sequential(
            Conv2DBNActiv(num_input_channels, num_output_channels, kernel_size, stride, padding, activation_function=activation_function), nn.Dropout2d(0.1) if include_dropout else nn.Identity()
        )

    def __call__(self, input_tensor):
        # Forward pass through the convolutional layer and optional dropout
        hidden = self.convolution_with_activation(input_tensor)
        return hidden


class ASPPModule(nn.Module):
    """
    ASPPModule Class:
    This class implements the Atrous Spatial Pyramid Pooling (ASPP) module, which is useful for semantic image segmentation tasks.
    It captures multi-scale contextual information by applying convolutions at multiple dilation rates.
    """

    def __init__(self, num_input_channels, num_output_channels, dilation_rates=(4, 8, 12), activation_function=nn.ReLU, include_dropout=False):
        super(ASPPModule, self).__init__()
        # Global context convolution captures the overall context
        self.global_context_convolution = nn.Sequential(nn.AdaptiveAvgPool2d((1, None)), Conv2DBNActiv(num_input_channels, num_output_channels, 1, 1, 0, activation_function=activation_function))
        # Local context convolution focuses on finer details
        self.local_context_convolution = Conv2DBNActiv(num_input_channels, num_output_channels, 1, 1, 0, activation_function=activation_function)
        # Multi-scale context convolutions at different dilation rates
        self.multi_scale_context_convolution_1 = Conv2DBNActiv(num_input_channels, num_output_channels, 3, 1, dilation_rates[0], dilation_rates[0], activation_function=activation_function)
        self.multi_scale_context_convolution_2 = Conv2DBNActiv(num_input_channels, num_output_channels, 3, 1, dilation_rates[1], dilation_rates[1], activation_function=activation_function)
        self.multi_scale_context_convolution_3 = Conv2DBNActiv(num_input_channels, num_output_channels, 3, 1, dilation_rates[2], dilation_rates[2], activation_function=activation_function)
        # Dimensionality reduction convolution combines features from all scales
        self.dimensionality_reduction_convolution = Conv2DBNActiv(num_output_channels * 5, num_output_channels, 1, 1, 0, activation_function=activation_function)
        # Optional dropout layer for regularization
        self.dropout_layer = nn.Dropout2d(0.1) if include_dropout else None

    def forward(self, input_tensor):
        _, _, height, width = input_tensor.size()
        # Upsample global context to match input size and combine with local and multi-scale features
        feature_1 = F.interpolate(self.global_context_convolution(input_tensor), size=(height, width), mode="bilinear", align_corners=True)
        feature_2 = self.local_context_convolution(input_tensor)
        feature_3 = self.multi_scale_context_convolution_1(input_tensor)
        feature_4 = self.multi_scale_context_convolution_2(input_tensor)
        feature_5 = self.multi_scale_context_convolution_3(input_tensor)
        concatenated_features = torch.cat((feature_1, feature_2, feature_3, feature_4, feature_5), dim=1)
        output = self.dimensionality_reduction_convolution(concatenated_features)

        if self.dropout_layer is not None:
            output = self.dropout_layer(output)

        return output


class LSTMModule(nn.Module):
    """
    LSTMModule Class:
    This class defines a module that combines convolutional feature extraction with a bidirectional LSTM for sequence modeling.
    It is useful for tasks that require understanding temporal dynamics in data, such as speech and audio processing.
    """

    def __init__(self, num_input_channels_convolution, num_input_features_lstm, num_output_features_lstm):
        super(LSTMModule, self).__init__()
        # Convolutional layer for initial feature extraction
        self.feature_extraction_convolution = Conv2DBNActiv(num_input_channels_convolution, 1, 1, 1, 0)
        # Bidirectional LSTM for capturing temporal dynamics
        self.bidirectional_lstm = nn.LSTM(input_size=num_input_features_lstm, hidden_size=num_output_features_lstm // 2, bidirectional=True)
        # Dense layer for output dimensionality matching
        self.dense_layer = nn.Sequential(nn.Linear(num_output_features_lstm, num_input_features_lstm), nn.BatchNorm1d(num_input_features_lstm), nn.ReLU())

    def forward(self, input_tensor):
        batch_size, _, num_bins, num_frames = input_tensor.size()
        # Extract features and prepare for LSTM
        hidden = self.feature_extraction_convolution(input_tensor)[:, 0]
        hidden = hidden.permute(2, 0, 1)
        hidden, _ = self.bidirectional_lstm(hidden)
        # Apply dense layer and reshape to match expected output format
        hidden = self.dense_layer(hidden.reshape(-1, hidden.size()[-1]))
        hidden = hidden.reshape(num_frames, batch_size, 1, num_bins)
        hidden = hidden.permute(1, 2, 3, 0)

        return hidden
