import torch
from torch import nn
import torch.nn.functional as F

from ultimatevocalremover.lib_v5 import spec_utils


class Conv2DBNActiv(nn.Module):
    """
    The Conv2DBNActiv class defines a custom convolutional layer with batch normalization and activation.
    It is designed to be a modular component for building neural networks, particularly for image and audio processing tasks.
    This layer encapsulates a common pattern in deep learning models, which is a convolution followed by batch normalization and a non-linear activation function.
    """

    # The constructor takes several parameters to configure the convolutional layer.
    # nin: Number of input channels.
    # nout: Number of output channels.
    # ksize: Kernel size for the convolution. Defaults to 3 for a 3x3 kernel.
    # stride: Stride of the convolution. Defaults to 1.
    # pad: Padding added to all four sides of the input. Defaults to 1.
    # dilation: Spacing between kernel elements. Defaults to 1.
    # activ: The activation function to use. Defaults to nn.ReLU.
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        # The convolutional layer is defined here without bias because batch normalization will be applied.
        # Batch normalization makes the bias term redundant.
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin,
                nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                nout
            ),  # Batch normalization is applied to the output channels.
            activ(),  # The activation function is applied last.
        )

    # The forward method defines how the data flows through this layer.
    # x: The input tensor.
    def __call__(self, x):
        return self.conv(
            x
        )  # The input is passed through the convolution, batch normalization, and activation in sequence.


class Encoder(nn.Module):
    """
    The Encoder class is a custom module that represents an encoder in an encoder-decoder architecture.
    It is designed to extract features from the input data through a series of convolutional layers.
    The encoder is typically used to downsample the input data and extract a compact representation that captures the most important features.
    """

    # The constructor takes several parameters to configure the convolutional layers:
    # nin: Number of input channels.
    # nout: Number of output channels.
    # ksize: Kernel size for the convolution. Defaults to 3 for a 3x3 kernel.
    # stride: Stride of the convolution. Defaults to 1.
    # pad: Padding added to all four sides of the input. Defaults to 1.
    # activ: The activation function to use. Defaults to nn.LeakyReLU.
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(Encoder, self).__init__()
        # The first convolutional layer takes the input data and applies a convolution operation.
        # The number of output channels is specified by nout.
        # The kernel size, stride, padding, and activation function are configurable.
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        # The second convolutional layer takes the output of the first layer and applies another convolution operation.
        # The number of input and output channels is the same (nout).
        # The kernel size, stride, padding, and activation function are configurable.
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    # The __call__ method defines how the data flows through the encoder.
    # x: The input tensor.
    def __call__(self, x):
        # The input data is passed through the first convolutional layer.
        h = self.conv1(x)
        # The output of the first layer is passed through the second convolutional layer.
        h = self.conv2(h)
        # The output of the second layer is returned as the final output of the encoder.
        return h


class Decoder(nn.Module):
    """
    The Decoder class is designed to perform the inverse operation of an encoder in an encoder-decoder architecture.
    It aims to reconstruct or generate data from the encoded representations. This class is typically used in tasks
    such as image segmentation, where the goal is to generate pixel-wise labels from the encoded features, or in
    generative models where the goal is to generate data samples from latent representations.

    Attributes:
        conv1 (nn.Module): A convolutional layer with batch normalization and activation.
        dropout (nn.Module, optional): A dropout layer for regularization, applied to the output of conv1.
    """

    def __init__(
        self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False
    ):
        """
        Initializes the Decoder module.

        Parameters:
            nin (int): Number of input channels.
            nout (int): Number of output channels.
            ksize (int, optional): Kernel size for the convolutional layers. Defaults to 3.
            stride (int, optional): Stride for the convolutional layers. Defaults to 1.
            pad (int, optional): Padding for the convolutional layers. Defaults to 1.
            activ (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            dropout (bool, optional): Whether to include a dropout layer. Defaults to False.
        """
        super(Decoder, self).__init__()
        # Convolutional layer with batch normalization and activation function.
        # This layer will process the input tensor and apply a series of transformations.
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)

        # Optional dropout layer for regularization, helps prevent overfitting.
        # Only included if dropout is set to True.
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        """
        Defines the forward pass of the Decoder.

        Parameters:
            x (Tensor): The input tensor to the Decoder.
            skip (Tensor, optional): An optional skip connection tensor that can be concatenated with the input tensor.
                                     This is typically used in U-Net architectures to provide local information to the
                                     global context captured by the encoder.

        Returns:
            Tensor: The output tensor after processing through the Decoder.
        """
        # Upsample the input tensor to a higher resolution using bilinear interpolation.
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # If a skip connection tensor is provided, it is first cropped to match the size of 'x' and then concatenated.
        # This enriches 'x' with features from earlier in the network, which can help in reconstructing finer details.
        if skip is not None:
            skip = spec_utils.crop_center(
                skip, x
            )  # Crop 'skip' to match the size of 'x'.
            x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension.

        # Pass the tensor through the first convolutional layer.
        h = self.conv1(x)

        # If dropout is enabled, apply it to the output of the convolutional layer.
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class ASPPModule(nn.Module):
    """
    ASPPModule is a class that implements the Atrous Spatial Pyramid Pooling (ASPP) module.
    ASPP is a technique used in semantic segmentation tasks to capture multi-scale context by applying
    parallel dilated convolutions with different dilation rates to the input.
    This class is a PyTorch module, and can be integrated into any PyTorch model.
    """

    # The constructor for the ASPPModule class.
    # It initializes the five convolutional layers used in the ASPP module, as well as an optional dropout layer.
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()
        # The first convolutional layer applies an adaptive average pooling operation to the input,
        # followed by a 1x1 convolution. This is used to capture the global context of the input.
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),
        )
        # The second convolutional layer applies a 1x1 convolution to the input. This is used to capture the local context.
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        # The next three convolutional layers apply 3x3 convolutions with different dilation rates.
        # These are used to capture multi-scale context.
        self.conv3 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[0], dilations[0], activ=activ
        )
        self.conv4 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[1], dilations[1], activ=activ
        )
        self.conv5 = Conv2DBNActiv(
            nin, nout, 3, 1, dilations[2], dilations[2], activ=activ
        )
        # The bottleneck layer applies a 1x1 convolution to the concatenated output of the previous layers.
        # This is used to reduce the dimensionality of the output and to integrate the multi-scale context.
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        # Optional dropout layer for regularization, helps prevent overfitting.
        # Only included if dropout is set to True.
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    # The forward method defines the forward pass of the ASPPModule.
    # It applies the five convolutional layers to the input, concatenates the outputs,
    # passes the result through the bottleneck layer, and applies dropout if enabled.
    def forward(self, x):
        _, _, h, w = x.size()
        # Apply the first convolutional layer and upsample the output to match the input size.
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        # Apply the next four convolutional layers to the input.
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        # Concatenate the outputs of the five convolutional layers along the channel dimension.
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        # Pass the concatenated output through the bottleneck layer.
        out = self.bottleneck(out)

        # If dropout is enabled, apply it to the output of the bottleneck layer.
        if self.dropout is not None:
            out = self.dropout(out)

        # Return the final output tensor.
        return out

    class LSTMModule(nn.Module):
        """
        LSTMModule is a class that defines a custom module for a Long Short-Term Memory (LSTM) network.
        LSTM networks are a type of recurrent neural network (RNN) that are capable of learning long-term dependencies.
        This is particularly useful for tasks that involve sequential data, such as time series analysis or natural language processing.
        """

        # The constructor for the LSTMModule class.
        # It initializes the convolutional layer, LSTM layer, and dense layer.
        # nin_conv: Number of input channels for the convolutional layer.
        # nin_lstm: Number of input features for the LSTM layer.
        # nout_lstm: Number of output features for the LSTM layer.
        def __init__(self, nin_conv, nin_lstm, nout_lstm):
            super(LSTMModule, self).__init__()
            # The convolutional layer is used to extract features from the input data.
            self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
            # The LSTM layer is used to process the features extracted by the convolutional layer.
            # It is bidirectional, meaning that it processes the data in both forward and backward directions.
            self.lstm = nn.LSTM(
                input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True
            )
            # The dense layer is used to map the output of the LSTM layer to the desired output size.
            # It includes batch normalization and a ReLU activation function.
            self.dense = nn.Sequential(
                nn.Linear(nout_lstm, nin_lstm), nn.BatchNorm1d(nin_lstm), nn.ReLU()
            )

        # The forward method defines the forward pass of the LSTMModule.
        # It applies the convolutional layer, LSTM layer, and dense layer to the input in sequence.
        def forward(self, x):
            # Get the size of the input tensor.
            N, _, nbins, nframes = x.size()
            # Apply the convolutional layer to the input tensor.
            h = self.conv(x)[:, 0]  # N, nbins, nframes
            # Permute the dimensions of the tensor for the LSTM layer.
            h = h.permute(2, 0, 1)  # nframes, N, nbins
            # Apply the LSTM layer to the tensor.
            h, _ = self.lstm(h)
            # Reshape the tensor and apply the dense layer.
            h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
            # Reshape the tensor to its original shape.
            h = h.reshape(nframes, N, 1, nbins)
            # Permute the dimensions of the tensor to match the original input tensor.
            h = h.permute(1, 2, 3, 0)

            # Return the final output tensor.
            return h
