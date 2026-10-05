import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models


def resblock(x, filters, kernel_size, dilation_rate=1, first_conv_activation='relu'):
    """
    Residual block with two convolutional layers and a skip connection.

    Parameters
    ----------
    x : tensorflow.Tensor
        Input tensor.
    filters : int
        Number of convolutional filters.
    kernel_size : int
        Size of the convolutional kernels.
    dilation_rate : int, optional
        Dilation rate of the convolutional layers.
    first_conv_activation : str, optional
        Activation function before the first convolution.

    Returns
    -------
    tensorflow.Tensor
        Output tensor.

    """
    conv_x = layers.BatchNormalization()(x)
    conv_x = layers.Activation(first_conv_activation)(conv_x)
    conv_x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        padding='same',
        activation='linear',
        dilation_rate=dilation_rate,
        kernel_initializer='glorot_normal',
    )(conv_x)

    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.ReLU()(conv_x)
    conv_x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        padding='same',
        activation='linear',
        dilation_rate=dilation_rate,
        kernel_initializer='glorot_normal',
    )(conv_x)

    return layers.add([conv_x, x])


def resgroup(x, n_blocks, filters, kernel_size, dilation_rate=1, first_conv_activation='relu'):
    """
    Group of consecutive residual blocks with the same parameters.

    Parameters
    ----------
    x : tensorflow.Tensor
        Input tensor.
    n_blocks : int
        Number of residual blocks.
    filters, kernel_size, dilation_rate, first_conv_activation
        Parameters of each residual block.

    Returns
    -------
    tensorflow.Tensor
        Output tensor.

    """
    for _ in range(n_blocks):
        x = resblock(x, filters, kernel_size, dilation_rate, first_conv_activation)
    return x


def make_resnet(
        input_seq_length,
        groups=4,
        blocks_per_group=3,
        filters=128,
        kernel_size=13,
        dilation_rates=[1, 2, 4, 8],
        first_conv_activation='relu',
        n_outputs=8,
        output_activation='linear',
    ):
    """
    Residual neural network with a one-hot encoded DNA sequence input.

    The output of each residual group, and of the initial convolution, is
    projected with a 1x1 convolution and summed to the final convolution
    output, then averaged across the sequence and passed to a dense layer.

    Parameters
    ----------
    input_seq_length : int
        Length of the input sequence.
    groups : int, optional
        Number of residual groups.
    blocks_per_group : int, optional
        Number of residual blocks in each group.
    filters : int, optional
        Number of convolutional filters.
    kernel_size : int, optional
        Size of the convolutional kernels.
    dilation_rates : list of int, optional
        Dilation rate of each residual group.
    first_conv_activation : str, optional
        Activation before the first convolution of the first residual group.
    n_outputs : int, optional
        Number of outputs.
    output_activation : str, optional
        Activation of the output layer.

    Returns
    -------
    tensorflow.keras.Model
        Keras model.

    """
    model_input = layers.Input(shape=(input_seq_length, 4))

    # 1x1 convolution to get to the number of filters
    x = layers.Conv1D(
        filters,
        kernel_size=1,
        padding='same',
        activation='linear',
        kernel_initializer='glorot_normal',
    )(model_input)
    skip_convs = [layers.Conv1D(filters, kernel_size=1, padding='same')(x)]

    # Residual groups
    for group_idx in range(groups):
        x = resgroup(
            x,
            blocks_per_group,
            filters,
            kernel_size,
            dilation_rate=dilation_rates[group_idx],
            first_conv_activation=first_conv_activation if group_idx == 0 else 'relu',
        )
        skip_convs.append(
            layers.Conv1D(filters, kernel_size=1, padding='same', kernel_initializer='glorot_normal')(x)
        )

    # Final convolution summed with skip connections
    x = layers.Conv1D(filters, kernel_size=1, padding='same', kernel_initializer='glorot_normal')(x)
    for skip_conv in skip_convs:
        x = layers.add([x, skip_conv])

    # Average across sequence, then output layer
    x = layers.GlobalAvgPool1D()(x)
    model_output = layers.Dense(n_outputs, activation=output_activation)(x)

    return models.Model(model_input, model_output)


def load_model(model_path):
    """
    Load a trained model without its training configuration.

    Parameters
    ----------
    model_path : str or pathlib.Path
        Path to the saved Keras model.

    Returns
    -------
    tensorflow.keras.Model
        Loaded model.

    """
    return tensorflow.keras.models.load_model(model_path, compile=False)


class MSE_Cosine_Loss(tensorflow.keras.losses.Loss):
    """
    Weighted sum of mean squared error and negative cosine similarity.

    Cosine similarity is calculated across outputs after subtracting each
    sample's mean, and is therefore equivalent to the Pearson correlation.

    Parameters
    ----------
    w_mse, w_cosine : float
        Weights of the mean squared error and cosine similarity terms.

    """
    def __init__(self, w_mse, w_cosine):
        super().__init__()
        self.w_mse = w_mse
        self.w_cosine = w_cosine

    def call(self, y_true, y_pred):
        mse_loss = tensorflow.keras.losses.mean_squared_error(y_true, y_pred)
        ym_true = tensorflow.math.reduce_mean(y_true, axis=1, keepdims=True)
        ym_pred = tensorflow.math.reduce_mean(y_pred, axis=1, keepdims=True)
        # Keras cosine_similarity returns the negative similarity
        cosine_loss = tensorflow.keras.losses.cosine_similarity(y_true - ym_true, y_pred - ym_pred, axis=1)
        return self.w_mse * mse_loss + self.w_cosine * cosine_loss
