from model_builder import BaseModelBuilder
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, LeakyReLU, concatenate

class MarketPolicyGradientModelBuilder(BaseModelBuilder):

    def build_model(self):
        """
        Builds a Convolutional Neural Network model for policy gradient in market environments.
        """
        # Input for basic features
        basic_input = Input(shape=(3,))
        basic_dense = Dense(5, activation="relu")(basic_input)

        inputs = [basic_input]
        merged_layers = [basic_dense]

        for i in range(1):
            # Input for sequential data (e.g., market history)
            seq_input = Input(shape=[2, 60, 1])
            inputs.append(seq_input)

            conv1 = Conv2D(2048, (3, 1), padding='valid')(seq_input)
            conv1 = LeakyReLU(0.001)(conv1)

            conv2 = Conv2D(2048, (5, 1), padding='valid')(seq_input)
            conv2 = LeakyReLU(0.001)(conv2)

            conv3 = Conv2D(2048, (10, 1), padding='valid')(seq_input)
            conv3 = LeakyReLU(0.001)(conv3)

            conv4 = Conv2D(2048, (20, 1), padding='valid')(seq_input)
            conv4 = LeakyReLU(0.001)(conv4)

            conv5 = Conv2D(2048, (40, 1), padding='valid')(seq_input)
            conv5 = LeakyReLU(0.001)(conv5)

            # Flatten and merge convolutional layers
            flattened = Flatten()(conv5)
            dense_layer = Dense(512)(flattened)
            dense_layer = LeakyReLU(0.001)(dense_layer)
            merged_layers.append(dense_layer)

            # Additional convolutional processing
            conv_final = Conv2D(2048, (60, 1), padding='valid')(seq_input)
            conv_final = LeakyReLU(0.001)(conv_final)

            flattened_final = Flatten()(conv_final)
            dense_final = Dense(512)(flattened_final)
            dense_final = LeakyReLU(0.001)(dense_final)
            merged_layers.append(dense_final)

        # Merge all layers and finalize the model
        merged_output = concatenate(merged_layers, axis=1)
        dense1 = Dense(1024)(merged_output)
        dense1 = LeakyReLU(0.001)(dense1)
        dense2 = Dense(512)(dense1)
        dense2 = LeakyReLU(0.001)(dense2)
        dense3 = Dense(256)(dense2)
        dense3 = LeakyReLU(0.001)(dense3)
        output = Dense(2, activation='softmax')(dense3)

        model = Model(inputs=inputs, outputs=output)
        return model


class MarketModelBuilder(BaseModelBuilder):

    def build_model(self):
        """
        Builds a Convolutional Neural Network model for market prediction.
        """
        dropout_rate = 0.0

        # Input for basic features
        basic_input = Input(shape=(3,))
        basic_dense = Dense(5, activation="relu")(basic_input)

        inputs = [basic_input]
        merged_layers = [basic_dense]

        for i in range(1):
            # Input for sequential data (e.g., market history)
            seq_input = Input(shape=[2, 60, 1])  # Ensure this input height is at least as large as the largest filter height
            inputs.append(seq_input)

            conv1 = Conv2D(64, (2, 1), padding='same')(seq_input)
            conv1 = LeakyReLU(0.001)(conv1)

            conv2 = Conv2D(128, (2, 1), padding='same')(seq_input)
            conv2 = LeakyReLU(0.001)(conv2)

            conv3 = Conv2D(256, (2, 1), padding='same')(seq_input)
            conv3 = LeakyReLU(0.001)(conv3)

            conv4 = Conv2D(512, (2, 1), padding='same')(seq_input)
            conv4 = LeakyReLU(0.001)(conv4)

            conv5 = Conv2D(1024, (2, 1), padding='same')(seq_input)
            conv5 = LeakyReLU(0.001)(conv5)

            # Flatten and merge convolutional layers
            flattened = Flatten()(conv5)
            dense_layer = Dense(2048)(flattened)
            dense_layer = LeakyReLU(0.001)(dense_layer)
            dense_layer = Dropout(dropout_rate)(dense_layer)
            merged_layers.append(dense_layer)

            # Additional convolutional processing
            conv_final = Conv2D(2048, (2, 1), padding='same')(seq_input)
            conv_final = LeakyReLU(0.001)(conv_final)

            flattened_final = Flatten()(conv_final)
            dense_final = Dense(4096)(flattened_final)
            dense_final = LeakyReLU(0.001)(dense_final)
            dense_final = Dropout(dropout_rate)(dense_final)
            merged_layers.append(dense_final)

        # Merge all layers and finalize the model
        merged_output = concatenate(merged_layers, axis=1)
        dense1 = Dense(1024)(merged_output)
        dense1 = LeakyReLU(0.001)(dense1)
        dense1 = Dropout(dropout_rate)(dense1)
        dense2 = Dense(512)(dense1)
        dense2 = LeakyReLU(0.001)(dense2)
        dense2 = Dropout(dropout_rate)(dense2)
        dense3 = Dense(256)(dense2)
        dense3 = LeakyReLU(0.001)(dense3)
        dense3 = Dropout(dropout_rate)(dense3)
        output = Dense(2, activation='linear')(dense3)

        model = Model(inputs=inputs, outputs=output)
        return model