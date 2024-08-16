import numpy as np
from os import path
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout

# This is an abstract class. You need to implement your own model.
class BaseModelBuilder:
    
    def __init__(self, model_weights_path=None):
        self.model_weights_path = model_weights_path

    def get_model(self):
        """
        Returns a compiled model. If weights_path is provided and the file exists, 
        the model's weights are loaded from that file.
        """
        model = self.build_model()

        if self.model_weights_path and path.isfile(self.model_weights_path):
            try:
                model.load_weights(self.model_weights_path)
            except Exception as e:
                print(f"Error loading weights: {e}")

        return model

    # This method should be overridden by subclasses.
    def build_model(self):
        """
        Builds and returns a Keras model. This method must be overridden by subclasses.
        """
        raise NotImplementedError("You must implement your own model by overriding the 'build_model' method.")