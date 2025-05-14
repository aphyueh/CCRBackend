# model/model.py
import tensorflow as tf

@tf.keras.saving.register_keras_serializable(package="trainer.model")
class ColorCastRemoval(tf.keras.Model):
    """
    Minimal version of ColorCastRemoval class for model loading.
    This class contains just enough structure to deserialize the saved model.
    """
    def __init__(self, decomnet_layer_num=5, **kwargs):
        super(ColorCastRemoval, self).__init__(**kwargs)
        self.DecomNet_layer_num = decomnet_layer_num
        
        # We don't need to recreate all the layers since they'll be loaded from the saved model
        # This is just for the class structure to be recognized
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "decomnet_layer_num": self.DecomNet_layer_num,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=False):
        # This is a placeholder that will be replaced by the loaded weights
        # The actual implementation comes from the saved model
        return None