import tensorflow as tf
import numpy as np

from tensorflow import keras

from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed


from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.metrics import categorical_accuracy


class ModelGenerator(object):
    def __init__(self,
                 imageSize: int = 100,
                 greyScale: bool = True,
                 dictionaryLength: int = 200):
        """
        This class is responsible for generating a neural net model capable of
        embedding text information within images and recovering the original text

        :param imageSize: Size of the images the neural net is to be trained on (will be square images)
        :param greyScale: Whether or not the images will be greyscale
        :param dictionaryLength: Length of the dictionary of characters the neural net will train on
        """
        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = imageSize
        self.dictionaryLength = dictionaryLength

    def getModel(self) -> tuple:
        """
        This method is responsible for creating the neural net model, as well as the
        encryptor and decryptor segments of the model.

        :return: Full model, encoder model, decoder model
        """
        # Construct the layers, inputs, and outputs
        inputImage = Input(self.imageSize)
        inputSentence = Input((self.sentenceLength, ))
        embeddedSentence = Embedding(input_dim=self.dictionaryLength, output_dim=self.sentenceLength)(inputSentence)
        embeddedSentence = Flatten()(embeddedSentence)
        embeddedSentence = Reshape(target_shape=(self.imageSize[0], self.imageSize[1], 1))(embeddedSentence)
        convolvedImage = Conv2D(20, 1, activation="relu")(inputImage)
        concatenated = Concatenate(axis=-1)([embeddedSentence, convolvedImage])
        outputImage = Conv2D(3, 1, activation="relu", name="imageReconstruction")(concatenated)

        # Construct the decoder model
        decoderModel = Sequential(name="sentenceReconstruction")
        decoderModel.add(Conv2D(1, 1, input_shape=self.imageSize))
        decoderModel.add(Reshape((self.sentenceLength, self.sentenceLength)))
        decoderModel.add(TimeDistributed(Dense(self.dictionaryLength, activation="softmax")))
        outputSentence = decoderModel(outputImage)

        # Construct the encoder model
        model = Model(inputs=[inputImage, inputSentence], outputs=[outputImage, outputSentence])
        model.compile(
            optimizer="adam",
            loss=[mean_absolute_error, categorical_crossentropy],
            metrics={"sentenceReconstruction": categorical_accuracy}
        )

        encoderModel = Model(inputs=[inputImage, inputSentence], outputs=[outputImage])

        return model, encoderModel, decoderModel


if __name__ == "__main__":
    pass