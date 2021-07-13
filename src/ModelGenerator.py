import tensorflow as tf
import numpy as np

from tensorflow import keras
from DataGenerator import DataGenerator

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

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.metrics import categorical_accuracy


class ModelGenerator(object):
    def __init__(self, imageSize: int = 100, greyScale: bool = True, sentenceLength: int = 100, dictionaryLength: int = 200):
        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = sentenceLength
        self.dictionaryLength = dictionaryLength

    def getModel(self):
        # Construct the layers, inputs, and outputs
        inputImage = Input(self.imageSize)
        inputSentence = Input((self.sentenceLength, ))
        embeddedSentence = Embedding(self.dictionaryLength, 100)(inputSentence)
        embeddedSentence = Flatten()(embeddedSentence)
        embeddedSentence = Reshape((self.imageSize[0], self.imageSize[1], 1))(embeddedSentence)
        convolvedImage = Conv2D(20, 1, activation="relu")(inputImage)
        concatenated = Concatenate(axis=-1)([embeddedSentence, convolvedImage])
        outputImage = Conv2D(3, 1, activation="relu", name="imageReconstruction")(concatenated)

        # Construct the decoder model
        decoderModel = Sequential(name="sentenceReconstruction")
        decoderModel.add(Conv2D(1, 1, input_shape=self.imageSize))
        decoderModel.add(Reshape(self.sentenceLength, 100))
        decoderModel.add(TimeDistributed(Dense(self.dictionaryLength, activation="softmax")))
        outputSentence = decoderModel(outputImage)

        # Create the encoder model
        model = Model(inputs=[inputImage, inputSentence], outputs=[outputImage, outputSentence])
        model.compile(
            optimizer="adam",
            loss=[mean_absolute_error, categorical_crossentropy],
            metrics={"Sentence Reconstruction": categorical_accuracy}
        )

        encoderModel = Model(inputs=[inputImage, inputSentence], outputs=[outputImage])

        return model, encoderModel, decoderModel