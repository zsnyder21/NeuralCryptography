import pickle

from ModelGenerator import ModelGenerator
from DataGenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback


class ModelTrainer(object):
    def __init__(self,
                 modelSavePath: str,
                 imageSize: int = 100,
                 greyScale: bool = True,
                 dictionaryLength: int = 200,
                 batchSize: int = 32,
                 loadExistingModel: bool = False):
        """
        This class is responsible for training a model to encrypt/decrypt
        string information within an image. This method will save the parameters
        expected by CryptoNet in a file named identically to modelSavePath (with .p extension).

        :param modelSavePath: Location to save the model weights to
        :param imageSize: Size of images to train on (images will be square)
        :param greyScale: Whether or not the images will be greyscale
        :param dictionaryLength: Number of distinct characters to use within sentences
        :param batchSize: Number of images used per batch when training
        :param loadExistingModel: Whether or not to load an existing model
        """
        self.modelSavePath = modelSavePath
        self.loadExistingModel = loadExistingModel

        if self.loadExistingModel:
            file = open(f"{self.modelSavePath}.p", "rb")
            modelParameters = pickle.load(file=file)
            file.close()

            self.imageSize = modelParameters["imageSize"]
            self.greyScale = modelParameters["greyScale"]
            self.sentenceLength = modelParameters["sentenceLength"]
            self.dictionaryLength = modelParameters["dictionaryLength"]
            self.batchSize = modelParameters["batchSize"]

            if self.greyScale:
                self.imageSize = (self.imageSize, self.imageSize, 1)
            else:
                self.imageSize = (self.imageSize, self.imageSize, 3)

        else:
            if greyScale:
                self.imageSize = (imageSize, imageSize, 1)
            else:
                self.imageSize = (imageSize, imageSize, 3)

            self.greyScale = greyScale
            self.sentenceLength = imageSize
            self.dictionaryLength = dictionaryLength
            self.batchSize = batchSize

            modelParameters = {
                "imageSize": imageSize,
                "greyScale": greyScale,
                "sentenceLength": imageSize,
                "dictionaryLength": dictionaryLength,
                "batchSize": batchSize
            }

            file = open(f"{modelSavePath}.p", "wb")
            pickle.dump(obj=modelParameters, file=file)
            file.close()

        self.modelGenerator = ModelGenerator(
            imageSize=self.imageSize[0],
            greyScale=self.greyScale,
            dictionaryLength=self.dictionaryLength
        )

        self.dataGenerator = DataGenerator(
            imageSize=self.imageSize[0],
            greyScale=self.greyScale,
            dictionaryLength=self.dictionaryLength
        )

        self.model = None,
        self.encoder = None,
        self.decoder = None

    def trainModels(self, epochs: int, stepsPerEpoch: int, verbose: int = 1, threshold: float = 0.01) -> None:
        """
        This method is responsible for training the models

        :param epochs: Number of epochs to use when training
        :param stepsPerEpoch: Number of steps per epoch
        :param verbose: Whether or not to be verbose while training
        :return:
        """
        dataGenerator = self.dataGenerator.generateData(batchSize=self.batchSize)
        self.model, self.encoder, self.decoder = self.modelGenerator.getModel()

        if self.loadExistingModel:
            self.model.load_weights(filepath=self.modelSavePath)

        self.model.fit(
            x=dataGenerator,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            callbacks=[
                ModelCheckpoint(
                    filepath=self.modelSavePath,
                    monitor="imageReconstruction_loss",
                    verbose=verbose,
                    save_weights_only=True,
                    save_best_only=True
                ),
                EarlyStoppingThreshold(
                    monitor="imageReconstruction_loss",
                    mode="min",
                    threshold=threshold
                )
            ]
        )


class EarlyStoppingThreshold(Callback):
    def __init__(self, monitor: str, mode: str, threshold: float,):
        """
        This class is a custom implementation of a Keras callback
        intended to stop training the model when a certain threshold
        metric is attained

        :param monitor: The name of the metric to monitor
        :param mode: Either 'min' or 'max' depending whether the metric should increase or decrease
        :param threshold: The threshold value for which to stop training
        """
        super(EarlyStoppingThreshold, self).__init__()

        if mode.lower() not in {"min", "max"}:
            raise ValueError("Parameter mode must be either 'min' or 'max'.")

        self.monitor = monitor
        self.mode = mode.lower()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        """
        Determine whether or not we can stop training the model

        :param epoch: Epoch we are on
        :param logs: The logs
        """
        metric = logs[self.monitor]

        if self.mode == "min":
            if metric < self.threshold:
                self.model.stop_training = True
        else:
            if metric > self.threshold:
                self.model.stop_training = True


if __name__ == "__main__":
    trainer = ModelTrainer(
        modelSavePath="../data/ModelWeights/alternativeModel.h5",
        imageSize=2000,
        greyScale=False,
        dictionaryLength=1000,
        batchSize=4,
        loadExistingModel=False
    )

    trainer.trainModels(
        epochs=512,
        stepsPerEpoch=64,
        threshold=0.2
    )