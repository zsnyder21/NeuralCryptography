import pickle

from ModelGenerator import ModelGenerator
from DataGenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


class ModelTrainer(object):
    def __init__(self,
                 modelSavePath: str,
                 imageSize: int = 100,
                 greyScale: bool = True,
                 dictionaryLength: int = 200,
                 batchSize: int = 32):
        """
        This class is responsible for training a model to encrypt/decrypt
        string information within an image. This method will save the parameters
        expected by CryptoNet in a file named identically to modelSavePath (with .p extension).

        :param modelSavePath: Location to save the model weights to
        :param imageSize: Size of images to train on (images will be square)
        :param greyScale: Whether or not the images will be greyscale
        :param dictionaryLength: Number of distinct characters to use within sentences
        :param batchSize: Number of images used per batch when training
        """
        self.modelSavePath = modelSavePath

        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = imageSize
        self.dictionaryLength = dictionaryLength
        self.batchSize = batchSize

        self.modelGenerator = ModelGenerator(
            imageSize=imageSize,
            greyScale=greyScale,
            dictionaryLength=dictionaryLength
        )

        self.dataGenerator = DataGenerator(
            imageSize=imageSize,
            greyScale=greyScale,
            dictionaryLength=dictionaryLength
        )

        self.model = None,
        self.encoder = None,
        self.decoder = None

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

    def trainModels(self, epochs: int, stepsPerEpoch: int, verbose: int = 1):
        """
        This method is responsible for training the models

        :param epochs: Number of epochs to use when training
        :param stepsPerEpoch: Number of steps per epoch
        :param verbose: Whether or not to be verbose while training
        :return:
        """
        dataGenerator = self.dataGenerator.generateData(batchSize=self.batchSize)
        self.model, self.encoder, self.decoder = self.modelGenerator.getModel()

        self.model.fit(
            x=dataGenerator,
            steps_per_epoch=stepsPerEpoch,
            epochs=epochs,
            callbacks=[
                ModelCheckpoint(
                    filepath=self.modelSavePath,
                    monitor="loss",
                    verbose=verbose,
                    save_weights_only=True,
                    save_best_only=True
                )
            ]
        )

if __name__ == "__main__":
    trainer = ModelTrainer(
        modelSavePath="../data/ModelWeights/someRandomModelTest.h5",
        imageSize=2000,
        greyScale=False,
        dictionaryLength=1000,
        batchSize=6
    )

    trainer.trainModels(
        epochs=512,
        steps_per_epoch=64
    )