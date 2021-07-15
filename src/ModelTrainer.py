import pickle

from ModelGenerator import ModelGenerator
from DataGenerator import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


class ModelTrainer(object):
    def __init__(self,
                 modelSavePath: str,
                 imageSize: int = 100,
                 greyScale: bool = True,
                 sentenceLength: int = 100,
                 dictionaryLength: int = 200,
                 batchSize: int = 32):

        self.modelSavePath = modelSavePath

        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = sentenceLength
        self.dictionaryLength = dictionaryLength
        self.batchSize = batchSize

        self.modelGenerator = ModelGenerator(
            imageSize=imageSize,
            greyScale=greyScale,
            sentenceLength=sentenceLength,
            dictionaryLength=dictionaryLength
        )

        self.dataGenerator = DataGenerator(
            imageSize=imageSize,
            greyScale=greyScale,
            sentenceLength=sentenceLength,
            dictionaryLength=dictionaryLength
        )

        self.model = None,
        self.encoder = None,
        self.decoder = None

        modelParameters = {
            "imageSize": imageSize,
            "greyScale": greyScale,
            "sentenceLength": sentenceLength,
            "dictionaryLength": dictionaryLength,
            "batchSize": batchSize
        }

        file = open(f"{modelSavePath}.p", "wb")
        pickle.dump(obj=modelParameters, file=file)
        file.close()

    def trainModels(self, epochs: int, steps_per_epoch: int, verbose: int = 1):
        dataGenerator = self.dataGenerator.generateData(batchSize=self.batchSize)
        self.model, self.encoder, self.decoder = self.modelGenerator.getModel()

        self.model.fit(
            x=dataGenerator,
            steps_per_epoch=steps_per_epoch,
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
        modelSavePath="../data/ModelWeights/modelWeights.h5",
        imageSize=2000,
        greyScale=False,
        sentenceLength=2000,
        dictionaryLength=1000,
        batchSize=6
    )

    trainer.trainModels(
        epochs=128,
        steps_per_epoch=50
    )