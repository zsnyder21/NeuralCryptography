from ModelGenerator import ModelGenerator
from DataGenerator import DataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint

class ModelTrainer(object):
    def __init__(self, imageSize: int = 100, greyScale: bool = True, sentenceLength: int = 100, dictionaryLength: int = 200):
        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = sentenceLength
        self.dictionaryLength = dictionaryLength

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

    def trainModels(self, epochs: int, steps_per_epoch: int, modelSavePath: str, verbose: int = 1):
        dataGenerator = self.dataGenerator.generateData()
        self.model, self.encoder, self.decoder = self.modelGenerator.getModel()

        self.model.fit(
            x=dataGenerator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[
                ModelCheckpoint(
                    filepath=modelSavePath,
                    monitor="loss",
                    verbose=verbose,
                    save_weights_only=True,
                    save_best_only=True
                )
            ]
        )

if __name__ == "__main__":
    trainer = ModelTrainer(greyScale=False)
    trainer.trainModels(epochs=20, steps_per_epoch=100, modelSavePath="../data/ModelWeights/bestWeights_color_20_100.h5")