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

    def trainModels(self):
        dataGenerator = self.dataGenerator.generateData()
        model, encoder, decoder = self.modelGenerator.getModel()

        model.fit(
            x=dataGenerator,
            steps_per_epoch=200,
            epochs=512,
            callbacks=[
                ModelCheckpoint(
                    filepath="../data/ModelWeights/bestWeights.h5",
                    monitor="loss",
                    verbose=1,
                    save_weights_only=True,
                    save_best_only=True
                )
            ]
        )

if __name__ == "__main__":
    trainer = ModelTrainer(greyScale=False)

    trainer.trainModels()