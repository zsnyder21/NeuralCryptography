from ModelGenerator import ModelGenerator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

class CryptoNet(object):
    def __init__(self,
                 weightsFilePath: str,
                 imageSize: int = 100,
                 greyScale: bool = True,
                 sentenceLength: int = 100,
                 dictionaryLength: int = 200):

        modelGenerator = ModelGenerator(
            imageSize=imageSize,
            greyScale=greyScale,
            sentenceLength=sentenceLength,
            dictionaryLength=dictionaryLength
        )

        self.dictionaryLength = dictionaryLength
        self.model, self.encoder, self.decoder = modelGenerator.getModel()
        self.model.load_weights(filepath=weightsFilePath)

    @staticmethod
    def messageEncode(message: str) -> np.array:
        return np.array([ord(char) for char in message])

    @staticmethod
    def messageDecode(message: np.array) -> str:
        return "".join(chr(codePoint) for codePoint in message)

    def oneHotEncode(self, array: np.array) -> np.array:
        maxValue = self.dictionaryLength
        return np.eye(maxValue)[array]

    def encrypt(self, imageFilePath: str, sentence: str):
        pass

    def decrypt(self):
        pass

if __name__ == "__main__":
    img = imread("../img/Raw/Wizard_small.png")
    # img = np.expand_dims(img, axis=0)
    testImage = img[:, :, :3]
    print(testImage.shape)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(testImage)
    # fig.show()

    testSentence = "Gandalf? Yes... that was what they used to call me. Gandalf the Gray. That was my name." \
                   " I am Gandalf the White. And I come back to you now - at the turn of the tide."

    cryptoNet = CryptoNet(weightsFilePath="../data/ModelWeights/bestWeights_color_512_2000.h5", greyScale=False)
    testSentenceEncoded = cryptoNet.messageEncode(testSentence[:100])
    # test2 = cryptoNet.oneHotEncode(testSentenceEncoded)
    # test2 = np.expand_dims(test2, axis=0)
    testImage = np.expand_dims(testImage, axis=0)
    testSentenceEncoded = np.expand_dims(testSentenceEncoded, axis=0)
    # print(test2.shape)
    print(testImage.shape)
    encodedImage = cryptoNet.encoder.predict([testImage, testSentenceEncoded])

    fig2, ax2 = plt.subplots(figsize=(10,10))
    ax2.imshow(encodedImage[0])
    # fig2.show()
    # print(encodedImage)

    decodedSentence = cryptoNet.decoder.predict(encodedImage)
    print(decodedSentence[0].argmax(-1))

    test3 = cryptoNet.messageDecode(decodedSentence[0].argmax(-1))
    print(test3)