import numpy as np
import matplotlib.pyplot as plt
import os
import random


class DataGenerator(object):
    def __init__(self, imageSize: int = 100, greyScale: bool = True, sentenceLength: int = 100, dictionaryLength: int = 200):
        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = sentenceLength
        self.dictionaryLength = dictionaryLength

    def generateImage(self) -> np.array:
        return np.random.randint(0, 256, self.imageSize) / 255.0

    def oneHotEncode(self, array: np.array) -> np.array:
        maxValue = self.dictionaryLength
        return np.eye(maxValue)[array]

    def generateSentence(self) -> np.array:
        return np.array([np.random.randint(low=0, high=self.dictionaryLength) for _ in range(self.sentenceLength)])

    def messageEncode(self, message: str) -> np.array:
        return np.array([ord(char) for char in message])

    def messageDecode(self, message: np.array) -> str:
        return "".join(chr(codePoint) for codePoint in message)

    def generateData(self, batchSize: int = 32):
        while True:
            Ximage = np.zeros(shape=(batchSize, self.imageSize[0], self.imageSize[1], self.imageSize[2]))
            Xsentence = np.zeros(shape=(batchSize, self.sentenceLength))
            Yimage = np.zeros(shape=(batchSize, self.imageSize[0], self.imageSize[1], self.imageSize[2]))
            Ysentence = np.zeros(shape=(batchSize, self.sentenceLength, self.dictionaryLength))

            for idx in range(batchSize):
                image = self.generateImage()
                sentence = self.generateSentence()
                sentenceOneHot = self.oneHotEncode(sentence)

                Ximage[idx] = image
                Xsentence[idx] = sentence
                Yimage[idx] = image
                Ysentence[idx] = sentenceOneHot

            yield [[Ximage, Xsentence], [Yimage, Ysentence]]


if __name__ == "__main__":
    generator = DataGenerator(
        imageSize=300,
        greyScale=False,
        sentenceLength=100,
        dictionaryLength=200
    )

    # img = generator.generateImage()

    # file = open("../data/RandomSentences/random.txt", "a")
    #
    # for idx in range(1, 11):
    #     sentence = generator.generateSentence()
    #     sentenceDecoded = generator.messageDecode(sentence)
    #     print(sentenceDecoded)
    #     file.write(f"Random sentence {idx}: {sentenceDecoded}\r\n\r\n")
    #
    # file.close()


    # sentenceDecoded = generator.messageDecode(sentence)
    # print(sentenceDecoded)
    # sentenceEncoded = generator.messageEncode(sentenceDecoded)
    # print(sentenceEncoded)
    #
    # test = generator.oneHotEncode(sentence)
    # print(test.shape)
    # print(test[0], len(test)) #, len(test[0]))

    # test = generator.generateData()
    #
    # print(next(test))

    # plt.rc("xtick", labelsize=22)
    # plt.rc("ytick", labelsize=22)
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(img, cmap="gray")
    # ax.set_xlabel("X", fontsize=24)
    # ax.set_ylabel("Y", fontsize=24)
    # ax.set_title("Randomly Generated Image", fontsize=28)
    # plt.show()
    #
    # fig.savefig("../img/Raw/Random")
