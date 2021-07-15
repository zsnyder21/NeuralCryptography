import numpy as np
import matplotlib.pyplot as plt
import os
import random


class DataGenerator(object):
    def __init__(self, imageSize: int = 100, greyScale: bool = True, dictionaryLength: int = 200):
        """
        This class is responsible for creating a generator that generates
        random sentences and images.

        :param imageSize: Size of the images to be generated
        :param greyScale: Whether or not the images are greyscale
        :param dictionaryLength: Length of the dictioanry of characters to be used
        """
        if greyScale:
            self.imageSize = (imageSize, imageSize, 1)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.sentenceLength = imageSize
        self.dictionaryLength = dictionaryLength

    def generateImage(self) -> np.array:
        """
        Generates a random square image

        :return: Numpy array representing a random image
        """
        return np.random.randint(0, 256, self.imageSize) / 255.0

    def oneHotEncode(self, array: np.array) -> np.array:
        """
        One hot encodes a sentence represented as a numpy array.
        Uses the length of the character dictionary to determine
        how many distinct classes can be represented

        :param array: Numpy array representing our sentence to one-hot encode
        :return: Numpy array representing the one-hot encoded sentence
        """
        maxValue = self.dictionaryLength
        return np.eye(maxValue)[array]

    def generateSentence(self) -> np.array:
        """
        Generates a random sentence represented as a numpy array

        :return: Numpy array representing a random sentence
        """
        return np.array([np.random.randint(low=0, high=self.dictionaryLength) for _ in range(self.sentenceLength)])

    @staticmethod
    def messageEncode(message: str) -> np.array:
        """
        This method takes a string and converts it to a numpy array using ord

        :param message: String to be converted to a numpy array
        :return: Numpy array
        """
        return np.array([ord(char) for char in message])

    @staticmethod
    def messageDecode(message: np.array) -> str:
        """
        This method takes a numpy array and converts it to a string

        :param message: Numpy array to be decoded
        :return: Decoded message string
        """
        return "".join(chr(codePoint) for codePoint in message)

    def generateData(self, batchSize: int = 32):
        """
        This method is responsible for constructing the data generator.
        The data is output as a tuple of lists. The first list contains
        The random images and the random sentences. The second list contains
        random images and the one-hot encoded sentences. Each item in each of
        these lists is a numpy array of size batchSize.


        :param batchSize: Number of images and sentences to create per batch
        :return: Generator holding image and sentence data
        """
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

            yield ([Ximage, Xsentence], [Yimage, Ysentence])


if __name__ == "__main__":
    generator = DataGenerator(
        imageSize=300,
        greyScale=False,
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
