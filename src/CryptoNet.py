from ModelGenerator import ModelGenerator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.image import imsave

import pickle

class CryptoNet(object):
    def __init__(self, weightsFilePath: str):
        file = open(f"{weightsFilePath}.p", "rb")
        modelParameters = pickle.load(file=file)
        file.close()

        self.imageSize = modelParameters["imageSize"]
        self.greyScale = modelParameters["greyScale"]
        self.sentenceLength = modelParameters["sentenceLength"]
        self.dictionaryLength = modelParameters["dictionaryLength"]
        self.batchSize = modelParameters["batchSize"]
        self.pepper = [chr(idx) for idx in range(int(0.8 * self.dictionaryLength), self.dictionaryLength)]

        modelGenerator = ModelGenerator(
            imageSize=self.imageSize,
            greyScale=self.greyScale,
            sentenceLength=self.sentenceLength,
            dictionaryLength=self.dictionaryLength
        )

        if self.greyScale:
            self.imageSize = (self.imageSize, self.imageSize, 1)
        else:
            self.imageSize = (self.imageSize, self.imageSize, 3)

        self.model, self.encoder, self.decoder = modelGenerator.getModel()
        self.model.load_weights(filepath=weightsFilePath)

    @staticmethod
    def messageEncode(message: str) -> np.array:
        return np.array([ord(char) for char in message])

    @staticmethod
    def messageDecode(message: np.array) -> str:
        return "".join(chr(codePoint) for codePoint in message)

    def preprocessSentence(self, sentence: str) -> np.array:
        # The final 500 (chr(501) to chr(1000)) items in the dictionary are for sprinkling within your sentence.
        # As a result, if any of these items exist within our sentence, we need to fail.

        if not 0 < len(sentence) <= self.sentenceLength:
            raise ValueError(f"Length of string must be between 0 and {self.sentenceLength + 1}.")

        characters = set(sentence)
        if any(character in self.pepper for character in characters):
            raise ValueError(f"String contains invalid characters.")

        pepperedSentence = sentence
        count = 0
        while len(pepperedSentence) < self.sentenceLength:
            count += 1
            idx = np.random.randint(low=0, high=len(pepperedSentence) + 1)
            character = np.random.choice(self.pepper)
            pepperedSentence = pepperedSentence[:idx] + character + pepperedSentence[idx:]

        encodedSentence = np.expand_dims(self.messageEncode(pepperedSentence), axis=0)

        return encodedSentence

    def preprocessImage(self, imageFilePath: str) -> np.array:
        img = imread(imageFilePath)

        # Crop
        if img.shape[0] > self.imageSize[0]:
            img = img[:self.imageSize[0], :, :]

        if img.shape[1] > self.imageSize[1]:
            img = img[:, :self.imageSize[1], :]

        if img.shape[2] > self.imageSize[2]:
            img = img[:, :, :self.imageSize[2]]

        # Pad
        averageColor = img.mean(axis=1).mean(axis=0)
        paddedImage = np.full(shape=self.imageSize, fill_value=averageColor)
        imgWidthOffset = (self.imageSize[0] - img.shape[0]) // 2
        imgHeightOffset = (self.imageSize[1] - img.shape[1]) // 2

        paddedImage[
            imgWidthOffset:imgWidthOffset + img.shape[0],
            imgHeightOffset:imgHeightOffset + img.shape[1],
            :
        ] = img

        return np.expand_dims(paddedImage, axis=0)

    def encrypt(self, imageFilePath: str, sentence: str, saveOutput: bool = False, embeddedOutputPath: str = None, preprocessedOutputPath: str = None):
        if saveOutput and embeddedOutputPath is None:
            embeddedOutputPath = imageFilePath.replace(r"img/Raw/", r"img/Embedded/")

        if saveOutput and preprocessedOutputPath is None:
            preprocessedOutputPath = imageFilePath.replace(r"img/Raw/", r"img/PreProcessed/")

        # Pad the sentence appropriately
        encodedSentence = self.preprocessSentence(sentence=sentence)

        # Adjust image size as necessary
        img = self.preprocessImage(imageFilePath=imageFilePath)

        imageWithEmbeddedText = self.encoder.predict([img, encodedSentence])[0]

        if saveOutput:
            imsave(fname=preprocessedOutputPath, arr=img[0])
            imsave(fname=embeddedOutputPath, arr=np.clip(imageWithEmbeddedText, a_min=0.0, a_max=1.0))

        return img[0], imageWithEmbeddedText

    def decrypt(self, img: any([np.array, str])) -> str:
        if type(img) == str:
            if self.greyScale:
                img = imread(img)[:, :, :1]
            else:
                img = imread(img)[:, :, :3]

        decodedArray = self.decoder.predict(np.expand_dims(img, axis=0))[0].argmax(-1)
        decodedSentence = self.messageDecode(decodedArray)

        return "".join([char for char in decodedSentence if char not in self.pepper])


if __name__ == "__main__":
    # testSentence = "Gandalf? Yes... that was what they used to call me. Gandalf the Gray. That was my name." \
    #                " I am Gandalf the White. And I come back to you now - at the turn of the tide."

    # testSentence = "Hello there - General Kenobi"
    testSentence = 'Contrary to popular belief, \r\n' \
                   'Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.'

    cryptoNet = CryptoNet(weightsFilePath="../data/ModelWeights/maskTest.h5")

    originalImage, imageWithEmbeddedText = cryptoNet.encrypt(
        imageFilePath="../img/Raw/twister.png",
        sentence=testSentence,
        saveOutput=True
    )

    fig, ax = plt.subplots(1, 2, figsize=(10,10))

    ax[0].imshow(originalImage)
    ax[1].imshow(imageWithEmbeddedText)

    fig.show()

    # decodedMessage = cryptoNet.decrypt(img=imageWithEmbeddedText)
    decodedMessage = cryptoNet.decrypt(img="../img/Embedded/twister.png")
    print(decodedMessage)
    print(decodedMessage == testSentence)
