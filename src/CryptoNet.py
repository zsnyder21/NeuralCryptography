import matplotlib.pyplot as plt
import pickle
import numpy as np

from ModelGenerator import ModelGenerator
from matplotlib.image import imread
from matplotlib.image import imsave


class CryptoNet(object):
    def __init__(self, weightsFilePath: str):
        """
        This class is designed with the purpose of encrypting text into an image and decrypting the
        message from the image.

        You must specify weights for a model when instantiating this class. A pickle file is expected to
        have the same name (including .h5) that holds the information regarding the imageSize, greyScale,
        sentenceLength, dictionaryLength, and batchSize.


        :param weightsFilePath: Fully qualified path to the file in which the model weights are stores
        """
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

    def preprocessSentence(self, sentence: str) -> np.array:
        """
        This method is responsible for pre-processing a sentence to be encoded. The final 20% of the
        dictionary of characters being used is reserved for peppering the sentence. As such, a
        ValueError is thrown if the passed sentence contains any of these characters.

        :param sentence: String to be embedded into the image
        :return: The pre-processed string, now ready to be embedded into an image
        """

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
        """
        This method is responsible for pre-processing the image. It naively crops the image in any
        dimension that is too large, and appends the average pixel color to spatial dimensions that
        are too short.

        :param imageFilePath: File path to the image being used
        :return: The pre-processed image, now ready to have text embedded within it
        """
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
        """
        This method takes an image and sentence and encrypts the sentence by embedding it
        within the image. You can optionally specify whether you would like to save the
        output, as well as the output locations

        :param imageFilePath: File path of the image to embed text within
        :param sentence: Sentence to embed within the image
        :param saveOutput: Whether or not to save the output
        :param embeddedOutputPath: Location to save the image with embedded text
        :param preprocessedOutputPath: Location to save the pre-processed image
        :return: The pre-procsessed image and the image with the sentence embedded within it
        """
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
        """
        This method takes in an image (as an array or a filepath) and extracts the
        information embedded within it.

        :param img: Numpy array representing image with embedded text or
                    filepath to this image
        :return: String information that was embedded within the image
        """
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
