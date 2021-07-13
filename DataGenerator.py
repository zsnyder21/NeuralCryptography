import numpy as np
import matplotlib.pyplot as plt
import os
import random


class DataGenerator(object):
    def __init__(self, imageSize: int, outputDirectory: str, greyScale: bool = True):
        if greyScale:
            self.imageSize = (imageSize, imageSize)
        else:
            self.imageSize = (imageSize, imageSize, 3)

        self.outputDirectory = outputDirectory

        self.codePointRanges = [
            (0x0021, 0x0021),
            (0x0023, 0x0026),
            (0x0028, 0x007E),
            (0x00A1, 0x00AC),
            (0x00AE, 0x00FF),
            (0x0100, 0x017F),
            (0x0180, 0x024F),
            (0x2C60, 0x2C7F),
            (0x16A0, 0x16F0),
            (0x0370, 0x0377),
            (0x037A, 0x037E),
            (0x0384, 0x038A),
            (0x038C, 0x038C),
        ]

        self.alphabet = [
            chr(code_point) for current_range in self.codePointRanges
            for code_point in range(current_range[0], current_range[1] + 1)
        ]

    def generateImage(self) -> np.array:
        return np.random.randint(0, 256, self.imageSize) / 255.0

    def generateSentence(self, sentenceLength: int) -> np.array:
        return np.array([ord(np.random.choice(self.alphabet)) for _ in range(sentenceLength)])

    def messageEncode(self, message: str) -> np.array:
        return np.array([ord(char) for char in message])

    def messageDecode(self, message: np.array) -> str:
        return "".join(chr(codePoint) for codePoint in message)


if __name__ == "__main__":
    generator = DataGenerator(imageSize=300, greyScale=False, outputDirectory="Some Fake Place")
    # img = generator.generateImage()

    sentence = generator.generateSentence(100)
    print(sentence)
    sentenceDecoded = generator.messageDecode(sentence)
    print(sentenceDecoded)
    sentenceEncoded = generator.messageEncode(sentenceDecoded)
    print(sentenceEncoded)

    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.imshow(img, cmap="gray")
    # plt.show()
    #
    # print(os.urandom(30).decode("utf-8"))
    # print(sentence, len(sentence))
    # print(sentenceDecoded, len(sentenceDecoded))
    # print(sentenceEncoded)
    # print(generator.messageDecode(sentence))
    # print(generator.messageEncode())

