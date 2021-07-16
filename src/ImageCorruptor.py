import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread
from matplotlib.image import imsave


class ImageCorruptor(object):
    def __init__(self, greyScale: bool = False, corruptValue: tuple = (0, 0, 0)):
        """
        This class will take an image and corrupt random pixels to a supplied value

        :param greyScale: Whether or not the image is grey scale
        :param corruptValue: Value to fill corrupt values with. First value is used in the case of greyscale
        """
        self.greyScale = greyScale

        if self.greyScale:
            self.corruptValue = corruptValue[0]
        else:
            self.corruptValue = corruptValue

    def corruptImage(self, proportionToCorrupt: float, imageFilePath: str, saveOutput: bool = True, outputFilePath: str = None) -> np.array:
        if not 0 <= proportionToCorrupt <= 1:
            raise ValueError("percentageToCorrupt must be between 0 and 1.")

        img = imread(fname=imageFilePath)

        if self.greyScale:
            img = img[:, :, 0]
        else:
            img = img[:, :, :3]

        corruptImage = img
        corruptPixelValue = self.corruptValue

        for _ in range(int(proportionToCorrupt * img.shape[0] * img.shape[1])):
            x = np.random.randint(low=0, high=img.shape[0])
            y = np.random.randint(low=0, high=img.shape[1])

            corruptImage[x, y] = corruptPixelValue

        if saveOutput:
            imsave(fname=outputFilePath, arr=corruptImage)

        return corruptImage


if __name__ == "__main__":
    corruptor = ImageCorruptor(greyScale=False)

    corruptedImage = corruptor.corruptImage(
        proportionToCorrupt=0.30,
        imageFilePath="../img/Embedded/twister.png",
        saveOutput=True,
        outputFilePath="../img/Corrupted/twister.png"
    )

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(corruptedImage)
