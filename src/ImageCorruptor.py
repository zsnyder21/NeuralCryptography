import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread
from matplotlib.image import imsave


class ImageCorruptor(object):
    def __init__(self, greyScale: bool = True, corruptValue: tuple = (0, 0, 0)):
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

    def corruptImage(self,
                     proportionToCorrupt: float,
                     imageFilePath: str,
                     saveOutput: bool = True,
                     outputFilePath: str = None) -> np.array:
        """
        This method is responsible for corrupting a fixed percentage of an image

        :param proportionToCorrupt: Proportion of the image to corrupt. Must be between 0 and 1
        :param imageFilePath: File path to the image to corrupt
        :param saveOutput: Whether or not to save the output
        :param outputFilePath: Location to save the output
        :return: Numpy array representation of the corrupted image
        """
        if not 0 <= proportionToCorrupt <= 1:
            raise ValueError("proportionToCorruptX must be between 0 and 1.")

        img = imread(fname=imageFilePath)

        if self.greyScale:
            img = img[:, :, 0]
        else:
            img = img[:, :, :3]

        corruptImage = img
        corruptPixelValue = self.corruptValue

        points = np.array([[x, y] for x in range(img.shape[0]) for y in range(img.shape[1])])
        randomPoints = points[np.random.choice(a=points.shape[0], size=int(proportionToCorrupt * points.shape[0]), replace=False)]

        for x, y in randomPoints:
            corruptImage[x, y] = corruptPixelValue

        if saveOutput:
            imsave(fname=outputFilePath, arr=corruptImage)

        return corruptImage


if __name__ == "__main__":
    corruptor = ImageCorruptor(greyScale=False)

    corruptedImage = corruptor.corruptImage(
        proportionToCorrupt=0.45,
        imageFilePath="../img/Embedded/twister.png",
        saveOutput=True,
        outputFilePath="../img/Corrupted/twister.png"
    )

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(corruptedImage)

    fig.show()
