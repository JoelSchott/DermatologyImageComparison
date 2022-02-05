import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import wasserstein_distance, pearsonr
import imagehash
from PIL import Image


def resize_images(image1, image2):
    """
    Resizes the given images to the same dimension using the average of width and height
    :param image1: the first image to resize, as a 3D numpy array
    :param image2: the second image to resize, as a 3D numpy array
    :return: a tuple of (image1, image2), both resized to the same dimensions
    """
    num_rows = int((image1.shape[0] + image2.shape[0]) / 2)
    num_cols = int((image1.shape[1] + image2.shape[1]) / 2)
    image1 = resize(image1, (num_rows, num_cols), preserve_range=True).astype("uint8")
    image2 = resize(image2, (num_rows, num_cols), preserve_range=True).astype("uint8")
    return image1, image2


class ImageRequirements:
    """
    Possible requirements for each of the metrics
    """
    BOUNDING_BOX = 0
    HISTOGRAM = 1
    IMAGE = 2
    HASHES = 3
    STATS = 4
    MASK = 5
    ALL_REQUIREMENTS = [BOUNDING_BOX, HISTOGRAM, IMAGE, HASHES, STATS, MASK]


class ImageComparisonMetric:
    def __init__(self, weight):
        self.weight = weight


class AspectRatioComparison(ImageComparisonMetric):
    """
    Compares the aspect ratios of two images
    """
    def __call__(self, image1, image2):
        return -self.weight * abs(image1.calculate_aspect_ratio() - image2.calculate_aspect_ratio())

    def __str__(self):
        return "Aspect Ratio"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX]


class ExtentRatioComparison(ImageComparisonMetric):
    """
    Compares extents of two images
    """
    def __call__(self, image1, image2):
        return -self.weight * abs(image1.bounding_box.extent() - image2.bounding_box.extent())

    def __str__(self):
        return "Extent Ratio"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX]


class CorrelationHistogramComparison(ImageComparisonMetric):
    """
    Finds correlation of histograms of two images
    """
    def __call__(self, image1, image2):
        return self.weight * cv2.compareHist(image1.hist, image2.hist, cv2.HISTCMP_CORREL)

    def __str__(self):
        return "Histogram Correlation"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.HISTOGRAM]


class BhattacharyyaHistogramComparison(ImageComparisonMetric):
    """
    Finds Bhattacharyya metric of the two images
    """
    def __call__(self, image1, image2):
        return -self.weight * cv2.compareHist(image1.hist, image2.hist, cv2.HISTCMP_BHATTACHARYYA)

    def __str__(self):
        return "Histogram Bhattacharyya"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.HISTOGRAM]


class EMDHistogramComparison(ImageComparisonMetric):
    """
    Finds Earth Mover's Distance metric of the two images
    """
    def __call__(self, image1, image2):
        return -self.weight * wasserstein_distance(image1.hist, image2.hist)

    def __str__(self):
        return "Histogram Earth Mover's Distance"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.HISTOGRAM]


class MAEComparison(ImageComparisonMetric):
    """
    Mean Absolute Error comparison of the two images
    """
    def __init__(self, weight, use_bounding_box=True):
        super().__init__(weight)
        self.use_bounding_box = use_bounding_box

    def __call__(self, image1, image2):
        if self.use_bounding_box:
            image1 = image1.bounding_box.subimage(image1.image)
            image2 = image2.bounding_box.subimage(image2.image)
        else:
            image1 = image1.image
            image2 = image2.image
        image1, image2 = resize_images(image1, image2)
        error = np.sum(np.abs(image1.astype("float32") - image2.astype("float32")))
        error /= (255 * image1.size)
        return -self.weight * error

    def __str__(self):
        return "Mean Absolute Error"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.IMAGE]


class SSIMComparison(ImageComparisonMetric):
    """
    Structured Similarity Comparison of the two images
    """
    def __init__(self, weight, use_bounding_box=True):
        super().__init__(weight)
        self.use_bounding_box = use_bounding_box

    def __call__(self, image1, image2):
        if self.use_bounding_box:
            image1 = image1.bounding_box.subimage(image1.image)
            image2 = image2.bounding_box.subimage(image2.image)
        else:
            image1 = image1.image
            image2 = image2.image
        image1, image2 = resize_images(image1, image2)
        if len(image1.shape) == 2:
            return self.weight * ssim(image1, image2, multichannel=False)
        elif len(image1.shape) == 3:
            return self.weight * ssim(image1, image2, multichannel=True)

    def __str__(self):
        return "Structured Similarity"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.IMAGE]


class PixelCorrelation(ImageComparisonMetric):
    """
    Correlation across all of the pixels
    """
    def __init__(self, weight, p_value_weight=0, use_bounding_box=True):
        super().__init__(weight)
        self.p_value_weight = p_value_weight
        self.use_bounding_box = use_bounding_box

    def __call__(self, image1, image2):
        if self.use_bounding_box:
            image1 = image1.bounding_box.subimage(image1.image)
            image2 = image2.bounding_box.subimage(image2.image)
        else:
            image1 = image1.image
            image2 = image2.image
        image1, image2 = resize_images(image1, image2)
        image1 = rgb2gray(image1).flatten()
        image2 = rgb2gray(image2).flatten()
        corr = pearsonr(image1, image2)
        return (self.weight * corr[0]) - (self.p_value_weight * corr[1])

    def __str__(self):
        return "Pixel Correlation and P Value"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.IMAGE]


class TextureSimilarity(ImageComparisonMetric):
    """
    Texture similarity, measured with contrast, dissimilarity, homogeneity, ASM, energy, and correlation
    """
    def __init__(self, weights, use_bounding_box=True):
        super().__init__(weights)
        self.use_bounding_box = use_bounding_box

    def __call__(self, image1, image2):
        if self.use_bounding_box:
            image1 = image1.bounding_box.subimage(image1.image)
            image2 = image2.bounding_box.subimage(image2.image)
        else:
            image1 = image1.image
            image2 = image2.image
        image1, image2 = resize_images(image1, image2)
        image1 = (255 * rgb2gray(image1)).astype("uint8")
        image2 = (255 * rgb2gray(image2)).astype("uint8")
        image1_gray_comatrix = greycomatrix(image1, [1], [0], levels=256, normed=True)
        image2_gray_comatrix = greycomatrix(image2, [1], [0], levels=256, normed=True)
        difference = 0
        for prop_index, prop in enumerate(['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']):
            difference += self.weights[prop_index] * np.sum(np.abs(greycoprops(image1_gray_comatrix, prop) - greycoprops(image2_gray_comatrix, prop)))
        return -difference

    def __str__(self):
        return "Texture Similarity"

    def requires(self):
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.IMAGE]


class HashComparison(ImageComparisonMetric):
    """
    Uses image signatures to determine the difference between two images
    """
    def __init__(self, weight, use_bounding_box=True, hashes=None):
        super().__init__(weight)
        self.use_bounding_box = use_bounding_box
        if hashes is None:
            self.hashes = [imagehash.average_hash]
        else:
            self.hashes = hashes

    def __call__(self, image1, image2):
        if self.use_bounding_box:
            image1 = image1.bounding_box.subimage(image1.image)
            image2 = image2.bounding_box.subimage(image2.image)
        else:
            image1 = image1.image
            image2 = image2.image
        image1 = Image.fromarray(image1, mode='RGB')
        image2 = Image.fromarray(image2, mode='RGB')
        diff = 0
        for image_hash in self.hashes:
            first_hash = image_hash(image1)
            second_hash = image_hash(image2)
            diff += abs(first_hash - second_hash)
        return -(self.weight * diff)

    def __str__(self):
        return "Average Hash"

    @staticmethod
    def requires():
        return [ImageRequirements.BOUNDING_BOX, ImageRequirements.IMAGE]


class PHashComparison(ImageComparisonMetric):
    """
    Uses perceptual hashes to compare the two images
    """
    def __init__(self, weight, use_bounding_box=True):
        super().__init__(weight)
        self.use_bounding_box = use_bounding_box

    def __call__(self, image1, image2):
        for image in (image1, image2):
            if image.phash is None:
                image.calculate_phash(use_bounding_box=self.use_bounding_box)
        return -(self.weight * abs(image1.phash - image2.phash))

    def __str__(self):
        return "Perceptual Hashing"

    def requires(self):
        if self.use_bounding_box:
            return [ImageRequirements.BOUNDING_BOX, ImageRequirements.IMAGE]
        else:
            return [ImageRequirements.IMAGE]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from ImageComparison import load_image
    # testing metrics on example images
    bounding_boxes = pd.read_csv("InnerBounding.csv")
    all_reqs = [ImageRequirements.BOUNDING_BOX, ImageRequirements.HISTOGRAM, ImageRequirements.IMAGE]
    first_image = load_image(r"image\ISIC_0025467.jpg", bounding_boxes, all_reqs)
    second_image = load_image(r"image\ISIC_0026321.jpg", bounding_boxes, all_reqs)
    pixelCor = PixelCorrelation(1)
    print(pixelCor(first_image, second_image))
    mae = MAEComparison(1)
    print(mae(first_image, second_image))
    s = SSIMComparison(1)
    print(s(first_image, second_image))
    texture = TextureSimilarity(1)
    print(texture(first_image, second_image))