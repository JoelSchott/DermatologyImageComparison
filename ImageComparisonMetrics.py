import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


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


class MAEComparison(ImageComparisonMetric):
    """
    Mean Absolute Error comparison of the two images
    """
    def __call__(self, image1, image2, use_bounding_box=True):
        if use_bounding_box:
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
    def __call__(self, image1, image2, use_bounding_box=True):
        if use_bounding_box:
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

