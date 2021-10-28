from PIL import Image
import numpy as np
import cv2
import imagehash
from scipy.stats import skew, kurtosis


def normalize_image_by_channels(image):
    """
    Normalizes each channel of the given image from 0 to 255 using min-max normalization
    :param image: the 3D numpy array of the image to normalize
    :return: the image normalized by each of the channels
    """
    normed_image = image.copy().astype("float32")
    for channel in range(normed_image.shape[2]):
        channel_min = np.min(normed_image[:, :, channel])
        channel_max = np.max(normed_image[:, :, channel])
        normed_image[:, :, channel] = normed_image[:, :, channel] * 255 / (channel_max - channel_min) - channel_min
        normed_image[normed_image < 0] = 0
    return normed_image.astype("uint8")


def normalize_image(image):
    """
    Normalizes the image from 0 to 255 using min-max normalization across all channels
    :param image: 3D numpy array of image to normalize
    :return: normalized image
    """
    normed_image = image.copy().astype("float32")
    image_min = np.min(normed_image)
    image_max = np.max(normed_image)
    normed_image = normed_image * 255 / (image_max - image_min) - image_min
    normed_image[normed_image < 0] = 0
    return normed_image.astype("uint8")


class BoundingBox:
    """
    Represent a bounding box on an image
    """
    def __init__(self, xmin=None, ymin=None, xmax=None, ymax=None):
        """
        Represents a bounding box, where the dimensions are between 0 and 1
        :param xmin: left side of the bounding box, between 0 and 1
        :param ymin: top side of the bounding box, between 0 and 1
        :param xmax: right side of the bounding box, between 0 and 1
        :param ymax: bottom side of the bounding box, between 0 and 1
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def load_from_dataframe(self, image_name, dataframe):
        """
        loads the dimensions of the bounding box from the dataframe
        :param image_name: the name of the image, in the column 'image'
        :param dataframe: pandas dataframe with columns for 'image', 'xmin', 'ymin', 'xmax', and 'ymax'
        :return: if the dimensions were loaded successfully from the dataframe
        """
        table = dataframe[dataframe['image'] == image_name]
        if len(table) == 0:
            return False
        self.xmin = table['xmin'].iloc[0]
        self.ymin = table['ymin'].iloc[0]
        self.xmax = table['xmax'].iloc[0]
        self.ymax = table['ymax'].iloc[0]
        return True

    def as_mask(self, image_shape):
        """
        Returns a 2d numpy bool array, true over the part of the image given by the bounding box
        :param image_shape: tuple of rows, cols of the image
        :return: 2d numpy array, masking the bounding box
        """
        mask = np.zeros((image_shape[0], image_shape[1])).astype('uint8')
        mask_xmin = int(self.xmin * image_shape[1])
        mask_xmax = int(self.xmax * image_shape[1])
        mask_ymin = int(self.ymin * image_shape[0])
        mask_ymax = int(self.ymax * image_shape[0])
        mask[mask_ymin:mask_ymax, mask_xmin:mask_xmax] = 1
        return mask

    def subimage(self, image):
        """
        Returns the section of the image in the bounding box
        :param image: the image to find the subsection of
        :return: the subimage of the image that is covered by the bounding box
        """
        mask_xmin = int(self.xmin * image.shape[1])
        mask_xmax = int(self.xmax * image.shape[1])
        mask_ymin = int(self.ymin * image.shape[0])
        mask_ymax = int(self.ymax * image.shape[0])
        return image[mask_ymin:mask_ymax, mask_xmin:mask_xmax]

    def aspect_ratio(self):
        """
        The aspect ratio of the bounding box
        :return: height / width of the bounding box
        """
        return (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def extent(self):
        """
        Extent of the bounding box
        :return: area of bounding box / total area extent of bounding box
        """
        return (self.ymax - self.ymin) * (self.xmax - self.xmin)


class BoundingBoxImage:
    """
    Represents an image with a bounding box
    """
    def __init__(self, image=None, bounding_box=None):
        """
        Create a bounding box image with the given image and bounding box
        :param image: numpy array representing the image
        :param bounding_box: bounding box instance
        """
        self.image = image
        self.bounding_box = bounding_box
        self.hist = None

        self.phash = None
        self.phash_quad = None
        self.dhash = None
        self.avg_hash = None
        self.whash = None
        self.chash = None

        self.rgb_mean = None
        self.rgb_std = None
        self.rgb_median = None
        self.rgb_skew = None
        self.gray_mean = None
        self.gray_std = None
        self.gray_median = None
        self.gray_skew = None

        self.hist_mean = None
        self.hist_median = None
        self.hist_mode = None
        self.hist_std = None
        self.hist_skew = None
        self.hist_kurtosis = None

    def load_image_from_path(self, image_path, normalize=False, rgb2hsv=False):
        """
        Loads the image from the given path
        :param image_path: path to the image to load
        :param normalize: whether to normalize the image
        :param rgb2hsv: whether to convert the image from rgb to hsv
        :return: if the image was successfully loaded
        """
        try:
            self.image = np.array(Image.open(image_path)).astype("uint8")
            if normalize:
                self.image = normalize_image(self.image)
            if rgb2hsv:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)

            return True
        except FileNotFoundError as e:
            return False

    def calculate_histogram(self, use_bounding_box=True, normalize=False, bins=8, channels=None, rgb2hsv=False):
        """
        Calculates a histogram from the image
        :param use_bounding_box: whether or not to use the bounding box when creating the histogram
        :param normalize: whether to norm the image to the extent of the histogram
        :param bins: the number of bins for each channel of the histogram
        :param channels: list of the channels of the image to use, default is all of the channels
        :param rgb2hsv: whether to convert the rgb bounding box to hsv
        """
        assert self.image is not None, "Tried to make a histogram with an image that is None"
        if use_bounding_box:
            hist_image = self.bounding_box.subimage(self.image)
        else:
            hist_image = self.image
        if rgb2hsv:
            hist_image = cv2.cvtColor(hist_image, cv2.COLOR_RGB2HSV)
        if normalize:
            hist_image = normalize_image(hist_image)
        # if channels is not given, have the channels be all of the available channels
        if channels is None:
            if len(self.image.shape) == 3:
                channels = list(range(self.image.shape[2]))
            elif len(self.image.shape) == 2:
                channels = [0]
        # create a bin and a range for each channel
        all_bins = []
        all_ranges = []
        for _ in channels:
            all_bins.append(bins)
            all_ranges.extend([0, 256])
        # create the histogram based off of the channels, bins, and ranges
        hist = cv2.calcHist([hist_image], channels, None, all_bins, all_ranges)
        hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L2).flatten()
        self.hist = hist

    def calculate_histogram_metrics(self, use_bounding_box=True, normalize=False, bins=8, channels=None):
        """
        Calculates and stores metrics about a histogram generated from the image that has the given specifications
        :param use_bounding_box: whether to use the bounding box of the image to make the histogram
        :param normalize: whether to normalize the image before creating a histogram
        :param bins: the number of bins to use for the histogram
        :param channels: list of the indices of the channels to create metrics from, can include 'gray' to mean metrics
                         from a grayscaled image
        """
        if channels is None:
            channels = ['gray']

        if use_bounding_box:
            hist_image = self.bounding_box.subimage(self.image)
        else:
            hist_image = self.image
        if normalize:
            hist_image = normalize_image(hist_image)

        hist_values = []
        for channel in channels:
            if channel == 'gray':
                hist_values.append(cv2.cvtColor(hist_image, cv2.COLOR_RGB2GRAY))
            elif channel == 0 and len(hist_image.shape) == 2:
                hist_values.append(hist_image)
            else:
                hist_values.append(hist_image[:, :, channel])
        hist_values = np.array(hist_values)

        hist, bin_edges = np.histogram(hist_values, bins=bins, range=(0,255))
        # use weights and histogram functions to find metrics about the histogram
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(bins)]

        self.hist_mean = np.average(bin_centers, weights=hist)
        self.hist_std = np.sqrt(np.average((bin_centers - self.hist_mean)**2, weights=hist))
        self.hist_mode = bin_centers[np.argmax(hist)]

        hist_cum = np.cumsum(hist)
        median_place = int(np.sum(hist)/2) + 1
        median_index = np.where(hist_cum >= median_place)[0][0]
        self.hist_median = bin_centers[median_index]

        self.hist_skew = np.average(((bin_centers - self.hist_mean)/self.hist_std)**3, weights=hist)
        self.hist_kurtosis = np.average(((bin_centers - self.hist_mean)/self.hist_std)**4-3, weights=hist)

    def calculate_aspect_ratio(self, use_bounding_box=True):
        """
        Calculates the aspect ratio from the image
        :param use_bounding_box: whether to use the aspect ratio of the bounding box or the image
        :return: returns height / width aspect ratio of the image
        """
        if use_bounding_box:
            assert self.bounding_box is not None, "Tried to calculate aspect ratio when bounding box is None"
            return self.bounding_box.aspect_ratio()
        else:
            assert self.image is not None, "Tried to calculate aspect ratio when image is None"
            return self.image.shape[0] / self.image.shape[1]

    def calculate_hashes(self, use_bounding_box=True):
        """
        Calculates the perceptual hash of the image, stored as an attribute phash
        :param use_bounding_box: whether to calculate the hash for the entire image or just the bounding box
        """
        if use_bounding_box:
            hash_image = self.bounding_box.subimage(self.image)
        else:
            hash_image = self.image
        # make quadrants for the given image
        rows, cols, channels = hash_image.shape
        row_mid = int(rows / 2)
        col_mid = int(cols / 2)
        quads = [[0, 0, row_mid, col_mid],
                 [row_mid, 0, rows, col_mid],
                 [0, col_mid, row_mid, cols],
                 [row_mid, col_mid, rows, cols]]
        self.phash_quad = []
        for quad in quads:
            quad_image = hash_image[quad[0]: quad[2], quad[1]: quad[3], :]
            quad_image = Image.fromarray(quad_image, mode="RGB")
            self.phash_quad.append(imagehash.phash(quad_image))

        hash_image = Image.fromarray(hash_image, mode='RGB')
        self.phash = imagehash.phash(hash_image)
        self.dhash = imagehash.dhash(hash_image)
        self.avg_hash = imagehash.average_hash(hash_image)
        self.whash = imagehash.whash(hash_image)
        self.chash = imagehash.colorhash(hash_image)
        self.crop_resistant_hash = imagehash.crop_resistant_hash(hash_image)

    def calculate_stats(self, use_bounding_box=True):
        """
        Calculates the mean, median, and std of the image values for the standard image and for a grayscaled version
        :param use_bounding_box: whether to use the bounding box of the image for the metrics or the whole image
        """
        if use_bounding_box:
            stats_image = self.bounding_box.subimage(self.image)
        else:
            stats_image = self.image
        self.rgb_mean = np.mean(stats_image)
        self.rgb_median = np.median(stats_image)
        self.rgb_std = np.std(stats_image)
        self.rgb_skew = skew(stats_image.flatten())
        stats_image = cv2.cvtColor(stats_image, cv2.COLOR_RGB2GRAY)
        self.gray_mean = np.mean(stats_image)
        self.gray_median = np.median(stats_image)
        self.gray_std = np.std(stats_image)
        self.gray_skew = skew(stats_image.flatten())

