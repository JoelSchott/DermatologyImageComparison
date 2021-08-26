from BoundingBoxImage import BoundingBoxImage, BoundingBox
from ImageComparisonMetrics import *
import os
import pandas as pd
METRICS = [#AspectRatioComparison(weight=0),
           #ExtentRatioComparison(weight=0),
           #CorrelationHistogramComparison(weight=1),
           #BhattacharyyaHistogramComparison(weight=1),
#           EMDHistogramComparison(weight=1),
#           MAEComparison(weight=1),
#           SSIMComparison(weight=1)
            PHashComparison(weight=1, use_bounding_box=True)
          ]

GROUP1_FOLDER = r"ISIC_BCC_2019_no_repeats"
#ROUP1_IMAGE_PATHS = [f.path for f in os.scandir(GROUP1_FOLDER)]
GROUP1_IMAGE_PATHS = [os.path.join('image', f) for f in os.listdir(GROUP1_FOLDER)]
GROUP1_BB_PATH = "InnerBounding.csv"

GROUP2_FOLDER = r"ISIC_BCC_2018"
#GROUP2_IMAGE_PATHS = [f.path for f in os.scandir(GROUP2_FOLDER)]
GROUP2_IMAGE_PATHS = [os.path.join('image', f) for f in os.listdir(GROUP2_FOLDER)]
GROUP2_BB_PATH = "InnerBounding.csv"

# the size of the largest amount of images to hold in memory at one point
GROUP1_BATCH_SIZE = len(GROUP1_IMAGE_PATHS)

OUTPUT_FILE = "results/2019_2018_rgb_padded_hash_results2.csv"


def image_name_from_path(image_path):
    """
    Returns the name of the image (what will be in the csv file) from the path to the image
    :param image_path: the path to the image
    :return: the name of the image
    """
    return os.path.basename(image_path)[:-4]


def load_image(image_path, bounding_box_dataframe, requirements):
    """
    Loads an image and all needed information
    :param image_path: path to the image to load
    :param bounding_box_dataframe: dataframe with bounding box information
    :param requirements: list of requirements needed for the image
    :return: BoundingBoxImage if image and information successfully made, otherwise None
    """
    if ImageRequirements.BOUNDING_BOX in requirements:
        bounding_box = BoundingBox()
        # if the bounding box is required and cannot be found, return None
        if not bounding_box.load_from_dataframe(image_name_from_path(image_path), bounding_box_dataframe):
            return None
    else:
        bounding_box = None

    image = BoundingBoxImage(bounding_box=bounding_box)

    if (ImageRequirements.IMAGE in requirements) or (ImageRequirements.HISTOGRAM in requirements):
        # if the image is required and cannot be found, continue to the next image
        if not image.load_image_from_path(image_path, rgb2hsv=False):
            return None

    if ImageRequirements.HISTOGRAM in requirements:
        image.calculate_histogram(normalize=False, bins=8, channels=[0, 1, 2], rgb2hsv=False)

    # if the image is not needed, the image can be removed to save memory
    if ImageRequirements.IMAGE not in requirements:
        image.image = None

    return image


def load_images(image_paths, bounding_box_dataframe, requirements):
    """
    Loads images and associated information
    :param image_paths: list of image paths
    :param bounding_box_dataframe: pandas dataframe with bounding box information for the images
    :param requirements: list of requirements that each image needs to have
    :return: list of BoundingBoxImages, one for each image_path
    """
    images = []
    for image_path in image_paths:
        image = load_image(image_path, bounding_box_dataframe, requirements)
        images.append(image)

    return images


def requirements_for_metrics(metrics):
    """
    Returns a set of all requirements needed for the given metrics
    :param metrics: list of metrics to find the requirements of
    :return: minimal set of requirements needed
    """
    all_requirements = set()
    for metric in metrics:
        for requirement in metric.requires():
            all_requirements.add(requirement)
    return all_requirements


def main():
    all_requirements = requirements_for_metrics(METRICS)

    group1_bb_dataframe = pd.read_csv(GROUP1_BB_PATH)
    group2_bb_dataframe = pd.read_csv(GROUP2_BB_PATH)

    group2_images = load_images(GROUP2_IMAGE_PATHS, group2_bb_dataframe, all_requirements)

    with open(OUTPUT_FILE, 'w') as f:
        f.write("Group 1 Images,Group 2 Images,Score\n")

    for group1_image_index in range(0, len(GROUP1_IMAGE_PATHS), GROUP1_BATCH_SIZE):
        group1_last_index = min(group1_image_index + GROUP1_BATCH_SIZE, len(GROUP1_IMAGE_PATHS))
        group1_images = load_images(GROUP1_IMAGE_PATHS[group1_image_index:group1_last_index], group1_bb_dataframe, all_requirements)

        for index, group1_image in enumerate(group1_images):
            if group1_image is None:
                continue
            scores = []
            for group2_image in group2_images:
                if group2_image is None:
                    scores.append(-np.inf)
                    continue
                score = 0
                for metric in METRICS:
                    score += metric(group1_image, group2_image)
                scores.append(score)
            scores = np.array(scores)
            best_match_index = np.argmax(scores)

            group1_image_path = GROUP1_IMAGE_PATHS[group1_image_index + index]
            group2_image_path = GROUP2_IMAGE_PATHS[best_match_index]
            best_score = scores[best_match_index]
            print(f"Group 1 Image with path {group1_image_path} matched with {group2_image_path} with score {best_score}")
            with open(OUTPUT_FILE, 'a') as f:
                f.write(f"{group1_image_path},{group2_image_path},{best_score}\n")


if __name__ == '__main__':
    main()

