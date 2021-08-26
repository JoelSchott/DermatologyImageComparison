import ImageComparison
import BoundingBoxImage
from ImageComparisonMetrics import ImageRequirements
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import cv2


def examine_images(images):
    """
    Displays image attributes and comparisons between the list of images
    """
    cols = len(images)
    rows = 3
    images_made = 0
    plt.figure(0)
    for image_index, image in enumerate(images):
        fig = plt.subplot(rows, cols, images_made+1)
        plt.imshow(image.image)
        plt.colorbar()
        plt.title(f"Image {image_index + 1}")
        mask_rows, mask_cols = np.where(image.bounding_box.as_mask(image.image.shape) == 1)
        min_row, max_row = np.min(mask_rows), np.max(mask_rows) + 1
        min_col, max_col = np.min(mask_cols), np.max(mask_cols) + 1
        rect = patches.Rectangle((min_col, min_row), (max_col - min_col), (max_row - min_row), edgecolor='r', linewidth=1, facecolor='none')
        fig.add_patch(rect)
        fig = plt.subplot(rows, cols, images_made+1+cols)
        bounding_box_image = image.image[min_row:max_row, min_col:max_col]
        # normalize the bounding box image
        print("Bounding box minimum:", np.min(bounding_box_image[:,:,0]))
        print("Bounding box maximum:", np.max(bounding_box_image[:,:,0]))
        #bounding_box_image = BoundingBoxImage.normalize_image(bounding_box_image)
        plt.imshow(bounding_box_image)
        plt.colorbar()
        plt.title(f"Image {image_index + 1}")

        hist = cv2.calcHist([bounding_box_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        #hist = cv2.calcHist([bounding_box_image], [0, 1], None, [8, 8], [0, 256, 0, 256])
        #hist = cv2.calcHist([bounding_box_image], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()

        plt.subplot(rows, cols, images_made+1+(2*cols))
        plt.bar(range(1, len(hist) + 1), hist)

        hist_difference = np.sum(np.abs(hist - image.hist))
        if hist_difference == 0:
            print(f"Passed! For Image {image_index + 1}, there is no difference between the histograms regarding the bounding box")
        else:
            print(f"WARNING! For Image {image_index + 1}, the difference between the histograms regarding the bounding box is {hist_difference}")
        images_made += 1

    plt.figure(1)

    rows = 3
    cols = 3
    images_made = 0
    scores = [0 for _ in range(len(images))]
    for metric in ImageComparison.METRICS:
        metric_scores = []
        print(f"Metric {metric} Comparison:")
        for image_index, image in enumerate(images):
            metric_score = metric(images[0], image)
            metric_scores.append(metric_score)
            scores[image_index] += metric_score
        print(f"\t{metric_score}")
        plt.subplot(rows, cols, images_made + 1)
        plt.bar(range(1, len(metric_scores) + 1), metric_scores)
        plt.title(metric)
        images_made += 1

    plt.subplot(rows, cols, images_made + 1)
    plt.bar(range(1, len(images) + 1), scores)
    plt.title("Scores")

    plt.show()


def main():
    bounding_box_dataframe = pd.read_csv("InnerBounding.csv")
    all_requirements = [ImageRequirements.IMAGE, ImageRequirements.HISTOGRAM, ImageRequirements.BOUNDING_BOX]
    #image1 = ImageComparison.load_image(r"D:\DermatologyResearchData\ISIC_BCC_2018\ISIC_0026668.jpg", bounding_box_dataframe, all_requirements)
    #image1 = ImageComparison.load_image(r"D:\DermatologyResearchData\ISIC_BCC_2018\ISIC_0026321.jpg", bounding_box_dataframe, all_requirements)
    image1 = ImageComparison.load_image(r"ISIC_BCC_2018\ISIC_0031470.jpg", bounding_box_dataframe, all_requirements)
    image2 = ImageComparison.load_image(r"ISIC_BCC_2019_no_repeats\ISIC_0062080.jpg", bounding_box_dataframe, all_requirements)
    examine_images([image1, image2])


if __name__ == "__main__":
    main()