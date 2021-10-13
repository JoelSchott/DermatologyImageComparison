import ImageComparison
import os
import pandas as pd
import numpy as np
import json
import cv2
from ImageComparisonMetrics import ImageRequirements


def load_dataset_images(dataset_folder, bb_dataframe):
    """
    Loads the images in the given dataset
    :param dataset_folder: the path to the dataset folder
    :param bb_dataframe: pandas dataframe with bounding box data
    :return: a dictionary of the image name and the image object that goes with it
    """
    image_paths = [f.path for f in os.scandir(os.path.join(dataset_folder, "images"))]
    all_requirements = ImageRequirements.ALL_REQUIREMENTS
    images = ImageComparison.load_images(image_paths, bb_dataframe, all_requirements)
    image_dict = {}
    for image_path, image in zip(image_paths, images):
        image_dict[os.path.basename(image_path)[:-4]] = image
    return image_dict


def p_hash_compare(image1, image2):
    return abs(image1.phash - image2.phash)


def p_hash_quad_compare(image1, image2):
    diff = 0
    for q_phash1, q_phash2 in zip(image1.phash_quad, image2.phash_quad):
        diff += abs(q_phash1 - q_phash2)
    return diff


def p_hash_quad_and_total(image1, image2):
    quad_diff = p_hash_quad_compare(image1, image2)
    quad_diff /= 4
    total_diff = p_hash_compare(image1, image2)
    return quad_diff + total_diff


def hist_corr_compare(image1, image2):
    corr = cv2.compareHist(image1.hist, image2.hist, cv2.HISTCMP_CORREL)
    return -corr


def hist_corr_bhatt_compare(image1, image2):
    corr = hist_corr_compare(image1, image2)
    bhatt = cv2.compareHist(image1.hist, image2.hist, cv2.HISTCMP_BHATTACHARYYA)
    return bhatt + corr


def hist_corr_bhatt_p_hash_compare(image1, image2):
    p_hash = p_hash_compare(image1, image2)
    hist = hist_corr_bhatt_compare(image1, image2)
    return p_hash + hist


def find_pair_rankings(dataset_folder, images, compare_function):
    """
    Finds the ranking for each pair and accesses the ranking
    :param dataset_folder: path to the dataset folder
    :param images: list of the images to use
    :param compare_function: the function to compare the two images, this function should have two parameters, one
                             for each image
    :return: the rankings of the pairs
    """
    data_folder = os.path.basename(dataset_folder)
    with open(os.path.join(dataset_folder, "data", f"{data_folder}_metadata.json"), 'r') as f:
        metadata = json.load(f)
    pair_false_positives = []  # a list of the number of false positives for each image with a pair
    for pair in metadata["Pairs"]:
        for first, second in [(pair[0], pair[1]), (pair[1], pair[0])]:
            # rankings here for all the images
            diffs = []
            for image_name, image in images.items():
                if image_name == first:
                    continue
                diff = compare_function(image, images[first])
                if image_name == second:
                    matching_diff = diff
                else:
                    diffs.append(diff)
            diffs.sort()
            false_positives = 0
            for d in diffs:
                if d < matching_diff:
                    false_positives += 1
                else:
                    break
            pair_false_positives.append(false_positives)
    return pair_false_positives


def analyze_rankings(dataset_folder, rankings):
    """ Performs analysis on the rankings of the images in the given dataset """
    print("Rankings for folder", dataset_folder)
    print(rankings)
    print("There are", len(rankings), "rankings")
    rankings = np.array(rankings)
    print("Rankings unique values:")
    vals, counts = np.unique(rankings, return_counts=True)
    for val, count in zip(vals, counts):
        print("False Positives:", val, "Count:", count)
    folder_basename = os.path.basename(dataset_folder)
    with open(os.path.join(dataset_folder, "data", f"{folder_basename}_metadata.json"), 'r') as f:
        metadata = json.load(f)
    i = 0
    for pair in metadata['Pairs']:
        for image in pair:
            if rankings[i] != 0:
                print(f"{image}:", rankings[i])
            i += 1


def rank_dataset(dataset_folder, bb_data, compare_func):
    images = load_dataset_images(dataset_folder, bb_data)
    rankings = find_pair_rankings(dataset_folder, images, compare_func)
    return rankings


def main():
    dataset_path = r"C:\Users\joels\IdeaProjects\DermatologyImageComparison\Datasets\dataset"
    bb_data_path = r"C:\Users\joels\IdeaProjects\DermatologyImageComparison\InnerBounding.csv"
    bb_data = pd.read_csv(bb_data_path)
    for i in range(5):
        print("Dataset Number", i)
        full_path = dataset_path + str(i)
        compare_funcs = [p_hash_quad_and_total]
        names = ['P-Hash Quad and Total']
        for f, name in zip(compare_funcs, names):
            rankings = rank_dataset(full_path, bb_data, f)
            rankings = np.array(rankings)
            matches = np.count_nonzero(rankings == 0)
            print(f"\t{name}: {matches}")


if __name__ == "__main__":
    main()


