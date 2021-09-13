import pandas
import pandas as pd
import numpy as np
from typing import Dict, List
import os
import shutil
import json

from ImageComparison import load_images
from ImageComparisonMetrics import ImageRequirements

DATA_FOLDER = "data"
IMAGE_FOLDER = "images"


def group_table_by_col(table, key_col, val_col):
    """
    Returns a dictionary where the keys are the unique elements of column key_col
    and the values are a list of all the values of column val_col corresponding with that key
    :param table: pandas dataframe with two columns
    :param key_col: name of the column to get the unique keys from
    :param val_col: name of the column to get the values from
    :return: dictionary with unique values of first column and values of a list of corresponding values in the second column
    """
    table = table.sort_values(by=key_col, ignore_index=True)
    groups = {}
    if len(table) == 0:
        return groups
    start = 0
    current_key = table[key_col][0]
    for end in range(1, len(table) + 1):
        if end != len(table):
            new_key = table[key_col][end]
        if (end == len(table)) or (new_key != current_key):
            groups[current_key] = list(table[val_col][start:end])
            current_key = new_key
            start = end
    return groups


def create_dataset(image_groups: Dict[str, List[str]], dataset_folder: str, num_paired_images: int, num_singular_images: int ,
                   directory: str, extension: str = ".jpg", meta: str = "metadata"):
    """
    Creates a dataset in the folder given by dataset_folder using the images in image_groups. The used images are
    removed from image_groups.
    :param image_groups: dictionary of lesion IDs as keys and a list of lesion image paths as values
    :param dataset_folder: path to the folder where the dataset should be created
    :param num_paired_images: the number of images in the dataset that will have a pair image with the same lesion ID
    :param num_singular_images: the number of images in the dataset with no paired image
    :param directory: path to the folder where the images are stored
    :param extension: ending of each ISIC image file
    :param meta: the name of the metadata file, without an extension, that will be created, can be None for no metadata file
    """
    lesion_ids_singular = []
    lesion_ids_with_pairs = []
    for lesion_id in image_groups.keys():
        num_images = len(image_groups[lesion_id])
        if num_images > 1:
            lesion_ids_with_pairs.append(lesion_id)
        elif num_images == 1:
            lesion_ids_singular.append(lesion_id)
    num_pairs = int(num_paired_images / 2)
    available_pairs = len(lesion_ids_with_pairs) - num_pairs
    assert available_pairs >= 0, "Not enough images with pairs to make the dataset"
    np.random.shuffle(lesion_ids_singular)
    np.random.shuffle(lesion_ids_with_pairs)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    data_folder = os.path.join(dataset_folder, DATA_FOLDER)
    images_folder = os.path.join(dataset_folder, IMAGE_FOLDER)
    meta_data = {'Number of paired images': num_paired_images, 'Number of singular images': num_singular_images,
                 'Pairs': [], 'Singulars': []}
    for i in range(num_pairs):
        lesion_images = image_groups[lesion_ids_with_pairs[i]]
        np.random.shuffle(lesion_images)
        meta_data['Pairs'].append([lesion_images[-1], lesion_images[-2]])
        for _ in range(2):
            src = os.path.join(directory, lesion_images[-1] + extension)
            des = os.path.join(images_folder, lesion_images[-1] + extension)
            shutil.copyfile(src, des)
            lesion_images.pop()
    for i in range(num_singular_images):
        if i < len(lesion_ids_singular):
            lesion_image = image_groups[lesion_ids_singular[i]][0]
            meta_data['Singulars'].append(lesion_image)
            src = os.path.join(directory, lesion_image + extension)
            des = os.path.join(images_folder, lesion_image + extension)
            shutil.copyfile(src, des)
            image_groups[lesion_ids_singular[i]] = []
        else:
            # go through the pairs and get singular images from the pairs
            # add num_pairs to the index so it starts with new images
            index = i - len(lesion_ids_singular) + num_pairs
            lesion_images = image_groups[lesion_ids_with_pairs[index]]
            np.random.shuffle(lesion_images)
            meta_data['Singulars'].append(lesion_images[-1])
            src = os.path.join(directory, lesion_images[-1] + extension)
            des = os.path.join(images_folder, lesion_images[-1] + extension)
            shutil.copyfile(src, des)
            lesion_images.pop()
    if meta is not None:
        folder_name = os.path.basename(dataset_folder)
        with open(os.path.join(data_folder, folder_name + '_' + meta + '.json'), 'w') as f:
            json.dump(meta_data, f, indent=4)


def create_dataset_metrics(dataset_folder: str, bounding_box_df: pandas.DataFrame):
    """
    Calculates metrics for each of the images in the dataset and saves the metrics in the data subfolder of the dataset
    :param dataset_folder: the folder where the images and metadata are stored
    :param bounding_box_df: pandas dataframe of the bounding boxes to use when making image metrics
    """
    image_paths = [i.path for i in os.scandir(os.path.join(dataset_folder, IMAGE_FOLDER))]
    print(image_paths)
    requirements = [ImageRequirements.BOUNDING_BOX, ImageRequirements.HISTOGRAM,
                    ImageRequirements.IMAGE, ImageRequirements.STATS,
                    ImageRequirements.HASHES]
    images = load_images(image_paths, bounding_box_df, requirements)
    with open(os.path.join(dataset_folder, DATA_FOLDER, "metrics_with_histogram.csv"), 'w') as f:
        header = "Image ID,Perceptual Hash,Difference Hash,Average Hash,Wavelet Hash,RGB Mean,RGB Median,RGB STD,RGB Skew," + \
                 "Gray Mean,Gray Median,Gray STD,Gray Skew,Aspect Ratio,Hist Mean,Hist Median,Hist Mode,Hist STD,Hist Skew,Hist Kurtosis\n"
        f.write(header)
        for image_path, image in zip(image_paths, images):
            f.write(f'{os.path.basename(image_path)[:-4]},{image.phash},{image.dhash},{image.avg_hash},{image.whash},{image.rgb_mean},'
                    f'{image.rgb_median},{image.rgb_std},{image.rgb_skew},{image.gray_mean},{image.gray_median},{image.gray_std},{image.gray_skew},{image.calculate_aspect_ratio()},'
                    f'{image.hist_mean},{image.hist_median},{image.hist_mode},{image.hist_std},{image.hist_skew},{image.hist_kurtosis}\n')


def label_dataset_matches(dataset_folder: str):
    """
    Edits the data csv file of the dataset folder to include a column describing the matching image (if any) for each image
    :param dataset_folder: path to the dataset folder that will be modified
    """
    matches_data_path = os.path.join(dataset_folder, "data", os.path.basename(dataset_folder) + '_metadata.json')
    with open(matches_data_path, 'r') as f:
        matches_data = json.load(f)
    metrics_data = pd.read_csv(os.path.join(dataset_folder, "data", "metrics_with_histogram.csv"))
    num_metrics = len(metrics_data)
    metrics_data['Image ID'] = metrics_data['Image ID'].map(lambda x: os.path.basename(x))
    metrics_data.insert(1, 'Matching Image ID', ['NA' for _ in range(num_metrics)])
    for match in matches_data['Pairs']:
        first_image = match[0]
        second_image = match[1]
        found_images = 0
        for i in range(num_metrics):
            current_image = metrics_data['Image ID'][i]
            if first_image == current_image:
                metrics_data.loc[i, 'Matching Image ID'] = second_image
                found_images += 1
                if found_images == 2:
                    break
            elif second_image == current_image:
                metrics_data.loc[i, 'Matching Image ID'] = first_image
                found_images += 1
                if found_images == 2:
                    break

    metrics_data.to_csv(os.path.join(dataset_folder, "data", "metrics_with_histogram_with_matches.csv"), index=False)


def create_dataset_main():
    IMAGE_DATA_PATH = "lesion_type_data.csv"
    image_data = pd.read_csv(IMAGE_DATA_PATH).drop(["class", "diagnosis_confirm_type"], axis=1)
    image_groups = group_table_by_col(image_data, 'lesion_id', 'image')
    for i in range(5):
        dataset_folder = os.path.join("Datasets", f"dataset{i}")
        create_dataset(image_groups, dataset_folder, 100, 100, "image")


def dataset_metrics_main():
    bounding_boxs = pd.read_csv("InnerBounding.csv")
    for i in range(5):
        dataset_folder = os.path.join("Datasets", f"dataset{i}")
        create_dataset_metrics(dataset_folder, bounding_boxs)


def label_datasets_main():
    for i in range(5):
        label_dataset_matches(os.path.join("Datasets", f"dataset{i}"))


if __name__ == "__main__":
    #create_dataset_main()
    #dataset_metrics_main()
    label_datasets_main()
