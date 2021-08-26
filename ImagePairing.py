import pandas as pd
import numpy as np


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


def create_training_dataset(image_groups, dataset_folder):
    """

    :param image_groups:
    :param dataset_folder:
    :return:
    """


IMAGE_ID = r"C:\Users\joels\Downloads\10k_lesion_id_and_diagnosis_type_info - 10k_lesion_id_and_diagnosis_type_info.csv"

image_labels = pd.read_csv(IMAGE_ID).drop(["class", "diagnosis_confirm_type"], axis=1)

image_groups = group_table_by_col(image_labels, 'lesion_id', 'image')

lengths = []
for val in image_groups.values():
    lengths.append(len(val))

uniqs, counts = np.unique(lengths, return_counts=True)
for uniq, count in zip(uniqs, counts):
    print("Unique value", uniq, "has count", count)
