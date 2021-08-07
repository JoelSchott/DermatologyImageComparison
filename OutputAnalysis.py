import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ImageComparison
from ImageComparisonMetrics import MAEComparison, SSIMComparison

ANALYSIS_FILE = "results/2019_2018_red_results.csv"


def display_best_matches(output_file, n=10, sort_col='Score'):
    """
    Displays the matches that have the highest score
    :param output_file: path to the output csv file where the image paths and the scores are
    :param n: the number of matches to display
    :param sort_col: name of the column to sort results by
    """
    plt.subplots(figsize=(6, 20))
    data = pd.read_csv(output_file)
    data = data.sort_values(by=sort_col, ascending=False)
    for col in range(n):
        plt.subplot(n, 2, col * 2 + 1)
        group1_image = Image.open(data.iloc[col]['Group 1 Images'])
        plt.imshow(group1_image)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(n, 2, col * 2 + 2)
        group2_image = Image.open(data.iloc[col]['Group 2 Images'])
        plt.imshow(group2_image)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def calculate_match_metrics(matches, metrics, first_bounding_box_dataframe, second_bounding_box_dataframe):
    """
    Calculates a score based on the given metrics for each match in matches
    :param matches: pandas dataframe where the first column is one image path and the second column
                    is the path to the matching image
    :param metrics: list of metrics to use to compare the matches
    :param first_bounding_box_dataframe: pandas dataframe describing bounding boxes for the images in the first column
    :param second_bounding_box_dataframe: pandas dataframe describing bounding boxes for the images in the second column
    :return: pandas dataframe with an additional column, Match Score, based off of the given metrics
    """
    matches['Match Score'] = [0 for _ in range(len(matches))]
    # convert the match score column to floats
    matches = matches.astype({'Match Score': np.float32})
    all_requirements = ImageComparison.requirements_for_metrics(metrics)
    for match_index in range(len(matches)):
        first_image_path = matches.iloc[match_index, 0]
        first_image = ImageComparison.load_image(first_image_path, first_bounding_box_dataframe, all_requirements)
        second_image_path = matches.iloc[match_index, 1]
        second_image = ImageComparison.load_image(second_image_path, second_bounding_box_dataframe, all_requirements)
        match_score = 0
        for metric in metrics:
            match_score += metric(first_image, second_image)
        matches.at[match_index, 'Match Score'] = match_score
    return matches


def main():
    display_best_matches(ANALYSIS_FILE)
    bounding_box_dataframe = pd.read_csv("InnerBounding.csv")
    match_metrics = [MAEComparison(weight=1),
                     SSIMComparison(weight=1)]
    matches = pd.read_csv(ANALYSIS_FILE)
    matches_new_scores = calculate_match_metrics(matches, match_metrics, bounding_box_dataframe, bounding_box_dataframe)
    output_path = ANALYSIS_FILE[:-4] + '_match_metrics.csv'
    matches_new_scores.to_csv(output_path)


if __name__ == '__main__':
    #main()
    display_best_matches(ANALYSIS_FILE, sort_col='Score')
