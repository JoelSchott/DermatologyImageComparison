import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ImageComparisonMetrics import ImageRequirements
from BoundingBoxImage import BoundingBoxImage, BoundingBox
import ImageComparison


image = np.array(Image.open(r"group2_images\ISIC_0026439.jpg"))
dark_image = image.copy()
dark_image += 50
dark_image = cv2.normalize(image, dark_image, 0, 255, norm_type=cv2.NORM_MINMAX)

bounding_box_dataframe = pd.read_csv("InnerBounding.csv")

image_bounding_box = BoundingBox()
image_bounding_box.load_from_dataframe("ISIC_0026439", bounding_box_dataframe)
image = BoundingBoxImage(image=image, bounding_box=image_bounding_box)
image.calculate_histogram(normalize=False, bins=8, channels=[0, 1, 2], rgb2hsv=False)

dark_image_bounding_box = BoundingBox()
dark_image_bounding_box.load_from_dataframe("ISIC_0026439", bounding_box_dataframe)
dark_image = BoundingBoxImage(image=dark_image, bounding_box=dark_image_bounding_box)
dark_image.calculate_histogram(normalize=False, bins=8, channels=[0, 1, 2], rgb2hsv=False)

for metric in ImageComparison.METRICS:
    print(metric)
    print(metric(image, dark_image))

plt.subplot(2,1,1)
plt.imshow(image.bounding_box.subimage(image.image))
plt.subplot(2,1,2)
plt.imshow(dark_image.bounding_box.subimage(dark_image.image))
plt.show()




