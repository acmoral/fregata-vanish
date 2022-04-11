import numpy as np
import io
import cv2  # to create video
import gdal
import re
from skimage import exposure  # nivelate exposure in tiff->png
from PIL import Image
import rasterio
import matplotlib.pyplot as plt
import matplotlib
from typing import List
from os import listdir
from os.path import isfile, join


def imagToMat(image) -> np.array:
    img = rasterio.open(image)
    arr = np.array(img.read())
    return arr[0]


def make_video(image_folder: str):
    video_name = image_folder + "/transition.mp4"
    images = [img for img in listdir(image_folder) if img.endswith(".png")]
    names=[int(i[:-4]) for i in images]
    file_names=sorted(names)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame = cv2.imread(image_folder + "/" + images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))
    for file in file_names:
        video.write(cv2.imread(image_folder + "/" + str(file)+'.png'))
    cv2.destroyAllWindows()
    video.release()


def interpolate_images(image1: np.ndarray, image2: np.ndarray, step: int) -> List[any]:
    mycm=matplotlib.colors.LinearSegmentedColormap.from_list('',['#582f91', '#00aeef'])
    if image1.shape != image2.shape:
        raise Exception("Image must have same shape!")
    alphas = np.linspace(0, 1, step)
    alphas = alphas[::-1]
    images = list()
    images_PNG = list()
    for alpha in alphas:
        new_image = np.sum(
            [alpha * np.nan_to_num(image1), (1 - alpha) * np.nan_to_num(image2)], axis=0
        )
        equalized = exposure.rescale_intensity(
            new_image,
            # in_range=(np.nanmin(new_image), np.nanmax(new_image)),
            out_range="uint16",
        )
        imgrad=mycm(equalized)
        new_image[new_image == 0] = "nan"
        images.append(new_image)
        images_PNG.append(Image.fromarray(imgrad))
    return images, images_PNG


def get_day(name: str):
    match = re.search(r"\d{4}\d{2}\d{2}", name)
    day = match.group()[-2:]
    return int(day)


def img_name(name_0: str, i: int):
    return name_0[:17] + str(get_day(name_0) + i)


def save_parameters(data_folder: str, file: str):
    dataset = rasterio.open(data_folder + file)
    parameters = {
        "height": dataset.height,
        "width": dataset.width,
        "count": dataset.count,
        "dtype": dataset.dtypes[0],
        "crs": dataset.crs,
        "transform": dataset.transform,
    }
    return parameters


def write_raster(interpolation_image: np.ndarray, name: str, params: dict, PNG=False):
    driver = "GTiff"
    if PNG:
        driver = "PNG"
    new_dataset = rasterio.open(name, "w", driver=driver, **params)
    new_dataset.write(interpolation_image, 1)
    new_dataset.close()


def interpolate_set_images(data_folder=str, save_folder=str):
    datapiro = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]


def interpolate_set_images(data_folder: str, save_folder: str):
    datapiro = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
    images: List[np.array] = list()
    params = save_parameters(data_folder, datapiro[0])
    for file in datapiro:
        # print(file)
        image = imagToMat(data_folder + "/" + file)
        images.append(image)

    for i in range(len(images) - 1):
        image1 = images[i]
        image2 = images[i + 1]
        # Image.fromarray(image1).convert("L").save("data/interpolation/1.png")
        # Image.fromarray(image2).convert("L").savse("data/interpolation/2.png")
        # sys.exit()
        step =40# get_day(datapiro[i + 1]) - get_day(datapiro[i]) + 1
        # import sys
        # sys.exit()

        interpolation_images, interpolation_images_PNG = interpolate_images(
            image1, image2, step=step
        )

        j = 0
        for interpolation_image in interpolation_images:
            #name = "{save_folder}/{j}.tiff".format(
             #   save_folder=save_folder, j=j#=img_name(datapiro[0], j)
            #)
            #write_raster(interpolation_image, name, params, PNG=False)
            interpolation_image_PNG = interpolation_images_PNG[j]
            interpolation_image_PNG.save(
                "{save_folder}/{j}.png".format(save_folder=save_folder, j=j)
            )
            # print("-----")
            # sys.exit()
            j += 1


if __name__ == "__main__":

    """
    image1_filepath = "image1.tif"
    image2_filepath = "image2.tif"
    image1 = imagToMat(image1_filepath)
    image2 = imagToMat(image2_filepath)
    artificial_images = interpolate_images(image1, image2, average = False)
    for i in range(len(artificial_images)):
        artificial_images[i].save("./artificial_average/temp{i}.png".format(i = i))
    """
    interpolate_set_images(data_folder="./data/", save_folder="./interpolation/")
    make_video(image_folder="./interpolation/")
