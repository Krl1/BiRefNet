import random
from PIL import Image, ImageEnhance
import numpy as np
import cv2


def fb_blur_fusion_foreground_estimator_2(image, alpha, blur_radius=90):
    """
    Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    Estimate the foreground image by applying a blur fusion method.

    Args:
        image (numpy.ndarray): The input image.
        alpha (numpy.ndarray): The alpha matte.
        blur_radius (int, optional): The blur radius for the fusion. Default is 90.

    Returns:
        numpy.ndarray: The estimated foreground image.
    """
    alpha = alpha[:, :, None]
    foreground, blurred_background = fb_blur_fusion_foreground_estimator(
        image, image, image, alpha, blur_radius
    )
    return fb_blur_fusion_foreground_estimator(
        image, foreground, blurred_background, alpha, blur_radius=6
    )[0]


def fb_blur_fusion_foreground_estimator(image, foreground, background, alpha, blur_radius=90):
    """
    Perform blur fusion to estimate the foreground and background images.

    Args:
        image (numpy.ndarray): The input image.
        foreground (numpy.ndarray): The initial foreground estimate.
        background (numpy.ndarray): The initial background estimate.
        alpha (numpy.ndarray): The alpha matte.
        blur_radius (int, optional): The blur radius for the fusion. Default is 90.

    Returns:
        tuple: A tuple containing the estimated foreground and blurred background images.
    """
    blurred_alpha = cv2.blur(alpha, (blur_radius, blur_radius))[:, :, None]

    blurred_foreground_alpha = cv2.blur(foreground * alpha, (blur_radius, blur_radius))
    blurred_foreground = blurred_foreground_alpha / (blurred_alpha + 1e-5)

    blurred_background_alpha = cv2.blur(background * (1 - alpha), (blur_radius, blur_radius))
    blurred_background = blurred_background_alpha / ((1 - blurred_alpha) + 1e-5)

    foreground = blurred_foreground + alpha * (
            image - alpha * blurred_foreground - (1 - alpha) * blurred_background
    )
    foreground = np.clip(foreground, 0, 1)

    return foreground, blurred_background


def preproc(image, label, preproc_methods=['flip']):
    if 'flip' in preproc_methods:
        image, label = cv_random_flip(image, label)
    if 'crop' in preproc_methods:
        image, label = random_crop(image, label)
    if 'rotate' in preproc_methods:
        image, label = random_rotate(image, label)
    if 'enhance' in preproc_methods:
        image = color_enhance(image)
    if 'pepper' in preproc_methods:
        label = random_pepper(label)
    return image, label


def cv_random_flip(img, label):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def random_crop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    border = int(min(image_width, image_height) * 0.1)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def random_rotate(image, label, angle=15):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-angle, angle)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def color_enhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def random_gaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def random_pepper(img, N=0.0015):
    img = np.array(img)
    noiseNum = int(N * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)