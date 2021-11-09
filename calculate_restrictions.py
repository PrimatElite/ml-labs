import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io           # Only needed for web grabbing images, use cv2.imread for local images


def noisy(image, sigma=0.05, mean=0):
    row, col, ch = image.shape
    orig = cv2.cvtColor(io.imread('https://i.stack.imgur.com/0FNPQ.jpg'), cv2.COLOR_RGB2BGR)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = image + gauss
    print(f"sigma: {np.std(noisy_img - orig)}")
    # noisy_img = np.apply_over_axes(fix_ndarray_types, noisy_img, [0, 1, 2])
    return np.rint(noisy_img)


def gauss_noise_check():
    test_image = cv2.cvtColor(io.imread('https://i.stack.imgur.com/0FNPQ.jpg'), cv2.COLOR_RGB2BGR)
    noisy_image = noisy(test_image, sigma=200)
    noisy_image2 = noisy(noisy_image, sigma=200)
    noisy_image3 = noisy(test_image, sigma=400)

    plt.imshow(test_image)
    plt.show()
    plt.imshow(noisy_image)
    plt.show()
    plt.imshow(noisy_image2)
    plt.show()
    plt.imshow(noisy_image3)
    plt.show()


def is_valid(image):
    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    ##### Just for visualization and debug; remove in final
    plt.plot(s)
    plt.plot([p * 255, p * 255], [0, np.max(s)], 'r')
    plt.text(p * 255 + 5, 0.9 * np.max(s), str(s_perc))
    plt.show()
    ##### Just for visualization and debug; remove in final

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5
    return s_perc > s_thr


def noise_validation_test():
    # Read example images; convert to grayscale
    noise1 = cv2.cvtColor(io.imread('https://i.stack.imgur.com/Xz9l0.png'), cv2.COLOR_RGB2BGR)
    noise2 = cv2.cvtColor(io.imread('https://i.stack.imgur.com/9ZPAj.jpg'), cv2.COLOR_RGB2BGR)
    valid = cv2.cvtColor(io.imread('https://i.stack.imgur.com/0FNPQ.jpg'), cv2.COLOR_RGB2BGR)

    for img in [noise1, noise2, valid]:
        print(is_valid(img))


if __name__ == "__main__":
    # run_noise_validation_test()
    gauss_noise_check()
