import cv2
import numpy as np

from imutils import perspective, rotate_bound
from pymatting import estimate_alpha_knn, estimate_foreground_ml, stack_images
from typing import Tuple


PAPER_SIZE = (1485, 1050)


def find_paper(image_bgr: np.ndarray) -> np.ndarray:
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    paper_mask = cv2.inRange(image_hsv, (0, 0, 90), (180, 60, 255))
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_contour = max(contours, key=cv2.contourArea)
    for i in range(100):
        paper_contour = cv2.approxPolyDP(paper_contour, i, True)
    paper_contour = np.squeeze(paper_contour)
    paper_image_bgr = perspective.four_point_transform(image_bgr, paper_contour)
    return cv2.resize(paper_image_bgr, PAPER_SIZE if image_bgr.shape[1] > image_bgr.shape[0] else PAPER_SIZE[::-1])


def get_object_trimap(paper_image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    paper_image_gray = cv2.cvtColor(paper_image_bgr, cv2.COLOR_BGR2GRAY)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = paper_image_gray.reshape((-1, 1))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, None, None, cv2.KMEANS_PP_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)

    darkest_component_mask = np.uint8(np.ones(paper_image_gray.shape) * 255)
    darkest_component_mask[labels.reshape(paper_image_gray.shape) == np.argmin(centers)] = 0

    contours, _ = cv2.findContours(darkest_component_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_new = []
    border_size = 5
    for contour in contours:
        if np.min(contour[:, :, 0]) > border_size \
                and np.min(contour[:, :, 1]) > border_size \
                and np.max(contour[:, :, 0]) < darkest_component_mask.shape[1] - border_size \
                and np.max(contour[:, :, 1]) < darkest_component_mask.shape[0] - border_size \
                and cv2.contourArea(contour) > 150:
            contours_new.append(contour)

    convex_hulls = []
    for contour_new in contours_new:
        convex_hulls.append(cv2.convexHull(contour_new))
    convex_hull = cv2.convexHull(np.concatenate(convex_hulls))

    mask_by_countour = np.uint8(np.ones(paper_image_gray.shape) * 255)
    cv2.drawContours(mask_by_countour, [convex_hull], -1, 0, -1)
    eroded_mask_by_countour = cv2.erode(mask_by_countour, (30, 30), iterations=9)
    trimap = 255 - eroded_mask_by_countour
    trimap[trimap == 255] = 128
    trimap[np.logical_and(trimap == 128, labels.reshape(paper_image_gray.shape) == np.argmin(centers))] = 255
    return trimap, convex_hull


def find_object(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_bgr = image
    image_bgr = cv2.resize(image_bgr, (1920, 1080) if image_bgr.shape[1] > image_bgr.shape[0] else (1080, 1920))

    paper_image_bgr = find_paper(image_bgr)
    trimap, convex_hull = get_object_trimap(paper_image_bgr)

    paper_image_bgr_scaled = cv2.cvtColor(paper_image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    trimap_scaled = trimap / 255.0

    # alpha = estimate_alpha_knn(paper_image_bgr_scaled, trimap_scaled)
    alpha = np.zeros_like(trimap_scaled)
    alpha[trimap_scaled > 0] = 1

    return paper_image_bgr, np.squeeze(convex_hull, 1), np.uint8(alpha * 255)
