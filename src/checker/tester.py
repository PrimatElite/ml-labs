from __future__ import annotations

import cv2
import numpy as np
import os

from imutils import rotate_bound
from intelligent_placer_lib import intelligent_placer
from pymatting import blend
from typing import Dict, List, Optional

from ..common import load_image, Object, PAPER_SIZE
from ..utils import get_config


DEFAULT_HEIGHT = 232.5
PIXELS_PER_MM = PAPER_SIZE[0] / 297
PIXELS_PER_CM = PIXELS_PER_MM * 10
SEGMENT_RATIO = 0.01
TEST_CONSTRAINTS = ['shooting_height', 'rotation', 'noise', 'blur']
X_SHIFT_CM = 3

# - name: obj_num
# - name: same_obj_num
# - name: back_diff_obj


class Tester:
    background_image: np.ndarray
    config: Dict[str, List[float | int]]
    objects: List[Object]
    segments: Dict[str, List[float]]

    def __init__(self, path_to_objects: str, path_to_config: str):
        self.background_image = load_image('src/checker/background.png')
        config = get_config(path_to_config)
        self.config = {constraint['name']: constraint['value'] for constraint in config['restrictions']}
        self.objects = [Object(os.path.join(path_to_objects, file)) for file in os.listdir(path_to_objects)]
        self.segments = {}

    def generate_polygon(self, objects: List[Object], constraints: Dict[str, float | int]) -> np.ndarray:
        # - name: polygon_vertex_num
        # - name: polygon_angle
        # - name: area_ratio
        return np.array([])

    def generate_image(self, objects: List[Object], constraints: Dict[str, float | int],
                       polygon: Optional[np.ndarray] = None) -> np.ndarray:
        # - name: min_dist_between_obj - минимальное расстояние между объектами
        # - name: max_dist_between_obj_center - максимальное расстояние от центра объектов до дальних объектов
        # - name: min_dist_between_obj_polygon
        # - name: line_width
        # - name: resolution
        # - name: aspect_ratio
        # - name: back_shadows
        # - name: obj_shadows
        # - name: camera_shift - сдвиг камеры относительно предметов
        # - name: shooting_angle - перспектива
        # - name: noise
        # - name: blur
        objects_images: List[np.ndarray] = []
        objects_alphas: List[np.ndarray] = []
        for i, object_ in enumerate(objects):
            x, y, w, h = cv2.boundingRect(object_.convex_hull)
            objects_images.append(object_.image[y:y + h, x:x + w])
            objects_alphas.append(object_.alpha[y:y + h, x:x + w])

        if 'shooting_height' in constraints:
            scale = DEFAULT_HEIGHT / constraints['shooting_height']
            for i in range(len(objects)):
                h, w = round(objects_images[i].shape[0] * scale), round(objects_images[i].shape[1] * scale)
                objects_images[i] = cv2.resize(objects_images[i], (w, h))
                objects_alphas[i] = cv2.resize(objects_alphas[i], (w, h))

        if 'rotation' in constraints:
            for i in range(len(objects)):
                objects_images[i] = rotate_bound(objects_images[i], constraints['rotation'])
                objects_alphas[i] = rotate_bound(objects_alphas[i], constraints['rotation'])

        height = max(objects_images, key=lambda o: o.shape[0]).shape[0]
        width = sum(o.shape[1] for o in objects_images) + round(X_SHIFT_CM * PIXELS_PER_CM * (len(objects) - 1))
        objects_image = np.zeros((height, width, 3), np.uint8)
        objects_alpha = np.zeros((height, width), np.uint8)
        x = 0
        for i in range(len(objects)):
            objects_image[:objects_images[i].shape[0], x:x + objects_images[i].shape[1]] = objects_images[i]
            objects_alpha[:objects_alphas[i].shape[0], x:x + objects_alphas[i].shape[1]] = objects_alphas[i]
            x += objects_images[i].shape[1]
        foreground_height = round(width / self.config['aspect_ratio'][0])
        foreground = np.zeros((foreground_height, width, 3), np.uint8)
        alpha = np.zeros((foreground_height, width), np.uint8)
        y_shift = (foreground_height - height) // 2
        foreground[y_shift:y_shift + height] = objects_image
        alpha[y_shift:y_shift + height] = objects_alpha
        background = cv2.resize(self.background_image, (width, foreground_height))
        image = blend(foreground, background, alpha / 255.0).astype(np.uint8)
        cv2.imwrite('../alpha.png', alpha)
        cv2.imwrite('../foreground.png', foreground)
        cv2.imwrite('../background.png', background)
        cv2.imwrite('../image.png', image)
        cv2.imshow('win', image)
        cv2.waitKey()
        return image

    def do_first_step(self):
        for constraint in TEST_CONSTRAINTS:
            left_value = self.config[constraint][0]
            right_value = self.config[constraint][1]
            left, center_left, center_right, right = left_value, right_value, left_value, right_value
            for object_ in self.objects:
                x, y, w, h = cv2.boundingRect(object_.convex_hull)
                w, h = w / PIXELS_PER_MM, h / PIXELS_PER_MM
                good_polygon = np.array([[0, 0], [0, h + 10], [w + 10, h + 10], [w + 10, 0]])
                segment_end_len = (right_value - left_value) * SEGMENT_RATIO
                l, r = left_value, right_value
                left_image = self.generate_image([object_], {constraint: l})
                right_image = self.generate_image([object_], {constraint: r})
                left_good_result = intelligent_placer.check_image(left_image, good_polygon)
                right_good_result = intelligent_placer.check_image(right_image, good_polygon)
                if not left_good_result:
                    center_left, center_right = left_value, left_value
                    break
                elif right_good_result:
                    center_left, center_right = right_value, right_value
                    break
                while r - l > segment_end_len:
                    m = (l + r) / 2
                    m_image = self.generate_image([object_], {constraint: m})
                    m_result = intelligent_placer.check_image(m_image, good_polygon)
                    if m_result:
                        l = m
                    else:
                        r = m
                center_left, center_right = min(l, center_left), max(r, center_right)
            self.segments[constraint] = [left, center_left, center_right, right]

    def run(self):
        self.segments = {}
        self.do_first_step()
        print(self.segments)
