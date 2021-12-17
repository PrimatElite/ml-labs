from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import time

from functools import reduce
from imutils import rotate_bound
from intelligent_placer_lib import intelligent_placer
from itertools import islice
from pymatting import blend
from shapely.geometry import MultiPolygon, Polygon
from typing import Dict, List, Optional, Tuple

from ..common import load_image, Object, PIXELS_PER_MM
from ..utils import get_config


DEFAULT_HEIGHT = 232.5
PIXELS_PER_CM = PIXELS_PER_MM * 10
SEGMENT_RATIO = 0.01
FIRST_TEST_CONSTRAINTS = ['same_obj_num', 'shooting_height', 'rotation', 'noise', 'blur']
SECOND_TEST_CONSTRAINTS = [
    ['noise', 'blur'],
    ['same_obj_num', 'shooting_height', 'rotation'],
    ['noise', 'shooting_height'],
]
X_SHIFT_CM = 3
POLYGON_SHIFT = 10


# - name: back_diff_obj


class Tester:
    background_image: np.ndarray
    config: Dict[str, List[float | int]]
    objects: List[Object]
    segments: Dict[str, List[float]]
    test_results: List[List[Tuple[bool, bool]]]
    times: List[Tuple[float, List[int], Dict[str, float | int], np.ndarray]]
    path_to_images: str
    path_to_polygons: str
    path_to_charts: str
    default_height: float

    def __init__(self, path_to_objects: str, path_to_config: str, path_to_images: str, path_to_polygons: str,
                 path_to_charts: str):
        self.path_to_images = path_to_images
        self.path_to_polygons = path_to_polygons
        self.path_to_charts = path_to_charts
        for path in [path_to_images, path_to_polygons, path_to_charts]:
            if not os.path.exists(path):
                os.mkdir(path)
        self.background_image = load_image('src/checker/background.png')
        config = get_config(path_to_config)
        self.config = {constraint['name']: constraint['value'] for constraint in config['restrictions']}
        self.config['shooting_height'] = [self.config['shooting_height'][0] * 10,
                                          self.config['shooting_height'][1] * 10]
        self.config['resolution'] = [self.config['resolution'][0] * 1e6, self.config['resolution'][1] * 1e6]
        self.objects = [Object(os.path.join(path_to_objects, file)) for file in os.listdir(path_to_objects)]
        self.segments = {}
        self.test_results = []
        self.times = []
        self.default_height = np.clip(DEFAULT_HEIGHT, self.config['shooting_height'][0],  # type: ignore
                                      self.config['shooting_height'][1])

    @classmethod
    def _find_max_polygon_edge(cls, polygon: np.ndarray) -> int:
        max_edge = -1
        max_edge_len = np.linalg.norm(polygon[max_edge + 1] - polygon[max_edge])
        for edge in range(polygon.shape[0] - 1):
            edge_len = np.linalg.norm(polygon[edge + 1] - polygon[edge])
            if edge_len > max_edge_len:
                max_edge = edge
                max_edge_len = edge_len
        return max_edge

    @classmethod
    def _combine_two_convex_hulls(cls, convex_hull1: np.ndarray, convex_hull2: np.ndarray) -> np.ndarray:
        points = list(zip(*MultiPolygon([Polygon(convex_hull1), Polygon(convex_hull2)]).convex_hull.boundary.xy))[:-1]
        return np.array(points[::-1])

    def _combine_two_polygons(self, polygon1: np.ndarray, max_edge1: int, polygon2: np.ndarray,
                              max_edge2: int) -> np.ndarray:
        vector1 = polygon1[max_edge1 + 1] - polygon1[max_edge1]
        vector2 = polygon2[max_edge2 + 1] - polygon2[max_edge2]
        vector1_len = np.linalg.norm(vector1)
        vector2_len = np.linalg.norm(vector2)
        normalized_vector1 = vector1 / vector1_len
        normalized_vector2 = vector2 / vector2_len
        angle = np.arccos(np.clip(np.dot(normalized_vector1, normalized_vector2), -1.0, 1.0))
        cross = np.cross(np.append(normalized_vector1, 0), np.append(normalized_vector2, 0))
        rotation_angle = np.pi - angle
        if cross[2] < 0:
            rotation_angle = -rotation_angle
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])

        vector1_mid = polygon1[max_edge1] + normalized_vector1 * vector1_len / 2
        rotated_vector2_mid = polygon2[max_edge2] + normalized_vector2 * vector2_len / 2

        prev_points = [vector1_mid + rotation_matrix.dot(polygon2[idx] - rotated_vector2_mid)
                       for idx in range(-1, max_edge2)]
        next_points = [vector1_mid + rotation_matrix.dot(polygon2[idx] - rotated_vector2_mid)
                       for idx in range(max_edge2 + 1, polygon2.shape[0] - 1)]
        points = (prev_points, [vector1_mid]) if len(prev_points) > 0 else ([vector1_mid],)
        points = (*points, next_points) if len(next_points) > 0 else points
        new_polygon2 = np.concatenate(points)

        center1 = np.mean(polygon1, 0)
        center2 = np.mean(new_polygon2, 0)
        translation_vector = center2 - center1
        translation_vector_mini = translation_vector / np.linalg.norm(translation_vector) * 3.0
        new_polygon2 += translation_vector_mini

        plt.fill(new_polygon2[:, 0], new_polygon2[:, 1], fill=False)

        return self._combine_two_convex_hulls(polygon1, new_polygon2)

    @classmethod
    def _generate_rectangle_for_object(cls, object_: Object) -> np.ndarray:
        min_x, min_y, max_x, max_y = object_.bounds
        w, h = max_x - min_x, max_y - min_y
        rectangle = np.array([[0, 0], [0, h + POLYGON_SHIFT], [w + POLYGON_SHIFT, h + POLYGON_SHIFT],
                              [w + POLYGON_SHIFT, 0]])
        object_convex_hull = np.array([p + [POLYGON_SHIFT // 2 - min_x, POLYGON_SHIFT // 2 - min_y]
                                       for p in object_.scaled_convex_hull])
        plt.fill(object_convex_hull[:, 0], object_convex_hull[:, 1], fill=False)
        plt.fill(rectangle[:, 0], rectangle[:, 1], edgecolor='b', fill=False)
        plt.gca().set_aspect('equal')
        print(rectangle)
        return rectangle

    def _generate_polygon(self, objects: List[Object], constraints: Dict[str, float | int]) -> np.ndarray:
        # - name: polygon_vertex_num
        # - name: polygon_angle
        # - name: area_ratio
        polygon = objects[0].scaled_convex_hull
        plt.fill(polygon[:, 0], polygon[:, 1], fill=False)
        for object_ in islice(objects, 1, len(objects)):
            max_edge = self._find_max_polygon_edge(polygon)
            object_polygon = object_.scaled_convex_hull
            max_object_edge = self._find_max_polygon_edge(object_polygon)
            polygon = self._combine_two_polygons(polygon, max_edge, object_polygon, max_object_edge)

        plt.fill(polygon[:, 0], polygon[:, 1], edgecolor='b', fill=False)
        plt.gca().set_aspect('equal')
        print(polygon)
        return polygon

    def _generate_image(self, objects: List[Object], constraints: Dict[str, float | int],
                        polygon: Optional[np.ndarray] = None) -> np.ndarray:
        # - name: max_dist_between_obj_center - максимальное расстояние от центра объектов до дальних объектов
        # - name: min_dist_between_obj_polygon
        # - name: line_width
        # - name: resolution
        # - name: aspect_ratio
        # - name: back_shadows
        # - name: obj_shadows
        # - name: camera_shift - сдвиг камеры относительно предметов
        # - name: shooting_angle - перспектива
        objects_images: List[np.ndarray] = []
        objects_alphas: List[np.ndarray] = []
        for i, object_ in enumerate(objects):
            x, y, w, h = cv2.boundingRect(object_.convex_hull)
            objects_images.append(object_.image[y:y + h, x:x + w])
            objects_alphas.append(object_.alpha[y:y + h, x:x + w])

        scale = 1
        if 'shooting_height' in constraints:
            scale = DEFAULT_HEIGHT / constraints['shooting_height']
        elif abs(self.default_height - DEFAULT_HEIGHT) > 1e-9:
            scale = DEFAULT_HEIGHT / self.default_height
        if abs(scale - 1) > 1e-9:
            for i in range(len(objects)):
                h, w = round(objects_images[i].shape[0] * scale), round(objects_images[i].shape[1] * scale)
                objects_images[i] = cv2.resize(objects_images[i], (w, h))
                objects_alphas[i] = cv2.resize(objects_alphas[i], (w, h))

        if 'rotation' in constraints:
            for i in range(len(objects)):
                objects_images[i] = rotate_bound(objects_images[i], constraints['rotation'])
                objects_alphas[i] = rotate_bound(objects_alphas[i], constraints['rotation'])

        x_shift = round(min(self.config['min_dist_between_obj'][0], X_SHIFT_CM * 10) * PIXELS_PER_MM * scale)
        height = max(objects_images, key=lambda o: o.shape[0]).shape[0] + x_shift * 2
        width = sum(o.shape[1] for o in objects_images) + round(x_shift * (len(objects) + 1))
        objects_image = np.zeros((height, width, 3), np.uint8)
        objects_alpha = np.zeros((height, width), np.uint8)
        x = x_shift
        y = x_shift
        for i in range(len(objects)):
            objects_image[y:objects_images[i].shape[0] + y, x:x + objects_images[i].shape[1]] = objects_images[i]
            objects_alpha[y:objects_alphas[i].shape[0] + y, x:x + objects_alphas[i].shape[1]] = objects_alphas[i]
            x += objects_images[i].shape[1] + x_shift
        foreground_height = height
        foreground = np.zeros((foreground_height, width, 3), np.uint8)
        alpha = np.zeros((foreground_height, width), np.uint8)
        y_shift = (foreground_height - height) // 2
        foreground[y_shift:y_shift + height] = objects_image
        alpha[y_shift:y_shift + height] = objects_alpha
        background = cv2.resize(self.background_image, (width, foreground_height))
        image = blend(foreground, background, alpha / 255.0).astype(np.uint8)

        if 'noise' in constraints:
            row, col, ch = image.shape
            gauss = np.random.normal(0, constraints['noise'], (row, col, ch))
            gauss = gauss.reshape((row, col, ch))
            image = np.clip(np.rint(image + gauss), 0, 255).astype(np.uint8)

        if 'blur' in constraints and constraints['blur'] > 1e-9:
            image = cv2.GaussianBlur(image, (21, 21), constraints['blur'])

        # cv2.imshow('win', image)
        # cv2.waitKey()
        return image

    @classmethod
    def _call_placer(cls, image_path: str, polygon: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        start = time.time()
        res = intelligent_placer.check_image(image_path, polygon)
        end = time.time()
        res_time = end - start
        return res, res_time

    @classmethod
    def _get_constraints_image_path(cls, root: str, constraints: Dict[str, float | int],
                                    case: Optional[int] = None) -> str:
        if case is None:
            constraints_str = '_'.join('%s_%.4f' % (name, value) for name, value in constraints.items())
            return os.path.join(root, f'{constraints_str}.png')
        else:
            constraints_str = '_'.join('%.4f' % value for value in constraints.values())
            return os.path.join(root, f'{case}_{constraints_str}.png')

    def _save_image(self, image: np.ndarray, constraints: Dict[str, float | int], case: Optional[int] = None) -> str:
        image_path = self._get_constraints_image_path(self.path_to_images, constraints, case)
        cv2.imwrite(image_path, image)
        return image_path

    def _save_polygon(self, constraints: Dict[str, float | int], case: Optional[int] = None):
        plt.savefig(self._get_constraints_image_path(self.path_to_polygons, constraints, case))
        plt.clf()

    def _do_first_step(self):
        for constraint in FIRST_TEST_CONSTRAINTS:
            left_value = self.config[constraint][0]
            right_value = self.config[constraint][1]
            left, center_left, center_right, right = left_value, right_value, left_value, right_value
            if abs(right_value - left_value) < 1e-9:
                self.segments[constraint] = [left, left, right, right]
                continue

            for i, object_ in enumerate(self.objects):
                if constraint == 'same_obj_num':
                    left_objects = [object_] * left_value
                    left_objects_idx = [i] * left_value
                    right_objects = [object_] * right_value
                    right_objects_idx = [i] * right_value
                    left_good_polygon = self._generate_polygon(left_objects, {})
                    self._save_polygon({'object': i, constraint: left_value})
                    right_good_polygon = self._generate_polygon(right_objects, {})
                    self._save_polygon({'object': i, constraint: right_value})
                else:
                    left_objects = right_objects = [object_]
                    left_objects_idx = right_objects_idx = [i]
                    good_polygon = self._generate_rectangle_for_object(object_)
                    self._save_polygon({'object': i, constraint: left_value})
                    left_good_polygon = right_good_polygon = good_polygon

                segment_end_len = (right_value - left_value) * SEGMENT_RATIO
                l, r = left_value, right_value

                left_image = self._generate_image(left_objects, {constraint: l})
                left_image_path = self._save_image(left_image, {'object': i, constraint: l})
                left_good_result, left_good_result_time = self._call_placer(left_image_path, left_good_polygon)
                self.times.append((left_good_result_time, left_objects_idx, {constraint: l}, left_good_polygon))
                if not left_good_result:
                    center_left, center_right = left_value, left_value
                    break

                right_image = self._generate_image(right_objects, {constraint: r})
                right_image_path = self._save_image(right_image, {'object': i, constraint: r})
                right_good_result, right_good_result_time = self._call_placer(right_image_path, right_good_polygon)
                self.times.append((right_good_result_time, right_objects_idx, {constraint: r}, right_good_polygon))
                if right_good_result:
                    center_right = right_value if abs(center_right - left_value) < 1e-9 else center_right
                    continue

                while r - l > segment_end_len:
                    m = (l + r) / 2
                    if constraint == 'same_obj_num':
                        objects = [object_] * m
                        objects_idx = [i] * m
                        good_polygon = self._generate_polygon(objects, {})
                    else:
                        objects = [object_]
                        objects_idx = [i]
                        good_polygon = self._generate_rectangle_for_object(object_)
                    self._save_polygon({'object': i, constraint: m})

                    m_image = self._generate_image(objects, {constraint: m})
                    image_path = self._save_image(m_image, {'object': i, constraint: m})
                    m_result, m_result_time = self._call_placer(image_path, good_polygon)
                    self.times.append((m_result_time, objects_idx, {constraint: m}, good_polygon))

                    if m_result:
                        l = m
                    else:
                        r = m

                center_left, center_right = min(l, center_left), max(r, center_right)
            self.segments[constraint] = [left, center_left, center_right, right]

    def _do_second_step_test(self, constraints: Dict[str, float | int], case: int) -> Tuple[bool, bool]:
        if 'same_obj_num' in constraints:
            same_obj_num = round(constraints['same_obj_num'])
            objects = self.objects[:-1] + [self.objects[-1]] * same_obj_num
            objects_idx = list(range(len(self.objects) - 1)) + [len(self.objects) - 1] * same_obj_num
        else:
            objects = self.objects
            objects_idx = list(range(len(self.objects)))
        if self.config['obj_num'][1] < len(objects):
            objects = objects[-self.config['obj_num'][1]:]
            objects_idx = objects_idx[-self.config['obj_num'][1]:]

        min_length = min(object_.min_length for object_ in self.objects) - POLYGON_SHIFT
        good_polygon = self._generate_polygon(objects, {})
        self._save_polygon(constraints, case)
        w = Polygon(good_polygon).area / min_length
        bad_polygon = np.array([[0, 0], [0, min_length], [w, min_length], [w, 0]])

        image = self._generate_image(objects, constraints)
        image_path = self._save_image(image, constraints, case)
        good_result, good_result_time = self._call_placer(image_path, good_polygon)
        bad_result, bad_result_time = self._call_placer(image_path, bad_polygon)
        self.times.append((good_result_time, objects_idx, constraints, good_polygon))
        self.times.append((bad_result_time, objects_idx, constraints, bad_polygon))
        return good_result is True, bad_result is False

    def _do_second_step(self):
        for case, constraints in enumerate(SECOND_TEST_CONSTRAINTS):
            left_results = self._do_second_step_test({constraint: self.segments[constraint][0]
                                                      for constraint in constraints}, case)

            if all(abs(self.segments[constraint][1] - self.segments[constraint][0]) < 1e-9
                   for constraint in constraints):
                left_middle_results = left_results
                center_left_results = left_results
            else:
                left_middle_results = self._do_second_step_test({constraint: (self.segments[constraint][0]
                                                                              + self.segments[constraint][1]) / 2
                                                                 for constraint in constraints}, case)
                center_left_results = self._do_second_step_test({constraint: self.segments[constraint][1]
                                                                 for constraint in constraints}, case)

            if all(abs(self.segments[constraint][2] - self.segments[constraint][1]) < 1e-9
                   for constraint in constraints):
                center_results = center_left_results
                center_right_results = center_left_results
            else:
                center_results = self._do_second_step_test({constraint: (self.segments[constraint][1]
                                                                         + self.segments[constraint][2]) / 2
                                                            for constraint in constraints}, case)
                center_right_results = self._do_second_step_test({constraint: self.segments[constraint][2]
                                                                  for constraint in constraints}, case)

            if all(abs(self.segments[constraint][3] - self.segments[constraint][2]) < 1e-9
                   for constraint in constraints):
                right_middle_results = center_right_results
                right_results = center_right_results
            else:
                right_middle_results = self._do_second_step_test({constraint: (self.segments[constraint][2]
                                                                               + self.segments[constraint][3]) / 2
                                                                  for constraint in constraints}, case)
                right_results = self._do_second_step_test({constraint: self.segments[constraint][3]
                                                           for constraint in constraints}, case)

            self.test_results.append([left_results, left_middle_results, center_left_results, center_results,
                                      center_right_results, right_middle_results, right_results])

    def run(self):
        self.segments = {}
        self.test_results = []
        self.times = []
        self._do_first_step()
        print(self.segments)
        self._do_second_step()
        print(self.test_results)
        print([times[:2] for times in self.times], sep='\n')

    def plot(self):
        answers = ['right answer', 'undefined', 'wrong answer']
        segments_df = pd.DataFrame(columns=['constraint', 'answer', 'percentage'])
        for constraint, values in self.segments.items():
            if abs(values[0] - values[3]) < 1e-9:
                continue
            if abs(values[0] - values[1]) < 1e-9:
                segments_df = segments_df.append({'constraint': constraint, 'answer': answers[2], 'percentage': 100},
                                                 ignore_index=True)
            elif abs(values[2] - values[3]) < 1e-9:
                segments_df = segments_df.append({'constraint': constraint, 'answer': answers[0], 'percentage': 100},
                                                 ignore_index=True)
            else:
                segment_len = abs(values[3] - values[0]) / 100
                segments_df = segments_df.append({'constraint': constraint, 'answer': answers[0],
                                                  'percentage': abs(values[1] - values[0]) / segment_len},
                                                 ignore_index=True)
                segments_df = segments_df.append({'constraint': constraint, 'answer': answers[1],
                                                  'percentage': abs(values[2] - values[1]) / segment_len},
                                                 ignore_index=True)
                segments_df = segments_df.append({'constraint': constraint, 'answer': answers[2],
                                                  'percentage': abs(values[3] - values[2]) / segment_len},
                                                 ignore_index=True)
        colors = {answers[0]: 'green', answers[1]: 'grey', answers[2]: 'red'}
        fig = px.bar(segments_df, x='constraint', y='percentage', color='answer', color_discrete_map=colors)
        fig.write_html(os.path.join(self.path_to_charts, 'first_step.html'))

        parts = ['left', 'center', 'right']
        tests_df = pd.DataFrame(columns=['case', 'part', 'percentage'])
        for case_idx, values in enumerate(self.test_results):
            values = reduce(lambda x, y: x + y, values)
            left = sum(values[:4]) * 25
            center = sum(values[4:10]) * 100 / 6
            right = sum(values[10:]) * 25
            case = ', '.join(SECOND_TEST_CONSTRAINTS[case_idx])
            tests_df = tests_df.append({'case': case, 'part': parts[0], 'percentage': left}, ignore_index=True)
            tests_df = tests_df.append({'case': case, 'part': parts[1], 'percentage': center}, ignore_index=True)
            tests_df = tests_df.append({'case': case, 'part': parts[2], 'percentage': right}, ignore_index=True)
        fig = px.bar(tests_df, x='case', y='percentage', color='part')
        fig.write_html(os.path.join(self.path_to_charts, 'second_step.html'))
