from enum import Enum

from pathlib import Path

import numpy as np
import pyrender as pr
import trimesh as tr
import random

from PIL import Image


class ShapeLabels(Enum):
    SPHERE = 'sphere'
    CUBE = 'cube'
    TETRAHEDRON = 'tetrahedron'

    def __str__(self):
        return self.value


def rotate_to_origin(position):
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = position

    return pose

def spherical_to_cartesian(radius: float, angles: tuple[float, float]) -> tuple[float, float, float]:
    return (radius * np.sin(angles[0]) * np.cos(angles[1]),
            radius * np.sin(angles[0]) * np.sin(angles[1]),
            radius * np.cos(angles[0]))

def generate_coordinates(bounds: tuple[float, float]) -> np.ndarray:
    angles = (random.uniform(0, np.pi * 0.8), random.uniform(0, np.pi * 0.8))
    xyz = spherical_to_cartesian(random.uniform(bounds[0], bounds[1]), angles)
    return rotate_to_origin(np.asarray(xyz))

def label_to_mesh(label: ShapeLabels) -> pr.Mesh:
    match label:
        case ShapeLabels.SPHERE:
            return pr.Mesh.from_trimesh(tr.creation.icosphere(subdivisions=4, radius=1.0))
        case ShapeLabels.CUBE:
            return pr.Mesh.from_trimesh(tr.creation.box(extents=[2 ** 0.5 for _ in range(3)]))
        case ShapeLabels.TETRAHEDRON:
            vertices = np.array([
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0]
            ])
            faces = np.array([
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2]
            ])
            return pr.Mesh.from_trimesh(tr.creation.Trimesh(vertices=vertices, faces=faces))
        case _:
            raise ValueError("Unknown mesh label")


class DatasetGenerator:
    def __init__(self, shape_label: ShapeLabels, save_path=''):
        self.save_path = save_path

        self.__scene = pr.Scene(bg_color=[0, 0, 0])
        self.__camera = self.__scene.add(pr.PerspectiveCamera(yfov=np.pi / 3.0))
        self.__light = self.__scene.add(pr.DirectionalLight(color=np.ones(3), intensity=10.0))
        self.__shape = self.__scene.add(label_to_mesh(shape_label))

        self.renderer = pr.OffscreenRenderer(256, 256)

        self.__label = shape_label

    def set_shape(self, label: ShapeLabels) -> None:
        self.__label = label
        self.__shape.mesh = label_to_mesh(label)

    def set_positions(self) -> None:
        bounds = (3, 7)
        self.__camera.matrix = generate_coordinates(bounds)
        self.__light.matrix = generate_coordinates(bounds)
        self.__shape.matrix = generate_coordinates((0, bounds[0] - 2))

    def generate_samples(self, number: int) -> None:
        Path(f'{self.save_path}/{self.__label}').mkdir(parents=True, exist_ok=True)

        for index in range(number):
            self.set_positions()
            color, depth = self.renderer.render(self.__scene)
            img = Image.fromarray(color)
            img.save(f'{self.save_path}/{self.__label}/{self.__label}_{index}.png')
