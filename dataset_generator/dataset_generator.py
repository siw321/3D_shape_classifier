from enum import Enum

from pathlib import Path

import numpy as np
import pyrender as pr
import trimesh as tr
import random
import json

from PIL import Image


class ShapeLabels(Enum):
    SPHERE = 'sphere'
    CUBE = 'cube'
    TETRAHEDRON = 'tetrahedron'

    def __str__(self):
        return self.value


def rotate_to_origin(position: np.ndarray) -> np.ndarray:
    """Rotate cartesian coordinates to origin.

    Args:
        position (np.array): Cartesian coordinates

    Returns:
        pose (np.array): 4x4 matrix with coordinates and rotations
    """
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
    """Convert spherical coordinates to cartesian coordinates."""
    return (radius * np.sin(angles[0]) * np.cos(angles[1]),
            radius * np.sin(angles[0]) * np.sin(angles[1]),
            radius * np.cos(angles[0]))


def generate_coordinates(bounds: tuple[float, float]) -> np.ndarray:
    """Generate random position oriented toward origin

    Args:
        bounds (tuple[float, float]): distance to the origin

    Returns:
        (np.array): 4x4 matrix with coordinates and rotations
    """
    angles = (random.uniform(0, np.pi * 0.9), random.uniform(0, np.pi * 0.9))
    xyz = spherical_to_cartesian(random.uniform(bounds[0], bounds[1]), angles)
    return rotate_to_origin(np.asarray(xyz))

def label_to_mesh(label: ShapeLabels | None) -> pr.Mesh:
    """Create pyrender mesh from label.

    Args:
        label (ShapeLabels | None): 3D shape label

    Returns:
        (pr.Mesh): pyrender mesh. Empty mesh if label is None

    Raises:
        ValueError: if label is unknown
    """
    material = pr.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.5
    )

    match label:
        case ShapeLabels.SPHERE:
            return pr.Mesh.from_trimesh(tr.creation.icosphere(subdivisions=4, radius=1.0), material=material)
        case ShapeLabels.CUBE:
            return pr.Mesh.from_trimesh(tr.creation.box(extents=[2 ** 0.5 for _ in range(3)]), material=material)
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
            return pr.Mesh.from_trimesh(tr.creation.Trimesh(vertices=vertices, faces=faces, material=material))
        case None:
            return pr.Mesh.from_trimesh(tr.Trimesh())
        case _:
            raise ValueError("Unknown mesh label")


class DatasetGenerator:
    def __init__(self, config='dataset_generator/dsg_config.json',
                 save_path: str | None = None, generate_immediately: bool = False) -> None:
        """Initialize new dataset generator.

        Args:
            config (str): path to the .json config file
            save_path (str | None): custom output path for the dataset.
                If none, default path from config is used instead
            generate_immediately (bool): if true, runs generate method after initialization
        """
        self.__config = json.load(open(config, 'r'))
        self.save_path = save_path
        self.__label = None

        self.__scene = pr.Scene(
            bg_color=np.ones((3, 1)) * self.__config['background_brightness'],
            ambient_light=np.ones((3, 1)) * 0.03
        )
        self.__light = self.__scene.add(pr.DirectionalLight(
            color=np.ones(3),
            intensity=self.__config['light_brightness'])
        )

        self.__camera = self.__scene.add(pr.PerspectiveCamera(yfov=np.pi / 3.0))
        self.__shape = self.__scene.add(label_to_mesh(None))
        self.__renderer = pr.OffscreenRenderer(*self.__config['image_size'])
        if generate_immediately:
            self.generate()

    def generate(self) -> None:
        """Generate dataset"""
        if not self.save_path:
            self.save_path = self.__config['default_output_directory']

        for item in self.__config['classes']:
            self.__label = ShapeLabels(item)
            self.__shape.mesh = label_to_mesh(self.__label)
            self.__generate_samples(self.__config['samples_per_class'])


    def __randomize_positions(self) -> None:
        """Set random positions for camera, light and shape"""
        self.__camera.matrix = generate_coordinates(self.__config['camera_distance'])
        self.__light.matrix = generate_coordinates(self.__config['camera_distance'])
        self.__shape.matrix = generate_coordinates((0, self.__config['camera_distance'][0] - 2))

    def __generate_samples(self, number: int) -> None:
        """Generate given number of images of the shape that is currently at the scene"""
        Path(f'{self.save_path}/train/{self.__label}').mkdir(parents=True, exist_ok=True)
        test_samples = round(number * self.__config['train_test_split'])

        for index in range(number - test_samples):
            self.__randomize_positions()
            color, depth = self.__renderer.render(self.__scene)
            img = Image.fromarray(color)
            img.save(f'{self.save_path}/train/{self.__label}/{self.__label}_{index}.png')

        if test_samples != 0:
            Path(f'{self.save_path}/test/{self.__label}').mkdir(parents=True, exist_ok=True)
        for index in range(test_samples):
            self.__randomize_positions()
            color, depth = self.__renderer.render(self.__scene)
            img = Image.fromarray(color)
            img.save(f'{self.save_path}/test/{self.__label}/{self.__label}_{index}.png')
