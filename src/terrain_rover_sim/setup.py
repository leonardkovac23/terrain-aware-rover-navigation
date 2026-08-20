import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'terrain_rover_sim'


def collect_model_files(model_root):
    data_files = []

    for root, _, files in os.walk(model_root):
        if not files:
            continue

        install_dir = os.path.join('share', package_name, root)
        source_files = [os.path.join(root, file_name) for file_name in files]
        data_files.append((install_dir, source_files))

    return data_files

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'LICENSE']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ] + collect_model_files('models/leo_models') + collect_model_files('models/world_models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonard',
    maintainer_email='leonardkovac.compe@gmail.com',
    description='Simulation, robot description, Gazebo world, and launch files for the terrain rover project',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ground_truth_node = terrain_rover_sim.ground_truth_node:main',
        ],
    },
)
