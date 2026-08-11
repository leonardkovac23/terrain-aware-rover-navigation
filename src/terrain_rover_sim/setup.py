import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'terrain_rover_sim'

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
        (os.path.join('share', package_name, 'models', 'leo_models'), glob('models/leo_models/*')),
        (os.path.join('share', package_name, 'models', 'world_models'), glob('models/world_models/*')),
    ],
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
