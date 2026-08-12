from setuptools import find_packages, setup

package_name = 'perception_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leonard',
    maintainer_email='leonardkovac.compe@gmail.com',
    description='Offline perception dataset and evaluation tools for terrain-aware rover navigation.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'extract_images_from_bag = perception_tools.extract_images_from_bag:main',
        ],
    },
)
