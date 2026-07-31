from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'ybimu_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shreyas',
    maintainer_email='shreyaspatel517@gmail.com',
    description='ROS 2 serial driver for the YaBoom 10-axis IMU',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ybimu_driver = ybimu_ros2.driver:main',
        ],
    },
)
