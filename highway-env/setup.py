#import os
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="highway_env",
    ext_modules = cythonize(["highway_env/road/lane.pyx",
                             "highway_env/road/road.pyx",
                              #"highway_env/vehicle/kinematics.py",
                             "highway_env/vehicle/controller.pyx",
                             "highway_env/utils.pyx"],
                             annotate=True)
)

