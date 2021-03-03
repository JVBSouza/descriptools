from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.rst")) as f:
    long_description = f.read()

setup(
    name="descriptools",
    packages=["descriptools"],
    version="0.0.1",
    license="MIT",
    description=
    "A gpu-based toolbox for terrain descriptor calculation/delineation",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    mantainers="JVBSouza, JHBoing",
    url="https://github.com/JVBSouza/descriptools",
    download_url="https://github.com/JVBSouza/descriptools/archive/master.zip",
    keywords=["hidrology", "gis", "flood"],
    install_requires=[
        "numpy",
        "numba",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)