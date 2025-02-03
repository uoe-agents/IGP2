import os
import setuptools

github_workspace = os.getenv("GITHUB_WORKSPACE")
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if github_workspace is not None:
    requirements.remove("carla==0.9.13")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='igp2',
                 version='0.3.1',
                 description='Open-source implementation of the goal recognition and motion planning algorithm IGP2 '
                             'from the paper: Interpretable Goal-based Prediction and Planning for Autonomous Driving',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author='Balint Gyevnar, Cillian Brewitt, Samuel Garcin, Massimiliano Tamborski, Stefano Albrecht',
                 author_email='balint.gyevnar@ed.ac.uk',
                 url='https://github.com/uoe-agents/IGP2',
                 packages=setuptools.find_packages(exclude=["tests", "scripts"]),
                 install_requires=requirements
                 )
