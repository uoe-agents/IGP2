import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='igp2',
                 version='0.2.0',
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
