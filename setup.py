import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='igp2',
                 version='0.1',
                 description='Reimplementation of IGP2 from the paper:'
                             ' Interpretable Goal-based Prediction and Planning for Autonomous Driving',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author='Cillian Brewitt, Balint Gyevnar, Stefano Albrecht',
                 author_email='cillian.brewitt@ed.ac.uk',
                 url='https://github.com/cbrewitt/igp2-dev',
                 packages=setuptools.find_packages(where="igp2"),
                 )
