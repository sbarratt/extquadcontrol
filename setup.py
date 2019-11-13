from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="extquadcontrol",
    version="0.1",
    description="Extended quadratic control.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    license="Apache License, Version 2.0",
    url="https://github.com/sbarratt/extquadcontrol",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
