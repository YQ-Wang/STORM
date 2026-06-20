import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

version = os.environ.get("STORM_VERSION", "1.0.0").removeprefix("v")

setup(
    name="storm-omics",
    version=version,
    description="Statistical Test for spatial patterns using k-nearest neighbors (STORM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("test", "test.*")),
    author="Yiqing Wang, Jinpu Li",
    author_email="yqw@wangemail.com, lijinp@health.missouri.edu",
    license="GPL-3.0-only",
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.24.4",
        "pandas>=1.5",
        "scipy>=1.10.1",
        "scikit-learn>=1.3.2",
    ],
    extras_require={
        "gpu": ["torch>=2.0"],
    },
    python_requires=">=3.10",
    project_urls={
        "Equivalent R implementation": "https://github.com/CastleLi/STORM",
    },
)
