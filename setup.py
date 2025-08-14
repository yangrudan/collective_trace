"""
Package setup.py for collective-trace
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="collective-trace",
    version="0.1.1",
    author="Cookie Yang",
    author_email="yangrudan@zhejianglab.org",
    description="A monkey-patching tool for tracing collective operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangrudan/collective_trace",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Pytest",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
)
