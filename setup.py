"""
Setup configuration for SPXQuery package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spxquery",
    version="0.1.0",
    author="SPXQuery Team",
    description="SPHEREx time-domain analysis package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spxquery",
    packages=find_packages(where='..'),
    package_dir={'': '..'},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "spxquery=spxquery.core.pipeline:run_pipeline",
        ],
    },
)