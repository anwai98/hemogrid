from setuptools import setup, find_packages

setup(
    name="hemogrid",
    version="0.0.1",
    description="Grid-aware cell counting for hemocytometer images",
    author="Anwai Archit",
    author_email="anwai.archit@gmail.com",
    url="https://github.com/anwai98/hemogrid",
    packages=find_packages(include=["hemogrid", "hemogrid.*"]),
    python_requires=">=3.11",
    install_requires=["numpy", "scipy", "scikit-image", "tifffile", "pillow", "rich", "napari[all]"],
    entry_points={"console_scripts": ["hemogrid=hemogrid.cli:main"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
    ],
)
