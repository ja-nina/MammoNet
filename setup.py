from setuptools import setup, find_packages

setup(
    name="MammoNet",
    version="0.1.0",
    description="Deep Learning for Breast Histology Classification",
    author="Nina Zukowska, Silvia Cambiago, Enise Irem Colak",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/mammonet",  # TODO: replace with link to github
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "scikit-learn",
        "Pillow",
        "numpy",
        "matplotlib",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
