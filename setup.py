from setuptools import find_packages, setup


setup(
    name="trainfake",
    version="1.0.0",
    description="Cinematic AI training logs for demos, tests, teaching, and suspiciously serious terminals.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="TrainFake contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=6.0.0",
        "rich>=13.7.0",
    ],
    entry_points={
        "console_scripts": [
            "trainfake=simulate_train.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
