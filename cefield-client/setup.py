from setuptools import setup, find_packages

setup(
    name="cefield-client",
    version="0.1.0",
    description="Edge client for the CEFIELD Global Resonator Genome",
    author="Carsten Ehlers",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "httpx>=0.24.0",
        "pandas>=2.0.0",
        "click>=8.0.0"
    ],
    entry_points={
        "console_scripts": [
            "cefield=cefield.cli:main",
        ],
    },
)
