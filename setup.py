from setuptools import setup, find_packages

setup(
    name="safedrive",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "pyarrow",
        "openpyxl",
        "requests",
        "xgboost",
        "lightgbm",
        "catboost",
        "folium",
        "scikit-learn",
        "google-generativeai"
    ],
)
