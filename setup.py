from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.15.0", "scipy>=1.0.0", "joblib>=1.0.1", "scikit-learn>=0.22"]

setup(
    name="sanssouci",
    version="0.1.0",
    author="Laurent Risser and Pierre Neuvial",
    author_email="pierre.neuvial@math.univ-toulouse.fr",
    description="Post hoc inference via multiple testing",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/sanssouci.python/",
    download_url="https://github.com/pneuvial/sanssouci.python/archive/refs/tags/0.1.0.tar.gz",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
