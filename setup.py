from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.25",
    "scipy>=1.6",
    "joblib>=1.2.0",
    "scikit-learn>=1.5"
]

setup(
    name="sanssouci",
    version="0.1.3",
    author="Alexandre Blain, Nicolas Enjalbert Courrech, Pierre Neuvial, Nils Peyrouset, Laurent Risser, Bertrand Thirion",
    author_email="pierre.neuvial@math.univ-toulouse.fr",
    description="Post hoc inference via multiple testing",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/pneuvial/sanssouci.python/",
    download_url="https://github.com/pneuvial/sanssouci.python/archive/refs/tags/0.1.3.tar.gz",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
