from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.19.4"]

setup(
    name="sanssouci",
    version="0.1.1",
    author="Laurent Risser and Pierre Neuvial",
    author_email="pierre.neuvial@math.univ-toulouse.fr",
    description="Post hoc inference via multiple testing",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/sanssouci.python/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
