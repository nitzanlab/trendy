from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='trendy',
    description='Learning representations of reaction diffusion models',
    version='0.1.0',
    packages=find_packages(),
    install_requires=required,
    python_requires='>=3.8',
)
