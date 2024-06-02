from setuptools import setup, find_packages

setup(
    name='map_gym',
    version='0.0.1',
    install_requires=['gymnasium', 'numpy', 'matplotlib', 'stable-baselines3', 'torch', 'pygame'],
    packages=find_packages(),
)
