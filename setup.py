"""
Module configuration.
"""

from setuptools import setup

setup(
    name='rns',
    version='0.0.1',
    description='Reptile for reinforcement meta-learning',
    long_description='Reptile for reinforcement meta-learning',
    url='https://github.com/matwilso/relation-networks',
    author='Matthew Wilson',
    author_email='mattwilsonmbw@gmail.com',
    license='MIT',
    keywords='ai machine learning',
    packages=['rns'],
    install_requires=[
        'numpy>=1.0.0,<2.0.0',
        'Pillow>=4.0.0,<5.0.0',
        'gym',
    ],
    extras_require={
        "tf": ["tensorflow>=1.0.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0.0"],
    }
)
