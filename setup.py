"""
Module configuration.
"""

from setuptools import setup

setup(
    name='relation-networks',
    version='0.0.1',
    description='Simple relation network',
    long_description='Simple relation network',
    url='https://github.com/matwilso/relation-network',
    author='Matthew Wilson',
    author_email='mattwilsonmbw@gmail.com',
    license='MIT',
    keywords='ai machine learning',
    packages=['relation_network'],
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
