"""
Setup script for NMT project.
"""
from setuptools import setup, find_packages

setup(
    name='nmt-transformer',
    version='1.0.0',
    description='Neural Machine Translation using Transformer',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'sentencepiece>=0.1.99',
        'sacrebleu>=2.3.1',
        'datasets>=2.14.0',
        'tokenizers>=0.13.3',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'tqdm>=4.65.0',
        'pyyaml>=6.0',
        'tensorboard>=2.13.0',
        'flask>=2.3.0',
        'flask-cors>=4.0.0',
        'nltk>=3.8.1',
        'scikit-learn>=1.3.0',
    ],
    python_requires='>=3.8',
)

