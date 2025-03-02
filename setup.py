from setuptools import setup, find_packages

setup(
    name='music-generator',
    version='0.1.0',
    packages=find_packages(where='music_generator'),
    package_dir={'': 'music_generator'},
    install_requires=[
        'tensorflow>=2.12.0',
        'numpy>=1.22.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.2.0',
        'matplotlib>=3.7.0',
        'music21>=8.0.0',
        'pretty_midi>=0.2.9',
        'librosa>=0.9.0'
    ],
    author='Your Name',
    description='AI Music Generation Project using RNN and TensorFlow',
    python_requires='>=3.10',
)