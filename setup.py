from setuptools import setup, find_packages

setup(
    name='multiclass_wp_fixer',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'plotly',
        'torch',
    ],
    author='Ignacio Serrano',
    author_email='ignaciosll96@gmail.com',
    description='This repository is designed to handle and fix the decision threshold or working point in a multiclass classification problem.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/isllabres/Multiclass-Working-Point-Fixer',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.11',
)
