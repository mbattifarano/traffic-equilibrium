import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy


SOURCE_DIR = 'src'
REQUIREMENTS_DIR = 'requirements'
PACKAGE = 'traffic_equilibrium'


def requirements(fname):
    with open(fname) as fp:
        return fp.read().splitlines()


ext_modules = [
    Extension(
        '.'.join([PACKAGE, '*']),
        [os.path.join(SOURCE_DIR, PACKAGE, f'*.pyx')],
        libraries=['igraph', 'm', 'dang'],
        library_dirs=[
            '/usr/local/lib'
        ],
        include_dirs=[
            '/usr/local/include/igraph',
            '/usr/local/include',
            numpy.get_include()
        ],
        extra_compile_args=[
            "-Ofast",
            "-march=native",
            "-fopenmp",
            "-ftree-vectorize",
            #"-axCOMMON-AVX512"
        ],
        extra_link_args=["-Ofast", "-fopenmp"],
    )
]


setup(
    name="traffic_equilibrium",
    package_dir={'': SOURCE_DIR},
    packages=find_packages(SOURCE_DIR),
    setup_requires=[
        "setuptools>=18.0",
        "cython"
    ],
    install_requires=requirements(os.path.join(REQUIREMENTS_DIR, 'install.txt')),
    tests_require=requirements(os.path.join(REQUIREMENTS_DIR, 'develop.txt')),
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
    ),
    zip_safe=False,
)

