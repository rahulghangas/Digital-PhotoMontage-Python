from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('photoMontage3.pyx', language_level = '3', annotate=True))
setup(ext_modules=cythonize('autoMontage.pyx'  , language_level = '3', annotate=True))

