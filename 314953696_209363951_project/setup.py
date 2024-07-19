from setuptools import setup, Extension

module = Extension('mysymnmf', sources=['symnmf.c','symnmfmodule.c'])

setup(
    name='mysymnmf',
    version='1.0',
    description='SymNMF extension module',
    ext_modules=[module]
)
