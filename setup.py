'''
Created on Aug 9, 2013

@author: kirian
'''

from distutils.core import setup

setup(name='pydiffract',
      version='0.0.1',
      author="Richard A. Kirian",
      author_email="rkirian@gmail.com",
      description='Diffraction analysis tools',
      packages=["pydiffract","pydiffract.viewers"],
      package_dir={"pydiffract": "pydiffract"})
