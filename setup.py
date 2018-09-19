from setuptools import setup
from setuptools import find_packages

long_description = '''unet models for image colorization'''

setup(name='image_colorization',
      version='1.0.0',
      description='unet models for image colorization',
      long_description=long_description,
      author='Maximilian Jeblick',
      author_email='NA',
      url='https://gitlab.com/sk8maxe/tgs_salt',
      download_url='https://gitlab.com/sk8maxe/',
      license='MIT',
      install_requires=[],
      extras_require={
      },
      classifiers=[
      ],
      packages=find_packages())
