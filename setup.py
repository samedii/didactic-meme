from setuptools import setup

setup(
   name='didactic-meme',
   version='0.1',
   description='modelling toolbox',
   author='Richard Löwenström',
   author_email='samedii@github.com',
   packages=['didactic_meme'],
   install_requires=['torch', 'dash', 'flask'],
)
