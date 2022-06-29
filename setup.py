from setuptools import setup

setup(name='karolos',
      version='0.3',
      description='Open-Source Robot-Task Learning Simulation Framework',
      url='https://github.com/tmdt-buw/karolos',
      author='Christian Bitter',
      author_email='bitter@uni-wuppertal.de',
      license='MIT',
      packages=['karolos'],
      install_requires=[
          "numpy>=1.19.5",
          "torch>=1.11.0",
          "tensorboard>=2.4.1",
          "tqdm>=4.56.0",
          "gym>=0.18.0",
          "pybullet>=3.0.8",
          "scipy>=1.6.0",
          "klampt",
      ],
      zip_safe=False)
