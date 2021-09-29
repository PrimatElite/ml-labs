from setuptools import setup

setup(
    name='intelligent_placer_lib',
    version='0.0.1',
    packages=['intelligent_placer_lib', 'intelligent_placer_lib.common', 'intelligent_placer_lib.placer'],
    package_dir={'intelligent_placer_lib': 'src'},
    install_requires=['kedro', 'numpy', 'opencv-python'],
    setup_requires=['kedro', 'numpy', 'opencv-python']
)
