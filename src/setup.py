from distutils.core import setup
from setuptools import find_packages

setup(
    name='RLmbpo',
    packages=find_packages(),
    version='0.0.1',
    description='P2P',
    long_description=open('./README.md').read(),
    author='Zifan Wu',
    entry_points={
        'console_scripts': (
            'mbpo=softlearning.scripts.console_scripts:main',
            'viskit=mbpo.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)