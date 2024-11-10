from setuptools import setup
import os


def find_stubs(package):
    stubs = []
    for root, dirs, files in os.walk(package):
        for file in files:
            path = os.path.join(root, file).replace(package + os.sep, '', 1)
            stubs.append(path)
    return dict(package=stubs)


setup(
    name='ryupy-stubs',
    maintainer="ryupy Developers",
    maintainer_email="example@python.org",
    description="PEP 561 type stubs for ryupy",
    version='1.0',
    packages=['ryupy-stubs'],
    # PEP 561 requires these
    install_requires=['ryupy'],
    package_data=find_stubs('ryupy-stubs'),
)