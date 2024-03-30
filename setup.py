from setuptools import setup, find_packages

setup(
    name = 'torchstudio',
    packages = find_packages(),
    version = '0.0.0a',
    license='GPL',
    description = 'torchstudio',
    author = 'JiauZhang',
    author_email = 'jiauzhang@163.com',
    url = 'https://github.com/JiauZhang/torchstudio',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type = 'text/markdown',
    keywords = [
        'PyTorch',
    ],
    install_requires=[
        'fakeset',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3.8',
    ],
)