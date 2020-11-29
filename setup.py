import setuptools

with open("README.md", "r") as fh:
    long_description=fh.read()

setuptools.setup(
        name="mlvtk",
        version="1.0.0",
        author="tm-schwartz",
        author_email="tschwartz@csumb.edu",
        description="loss surface visualization tool",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/tm-schwartz/mlvtk",
        license='MIT',
        packages=setuptools.find_packages(),
        classifiers=[ 
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
        python_requires='>=3.6',
        install_requires=['tensorflow', 'pandas', 'plotly', 'sklearn', 'matplotlib', 'tqdm']
        )
