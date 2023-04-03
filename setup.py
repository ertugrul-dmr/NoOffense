import setuptools

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setuptools.setup(
    name='no-offense',
    version='0.0.3',
    author='ertugrul',
    author_email='',
    description='A smart and fast offensive language detection tool for Turkish',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/ertugrul-dmr/NoOffense',
    project_urls = {
        "Bug Tracker": "https://github.com/ertugrul-dmr/NoOffense/issues"
    },
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        "torch>=2.0.0"
        "transformers>=4.6.0,<5.0.0"
        "pandas"
        "tqdm"
        "numpy"
        "sentencepiece",
        "huggingface-hub>=0.4.0"

    ],
)