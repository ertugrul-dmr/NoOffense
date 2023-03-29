import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='no-offense',
    version='0.0.3',
    author='ertugrul',
    author_email='',
    description='A smart and fast offensive language detection tool for Turkish',
    url='https://github.com/ertugrul-dmr/NoOffense',
    project_urls = {
        "Bug Tracker": "https://github.com/ertugrul-dmr/NoOffense/issues"
    },
    license='Apache',
    packages=['nooffense']
)