import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='NoOffense',
    version='0.0.1',
    author='ertugrul',
    author_email='',
    description='Testing installation of Package',
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    url='https://github.com/ertugrul-dmr/NoOffense',
    project_urls = {
        "Bug Tracker": "https://github.com/ertugrul-dmr/NoOffense/issues"
    },
    license='Apache',
    packages=['nooffense'],
    #install_requires=['string', 're'],
)