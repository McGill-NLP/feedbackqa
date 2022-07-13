import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="feedbackqa-app",
    version="0.1.0dev0",
    author="Zichao Li, Prakhar Sharma, Xing Han Lu, Jackie C.K. Cheung, Siva Reddy",
    author_email="",
    description="The web app and UI for FeedbackQA.",
    # TODO: Replace this with "long description"
    long_description="Please reach out to the author for more information.",
    long_description_content_type="text/markdown",
    # url="https://github.com/xhlulu/fqa-web-app",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "jupyter-dash>=0.3.0",
        "dash>=1.11.0",
        "gunicorn",
        "tomd",
        "lxml",
        "beautifulsoup4",
        "dash-bootstrap-components<1.0.0",
        "pyngrok"
    ],
    extra_require={"toy": ["scikit-learn"]},
)