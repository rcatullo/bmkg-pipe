from setuptools import setup, find_packages

setup(
    name="semantic-parser-framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.9",
    description="A unified semantic parsing framework using LLM with ReACT-style reasoning",
    author="Your Name",
    license="MIT",
)
