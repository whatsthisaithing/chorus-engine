from setuptools import setup, find_packages

setup(
    name="chorus-discord-bridge",
    version="0.1.0",
    description="Discord bridge service for Chorus Engine",
    author="Chorus Engine Team",
    packages=find_packages(),
    install_requires=[
        "discord.py>=2.3.2",
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "aiosqlite>=0.19.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "chorus-discord-bridge=bridge.main:main",
        ],
    },
)
