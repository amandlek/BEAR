from setuptools import setup, find_packages


setup(
    name="BEAR",
    packages=[
        package for package in find_packages() if package.startswith("BEAR")
    ],
    install_requires=[],
    eager_resources=['*'],
    include_package_data=True,
    description="BEAR: batch reinforcement learning",
)
