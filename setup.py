from setuptools import setup

setup(
    name="pxpguided-diffusion",
    py_modules=["guided_diffusion", "guided_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
