#pip install -e .

from setuptools import setup, find_packages

setup(
    name="usb_cap_48x48_hand",
    version="0.1",
    packages=find_packages(),  # 自动发现所有 Python 包（如 lib）
)