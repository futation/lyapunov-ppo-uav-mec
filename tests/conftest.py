"""tests/conftest.py — 让测试从仓库根目录导入模块"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
