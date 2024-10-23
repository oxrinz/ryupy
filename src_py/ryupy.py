import ctypes
import os

lib_path = os.path.abspath("./lib/ryupy.dll")
ryupy_lib = ctypes.CDLL(lib_path)

ryupy_lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
ryupy_lib.add.restype = ctypes.c_int

ryupy_lib.subtract.argtypes = [ctypes.c_int, ctypes.c_int]
ryupy_lib.subtract.restype = ctypes.c_int

def add(a, b):
    return ryupy_lib.add(a, b)

def subtract(a, b):
    return ryupy_lib.subtract(a, b)