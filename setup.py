import subprocess
import os
import sys
import inspect
import typing
import shutil

# CUDA and Python paths (updated for Linux)
cuda_path = os.getenv("CUDA_PATH")  # Use CUDA_PATH from nix-shell environment
src_dir = "src"
output_file = "ryupy.so"  # .so extension for Linux shared library

# Python executable path and includes for pybind11
python_executable = sys.executable

# Get pybind11 include paths
pybind_includes = (
    subprocess.check_output([python_executable, "-m", "pybind11", "--includes"])
    .decode()
    .split()
)

# Python include and library paths
python_include = (
    subprocess.check_output(
        [
            python_executable,
            "-c",
            "import sysconfig; print(sysconfig.get_path('include'))",
        ]
    )
    .decode()
    .strip()
)
python_lib_dir = (
    subprocess.check_output(
        [
            python_executable,
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
        ]
    )
    .decode()
    .strip()
)
python_lib = (
    subprocess.check_output(
        [
            python_executable,
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))",
        ]
    )
    .decode()
    .strip()
)

# Collect .cu and .cpp source files
source_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".cu") or file.endswith(".cpp"):
            source_files.append(os.path.join(root, file))

# Path to the directory containing libcudadevrt.a and libcudart_static.a
cuda_static_lib_dir = (
    "/nix/store/yp5wra915j9p5nxa631svxv0x1r5z3m3-cuda_cudart-11.8.89-static/lib"
)

command = (
    [
        "nvcc",
        "--verbose",
        "-shared", 
        "-arch=sm_75",
        "-DCUBLAS_ENABLED",
        "-lcublas",
        "-lcudnn",
        "-lcudadevrt",
        "-lcudart_static",
        "-L{}".format(cuda_static_lib_dir),
        "-o",
        output_file,
        "-std=c++14",
    ]
    + source_files
    + pybind_includes
    + [f"-I{python_include}", f"-L{python_lib_dir}", f"-lpython3.10"]
)


# Run the build command
subprocess.run(command, check=True)

# Cleanup any unnecessary files (Linux might not create .exp/.lib files)
for ext in [".exp", ".lib"]:
    file = output_file.replace(".so", ext)
    if os.path.exists(file):
        os.remove(file)

print(f"Successfully built {output_file}")

# Generate stubs if the shared library was built successfully
if os.path.exists(output_file):
    try:
        os.makedirs("ryupy", exist_ok=True)
        # Generate stubs with the target output directory set to "ryupy"
        subprocess.run(
            [sys.executable, "-m", "pybind11_stubgen", "ryupy", "--output-dir", "."],
            check=True,
        )

        if os.path.exists("ryupy"):
            shutil.rmtree("ryupy")  # Delete the existing ryupy directory

        # Rename the ryupy-stubs directory to ryupy
        os.rename("ryupy-stubs", "ryupy")

        with open("ryupy/py.typed", "w") as f:
            pass
        print("Successfully generated stubs")
    except subprocess.CalledProcessError:
        print("Warning: Failed to generate stubs. Is pybind11-stubgen installed?")
