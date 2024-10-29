import subprocess
import os
import sys
import inspect
import typing

src_dir = "src"
output_file = "ryupy.pyd"

python_executable = "C:/Users/rober/AppData/Local/Programs/Python/Python310/python.exe"

pybind_includes = (
    subprocess.check_output([python_executable, "-m", "pybind11", "--includes"])
    .decode()
    .split()
)

python_include = "C:/Users/rober/AppData/Local/Programs/Python/Python310/include"
python_lib_dir = "C:/Users/rober/AppData/Local/Programs/Python/Python310/libs"
python_lib = f"{python_lib_dir}/python310.lib"

source_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".cu") or file.endswith(".cpp"):
            source_files.append(os.path.join(root, file))

command = (
    ["nvcc", "-shared", "-arch=sm_75", "-lcublas", "-o", output_file, "-std=c++14"]
    + source_files
    + pybind_includes
    + [f"-I{python_include}", f"-L{python_lib_dir}", f"-lpython310"]
)

subprocess.run(command, check=True)

exp_file = output_file.replace(".pyd", ".exp")
lib_file = output_file.replace(".pyd", ".lib")

for file in [exp_file, lib_file]:
    if os.path.exists(file):
        os.remove(file)

print(f"Successfully built {output_file}")

if os.path.exists("ryupy.pyd"):
    try:
        # Create ryupy directory if it doesn't exist
        os.makedirs("ryupy", exist_ok=True)

        # Generate stubs directly in the ryupy directory
        subprocess.run(
            [sys.executable, "-m", "pybind11_stubgen", "ryupy", "--output-dir", "."],
            check=True,
        )

        # Create py.typed file
        with open("ryupy/py.typed", "w") as f:
            pass

        print("Successfully generated stubs")
    except subprocess.CalledProcessError:
        print("Warning: Failed to generate stubs. Is pybind11-stubgen installed?")
