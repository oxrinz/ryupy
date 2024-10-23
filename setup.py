import subprocess
import os

src_dir = "src"
output_file = "lib/ryupy.dll"

source_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".cu") or file.endswith(".cpp"):
            source_files.append(os.path.join(root, file))

command = ["nvcc", "-shared", "-o", output_file] + source_files
subprocess.run(command, check=True)

print(f"Successfully built {output_file}")

exp_file = output_file.replace(".dll", ".exp")
lib_file = output_file.replace(".dll", ".lib")

for file in [exp_file, lib_file]:
    if os.path.exists(file):
        os.remove(file)
        print(f"Removed {file}")

print("Cleaned up unnecessary files.")
