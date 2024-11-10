{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "ryupy-shell";
  buildInputs = with pkgs; [
    pkgs.gcc10                         
    pkgs.python310
    pkgs.python310Packages.pybind11
    pkgs.python310Packages.tqdm
    pkgs.python310Packages.mypy
    pkgs.python310Packages.numpy
    (pkgs.python310Packages.buildPythonPackage rec {
      pname = "pybind11-stubgen";
      version = "0.10.0";

      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "0ffq8qd4qc7bs5qm1i8c0v74s8p14qrladhj2rzysnj2in9b3whh";
      };

      propagatedBuildInputs = [ pkgs.python310Packages.pybind11 ];
    })
    pkgs.cudaPackages_11_8.cudatoolkit        
    pkgs.cudaPackages_11_8.cudnn        
    pkgs.linuxPackages.nvidia_x11            
    pkgs.libGLU pkgs.libGL
    pkgs.xorg.libXi pkgs.xorg.libXmu pkgs.freeglut
    pkgs.xorg.libXext pkgs.xorg.libX11 pkgs.xorg.libXv pkgs.xorg.libXrandr pkgs.zlib 
    pkgs.ncurses5 pkgs.binutils
  ];

  shellHook = ''
    export NIXPKGS_ALLOW_UNFREE=1
    export CUDA_PATH=${pkgs.cudaPackages_11_8.cudatoolkit}/bin
    
    # Add cuDNN include and library paths
    export CUDNN_INCLUDE_DIR=${pkgs.cudaPackages_11_8.cudnn}/include
    export CUDNN_LIB_DIR=${pkgs.cudaPackages_11_8.cudnn}/lib
    
    # Update compiler flags to include cuDNN paths
    export EXTRA_CCFLAGS="-I$CUDNN_INCLUDE_DIR -I${pkgs.cudaPackages_11_8.cudatoolkit}/include"
    export EXTRA_LDFLAGS="-L$CUDNN_LIB_DIR -L${pkgs.cudaPackages_11_8.cudatoolkit}/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    
    # Override the compilers for nvcc to use GCC 10
    export CC=${pkgs.gcc10}/bin/gcc    # Explicitly set GCC 10 as the compiler
    export CXX=${pkgs.gcc10}/bin/g++   # Set GCC 10 g++ as the C++ compiler
    export PATH=${pkgs.gcc10}/bin:$PATH  # Prepend GCC 10 to the PATH

    # Set compilers for nvcc
    export CUDACXX=${pkgs.gcc10}/bin/g++
    export HOST_COMPILER=${pkgs.gcc10}/bin/gcc
  '';
}
