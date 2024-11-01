# MIOpen Demo

Illustrates a simple Convolution using MIOpen.

For now, I'll focus my tests mostly on Windows

I'll assume you have:
- the HIP SDK 6.2 installed ( https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html )
- built and installed [LLVM-rocm](https://github.com/ROCm/llvm-project/) in this example path: `H:\PROJECTS\096_rocm\llvm\llvm-project\amd\comgr\INSTALL`.
- build and installed [MIOpen](https://github.com/ROCm/MIOpen) in this example path: `H:\PROJECTS\096_rocm\miOpen\git\MIOpen\build_release\INSTALL`


## Instruction to clone/build/run

```
git clone https://github.com/RichardGe/MIOpen_Demo.git
cd MIOpen_Demo

mkdir build
cd build

cmake .. ^
    -Dhip_DIR="C:\Program Files\AMD\ROCm\6.2\lib\cmake\hip" ^
    -Damd_comgr_DIR="H:\PROJECTS\096_rocm\llvm\llvm-project\amd\comgr\INSTALL\lib\cmake\amd_comgr" ^
    -Dmiopen_DIR="H:\PROJECTS\096_rocm\miOpen\git\MIOpen\build_release\INSTALL\lib\cmake\miopen"

```

open `build/MIOpenConvolutionExample.sln`

in the properties of the `convolution_example` project, got to `Debugging` and in `Environment`, and those lines:

```
MIOPEN_SYSTEM_DB_PATH=H:\PROJECTS\096_rocm\miOpen\git\MIOpen\build_release\INSTALL\bin
PATH=C:\Program Files\AMD\ROCm\6.2\bin;%PATH%
```

Now, you should be able to build and run.
The expected output of the demo is:
```
1 4 3
1 2 4
1 2 3
```

## Explainations

adding `C:\Program Files\AMD\ROCm\6.2\bin` to `PATH` is needed to access DLLs like `hiprtc0602.dll`

`MIOPEN_SYSTEM_DB_PATH` is the path that is needed to build some bitcode cache. If you don't set it, you may have error looking like that:
`MIOpen Error: ai_heuristics.cpp:39: Unable to load file: gfx908_metadata.tn.model`

Speaking of envvar, if your project is using rocblas, you may also need to add this one:
`ROCBLAS_TENSILE_LIBPATH=C:\Program Files\AMD\ROCm\6.2\bin\rocblas\library`

