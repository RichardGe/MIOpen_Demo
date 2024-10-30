# MIOpen_Demo

Illustrates a simple Convolution using MIOpen.


For now, I'll focus my tests mostly on Windows

instruction to clone/build/run:


git clone https://github.com/RichardGe/MIOpen_Demo.git
cd MIOpen_Demo


mkdir build
cd build


cmake .. ^
    -Dhip_DIR="C:\Program Files\AMD\ROCm\6.2\lib\cmake\hip" ^
    -Damd_comgr_DIR="H:\PROJECTS\096_rocm\llvm\llvm-project\amd\comgr\INSTALL\lib\cmake\amd_comgr" ^
    -Dmiopen_DIR="H:\PROJECTS\096_rocm\miOpen\git\MIOpen\build_release\INSTALL\lib\cmake\miopen"



MIOPEN_SYSTEM_DB_PATH=H:\PROJECTS\096_rocm\miOpen\git\MIOpen\build_release\INSTALL\bin
PATH=C:\Program Files\AMD\ROCm\6.2\bin;%PATH%



