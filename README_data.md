## How to prepare PSF / vertex data files for TCI4Keldysh
1. Open MuNRG in matlab. Run MuNRG/startup.m.
2. Run the function mpnrg4julia in matlab with the data path. Skip mpNRG_\*.mat files in the script, if any. It is recommended to run it twice (unclear why this is necessary).
3. Run check\_matfiles.jl to see whether all target files are now readable by MAT.jl. See whether any missing values or reading fails are reported. If so, repeat step 2.
4. Run change\_PSF\_layout() in change\_PSF\_layout.jl __if__ PSF files have keys ["Adisc", "PSF", "isdone"] to get the key "odisc".
5. mv all 4pt PSFs to a directory 4pt/
6. Run symmetry\_expand.jl on that 4pt/ directory.
7. Might need to copy the parameter file mpNRG.mat to mpNRG\_pp.mat mpNRG\_ph.mat mpNRG\_pht.mat
