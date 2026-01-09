## How to prepare PSF / vertex data files for TCI4Keldysh
1. Open MuNRG in matlab. Run `MuNRG/startup.m`.
2. Run the function `mpnrg4julia` in matlab with the data path. Skip `mpNRG_*.mat` files in the script, if any. It is recommended to run it twice (unclear why this is necessary).
3. Run `scrpts/check_matfiles.jl` to see whether all target files are now readable by MAT.jl, by specifying the PSF directory at the top. See whether any missing values or reading fails are reported. If so, repeat step 2.
4. Run `change_PSF_layout()` in `scrpts/change_PSF_layout.jl` __if__ PSF files have keys ["Adisc", "PSF", "isdone"] to get the key "odisc". Again, you have to specify the PSF the path in the script.
5. `mv` all 4pt PSFs to a sub-directory called `4pt/`.
6. Run `scripts/symmetry_expand.jl` on that `4pt/` directory. To that end, change the path on the top of the script.
7. Copy the parameter file `mpNRG.mat` to three distinct files `mpNRG_pp.mat` `mpNRG_ph.mat` `mpNRG_pht.mat`, if these files don't already exist.
