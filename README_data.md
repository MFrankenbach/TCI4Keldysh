## How to prepare PSF / vertex data files for TCI4Keldysh
The workflow has been simplified after the rehearsal of the mpNRG code in spring 2026.
One way to find out which version was used to generate your partial spectral function data (i.e., which steps you should follow) is to check
the PSF `.mat` files have keys `Adisc` or `PSFinfo` (_old_ version) or simply a key `PSFdata` holding a bare array of spectral peaks (_new_ version).
### For new (spring 2026) mpNRG code
1. Extract the frequency grid in `mpNRG.mat` from a `PSF_info` object and, for Keldysh calculations, specify default broadening parameters. This has to be done in `MuNRG` using, e.g., a function like this:
    ```
    function UnpackOdisc(mpNRGpath)
        % unpack
        load(mpNRGpath, 'PSFinfo_3p') % it does not matter where we load it from
        odisc_info = PSFinfo_3p.odisc_info
        odisc = odisc_info.odisc
        % append odisc
        save(mpNRGpath, 'odisc', '-append');

        % for Keldysh calculations: default logarithmic (sigma) and linear (gamma) broadening parameters
        % CHOOSE THESE FOR YOUR SPECIFIC DATASET
        vars = who('-file', mpNRGpath);

        if ismember('Lwidth', vars)
            S = load(mpNRGpath, Lwidth);
            Lwidth = S.(Lwidth); % don't override old value
        else
            Lwidth = 0.1; % new value
        end 

        if ismember('Hwidth', vars)
            S = load(mpNRGpath, Hwidth); % don't override old value
            Hwidth = S.(Hwidth);
        else
            Hwidth = 0.2; % new value
        end 
        save(mpNRGpath, 'Lwidth', '-append');
        save(mpNRGpath, 'Hwidth', '-append');
    end
    ```
    It does not matter from which `PSF_info` object you extract the grid as long as they are all the same.
    The broadening parameters can be overwritten in input files.
2. `mv` all 4-point PSFs to a subdirectory `4pt/`.
3. Export an environment variable as `MPNRG_VERSION=new` and run `scripts/symmetry_expand.jl`. You should now have 240 PSF files in `4pt` (for SIAM / single-orbital DMFT).
### For old mpNRG code
1. Open MuNRG in matlab. Run MuNRG/startup.m.
2. Run the function `mpNRGobj2struct` in matlab with the data path. Skip mpNRG_\*.mat files in the script, if any. It is recommended to run it twice (unclear why this is necessary).
3. Run check\_matfiles.jl to see whether all target files are now readable by MAT.jl. See whether any missing values or reading fails are reported. If so, repeat step 2.
4. Run change\_PSF\_layout() in change\_PSF\_layout.jl __if__ PSF files have keys ["Adisc", "PSF", "isdone"] to get the key "odisc".
5. mv all 4-point PSFs to a subdirectory `4pt/`
6. Run symmetry\_expand.jl on that `4pt/` directory.
