using TCI4Keldysh
using HDF5

const OUT_PATH="/scratch/m/M.Frankenbach/Mochi.jl/local_data/DGA_input/HM_beta10_U4"
const PSF_PATH=joinpath(TCI4Keldysh.datadir(), "/home/m/M.Frankenbach/Documents/MATLAB/mpNRG/data/2DHubbard_Fig8Kaufmann2020_2026-10-19_13-08-16/PSFdata_nz=2_conn_zavg")

TCI4Keldysh.set_mpNRG_file_settings()

function gen_Sigma(;R::Int=2^6, PSFpath=PSF_PATH, outpath=OUT_PATH)
    Nhalf = 2^(R-1)
    T = TCI4Keldysh.read_temperature(PSFpath)
    ωfer = TCI4Keldysh.MF_grid(T, Nhalf, true)
    @show (ωfer[2]-ωfer[1])/(2π)
    Σ = TCI4Keldysh.calc_Σ_MF_sIE(PSFpath, ωfer; T=T, flavor_idx=1)
    h5write(joinpath(outpath,"Sigma_dense_R$(R)_beta$(1/T).h5"), "Sigma", Σ)
    h5write(joinpath(outpath,"Sigma_dense_R$(R)_beta$(1/T).h5"), "om", ωfer)
end

gen_Sigma(;R=length(ARGS)>0 ? round(Int,parse(Float64,ARGS[1])) : 6, PSFpath=length(ARGS)>1 ? ARGS[2] : PSF_PATH, outpath=length(ARGS)>2 ? ARGS[3] : OUT_PATH)
