using TCI4Keldysh
using HDF5

const OUT_PATH="/scratch/m/M.Frankenbach/Mochi.jl/local_data/DGA_input/HM_beta10_U4"
const PSF_PATH=joinpath(TCI4Keldysh.datadir(), "/home/m/M.Frankenbach/Documents/MATLAB/mpNRG/data/2DHubbard_Fig8Kaufmann2020_2026-10-19_13-08-16/PSFdata_nz=2_conn_zavg")

TCI4Keldysh.set_mpNRG_file_settings()

function GF_MF(;Nhalf::Int=2^6, PSFpath=PSF_PATH, outpath=OUT_PATH)
    T = TCI4Keldysh.read_temperature(PSFpath)
    
    ω_fer = TCI4Keldysh.MF_grid(T, Nhalf, true)
    @show ω_fer[1]
    @show (ω_fer[2]-ω_fer[1])/(2π)
    G = TCI4Keldysh.FullCorrelator_MF(
        PSFpath, ["F1", "F1dag"];
        T, flavor_idx=1, ωs_ext=(ω_fer,),
        ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    Gval = TCI4Keldysh.precompute_all_values(G)
    @show size(Gval)
    R = Int(log2(2*Nhalf))
    h5write(joinpath(outpath,"GF_R$(R).h5"), "om", ω_fer)
    h5write(joinpath(outpath,"GF_R$(R).h5"), "G", Gval)
end

GF_MF(;Nhalf=length(ARGS)>0 ? round(Int,parse(Float64,ARGS[1])) : 2^6, PSFpath=length(ARGS)>1 ? ARGS[2] : PSF_PATH, outpath=length(ARGS)>2 ? ARGS[3] : OUT_PATH)
