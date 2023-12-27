using TCI4Keldysh
using HDF5

u = 0.5; 

PA_filename = "/home/Anxiang.Ge/Desktop/PhD/mfrg_Data/MF_finiteT/T=00p01_U=1/parquetInit4_U_over_Delta=1.570796_T=0.010000_eVg=0.000000_n1=4097_n2=513_n3=257_version3_final.h5"
file_PA = h5open(PA_filename, "r")
U_PA = 1.
Δ_PA = U_PA / (π * u)
Σ_PA = read(file_PA, "selflist")[:] / Δ_PA


PSFpath = "data/PSF_nz=2_conn_zavg/"

N_MF = div(length(Σ_PA), 2)
U = 0.05;
Δ = U / (π * u)
T = 0.01*U
#ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * im * π * T
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * π * T



G0_inv   = ω_fer .+ im*Δ*sign.(imag.(ω_fer)) .+ U/2
G        = TCI4Keldysh.FullCorrelator_MF(PSFpath             , ["F1", "F1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath*"withQ/2pt/", ["Q1", "F1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath*"withQ/2pt/", ["Q1", "Q1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");

G_data      = TCI4Keldysh.precompute_all_values(G)
G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)


Σ_calc_dir = TCI4Keldysh.calc_Σ_MF_dir(G_data, G0_inv)
Σ_calc_aIE = TCI4Keldysh.calc_Σ_MF_aIE(G_aux_data, G_data)
Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)

Σ_calc_dir - Σ_calc_aIE
Σ_calc_sIE - Σ_calc_aIE / TCI4Keldysh.maxabs(Σ_calc_sIE)

#@test Σ_calc_aIE ≈ Σ_HA
#@test Σ_calc_sIE ≈ Σ_HA

using Plots
plot(Σ_PA)
plot([Σ_PA, imag.(Σ_calc_aIE/Δ), imag.(Σ_calc_sIE/Δ)], label=["PA" "aIE" "sIE"])

G0_inv_aIE = 1 ./ G_data + Σ_calc_aIE
G0_inv_sIE = 1 ./ G_data + Σ_calc_sIE

plot([imag.(G0_inv - ω_fer), imag.(G0_inv_aIE - ω_fer), imag.(G0_inv_sIE - ω_fer)], label=["exact" "aIE" "sIE"])
