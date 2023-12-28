using TCI4Keldysh
using HDF5

u = 0.5; 

U_PA = 1.
Δ_PA = U_PA / (π * u)

# selfenergy:
PA_filename = "/home/Anxiang.Ge/Desktop/PhD/mfrg_Data/MF_finiteT/T=00p01_U=1/parquetInit4_U_over_Delta=1.570796_T=0.010000_eVg=0.000000_n1=4097_n2=513_n3=257_version3_final.h5"
#PA_filename = "/home/Anxiang.Ge/Desktop/PhD/mfrg_Data/MF_finiteT/T=00p01_U=1/parquetInit4_U_over_Delta=2.328759_T=0.010000_eVg=0.000000_n1=4097_n2=513_n3=257_version3_final.h5"
file_PA = h5open(PA_filename, "r")
Σ_PA = read(file_PA, "selflist")[:] / Δ_PA


PSFpath = "data/PSF_nz=2_conn_zavg/"

N_MF = div(length(Σ_PA), 2)
U = 0.05;
Δ = U / (π * u)
T = 0.01*U
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * π * T



G0_inv   = ω_fer .+ im*Δ*sign.(imag.(ω_fer)) .+ U/2
G        = TCI4Keldysh.FullCorrelator_MF(PSFpath*"2pt/", ["F1", "F1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath*"2pt/", ["Q1", "F1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath*"2pt/", ["Q1", "Q1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");

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
plot([Σ_PA, imag.(Σ_calc_aIE/Δ), imag.(Σ_calc_sIE/Δ)], label=["PA" "aIE" "sIE"])

G0_inv_aIE = 1 ./ G_data + Σ_calc_aIE
G0_inv_sIE = 1 ./ G_data + Σ_calc_sIE

plot([imag.(G0_inv - ω_fer), imag.(G0_inv_aIE - ω_fer), imag.(G0_inv_sIE - ω_fer)], label=["exact" "aIE" "sIE"])


# K1:
K1a_PA = read(file_PA, "K1_a")[:] / Δ_PA
K1p_PA = read(file_PA, "K1_p")[:] / Δ_PA
K1t_PA = read(file_PA, "K1_t")[:] / Δ_PA
N_MF = div(length(K1a_PA)-1, 2)
ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * im * π * T

K1a      = TCI4Keldysh.FullCorrelator_MF(PSFpath*"2pt/", ["Q14", "Q23"]; flavor_idx=2, ωs_ext=(ω_bos,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM K1a");
K1p      = TCI4Keldysh.FullCorrelator_MF(PSFpath*"2pt/", ["Q13", "Q24"]; flavor_idx=2, ωs_ext=(ω_bos,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM K1p");
K1t      = TCI4Keldysh.FullCorrelator_MF(PSFpath*"2pt/", ["Q12", "Q34"]; flavor_idx=2, ωs_ext=(ω_bos,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM K1t");
K1a_data = TCI4Keldysh.precompute_all_values(K1a)
K1p_data =-TCI4Keldysh.precompute_all_values(K1p)
K1t_data =-TCI4Keldysh.precompute_all_values(K1t)

plot([K1a_PA, real.(K1a_data/Δ)], labels=["PA" "NRG"], title="K1a")
plot([K1p_PA, real.(K1p_data/Δ)], labels=["PA" "NRG"], title="K1p")
plot([K1t_PA, real.(K1t_data/Δ)], labels=["PA" "NRG"], title="K1t")

max_NRG_K1a = TCI4Keldysh.maxabs(K1a_data/Δ)
max_Par_K1a = TCI4Keldysh.maxabs(K1a_PA)
max_Par_K1a / max_NRG_K1a
max_Par_K1a / max_NRG_K1a * pi^2

max_NRG_K1p = TCI4Keldysh.maxabs(K1p_data/Δ)
max_Par_K1p = TCI4Keldysh.maxabs(K1p_PA)
max_Par_K1p / max_NRG_K1p

max_NRG_K1t = TCI4Keldysh.maxabs(K1t_data/Δ)
max_Par_K1t = TCI4Keldysh.maxabs(K1t_PA)
1 / max_Par_K1t * max_NRG_K1t









# K2:


K2a_PA = permutedims(read(file_PA, "K2_a")[1,1,:,:,1,1] / Δ_PA, (2,1))
K2p_PA = permutedims(read(file_PA, "K2_p")[1,1,:,:,1,1] / Δ_PA, (2,1))
K2t_PA = permutedims(read(file_PA, "K2_t")[1,1,:,:,1,1] / Δ_PA, (2,1))
N_K2_bos, N_K2_fer = div.(size(K2a_PA), 2)
ω_bos = (collect(-N_K2_bos:N_K2_bos) * (2.)      ) * im * π * T
ω_fer = (collect(-N_K2_fer:N_K2_fer-1) * (2.) .+ 1.) * im * π * T
ωs_ext=(ω_bos, ω_fer)
ωconvMat=[ 1 0; 0 1; -1 -1]
flavor_idx = 2

K2a_data = TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath*"3pt/", ("Q14", "3", "1dag"), Σ_calc_aIE; ωs_ext, ωconvMat, flavor_idx)

K2p_data =-TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath*"3pt/", ("Q13", "1dag", "3dag"), Σ_calc_aIE; ωs_ext, ωconvMat, flavor_idx)
K2t_data =-TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath*"3pt/", ("Q12", "3", "3dag"), Σ_calc_aIE; ωs_ext, ωconvMat, flavor_idx)


K2a_data[N_K2_bos+1,:] / Δ
K2a_PA[N_K2_bos+1,:] / Δ_PA

fac_mod = 1.5
plot([real.(K2a_data[N_K2_bos+1,:] / Δ * fac_mod), K2a_PA[N_K2_bos+1,:] / Δ_PA])
plot([real.(K2p_data[N_K2_bos+1,:] / Δ * fac_mod), K2p_PA[N_K2_bos+1,:] / Δ_PA])
plot([real.(K2t_data[N_K2_bos+1,:] / Δ * fac_mod), K2t_PA[N_K2_bos+1,:] / Δ_PA])




N_K2_bos, N_K2_fer = 100, 100
ω_bos = (collect(-N_K2_bos:N_K2_bos) * (2.)      ) * im * π * T
ω_fer = (collect(-N_K2_fer:N_K2_fer-1) * (2.) .+ 1.) * im * π * T
ωs_ext=(ω_bos, ω_fer, ω_fer)
ωconvMat=[  0 -1  0; 
            1  1  0; 
           -1  0 -1; 
            0  0  1]
flavor_idx = 2

Γcore_data = TCI4Keldysh.compute_Γcore_symmetric_estimator(PSFpath*"4pt/", Σ_calc_aIE; ωs_ext, ωconvMat, flavor_idx)
