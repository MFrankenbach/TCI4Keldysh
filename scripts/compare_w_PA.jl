using TCI4Keldysh
using HDF5
using Plots
using OffsetArrays
using MAT

# NRG convention for fermionic legs:
# 1         1'
# \        /
#  \______/
#   |    |          = <1 1' 3 3'>
#   |____|
#  /      \
# /        \
# 3'        3



# NRG frequency convention:
# t- channel
# 
# ν         ν+ω
# \        /
#  \______/
#   |    |
#   |____|
#  /      \
# /        \
# ν′        ν′+ω


# p- channel
# 
# ν         -ν′-ω       ω_p = ν′_t - ν_t 
# \        /            ν_p = ν_t
#  \______/             ν′_p= ν′_t
#   |    |
#   |____|
#  /      \
# /        \
# ν′       -ν-ω


# a- channel
# 
# ν         ν′          ω_a = ν_t - ν′_t
# \        /            ν_a = ν_t
#  \______/             ν′_a= ν_t + ω_t
#   |    |
#   |____|
#  /      \
# /        \
# ν+ω      ν′+ω

begin
    # set frequency conventions
    
    ωconvMat_t = [
        0  1  0;
        -1 -1  0;
        1  0  1;
        0  0 -1;
    ]
    ωconvMat_p = [
        0 -1  0;
        -1  0 -1;
        1  1  0;
        0  0  1;
    ]
    ωconvMat_a = [
        0 -1  0;
        0  0  1;
        -1  0 -1;
        1  1  0;
    ]

    # K1t = ["Q12", "Q34"]
    # K1p = ["Q13", "Q24"]
    # K1a = ["Q14", "Q23"])
    ωconvMat_K1t = reshape([
        sum(view(ωconvMat_t, [1,2], 1), dims=1);
        sum(view(ωconvMat_t, [3,4], 1), dims=1);
    ], (2,1))
    ωconvMat_K1p = reshape([
        sum(view(ωconvMat_p, [1,3], 1), dims=1);
        sum(view(ωconvMat_p, [2,4], 1), dims=1);
    ], (2,1))
    ωconvMat_K1a =  reshape([
        sum(view(ωconvMat_a, [1,4], 1), dims=1);
        sum(view(ωconvMat_a, [2,3], 1), dims=1);
    ], (2,1))

    # K2t = ("Q34", "1", "1dag")
    # K2p = ("Q24", "1", "3")
    # K2a = ("Q23", "1", "3dag")

    ωconvMat_K2t = [
        sum(view(ωconvMat_t, [3,4], [1,2]), dims=1);
        view(ωconvMat_t, [1,2], [1,2])
    ]

    ωconvMat_K2p = [
        sum(view(ωconvMat_p, [2,4], [1,2]), dims=1);
        view(ωconvMat_p, [1,3], [1,2])
    ]


    ωconvMat_K2a = [
        sum(view(ωconvMat_a, [2,3], [1,2]), dims=1);
        view(ωconvMat_a, [1,4], [1,2])
    ]


    # K2′t = ("Q12", "3", "3dag")
    # K2′p = ("Q13", "1dag", "3dag")
    # K2′a = ("Q14", "3", "1dag")

    ωconvMat_K2′t = [
        1  0;
        0  1;
        -1 -1;
    ]
    ωconvMat_K2′t = [
        sum(view(ωconvMat_t, [1,2], [1,3]), dims=1);
        view(ωconvMat_t, [3,4], [1,3])
    ]

    ωconvMat_K2′p = [
        sum(view(ωconvMat_p, [1,3], [1,3]), dims=1);
        view(ωconvMat_p, [2,4], [1,3])
    ]


    ωconvMat_K2′a = [
        sum(view(ωconvMat_a, [1,4], [1,3]), dims=1);
        view(ωconvMat_a, [2,3], [1,3])
    ]
end

begin
    # set physical parameters
    u = 0.5; 
    #u = 1.0

    U_PA = 1.
    Δ_PA = U_PA / (π * u)

    U = 0.05;
    Δ = U / (π * u)
    T = 0.01*U
end

# selfenergy:
output_filename = "data/NRG_MF_vertex_dump.h5"
PA_filename = "/home/Anxiang.Ge/Desktop/PhD/mfrg_Data/MF_finiteT/T=00p01_U=1/parquetInit4_U_over_Delta=1.570796_T=0.010000_eVg=0.000000_n1=4097_n2=513_n3=257_version3_final.h5"
#PA_filename = "/home/Anxiang.Ge/Desktop/PhD/mfrg_Data/MF_finiteT/T=00p01_U=1/parquetInit4_U_over_Delta=2.328759_T=0.010000_eVg=0.000000_n1=4097_n2=513_n3=257_version3_final.h5"
file_PA = h5open(PA_filename, "r")
Σ_PA = read(file_PA, "selflist")[:] / Δ_PA


PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
#PSFpath = "data/PSF_nz=2_conn_zavg_u=1.00/"

N_MF = div(length(Σ_PA), 2)
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * π * T

function compute_Π_p_DK(G, N_ωpos)
    # Π_t(ω,ν) = G(ν) G(-ω+ν)


    M = div.(length(G), 2)
    G = OffsetArrays.OffsetVector(G, -M-1) # axes = -M:M-1
    Π_t = OffsetArrays.OffsetArray(zeros(ComplexF64, 2*N_ωpos+1, 2*M), (-N_ωpos-1, -M-1)) # axes = (-N_ωpos:N_ωpos, -M:M-1)
    #println("M = ", M)
    #println("axes(G)", axes(G))
    #println("axes(Π_t)", axes(Π_t))

    for i in axes(Π_t, 1)

        Π_t[i,-M + max(i, 0): M-1 + min(i,0)] .= G[-M-min(i, 0):M-1-max(i,0)]
    end
    Π_t .*= G'
    return Π_t
end


function compute_Π_t(G, N_ωpos)
    # Π_t(ω,ν) = G(ν) G(ω+ν)


    M = div.(length(G), 2)
    G = OffsetArrays.OffsetVector(G, -M-1) # axes = -M:M-1
    Π_t = OffsetArrays.OffsetArray(zeros(ComplexF64, 2*N_ωpos+1, 2*M), (-N_ωpos-1, -M-1)) # axes = (-N_ωpos:N_ωpos, -M:M-1)
    #println("M = ", M)
    #println("axes(G)", axes(G))
    #println("axes(Π_t)", axes(Π_t))

    for i in axes(Π_t, 1)

        Π_t[i,-M - min(i, 0): M-1 - max(i,0)] .= G[-M+max(i, 0):M-1+min(i,0)]
    end
    Π_t .*= G'
    return Π_t
end



G0_inv   = ω_fer .+ im*Δ*sign.(imag.(ω_fer)) .+ U/2
G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");

G_data      = TCI4Keldysh.precompute_all_values(G)
G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)


Σ_calc_dir = TCI4Keldysh.calc_Σ_MF_dir(G_data, G0_inv)
Σ_calc_aIE = TCI4Keldysh.calc_Σ_MF_aIE(G_aux_data, G_data)
Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)


#@test Σ_calc_aIE ≈ Σ_HA
#@test Σ_calc_sIE ≈ Σ_HA

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

K1a      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q14", "Q23"]; flavor_idx=i, ωs_ext=(ω_bos,), ωconvMat=ωconvMat_K1a, name="SIAM K1a") for i in 1:2];
K1p      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q13", "Q24"]; flavor_idx=i, ωs_ext=(ω_bos,), ωconvMat=ωconvMat_K1p, name="SIAM K1p") for i in 1:2];
K1t      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "Q34"]; flavor_idx=i, ωs_ext=(ω_bos,), ωconvMat=ωconvMat_K1t, name="SIAM K1t") for i in 1:2];
K1a_data = [ TCI4Keldysh.precompute_all_values(K1a[i]) for i in 1:2]
K1p_data = [-TCI4Keldysh.precompute_all_values(K1p[i]) for i in 1:2]
K1t_data = [-TCI4Keldysh.precompute_all_values(K1t[i]) for i in 1:2]

#plot([K1a_PA, real.(K1a_data[2]/Δ)], labels=["PA" "NRG"], title="K1a", xlim=[2000,2200])
plot([K1a_PA, real.(K1a_data[2]/Δ)], labels=["PA" "NRG"], title="K1a", xlim=[2040,2070])
plot([K1p_PA, real.(K1p_data[2]/Δ)], labels=["PA" "NRG"], title="K1p", xlim=[2040,2070])
plot([K1t_PA, real.(K1t_data[2]/Δ)], labels=["PA" "NRG"], title="K1t", xlim=[2040,2070])

function loadGgrid(filename)

    file_JS = matopen(filename)
    try
        keys(file_JS)
    catch
        keys(file_JS)
    end
    return read(file_JS, "CFdat")["Ggrid"]
end
K1t_JS = loadGgrid("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U2_1.mat")
#K1p_JS = loadGgrid("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U2_2.mat")
#K1a_JS = loadGgrid("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U2_3.mat")


file_JS = matopen("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U2_1.mat")
try
    keys(file_JS)
catch
    keys(file_JS)
end
read(file_JS, "CF")["PSF"]


plot([real.(K1t_JS[2][1,1,:]), -real.(K1t_data[2][N_MF+1-101:N_MF+1+101])], labels=["JS" "AG"], xlim=[95,105])
#plot([real.(K1p_JS[2][1,1,:]),  real.(K1p_data[2][N_MF+1-101:N_MF+1+101])], labels=["JS" "AG"], xlim=[95,105])
#plot([real.(K1a_JS[2][1,1,:]), -real.(K1a_data[2][N_MF+1-101:N_MF+1+101])], labels=["JS" "AG"], xlim=[95,105])

maximum(abs.(K1t_JS[2][1,1,:]+real.(K1t_data[2][N_MF+1-101:N_MF+1+101])))
#maximum(abs.(K1p_JS[2][1,1,:]+real.(K1p_data[2][N_MF+1-101:N_MF+1+101])))
#maximum(abs.(K1a_JS[2][1,1,:]+real.(K1a_data[2][N_MF+1-101:N_MF+1+101])))



#max_NRG_K1a = TCI4Keldysh.maxabs(K1a_data[2]/Δ)
#max_Par_K1a = TCI4Keldysh.maxabs(K1a_PA)
#max_Par_K1a / max_NRG_K1a
#max_Par_K1a / max_NRG_K1a * pi^2

#max_NRG_K1p = TCI4Keldysh.maxabs(K1p_data[2]/Δ)
#max_Par_K1p = TCI4Keldysh.maxabs(K1p_PA)
#max_Par_K1p / max_NRG_K1p

#max_NRG_K1t = TCI4Keldysh.maxabs(K1t_data[2]/Δ)
#max_Par_K1t = TCI4Keldysh.maxabs(K1t_PA)
#1 / max_Par_K1t * max_NRG_K1t


begin
    file = h5open(output_filename, "cw")
    file["K1a"] = hcat(K1a_data...)
    file["K1p"] = hcat(K1p_data...)
    file["K1t"] = hcat(K1t_data...)
    file["K1_w"] = ω_bos

    file["SE"] = Σ_calc_aIE
    file["SE_v"] = ω_fer
    file["G"] = G_data

    close(file)

end





# K2:


K2a_PA = permutedims(read(file_PA, "K2_a")[1,1,:,:,1,1], (2,1))
K2p_PA = permutedims(read(file_PA, "K2_p")[1,1,:,:,1,1], (2,1))
K2t_PA = permutedims(read(file_PA, "K2_t")[1,1,:,:,1,1], (2,1))
N_K2_bos, N_K2_fer = div.(size(K2a_PA), 2)
ω_bos = (collect(-N_K2_bos:N_K2_bos) * (2.)      ) * im * π * T
ω_fer = (collect(-N_K2_fer:N_K2_fer-1) * (2.) .+ 1.) * im * π * T
ωs_ext=(ω_bos, ω_fer)


#K2′a_data = [ TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath, ("Q14", "3", "1dag"), Σ_calc_aIE   ; ωs_ext, ωconvMat=ωconvMat_K2′a, flavor_idx=i) for i in 1:2]
#K2′p_data = [-TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath, ("Q13", "1dag", "3dag"), Σ_calc_aIE; ωs_ext, ωconvMat=ωconvMat_K2′p, flavor_idx=i) for i in 1:2]
K2′t_data = [-TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath, ("Q12", "3", "3dag"), Σ_calc_aIE   ; ωs_ext, ωconvMat=ωconvMat_K2′t, flavor_idx=i) for i in 1:2]

K2a_data = [ TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath, ("Q23", "1", "3dag"), Σ_calc_aIE; ωs_ext, ωconvMat=ωconvMat_K2a, flavor_idx=i) for i in 1:2]
K2p_data = [ TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath, ("Q24", "1", "3"   ), Σ_calc_aIE; ωs_ext, ωconvMat=ωconvMat_K2p, flavor_idx=i) for i in 1:2]
K2t_data = [-TCI4Keldysh.compute_K2r_symmetric_estimator(PSFpath, ("Q34", "1", "1dag"), Σ_calc_aIE; ωs_ext, ωconvMat=ωconvMat_K2t, flavor_idx=i) for i in 1:2]


begin
    file = h5open(output_filename, "cw")
    file["K2a"] = cat(K2a_data..., dims=3)
    file["K2p"] = cat(K2p_data..., dims=3)
    file["K2t"] = cat(K2t_data..., dims=3)
    file["K2_w"] = ω_bos
    file["K2_v"] = ω_fer
    close(file)

end

K2t_JS = loadGgrid("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U3_6.mat")

dat_JS = -real.(K2t_JS[2][:,1,:])'
dat_AG = real.(K2t_data[2])[N_K2_bos+1-101:N_K2_bos+1+101, N_K2_fer-100:N_K2_fer+101]
heatmap(dat_JS)
heatmap(dat_AG)

plot([dat_JS[102,:], dat_AG[102,:]], label=["JS" "AG"], xlim=[95,125])
maximum(abs.(dat_JS - dat_AG))



K2′t_JS = loadGgrid("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U3_1.mat")

dat_JS = -real.(K2′t_JS[2][1,:,:])'
dat_AG = real.(K2′t_data[2])[N_K2_bos+1-101:N_K2_bos+1+101, N_K2_fer-100:N_K2_fer+101]
heatmap(dat_JS)
heatmap(dat_AG)

plot([dat_JS[102,:], dat_AG[102,:]], label=["JS" "AG"], xlim=[95,125])
maximum(abs.(dat_JS - dat_AG))



K1D = K1t_data[1] + K1t_data[2]
K1M = K1t_data[1] - K1t_data[2]
K2D = K2t_data[1] + K2t_data[2]
K2M = K2t_data[1] - K2t_data[2]
λD = K2D ./ (-U + K1D[size(K2D, 1)])
λM = K2M ./ ( U + K1M[size(K2D, 1)])

N_check = 100
Π_t = compute_Π_t(G_data, N_check)
NK1_PA = div(length(K1t_data[1])-1, 2)
check_range = 1+NK1_PA-N_check:1+NK1_PA+N_check
Ns_K2 = size(K2t_data[1])
K1D_check = -T * U * ((-U .+ K1D)[check_range] .* OffsetArrays.no_offset_view(dropdims(sum(Π_t, dims=2), dims=2)) + sum(OffsetArrays.no_offset_view(Π_t[:,-div(Ns_K2[2],2):div(Ns_K2[2],2)-1]).*K2D[div(Ns_K2[1],2)-100:div(Ns_K2[1],2)+100,:], dims=2))
K1M_check =  T * U * (( U .+ K1M)[check_range] .* OffsetArrays.no_offset_view(dropdims(sum(Π_t, dims=2), dims=2)) + sum(OffsetArrays.no_offset_view(Π_t[:,-div(Ns_K2[2],2):div(Ns_K2[2],2)-1]).*K2M[div(Ns_K2[1],2)-100:div(Ns_K2[1],2)+100,:], dims=2))
P1D_check =  T     * (                            OffsetArrays.no_offset_view(dropdims(sum(Π_t, dims=2), dims=2)) + sum(OffsetArrays.no_offset_view(Π_t[:,-div(Ns_K2[2],2):div(Ns_K2[2],2)-1]).* λD[div(Ns_K2[1],2)-100:div(Ns_K2[1],2)+100,:], dims=2))
P1M_check =  T     * (                            OffsetArrays.no_offset_view(dropdims(sum(Π_t, dims=2), dims=2)) + sum(OffsetArrays.no_offset_view(Π_t[:,-div(Ns_K2[2],2):div(Ns_K2[2],2)-1]).* λM[div(Ns_K2[1],2)-100:div(Ns_K2[1],2)+100,:], dims=2))
ηD_check = ((1. .+ (-U .+ K1D)[check_range] .* P1D_check) .* (-U))[:]
ηM_check = ((1. .+ ( U .+ K1M)[check_range] .* P1M_check) .* ( U))[:]



plot([real.(K1D_check), real.(K1D)[check_range]], labels=["check" "orig"], title="K1D")
plot([real.(K1M_check), real.(K1M)[check_range]], labels=["check" "orig"], title="K1M")
plot([real.(ηD_check .+ U), real.(K1D)[check_range]], labels=["check" "orig"], title="K1D")
plot([real.(ηM_check .- U), real.(K1M)[check_range]], labels=["check" "orig"], title="K1M")





#K2a_data[2][N_K2_bos+1,:] / Δ
#K2a_PA[N_K2_bos+1,:] / Δ_PA

#fac_mod = 1#5
plot([real.(K2a_data[2][N_K2_bos+1,:] / Δ), K2a_PA[N_K2_bos+1,:] / Δ_PA], labels=["NRG" "PA"])
plot([real.(K2p_data[2][N_K2_bos+1,:] / Δ), K2p_PA[N_K2_bos+1,:] / Δ_PA], labels=["NRG" "PA"])
plot([real.(K2t_data[2][N_K2_bos+1,:] / Δ), K2t_PA[N_K2_bos+1,:] / Δ_PA], labels=["NRG" "PA"])



# compute vertex core in t-channel parametrization:
N_K2_bos, N_K2_fer = 100, 100
ω_bos = (collect(-N_K2_bos:N_K2_bos) * (2.)      ) * im * π * T
ω_fer = (collect(-N_K2_fer:N_K2_fer-1) * (2.) .+ 1.) * im * π * T
ωs_ext=(ω_bos, ω_fer, ω_fer)

Γcore_data = [TCI4Keldysh.compute_Γcore_symmetric_estimator(PSFpath*"4pt/", Σ_calc_aIE; ωs_ext, ωconvMat=ωconvMat_t, flavor_idx=i) for i in 1:2]


    
begin
    file = h5open(output_filename, "cw")
    file["R"] = cat(Γcore_data..., dims=4)
    file["R_w"] = ω_bos
    file["R_v"] = ω_fer
    close(file)

end

begin
    file = h5open(output_filename, "cw")
    file["core"] = cat(Γcore_data..., dims=4)
    file["R_w"] = ω_bos
    file["R_v"] = ω_fer
    close(file)

end


#begin 
#    file_PA = h5open("data/Vertex_core_dump.h5", "w")
#
#    write(file_PA, "Vertex_core", Γcore_data)
#    write(file_PA, "wbos", ω_bos)
#    write(file_PA, "wfer", ω_fer)
#    close(file_PA)
#end


Core_t_JS = loadGgrid("data/SIAM_u=0.50/V_MF_ph_new/V_MF_U4.mat")

dat_JS = -real.(Core_t_JS[2][2:end-1,2:end-1,102])

file = h5open(output_filename, "r")
core_AG = read(file, "core")
close(file)
#core_AG = cat(Γcore_data..., dims=4)
dat_AG = -real.(core_AG[101,:,:,2])

heatmap(dat_AG)
heatmap(dat_JS)

plot([dat_JS[:,101], dat_AG[:,101]], label=["JS" "AG"])


using MAT
f = matopen("data/PSF_nz=2_conn_zavg/PSF_((Q1234)).mat")
keys(f)
Γ0 = read(f, "Adisc")