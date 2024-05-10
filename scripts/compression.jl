###
# Try out compression with different methods
###
using Revise
using TCI4Keldysh
using HDF5
using QuanticsTCI
import TensorCrossInterpolation as TCI
using Plots

TCI4Keldysh.VERBOSE() = true
TCI4Keldysh.DEBUG() = true


begin

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



    # MBEsolver frequency convention:
    # t- channel: same as in NRG
    # 
    # p- channel: difference to NRG: ω_p --> -ω_p
    # 
    # ν         ω-ν′        ω_p = ω_t + ν_t + ν′_t
    # \        /            ν_p = ν_t
    #  \______/             ν′_p= ν′_t
    #   |    |
    #   |____|
    #  /      \
    # /        \
    # ν′        ω-ν
    #
    # a-channel: same as in NRG


    begin
        # set frequency conventions
        
        ωconvMat_t = [
            0 -1  0;
            1  1  0;
           -1  0 -1;
            0  0  1;
        ]
        #ωconvMat_p = [   # NRG convention
        #    0 -1  0;
        #    -1  0 -1;
        #    1  1  0;
        #    0  0  1;
        #]
        ωconvMat_p = [    # MBEsolver convention
            0 -1  0;
            1  0 -1;
        -1  1  0;
            0  0  1;
        ]
        ωconvMat_a = [
            0 -1  0;
            0  0  1;
            -1  0 -1;
            1  1  0;
        ]

        ### deduce frequency conventions for 2p and 3p vertex contributions:

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

    function get_ωcont(ωmax, Nωcont_pos)
        ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
        return ωcont
    end
    
    

    filename = "test/tests/data_PSF_2D.h5"
    f = h5open(filename, "r")
    Adisc = read(f, "Adisc")
    ωdisc = read(f, "ωdisc")
    close(f)

    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    ### System parameters of SIAM ### 
    D = 1.
    ## Keldysh paper:    u=0.5 OR u=1.0
     # set physical parameters
     u = 0.5; 
     #u = 1.0
 
     U = 0.05;
     Δ = U / (π * u)
     T = 0.01*U

    Rpos = 6
    R = Rpos + 1
    Nωcont_pos = 2^Rpos # 512#
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))
    
    # get functor which can evaluate broadened data pointwisely
    #broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    ωs_ext = (ωbos, ωfer)
    ωconv = [
         1  0;
         -1  -1
    
         ]
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Gs      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "F3", "F3dag"]; T, flavor_idx=i, ωs_ext=(ωbos,ωfer), ωconvMat=ωconvMat_K2′t, name="SIAM 3pG", is_compactAdisc=false) for i in 1:2];
    K1ts    = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "Q34"]; T, flavor_idx=i, ωs_ext=(ωbos,), ωconvMat=ωconvMat_K1t, name="SIAM 2pG") for i in 1:2];
    G3D     = TCI4Keldysh.FullCorrelator_MF(PSFpath*"/4pt/", ["F1","F1dag", "F3", "F3dag"]; T, flavor_idx=1, ωs_ext=(ωbos,ωfer,ωfer), ωconvMat=ωconvMat_t, name="SIAM 4pG", is_compactAdisc=false)
    #Gp = Gs[1].Gps[1]#TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, ωs_ext, ωconv)
end

using Lehmann
# partial fraction decomposition of Gp:

begin
    β = 1/T
    rtol=1e-8
    symmetry = :none
    Euv = D*5
    dlr_fer = DLRGrid(Euv, β, rtol, true) #initialize the DLR parameters and basis
    dlr_bos = DLRGrid(Euv, β, rtol, false) #initialize the DLR parameters and basis
    @assert maximum(abs.(dlr_fer.ω-dlr_bos.ω)) < 1e-13
end
sum(abs.(dlr_bos.ω) .< 1e-10)
dlr_bos.ω[30:40]

#begin
    #ip = 3
for ip in 1:6
    Gp_in = deepcopy(Gs[1].Gps[ip])
    TCI4Keldysh.discreteLehmannRep4Gp!(Gp_in; dlr_bos, dlr_fer)
    #abs.(Gp_in.tucker.ωs_center[1])[25:35]


    #reg(x) = expm1(x*100) / (exp(x*100)+1)
    #plot(Gp_in.tucker.ωs_legs[1], reg.(Gp_in.tucker.ωs_legs[1]))
    #plot(Gp_in.tucker.ωs_legs[1], tanh.(reg.(Gp_in.tucker.ωs_legs[1])))

    # check DLR compression:
    vals_Gp_orig = TCI4Keldysh.precompute_reg_values_MF_without_ωconv(Gs[1].Gps[ip]);
    vals_Gp_orig2= TCI4Keldysh.precompute_reg_values_MF_without_ωconv(Gs[1].Gps[ip]);

    vals_Gp_dlrd = TCI4Keldysh.precompute_reg_values_MF_without_ωconv(Gp_in);
    vals_Gp_dlrd2 = Gp_in.tucker[:,:];
    @assert maximum(abs.(vals_Gp_dlrd - vals_Gp_orig)) / maximum(abs.(vals_Gp_dlrd)) < rtol
    maximum(abs.(vals_Gp_dlrd2 - vals_Gp_orig)) / maximum(abs.(vals_Gp_dlrd2))
    maximum(abs.(vals_Gp_dlrd2 - vals_Gp_dlrd)) / maximum(abs.(vals_Gp_dlrd2))

    vals_ano_Gp_orig = TCI4Keldysh.precompute_ano_values_MF_without_ωconv(Gs[1].Gps[ip]);
    vals_ano_Gp_dlrd = TCI4Keldysh.precompute_ano_values_MF_without_ωconv(Gp_in);
    #maximum(abs.(vals_ano_Gp_dlrd - vals_ano_Gp_orig)) / maximum(abs.(vals_ano_Gp_orig))
#end

#Gp_in.Adisc_anoβ



idx1 = 1
idx2 = 2

#dlr_bos.ωn#[25:35]

#dlr_bos.n[25:35]
Gp_out1, Gp_out2 = TCI4Keldysh.partial_fraction_decomp(Gp_in; idx1, idx2, dlr_bos, dlr_fer);
end
#heatmap(real.(Gp_in.tucker.center))
#heatmap(real.(Gp_out1.tucker.center))
#heatmap(real.(Gp_out2.tucker.center))
# check whether partial fraction decomposition worked:
vals_orig = TCI4Keldysh.precompute_all_values_MF(Gp_in);
vals_pfd1 = TCI4Keldysh.precompute_all_values_MF(Gp_out1)
vals_pfd2 = TCI4Keldysh.precompute_all_values_MF(Gp_out2)
diff = vals_pfd1+vals_pfd2-vals_orig
maximum(abs.(diff))
imax = Tuple(argmax(abs.(diff))) 
#plot(real.([vals_orig[imax[1],:], (vals_pfd1)[imax[1],:], vals_pfd2[imax[1],:]]), labels=["orig" "new" "zero"])
plot(real.([vals_orig[imax[1],:], ((vals_pfd1)[imax[1],:] + vals_pfd2[imax[1],:])]), labels=["orig" "new" "zero"])
plot(real.([vals_orig[:,imax[2]]- (vals_pfd1)[:,imax[2]] + vals_pfd2[:,imax[2]]]), labels=["orig" "new" "zero"])

Gp_out2.ωconvMat
# check whether partial fraction decomposition worked for 3D:






#@assert 1 ≤ idx1 ≤ idx2 ≤ D

Gp_out1 = deepcopy(Gp_in)

# reorder internal frequencies to bring idx1 and idx2 to the front
begin
    idxs_rest = Vector{Int}([collect(1:idx1-1); collect(idx1+1:idx2-1); collect(idx2+1:D)])
    idx_order_new = [[idx1, idx2]; idxs_rest]
    TCI4Keldysh.permute_Adisc_indices!(Gp_out1, idx_order_new)
end

Gp_out2 = deepcopy(Gp_out1)

# decide whether the new frequency should be (v1 - v2) or (v2 - v1)
# ==> make sure that the direction of fermionic frequency directions don't flip for anomalous contributions ∝ β
prefac = Gp_in.isFermi[2] ? 1 : -1
# frequencies for new PartialCorrelators from partial fraction decomposition
Gp_out1.ωconvMat = vcat(vcat(prefac.*(Gp_in.ωconvMat[1:1,:]-Gp_in.ωconvMat[2:2,:]), Gp_in.ωconvMat[1:1,:]), Gp_in.ωconvMat[3:D,:])
Gp_out2.ωconvMat = vcat(vcat(prefac.*(Gp_in.ωconvMat[1:1,:]-Gp_in.ωconvMat[2:2,:]), Gp_in.ωconvMat[2:2,:]), Gp_in.ωconvMat[3:D,:])


# prepare frequency arguments to compute Kernel_t (see below) on sparse sampling points
TCI4Keldysh.update_frequency_args!(Gp_out1)
TCI4Keldysh.update_frequency_args!(Gp_out2)
TCI4Keldysh.set_DLR_MFfreqs!(Gp_out1; dlr_bos, dlr_fer)
TCI4Keldysh.set_DLR_MFfreqs!(Gp_out2; dlr_bos, dlr_fer)
TCI4Keldysh.update_kernels!(Gp_out1)
TCI4Keldysh.update_kernels!(Gp_out2)
@assert maximum(abs.(Gp_out1.tucker.ωs_legs[1] - Gp_out2.tucker.ωs_legs[1])) < 1e-14
@assert sum(.!Gp_out1.isFermi) ≤ 1 # it is not allowed to have more than one bosonic frequency!
@assert sum(.!Gp_out2.isFermi) ≤ 1 # it is not allowed to have more than one bosonic frequency!


# construct Kernel_{n,i,j} := K(i ωₙ - ϵᵢ + ϵⱼ)
sz_center = size(Gp_out1.tucker.center)
ωn1 = (Gp_out1.isFermi[1] ? dlr_fer : dlr_bos).ωn
Kernel_t = 1. ./ (ωn1*im .- Gp_in.tucker.ωs_center[1]' .+ reshape(Gp_in.tucker.ωs_center[2], (1,1,sz_center[2])))
is_nan = .!isfinite.(Kernel_t)
Kernel_t[is_nan] .= 0.
iω0 = argmax(is_nan)[1]
println("nan at", argmax(is_nan))


####Adisc_ano_temp = Gp_in.tucker.center[is_nan[iω0,:,:]]
####
####Gp_ano = TCI4Keldysh.precompute_reg_div_values_MF_without_ωconv(1, Adisc_ano_temp, Gp_in.tucker.legs)
####
####Gp_in.Adisc_anoβ = fit_tucker_center(Gp_ano_data, Gp_in.tucker.legs)

####D= 2
####d = 1
####
####Kernels_ano = [Gp_in.tucker.legs[1:d-1]..., Gp_in.tucker.legs[d+1:D]...]
####        
####            values_ano = zeros(ComplexF64, size.(Kernels_ano, 1)...)
####            for dd in 1:D-1
####                Kernels_tmp = [Kernels_ano...]
####                #println("maxima before: ", maximum(abs.(Kernels_tmp[dd])))
####                Kernels_tmp[dd] = Kernels_tmp[dd].^2
####                #println("maxima after: ", maximum(abs.(Kernels_tmp[dd])))
####                values_ano .+= TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Kernels_tmp, Adisc_ano_temp)
####            end
####            values_ano .*= -0.5

sum(is_nan) > 0
#if sum(is_nan) > 0
    is_ωbos0 = is_nan[:,1,1]
    iω0 = argmax(is_nan)[1]
    #Adisc_ano_temp = Gp_in.tucker.center[is_ωbos0, [Colon() for _ in 2:D]...]
    Adisc_ano_temp = Gp_in.tucker.center[is_nan[iω0,:,:], [Colon() for _ in 3:D]...]
    println("size of Adisc_ano_temp: ", size(Adisc_ano_temp))

#    TCI4Keldysh.update_frequency_args!(Gp_out1)
    G_ano = TCI4Keldysh.precompute_reg_div_values_MF_without_ωconv(1, Adisc_ano_temp, Gp_out1.tucker.legs)
    #G_ano = reshape(G_ano, (1, size(G_ano)...))
    
    println("size of G_ano: ", size(G_ano))

    A_ano = TCI4Keldysh.fit_tucker_center(G_ano, Gp_out1.tucker.legs[2:end])

    view(G_1, iω0, [Colon() for _ in 2:D]...) .+= A_ano
    view(G_2, iω0, [Colon() for _ in 2:D]...) .+= A_ano

end
D = 2
G_ano
A_ano
view(G_1, iω0, [Colon() for _ in 2:D]...)
plot(dlr_fer.ωn[2:end], 2*reverse(-real.(G_ano))[1:end-1], xlims=[-0.2,0.2])
plot!(Gs[1].Gps[ip].tucker.ωs_legs[2], real.([vals_orig[imax[1],:]- ((vals_pfd1)[imax[1],:] + vals_pfd2[imax[1],:])]), labels=["orig" "new" "zero"])
plot(-real.(G_ano))
dlr_fer.ωn


# re-fit DLR-coefficients along free dimension for each partial fraction
G_1 = dropdims(sum(reshape(Gp_out1.tucker.center, (1, sz_center...)) .* Kernel_t, dims=3), dims=3)
G_2 = dropdims(sum(reshape(Gp_out1.tucker.center, (1, sz_center...)) .* Kernel_t, dims=2), dims=2)
Adisc_shift1 = reshape(Gp_out1.tucker.legs[1] \ reshape(G_1, (length(ωn1), div(length(G_1), length(ωn1)))), sz_center)
Adisc_shift2 = reshape(Gp_out1.tucker.legs[1] \ reshape(G_2, (length(ωn1), div(length(G_2), length(ωn1)))), sz_center)
# maximum(abs.(G_1 - Gp_out1.tucker.legs[1] * Adisc_shift1))
# maximum(abs.(G_2 - Gp_out1.tucker.legs[1] * Adisc_shift2))
Gp_out1.tucker.center = Adisc_shift1
Gp_out2.tucker.center = Adisc_shift2
if prefac == 1
    Gp_out1.tucker.center *= -1
else
    Gp_out2.tucker.center *= -1
end









# same for 3D: 

using Random

G3D.Gps[2].isFermi
G3D.Gps[2].tucker.ωs_center[1][12]
G3D.Gps[2].tucker.ωs_center[1][88]
vals = LinRange(-D,D,100)
view(G3D.Gps[2].tucker.center, 12:88, 12:88, 12:88)
randn!(view(G3D.Gps[2].tucker.center, 12:88, 12:88, 12:88))
Gp3D_in = deepcopy(G3D.Gps[2])
TCI4Keldysh.discreteLehmannRep4Gp!(Gp3D_in; dlr_bos, dlr_fer)


vals_Gp_3D_orig = TCI4Keldysh.precompute_reg_values_MF_without_ωconv(G3D.Gps[2]);
vals_Gp_3D_dlrd = TCI4Keldysh.precompute_reg_values_MF_without_ωconv(Gp3D_in);
vals_Gp_3D_dlrd2= Gp3D_in.tucker[:,:,:];
maximum(abs.(vals_Gp_3D_dlrd - vals_Gp_3D_orig)) / maximum(abs.(vals_Gp_3D_dlrd))
maximum(abs.(vals_Gp_3D_dlrd2 - vals_Gp_3D_orig)) / maximum(abs.(vals_Gp_3D_dlrd))

vals_ano_Gp_3D_orig = TCI4Keldysh.precompute_ano_values_MF_without_ωconv(G3D.Gps[2]);
vals_ano_Gp_3D_dlrd = TCI4Keldysh.precompute_ano_values_MF_without_ωconv(Gp3D_in);
maximum(abs.(vals_ano_Gp_3D_dlrd - vals_ano_Gp_3D_orig)) / maximum(abs.(vals_ano_Gp_3D_orig))




idx1 = 2
idx2 = 3

Gp3D_out1, Gp3D_out2 = TCI4Keldysh.partial_fraction_decomp(Gp3D_in; idx1, idx2, dlr_bos, dlr_fer);
vals_3D_orig = TCI4Keldysh.precompute_all_values_MF(Gp3D_in);
vals_3D_pfd1 = TCI4Keldysh.precompute_all_values_MF(Gp3D_out1);
vals_3D_pfd2 = TCI4Keldysh.precompute_all_values_MF(Gp3D_out2);
diff_3D = vals_pfd1+vals_pfd2-vals_orig;
maximum(abs.(diff_3D))

Gp3D_in.ωconvMat
Gp3D_out1.ωconvMat
Gp3D_out2.ωconvMat
Gp3D_in.isFermi
Gp3D_out2.isFermi

Gp3D_in.Adisc_anoβ

Gp3D_out2.Adisc_anoβ

# replace Gp with its partial fraction decomposition
G_out = deepcopy(Gs[1])
G_out.Gps[2] = Gp_out1
push!(G_out.Gps, Gp_out2)

vals_orig = TCI4Keldysh.precompute_all_values(Gs[1])
vals_pfrd = TCI4Keldysh.precompute_all_values(G_out)
maximum(abs.(vals_orig- vals_pfrd))





Gs_data = [TCI4Keldysh.precompute_all_values(Gs[i]) for i in 1:2]
K1ts_data = [TCI4Keldysh.precompute_all_values(K1ts[i]) for i in 1:2]


K1ts_data_phys = K1ts_data[1] + K1ts_data[2], K1ts_data[1] - K1ts_data[2]
Gs_data_phys = Gs_data[1] + Gs_data[2], Gs_data[1] - Gs_data[2]

maximum(abs.(imag.(Gs_data_phys[1])))
maximum(abs.(real.(Gs_data_phys[1])))
using Plots
plot(real.([K1ts_data_phys[1]- T*sum(Gs_data_phys[1], dims=2)*U]))
plot(real.([K1ts_data_phys[2], T*sum(Gs_data_phys[2], dims=2)*(-U)]))

heatmap(log.(abs.(imag.(Gs_data_phys[1]))))

sum(Gs_data_phys[1], dims=2)


########################
###### Reduce number of partial correlators
########################
G_in = deepcopy(Gs[1])
TCI4Keldysh.reduce_Gps!(G_in)


G_in_data = TCI4Keldysh.precompute_all_values(G_in)
G_predata = TCI4Keldysh.precompute_all_values(Gs[1])

maximum(abs.(G_in_data - G_predata))
maximum(abs.(imag.(G_in_data - G_predata)))
argmax(abs.(real.(G_in_data - G_predata)))
using Plots
plot(real.([G_in_data_ano[1025,:], G_predata_ano[1025,:]]))
plot(real.(G_in_data - G_predata)[1025,:])
plot(real.(G_in_data - G_predata)[:,2048])


###########################
#### compress with ID ####
###########################
atol = 1e-1
rtol = 1e-6
interpDecomps = [TCI4Keldysh.interpolDecomp4TD(G_in.Gps[ip]; atol, rtol) for ip in 1:3];


###########################
#### compress with DLR ####
###########################
Gp_in = deepcopy(Gp)

using Lehmann
# create DLR grids for bosonic and fermionic MF frequencies
β = 1/T
rtol=1e-8
symmetry = :none
Euv = D
dlr_fer = DLRGrid(Euv, β, rtol, true) #initialize the DLR parameters and basis
dlr_bos = DLRGrid(Euv, β, rtol, false) #initialize the DLR parameters and basis
maximum(abs.(dlr_fer.ω-dlr_bos.ω))


TCI4Keldysh.discreteLehmannRep4Gp!(Gp_in; dlr_bos, dlr_fer)

dev(g1, g2, idxs...) = maximum(abs.(g1.tucker[idxs...]-g2.tucker[idxs...]))
dev(Gp, Gp_in, 4000, :)
dev(Gp, Gp_in, :, 8000)



G_in.Gps[1].Adisc



Kernel_t = 1. ./ (reshape(G_in.ωs_ext[1], (1,1,length(G_in.ωs_ext[1]))) .- G_in.Gps[3].ωdiscs[1] .+ G_in.Gps[3].ωdiscs[2]')
is_nan = .!isfinite.(Kernel_t)
Kernel_t[is_nan] .= 0
G_1 = transpose(dropdims(sum(Adisc .* Kernel_t, dims=2), dims=2))
G_2 = transpose(dropdims(sum(Adisc .* Kernel_t, dims=1), dims=1))
p_iωs1[1]-p_iωs2[1]
Adisc_shift1 = Kernels_new1[1] \ G_1[p_iωs1[1],:]
Adisc_shift2 = Kernels_new1[1] \ G_2[p_iωs1[1],:]

maximum(abs.(G_1[p_iωs1[1],:] - Kernels_new1[1] * Adisc_shift1))
maximum(abs.(G_1 - G_in.Gps[1].Kernels[1][:,dlrs[1].p_ωdiscs[1]] * Adisc_shift1))
maximum(abs.(G_2 - G_in.Gps[1].Kernels[1][:,dlrs[1].p_ωdiscs[1]] * Adisc_shift2))
p_iωs1[1]
G_in.ωs_ext[1][p_iωs1[1]]

###################
### DLR: ##########
###################
rtols = 10. .^[-2, -3, -4, -5, -6, -7, -8, -9, -10]
rtols = 10. .^[-4]
N = length(rtols)
Nbasis_DLR1 = zeros(Int, N)
Nbasis_DLR2 = zeros(Int, N)
Gpdata_orig = Gp[:,:]
absdev_DLR = zeros(N)

for ir in 1:N
    atol = 1e2
    rtol = rtols[ir]
    Kernels_new, Adisc_new, p_iωs, p_ωdiscs = TCI4Keldysh.interpolDecomp4TD(Gp; atol, rtol);
    Nbasis_DLR1[ir] = length(p_iωs[1])
    Nbasis_DLR2[ir] = length(p_iωs[2])
    
    Kernels_new_large = [Gp.Kernels[i][:,p_ωdiscs[i]] for i in 1:2]
    Gpdata_DLR = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Kernels_new_large, Adisc_new)
    absdev_DLR[ir] = maximum(abs.(Gpdata_orig - Gpdata_DLR))
    
end

atol=1e-2;
rtol=rtols[1];
Kernels_new, Adisc_new, p_iωs, p_ωdiscs = TCI4Keldysh.interpolDecomp4TD(Gp; atol, rtol);
sizes_kernels = size.(Kernels_new)
Kernel2D = reshape(Kernels_new[1], (sizes_kernels[1][1], 1, sizes_kernels[1][2], 1)) .* reshape(Kernels_new[2], (1, sizes_kernels[2][1], 1, sizes_kernels[2][2]))
size_Kernel2D = size(Kernel2D)
Kernel2D = reshape(Kernel2D, (prod(size_Kernel2D[1:2]), prod(size_Kernel2D[3:4])))

# apply ID to 2D-Kernel again:
K_in = deepcopy(Kernel2D)
p_ωdisc = TCI4Keldysh.interp_decomp(K_in; atol, rtol)
K_interm = K_in[:,p_ωdisc]
println("length(p_ωdisc): ", length(p_ωdisc))
p_iω = TCI4Keldysh.interp_decomp(transpose(K_interm); atol, rtol, ncols_min=length(p_ωdisc))
K_new = K_interm[p_iω,:]
Gp_data_DLR = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Kernels_new, Adisc_new)[:]
adisc_DLR = TCI4Keldysh.linear_least_squares(K_new, Gp_data_DLR[p_iω])
Gp_data_reconstr = Kernel2D[:,p_ωdisc] * adisc_DLR

maximum(abs.(Gp_data_reconstr - Gp_data_DLR))


using Plots
plot(rtols, [Nbasis_DLR1, Nbasis_DLR2], labels=["Nbasis kernel bos" "Nbasis kernel fer"], xscale=:log10, xlabel="rtol", ylabel="Nbasis")
plot(absdev_DLR, [Nbasis_DLR1, Nbasis_DLR2], labels=["Nbasis kernel bos" "Nbasis kernel fer"], xscale=:log10, xlabel="adev", ylabel="Nbasis")
#savefig("scripts/plots/DLR42DGp.pdf")


###################
### SVD: ##########
###################
atols = 10. .^[-2, -3, -4, -5, -6, -7, -8, -9, -10]
N_svd = length(atols)
Nbasis_SVD1 = zeros(Int, N_svd)
Nbasis_SVD2 = zeros(Int, N_svd)
absdev_SVD = zeros(N_svd)

TCI4Keldysh.shift_singular_values_to_center!(Gp)

for i in 1:N_svd
    Gp_temp = deepcopy(Gp)
    TCI4Keldysh.svd_trunc_Adisc!(Gp_temp; atol=atols[i])
    Nbasis_SVD1[i], Nbasis_SVD2[i] = size(Gp_temp.Adisc)

    Gpdata_SVD = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Gp_temp.Kernels, Gp_temp.Adisc)
    absdev_SVD[i] = maximum(abs.(Gpdata_orig - Gpdata_SVD))
    
end

plot(absdev_DLR, [Nbasis_DLR1, Nbasis_DLR2], labels=["Nbasis kernel bos" "Nbasis kernel fer"], xscale=:log10, xlabel="adev", ylabel="Nbasis")
plot!(absdev_SVD, [Nbasis_SVD1, Nbasis_SVD2], labels=["Nbasis kernel bos (SVD)" "Nbasis kernel fer (SVD)"], xscale=:log10, xlabel="adev", ylabel="Nbasis")
savefig("scripts/plots/compare_compression_SVD_DLR.pdf")

Nbasis_SVD1



##########################################################
##### TCI ################################################
##########################################################

tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
colors = reshape(1:N_tols, (1, N_tols))
N_tols = length(tolerances)
linkdims = Vector{Vector{Int}}(undef, N_tols)
errors_Gs = zeros(N_tols)

for (it, tol) in enumerate(tolerances)
    #tolerance = 1e-5
    qtt, ranks, errors = quanticscrossinterpolate(
            Gs_data[1][1:end-1,:],
            tolerance=tol;
            maxiter=400
        ) 

    linkdims[it] = TCI.linkdims(qtt.tci)
    errors_Gs[it] = maximum(abs.(qtt[:,:]-Gs_data[1][1:end-1,:]))

end


linkdims_Gp = Vector{Vector{Int}}(undef, N_tols)
errors_Gp = zeros(N_tols)
linkdims_Gprot = Vector{Vector{Int}}(undef, N_tols)
errors_Gprot = zeros(N_tols)

Gp_data = Gs[1].Gps[2].tucker[:,:][1:end-1,1024:3071]
Gprot_data = TCI4Keldysh.precompute_all_values_MF(Gs[1].Gps[2])[1:end-1,:]
for (it, tol) in enumerate(tolerances)
    #tolerance = 1e-5
    qtt, ranks, errors = quanticscrossinterpolate(
            Gp_data,
            tolerance=tol
        ) 

    linkdims_Gp[it] = TCI.linkdims(qtt.tci)
    errors_Gp[it] = maximum(abs.(qtt[:,:]-Gp_data))


    qtt, ranks, errors = quanticscrossinterpolate(
        Gprot_data,
        tolerance=tol
    ) 

    linkdims_Gprot[it] = TCI.linkdims(qtt.tci)
    errors_Gprot[it] = maximum(abs.(qtt[:,:]-Gprot_data))

end

Gs[1].Gps[2].ωconvMat


plot(tolerances, [errors_Gs, errors_Gp, errors_Gprot], labels=["G" "Gp" "Gprot"], xscale=:log10, yscale=:log10, seriestype=:scatter)
plot!(tolerances, labels="diagonal", tolerances, xscale=:log10, yscale=:log10, seriestype=:path, linestyle=:dash, xlabel="tol (TCI)", ylabel="abs. dev.")
savefig("scripts/plots/Gp_error_vs_tol.pdf")

plot(linkdims, labels=string.(tolerances'), linecolor=colors)
plot!(linkdims_Gp, labels=string.(tolerances'); linestyle=:dash, linecolor=colors)
plot!(linkdims_Gprot, labels=string.(tolerances'); linestyle=:dot, linecolor=colors, xlabel="link", ylabel="linkdims")
savefig("scripts/plots/Gp_linkdims_vs_tol.pdf")

heatmap(log.(abs.(real.(Gp_data))), c=:bluesreds)
heatmap(log.(abs.(real.(Gprot_data))), c=:bluesreds)


using LinearAlgebra
singvals_bos = zeros(N_tols)
singvals_fer = zeros(N_tols)
u, s_bos, v = svd(Gs[1].Gps[2].tucker.legs[1])
u, s_fer, v = svd(Gs[1].Gps[2].tucker.legs[2])
for (it, tol) in enumerate(tolerances)
    println("|s_bos>tol=$tol|: \t", sum(s_bos .> tol), "\t|s_fer>tol=$tol|: \t", sum(s_fer .> tol))
    singvals_bos[it] = sum(s_bos .> tol)
    singvals_fer[it] = sum(s_fer .> tol)
end


dlr_Nbasis_bos = zeros(N_tols)
dlr_Nbasis_fer = zeros(N_tols)

for (it, tol) in enumerate(tolerances)
    interpDecomps = TCI4Keldysh.interpolDecomp4TD(Gs[1].Gps[2].tucker; atol=tol, rtol=1e-1);
    dlr_Nbasis_bos[it] = size(interpDecomps.center, 1)
    dlr_Nbasis_fer[it] = size(interpDecomps.center, 2)
end

plot(tolerances, [singvals_bos, singvals_fer], labels=["bos. kernel (SVD)" "fer. kernel (SVD)"], xscale=:log10)
plot!(tolerances, [dlr_Nbasis_bos, dlr_Nbasis_fer], labels=["bos. kernel (ID)" "fer. kernel (ID)"], linestyle=:dash, xscale=:log10, xlabel="tol(SVD/ID)", ylabel="N_basis")
savefig("scripts/plots/Nbasis_SVD_vs_ID.pdf")