###
# Try out compression with different methods
###
using Revise
using TCI4Keldysh
using HDF5

TCI4Keldysh.VERBOSE() = true
TCI4Keldysh.DEBUG() = false


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

    Rpos = 13
    R = Rpos + 1
    Nωcont_pos = 2^Rpos # 512#
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))
    
    # get functor which can evaluate broadened data pointwisely
    #broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos*2:Nωcont_pos*2-1) * 2 .+ 1)
    ωs_ext = (ωbos, ωfer)
    ωconv = [
         1  0;
         -1  -1
    
         ]
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Gs      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "F3", "F3dag"]; flavor_idx=i, ωs_ext=(ωbos,ωfer), ωconvMat=ωconvMat_K2′t, name="SIAM 3pG", is_compactAdisc=false) for i in 1:2];
    K1ts    = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "Q34"]; flavor_idx=i, ωs_ext=(ωbos,), ωconvMat=ωconvMat_K1t, name="SIAM 2pG") for i in 1:2];
    Gp = Gs[1].Gps[1]#TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, ωs_ext, ωconv)
end




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