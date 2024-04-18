###
# Try out compression with different methods
###
using Revise
using TCI4Keldysh
using HDF5

TCI4Keldysh.VERBOSE() = true
TCI4Keldysh.DEBUG() = false


begin

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
    # Keldysh paper:    u=0.5 OR u=1.0
    U = 1. / 20.
    T = 0.01 * U
    #Δ = (U/pi)/0.5

    Rpos = 10
    R = Rpos + 1
    Nωcont_pos = 2^Rpos # 512#
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))
    
    # get functor which can evaluate broadened data pointwisely
    #broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    ωbos = im * π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = im * π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    ωs_ext = (ωbos, ωfer)
    ωconv = [
         1  0;
         -1  -1
    ]
    Gp = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, ωs_ext, ωconv)
end

###################
### DLR: ##########
###################
rtols = 10. .^[-2, -3, -4, -5, -6, -7, -8, -9, -10]
N = length(rtols)
Nbasis_DLR1 = zeros(Int, N)
Nbasis_DLR2 = zeros(Int, N)
Gpdata_orig = Gp[:,:]
absdev_DLR = zeros(N)

for ir in 1:N
    atol = 1e2
    rtol = rtols[ir]
    Kernels_new, Adisc_new, p_iωs, p_ωdiscs = TCI4Keldysh.discreteLehmann4TD(Gp; atol, rtol);
    Nbasis_DLR1[ir] = length(p_iωs[1])
    Nbasis_DLR2[ir] = length(p_iωs[2])
    
    Kernels_new_large = [Gp.Kernels[i][:,p_ωdiscs[i]] for i in 1:2]
    Gpdata_DLR = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Kernels_new_large, Adisc_new)
    absdev_DLR[ir] = maximum(abs.(Gpdata_orig - Gpdata_DLR))
    
end

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