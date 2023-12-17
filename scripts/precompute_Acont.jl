using Revise
using MAT
using TCI4Keldysh
#using Plots
using HDF5



function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end


function save_Acont(filename, ωdisc, Adisc, ωcont, Acont)
    f_plot = h5open(filename*".h5", "w")
    f_plot["Adisc"] = Adisc
    f_plot["ωdisc"] = ωdisc
    f_plot["Acont"] = Acont
    f_plot["ωcont"] = ωcont
    close(f_plot)
    return nothing
end

PSFpath = "data/PSF_nz=2_conn_zavg/"

# 2p function
Ops = ["F1", "F1dag"  ]

# 3p function
Ops = ["F1", "F1dag", "Q34"]
Ops = ["F1dag", "F1", "Q34"]

# 4p function
Ops = ["F1", "F1dag", "F3", "F3dag"]


begin
   
    Adisc = TCI4Keldysh.load_Adisc(PSFpath, Ops, 1)
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))
    ωdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)
    
    ### System parameters of SIAM ### 
    D = 1.
    # Keldysh paper:    u=0.5 OR u=1.0
    U = 1. / 20.
    T = 0.01 * U
    Δ = (U/pi)/0.5
    # EOM paper:        U=5*Δ
    Δ = 0.1
    U = 0.5*Δ
    T = 0.01*Δ

    ### Broadening ######################
    #   parameters      σ       γ       #
    #       by JaeMo:   0.3     T/2     #
    #       by SSL:     0.6     T*3     #
    #   my choice:                      #
    #       u = 0.5:    0.6     T*3     #
    #       u = 1.0:    0.6     T*2     #
    #       u = 5*Δ:    0.6     T*0.5   #
    #####################################
    σ = 0.6
    sigmab = [σ]
    g = T * 1.
    tol = 1.e-14
    estep = 512
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Nωcont_pos = 2^6 # 512#
    #ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωcont = get_ωcont(D*2., Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))

    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)
end;


@time ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

GKeldysh = TCI4Keldysh.FullCorrelator_KF(PSFpath, ["F1", "F1dag"  ]; flavor_idx=1, ωs_ext=(ωcont,), ωconvMat=reshape([1 ; -1], (2, 1)), name="SIAM G", sigmak=sigmab, γ=g, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
GKeldysh = TCI4Keldysh.FullCorrelator_KF(PSFpath, ["F1", "F1dag", "Q34"]; flavor_idx=1, ωs_ext=(ωcont,ωcont,), ωconvMat=[1 1; 0 -1; -1 0], name="SIAM 3pG", sigmak=sigmab, γ=g, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

## evaluate Keldysh correlator:
gRtmp(idx) = TCI4Keldysh.evaluate(GKeldysh, 2, idx)
GRtmp_data = gRtmp.(collect(axes(ωcont)[1]))



## evaluate MF correlator:
#GM = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag", "F3", "F3dag"]; flavor_idx=1, ωs_ext=(ωcont.* im,ωcont.* im,ωcont.* im,), ωconvMat=[0 1 0;
#                                                                                                            -1 -1 0; 
#                                                                                                             0 0 -1;
#                                                                                                             1 0 1], name="SIAM 4pG");
#GM_dat = TCI4Keldysh.precompute_all_values(GM)
#GM = TCI4Keldysh.FullCorrelator_MF(PSFpath, Ops; flavor_idx=1, ωs_ext=(ωcont.* im,), ωconvMat=reshape([1 ; -1], (2, 1)), name="SIAM G")
#GMtmp_data = GM.(collect(axes(ωcont)[1]))


filename_broadened = "data/precomputedAcont_estep"*string(estep)*"_Nw"*string(Nωcont_pos)
save_Acont(filename_broadened, ωdisc, Adisc, ωcont, Acont)