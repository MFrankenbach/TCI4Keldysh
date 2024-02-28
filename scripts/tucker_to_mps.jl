using Revise
using TCI4Keldysh
TCI4Keldysh.TIME() = true
TCI4Keldysh.DEBUG() = true
TCI4Keldysh.VERBOSE() = true
using QuanticsTCI
using Quantics



function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end



# 2p function
PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
Ops = ["Q12", "Q34"  ]

# 3p function
#PSFpath = "data/PSF_nz=2_conn_zavg/"
Ops = ["F1", "F1dag", "Q34"]
#Ops = ["F1dag", "F1", "Q34"]

# 4p function
PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/"
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
    σ = 0.3
    sigmab = [σ]
    g = T * 0.5
    tol = 1.e-14
    estep = 512
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Rpos = 6
    R = Rpos+1
    Nωcont_pos = 2^Rpos
    #ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωcont = get_ωcont(D*2., Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))

    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)
end;


#@time ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);


# check 3D version:
mps_broadenedPsf = TCI4Keldysh.TD_to_MPS_via_TTworld(broadenedPsf)

broadened_via_TT_world = TCI4Keldysh.MPS_to_fatTensor(mps_broadenedPsf; tags=("ω1", "ω2", "ω3"))

broadened_oldschool = broadenedPsf[:,:,:]
diff = broadened_via_TT_world - broadened_oldschool[1:end-1,1:end-1,1:end-1]

using Plots
heatmap(broadened_via_TT_world[65,:,:])
heatmap(broadened_oldschool[65,:,:])

plot([broadened_oldschool[65,65,:], broadened_via_TT_world[65,65,:], diff[65,65,:]], labels=["old" "MPS" "diff"])
plot([broadened_oldschool[65,66,:], broadened_via_TT_world[65,66,:], diff[65,66,:]], labels=["old" "MPS" "diff"])



# check 2D version: 
mps_broadenedPsf = TCI4Keldysh.TD_to_MPS_via_TTworld(broadenedPsf)

broadened_via_TT_world = TCI4Keldysh.MPS_to_fatTensor(mps_broadenedPsf; tags=("ω1", "ω2"))

broadened_oldschool = broadenedPsf[:,:]
diff = broadened_via_TT_world - broadened_oldschool[1:end-1,1:end-1]
maximum(abs.(diff))

using Plots
heatmap(broadened_via_TT_world[:,:])
heatmap(broadened_oldschool[:,:])

plot([broadened_oldschool[65,:], broadened_via_TT_world[65,:], diff[65,:]], labels=["old" "MPS" "diff"])


