using Revise
using MAT
using TCI4Keldysh
#using Plots
using HDF5



function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    #Δωcont = get_Δω(ωcont)
    #Acont = zeros((ones(Int, D).*(Nωcont_pos*2+1))...)
    return ωcont
end

function get_Δω(ωs)
    Δωs = [ωs[2] - ωs[1]; (ωs[3:end] - ωs[1:end-2]) / 2; ωs[end] - ωs[end-1]]  # width of the frequency bin
    return Δωs 
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

f = matopen("./PSF_nz=2_conn_zavg/PSF_((F1,F1dag)).mat")        # 2p function
f = matopen("data/PSF_nz=2_conn_zavg/PSF_((F1,F1dag,Q34)).mat")      # 3p function
f = matopen("data/PSF_nz=2_conn_zavg_u=1.00/PSF_((F1dag,F1,Q34)).mat")
f = matopen("data/PSF_nz=4_conn_zavg_U=5Delta/PSF_((F1dag,F1,Q34)).mat")
f = matopen("PSF_nz=2_conn_zavg/PSF_((F1,F1dag,F3,F3dag)).mat") # 4p function
f = matopen("PSF_nz=2_conn_zavg/PSF_((Q1,Q3,F1dag,F3dag)).mat") # 4p function
f = matopen("data/PSF_nz=2_conn_zavg_u=1.00/PSF_((F1dag,F1,F3dag,F3)).mat")
f = matopen("data/PSF_nz=4_conn_zavg_U=5Delta/PSF_((F1dag,F1,F3dag,F3)).mat")

k = keys(f)


begin
   
    Adisc = read(f, "Adisc")[1]
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))
    ωdisc = read(f, "odisc")[:]
    
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
    estep = 2048
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Nωcont_pos = 2^13 # 512#
    #ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωcont = get_ωcont(D*2., Nωcont_pos)

    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)
end;


@time ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
#broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

filename_broadened = "data/precomputedAcont_estep"*string(estep)*"_Nw"*string(Nωcont_pos)
save_Acont(filename_broadened, ωdisc, Adisc, ωcont, Acont)