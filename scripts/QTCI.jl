using HDF5
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCI4Keldysh
using MAT
using JLD

begin 
function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end

function save_plotdata(filename, qtt, qttdata2D, qplotmesh, ωcont, dict_K)
    file = h5open(filename, "w")
    file["qttdata2D"] = qttdata2D
    file["qplotmesh"] = qplotmesh
    file["ωcont"] = ωcont
    file["pivoterrors"] = qtt.tt.pivoterrors
    file["linkdims"] = TCI.linkdims(qtt.tt)
    #save("qtt_dict_"*Kclass_str*".jld", "qtt_dict", dict_K)
    g_K = create_group(file, "parameters")
    for k in keys(dict_K)
        g_K[string(k)] = dict_K[k]

    end
    close(file)
    return nothing
end


function qtci_my_PSF(broadenedpsf::TCI4Keldysh.BroadenedPSF{1}, qmesh, tolerance)
    return quanticscrossinterpolate(
        Float64,
        x -> broadenedpsf(x),
        [qmesh],
        tolerance=tolerance
    )    
end
function qtci_my_PSF(broadenedpsf::TCI4Keldysh.BroadenedPSF{2}, qmesh, tolerance)
    return quanticscrossinterpolate(
        Float64,
        (x,y) -> broadenedpsf(x,y),
        [qmesh, qmesh],
        tolerance=tolerance
    )    
end
function qtci_my_PSF(broadenedpsf::TCI4Keldysh.BroadenedPSF{3}, qmesh, tolerance)
    return quanticscrossinterpolate(
        Float64,
        (x,y,z) -> broadenedpsf(x,y,z),
        [qmesh, qmesh, qmesh],
        tolerance=tolerance
    )    
end
end


filename_4p = "data/PSF_nz=2_conn_zavg/PSF_((F1,F1dag,Q34)).mat"
filename_4p = "data/PSF_nz=2_conn_zavg/PSF_((Q1,Q3,F1dag,F3dag)).mat"
filename_4p = "data/PSF_nz=4_conn_zavg_U=5Delta/PSF_((F1dag,F1,F3dag,F3)).mat"
#f = matopen("./PSF_nz=2_conn_zavg/PSF_((F1,F1dag)).mat")        # 2p function
#f = matopen("data/PSF_nz=2_conn_zavg/PSF_((F1,F1dag,Q34)).mat")      # 3p function
#f = matopen("data/PSF_nz=2_conn_zavg/PSF_((F1,F1dag,F3,F3dag)).mat") # 4p function
f = matopen(filename_4p) # 4p function

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
    g = T * 0.5
    tol = 1.e-14
    estep = 600
    emin = 1e-6; emax = 1e2;
    Lfun = "FD" 
    is2sum = false
    verbose = false


    dict_K3 = Dict("R"=>10, "PSFdata"=>filename_4p)
    R_K3 = dict_K3["R"]
    qR_K3 = 2^R_K3
    qmesh = collect(Int, range(1, qR_K3, length=qR_K3));
    Nωcont_pos = div(qR_K3,2)
    #ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωcont = get_ωcont(D*2., Nωcont_pos)

    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)

    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

end;



tol_high = 1e-3
tol_low  = 1e-4
qttK3a , _, _ = qtci_my_PSF(broadenedPsf,qmesh,tol_high)
qttK3a2, _, _ = qtci_my_PSF(broadenedPsf,qmesh,tol_low )


### evaluate on some slice --> can be plotted
Length = 218
qplotmesh = Int.(round.(range(2^(R_K3-1)-250, 2^(R_K3-1)+250, length=Length)))
qttdata = qttK3a.(qplotmesh, qplotmesh',2^(R_K3-1))
qttdata2 = qttK3a2.(qplotmesh, qplotmesh',2^(R_K3-1))
vmax = maximum(abs.(qttdata))
kwargs = Dict(:vmax=>vmax, :vmin=>-vmax)
save_plotdata("data/plotdata_4pPSFs1.h5", qttK3a , qttdata, qplotmesh, ωcont, dict_K3)
save_plotdata("data/plotdata_4pPSFs2.h5", qttK3a2, qttdata, qplotmesh, ωcont, dict_K3)

save("data/qtts.jld", "qtt1", qttK3a)
save("data/qtts.jld", "qtt2", qttK3a2)
save("data/qtts.jld", "broadenedPsf", broadenedPsf)




begin
    fig, axs = subplots(ncols=2, nrows=2, figsize=(300, 250)./72)

    axs[1, 1].set_title(L"\mathrm{Re}(K_{3a;\, \omega,\nu,\nu'})")
    axs[1, 1].imshow(real(qttK3adata)'; kwargs...) |> colorbar
    #axs[1, 2].set_title(L"\mathrm{Im}(K_{3a})")
    #axs[1, 2].imshow(imag(qttK3adata)'; extent=box, kwargs...) |> colorbar

    axs[1, 1].set_ylabel(L"\nu'")
    for ax in axs[1, :]
        ax.set_xlabel(L"\nu")
    end
    axs[2, 1].semilogy(1:TCI.rank(qttK3a.tt), qttK3a.tt.pivoterrors, label=L"\epsilon=10^{-3}")
    axs[2, 1].semilogy(1:TCI.rank(qttK3a2.tt), qttK3a2.tt.pivoterrors, label=L"\epsilon=10^{-4}")
    axs[2, 1].set_xlabel(L"D_{\max}")
    axs[2, 1].set_ylabel("abs. error")

    axs[2, 2].semilogy(1:3R_K3-1, 2 .^ min.(1:3R_K3-1, 3R_K3-1:-1:1), color="gray", linewidth=0.5)
    axs[2, 2].semilogy(1:3R_K3-1, TCI.linkdims(qttK3a.tt))
    axs[2, 2].semilogy(1:3R_K3-1, TCI.linkdims(qttK3a2.tt))
    axs[2, 2].set_xlabel(L"\ell")
    axs[2, 2].set_ylabel(L"D_\ell")
    #axs[2, 2].set_ylim(1.5, 300)
    axs[2, 2].set_xticks([1, 10, 19])

    axs[2, 1].legend()
    fig.suptitle("QTCI of 4p PSF")
    tight_layout()


    fig.savefig("PSF_4p.pdf")
end