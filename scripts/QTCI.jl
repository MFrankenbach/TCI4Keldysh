using HDF5
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCI4Keldysh
using MAT
using JLD
using Plots

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
    #Δ = 0.1
    #U = 0.5*Δ
    #T = 0.01*Δ

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
    emin = 1e-6; emax = 1e3;
    Lfun = "FD" 
    is2sum = false
    verbose = false


    dict_K3 = Dict("R"=>10, "PSFdata"=>filename_4p)
    R_K3 = dict_K3["R"]
    qR_K3 = 2^R_K3
    qmesh = collect(Int, range(1, qR_K3, length=qR_K3));
    Nωcont_pos = div(qR_K3,2)
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    #ωcont = get_ωcont(D*2., Nωcont_pos)

    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)

    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

end;

qttdat = broadenedPsf.(qmesh,qmesh')
#inputfile = load("./strange_lowres.h5")
#qttdat = inputfile["qttdat"]
#qttdat = qttdat[126:126+511, 126:126+511]
#qttdat = reverse(qttdat)
qmesh = collect(1:size(qttdat, 1))

dataaxes = collect(collect.(axes(qttdat)))
qttK3a, ranks, errors = quanticscrossinterpolate(Float64, broadenedPsf        , dataaxes; tolerance=1e-6)
#qttK3a2, ranks, errors = quanticscrossinterpolate(Float64, (i,j) -> qttdat[i,j], dataaxes; tolerance=1e-6)
qttdata = qttK3a.(qmesh,qmesh')


save_plotdata("data/plotdata_4pPSFs1.h5", qttK3a , qttdata , qmesh, ωcont, dict_K3)

#save("data/qtts.jld", "qtt1", qttK3a)
#save("data/qtts.jld", "qtt2", qttK3a2)
#save("data/qtts.jld", "broadenedPsf", broadenedPsf)



