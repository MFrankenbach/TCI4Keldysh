using Revise
using TCI4Keldysh
using Quantics
using QuanticsTCI
import TensorCrossInterpolation as TCI
using QuanticsGrids
using LinearAlgebra
using ITensors
using HDF5

TCI4Keldysh.VERBOSE() = true


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

    σ = 0.6
    sigmab = [σ]
    g = T * 1.
    tol = 1.e-14
    estep = 2048
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Rpos = 6
    R = Rpos + 1
    Nωcont_pos = 2^Rpos # 512#
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))
    
    # get functor which can evaluate broadened data pointwisely
    #broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    ωbos = im * π * T * collect(-Nωcont_pos+1:Nωcont_pos-1) * 2
    ωbos = im * π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = im * π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    ωs_ext = (ωbos, ωfer)
    ωconv = [
         1  0;
         -1  -1
    ]
    Gp = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, ωs_ext, ωconv)
end

qtt = TCI4Keldysh.TDtoQTCI(Gp)                  # convert Tucker decomposition to QTT
qtt_orig = deepcopy(qtt)

tags =("ω", "ν")
sites = TCI4Keldysh.getsitesforqtt(qtt; tags)
mps = TCI4Keldysh.TCItoMPS(qtt.tci; sites)       # convert QTT to MPS

qtt_unrot = TCI4Keldysh.MPStoQTCI(mps)          # convert MPS back to QTT and check that no information got lost
qttunrotdat = qtt_unrot[:,:]
qttcompar_unrot = (Gp[:,:])[1:end-1,65:192]
maximum(abs.(qttcompar_unrot - qttunrotdat))
plot([real.(qttcompar_unrot[64,:] - qttunrotdat[64,:]), imag.(qttcompar_unrot[64,:] - qttunrotdat[64,:])])

using Plots
heatmap(real.(qttunrotdat))
heatmap(real.(qttcompar_unrot[1:end-1, :]))

#############################################
### frequency conversion in MPS language: ###
#############################################
begin
    tags = collect(tags)
    halfN = 2^(qtt.grid.R - 1)

    mps_rot = Quantics.affinetransform(
        mps,
        tags,
        [
            Dict(tags[1] => 1, tags[2] => 0),
            Dict(tags[1] => 1, tags[2] => 1),
        ],
        [0, halfN],
        [1, 1]
    )

end

begin
    tags = collect(tags)
    halfN = 2^(qtt.grid.R - 1)

    mps_rot = Quantics.affinetransform(
        mps,
        tags,
        [
            Dict(tags[1] => 1, tags[2] => 0),
            Dict(tags[1] => 1, tags[2] => 1),
        ],
        [0, half],
        [1, 1]
    )

end

begin
    tags = collect(tags)
    halfN = 2^(qtt.grid.R - 1)

    mps_rot = Quantics.affinetransform(
        mps,
        tags,
        [
            Dict(tags[1] => 1, tags[2] => 0),
            Dict(tags[1] => 1, tags[2] => -1),
        ],
        [0, halfN-1],
        [1, 1]
    )

end


begin
    tags = collect(tags)
    halfN = 2^(qtt.grid.R - 1)

    mps_rot = Quantics.affinetransform(
        mps,
        tags,
        [
            Dict(tags[1] => 1, tags[2] => 0),
            Dict(tags[1] => -1, tags[2] => -1),
        ],
        [0, halfN-1],
        [1, 1]
    )

end

#d1 = qtt[:,:]
#mps_temp = TCI4Keldysh.TCItoMPS(qtt.tci)
#qtt_recov = TCI4Keldysh.MPStoQTCI(mps_temp)
#d2 = qtt_recov[:,:]
#heatmap(real.(d1))
#heatmap(abs.(d2-d1))

### convert MPS to fat tensor:
sitesx = [sites[findfirst(x -> hastags(x, "$(tags[1])=$n"), sites)] for n in 1:R]
sitesy = [sites[findfirst(x -> hastags(x, "$(tags[2])=$n"), sites)] for n in 1:R]
g_arr = Array(reduce(*, mps_rot), vcat(reverse(sitesx), reverse(sitesy)));
g_arr = Array(reduce(*, mps_rot), vcat(sitesx, sitesy));
g_vec = reshape(g_arr, 2^R, 2^R);
heatmap(real.(g_vec))

#qtt_rot = TCI4Keldysh.MPStoQTCI(mps_rot)
#qtt_prerot = TCI4Keldysh.MPStoQTCI(mps)
#qttrotdat = qtt_rot[:,:]
#qttprerotdat = qtt_prerot[:,:]
qttcompar_rot = TCI4Keldysh.precompute_all_values(Gp)[1:end-1,:]    # compare freq-rotated MPS with classically rotated data
#maximum(real.(qttrotdat))
#maximum(real.(qttprerotdat))

#heatmap(real.(qttrotdat))
#heatmap(real.(qttprerotdat))
heatmap(real.(qttcompar_rot))

plot(real.([g_vec[64,10:end], qttcompar_rot[64,10:end]]))#, xlim=[55,65])
heatmap(abs.(g_vec - qttcompar_rot))
abs.(g_vec - qttcompar_rot)