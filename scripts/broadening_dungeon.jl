using Combinatorics
using HDF5
using Plots
using LinearAlgebra
using Printf
using TCI4Keldysh
using QuanticsTCI
import TensorCrossInterpolation as TCI

"""
β=2000.0 in Seung-Sup's data
"""
function default_T()
    return 1.0/(2000.0)
end

function plot_ImGR_2pt()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    base_path = dirname(rstrip(PSFpath, '/'))
    npt = 2
    D = npt-1
    T = TCI4Keldysh.dir_to_T(PSFpath)
    Ops = ["F1", "F1dag"]

    R = 12
    ωmax = 0.3183098861837907
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = TCI4Keldysh.ωconvMat_K1()
    flavor_idx = 1
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(base_path)
    KFC = TCI4Keldysh.FullCorrelator_KF(
        PSFpath, Ops;
        T=T, ωs_ext=ωs_ext, flavor_idx=flavor_idx, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken",
        emin=2.5e-5, emax=50.0, estep=50
        )

    KFCval = TCI4Keldysh.precompute_all_values(KFC)

    PSF = imag.(KFCval[:,2,1])

    plot(only(ωs_ext), PSF)
    savefig("PSF.pdf")
end


function load_AcontAdisc(
    γ::Float64, sigmak::Float64, ωmax::Float64=1.0;
    basepath=joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50"),
    PSFpath=joinpath(basepath, "PSF_nz=4_conn_zavg/"),
    estep::Int=40,
    npt::Int=2
    )
    # create full correlator to get Acont
    T = TCI4Keldysh.dir_to_T(PSFpath)
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)

    R = 5
    channel = "p"
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    ωconvMat = if npt==2
        TCI4Keldysh.ωconvMat_K1()
    elseif npt==3
        TCI4Keldysh.channel_trafo_K2(channel, false)
    else
        TCI4Keldysh.channel_trafo(channel)
    end
    flavor_idx = 1
    Acont_folder = "Acontdata"
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath)
    broadening_kwargs[:estep]=estep
    if npt==4
        PSFpath = joinpath(PSFpath,"4pt")
    end
    KFC = TCI4Keldysh.FullCorrelator_KF(
        PSFpath, Ops;
        write_Aconts=Acont_folder,
        T=T,
        ωs_ext=ωs_ext,
        flavor_idx=flavor_idx,
        ωconvMat=ωconvMat,
        sigmak=[sigmak],
        γ=γ,
        broadening_kwargs...
        )

    # read Adisc
    perm_idx = 1
    perms = collect(permutations(1:npt))
    Adiscs = [TCI4Keldysh.load_Adisc(PSFpath, Ops[p], flavor_idx) for (_,p) in enumerate(perms)]
    Adisc = Adiscs[perm_idx]
    ωdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)
    @assert length(ωdisc)==size(Adisc,1)

    # read Acont
    Acont = h5read(TCI4Keldysh.Acont_h5fname(perm_idx, D; Acont_folder=Acont_folder), "Acont$perm_idx") 
    # we have D=1
    ωcont = h5read(TCI4Keldysh.Acont_h5fname(perm_idx, D; Acont_folder=Acont_folder), "omcont$(perm_idx)1") 

    return (Adisc, ωdisc, Acont, ωcont)
end

"""
Plot broadening of 2pt PSF for different parameters.
"""
function broaden_2pt(γ::Float64, sigmak::Float64; T=5*1.e-4, outsuffix::String="", save=true)
    
    (Adisc, ωdisc, Acont, ωcont) = load_AcontAdisc(γ, sigmak)

    printstyled("  #ωdisc points: $(length(ωdisc))\n"; color=:blue)
    printstyled("  #ωcont points: $(length(ωcont))\n"; color=:blue)
    printstyled("Adisc norm: $(sum(Adisc))\n"; color=:blue)
    dωcont = diff(ωcont)
    push!(dωcont, dωcont[end])
    printstyled("Acont norm: $(TCI4Keldysh.quadtrapz(ωcont, Acont))\n"; color=:blue)
    # printstyled("Adisc mean: $(dot(Adisc, ωdisc))\n"; color=:blue)
    # printstyled("Acont mean: $(TCI4Keldysh.quadtrapz(ωcont, Acont .* ωcont))\n"; color=:blue)

    @show maximum(Acont)
    @show ωcont[argmax(Acont)]
    @show maximum(Adisc)
    @show ωdisc[argmax(Adisc)]
    @show size(Acont)
    @show size(Adisc)

    # plot Adisc, Acont
    scfun = real
    p = TCI4Keldysh.default_plot()
    title!(p, "Broadening, γ=$(@sprintf("%.2e",γ)), σk=$sigmak, T=$(@sprintf("%.2e", T))")
    xlabel!(p, "log(ω)")
    ptwin = twinx(p)
    xscfun(x) = abs(x>1.e-14) ? log10(abs(x)) : 0.0
    poscont = findall(x -> x>0.0, ωcont)
    posdisc = findall(x -> x>0.0, ωdisc)
    plot!(ptwin, xscfun.(ωdisc[posdisc]), scfun.(Adisc[posdisc]); label="Adisc", linewidth=1, color=:red)
    plot!(p, xscfun.(ωcont[poscont]), scfun.(Acont[poscont]); label="Acont", linewidth=2, color=:blue)
    ylabel!(p, "Acont")
    ylabel!(ptwin, "Adisc")
    # xlims!(p, minimum(ωcont)*0.5, maximum(ωcont)*1.05)
    if save
        savefig(p, "AcontvsAdisc"*outsuffix*".pdf")
    end

    return p
end

function param_sweep_broaden2pt()
    T = default_T()
    gammas = T .* [0.5, 5, 50]
    sigmas = [0.01, 0.1, 1.0]
    plots = []
    for γ in gammas
        for σ in sigmas
            (Adisc, ωdisc, Acont, ωcont) = load_AcontAdisc(γ, σ)
            # plot Adisc, Acont
            scfun = real
            p = plot(dpi=300, xticks=false, yticks=false)
            # title!(p, "γ=$(@sprintf("%.2e",γ)), σk=$σ, T=$(@sprintf("%.2e", T))")
            # xlabel!(p, "ω")
            ptwin = twinx(p)
            plot!(ptwin, ωdisc, scfun.(Adisc[:]); yticks=false, label=nothing, linewidth=0.2, color=:red)
            plot!(p, ωcont, scfun.(Acont[:]); label=nothing, linewidth=0.5, color=:blue)
            # ylabel!(p, "Acont")
            # ylabel!(ptwin, "Adisc")
            xlims!(p, minimum(ωcont)*0.1, maximum(ωcont)*1.00)
            push!(plots, p)
        end
    end
    plot(plots...; layout=(length(gammas), length(sigmas)))
    savefig("foo.png")
end

function TCI_Acont()
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath)
    γ /= 2.0
    sigmak = [0.2]
    ωmax = 1.0
    (Adisc, omdisc, Acont, omcont) = load_AcontAdisc(
        γ, only(sigmak), ωmax;
        PSFpath=PSFpath, basepath=basepath
        )

    # TCI compression on log grid
    tolerance = 1.e-4
    R = TCI4Keldysh.padding_R(size(Acont))
    @show size(Acont)
    @show R
    Acont_pad = TCI4Keldysh.zeropad_array(Acont, R)
    qttlog, _, _ = quanticscrossinterpolate(Acont_pad; tolerance=tolerance)
    @show TCI4Keldysh.rank(qttlog)
    qttfat = TCI4Keldysh.qinterleaved_fattensor_to_regular(
        TCI4Keldysh.qtt_to_fattensor(qttlog.tci.sitetensors),
        R
    )


    # plot
    ompos = omcont .> 0.0
    omnonpos = omcont .< 0.0
    plot(omcont[ompos], Acont[ompos]; linewidth=2, color="red", label="positive ϵ", xscale=:log10)
    plot!(abs.(omcont[omnonpos]), Acont[omnonpos]; linewidth=2, color="blue", label="negative ϵ", xscale=:log10)
    plot!(omcont[ompos], qttfat[1:length(Acont)][ompos]; linewidth=2, linestyle=:dot, color="violet", label="TCI: positive ϵ", xscale=:log10)
    plot!(abs.(omcont[omnonpos]), qttfat[1:length(Acont)][omnonpos]; linewidth=2, linestyle=:dot, color="cyan", label="TCI: negative ϵ", xscale=:log10)
    xlabel!("log10|ϵ|")
    title!("Broadened PSF, log grid")
    savefig("Acontlog.pdf")

    # TCI compression on linear grid
    tolerance = 1.e-4
    # linearly interpolate Acont
    lin_interp = TCI4Keldysh.linear_interpolation(omcont, Acont)
    Rlin = 18
    wsize = 0.3
    omcont_lin = range(-wsize, wsize; length=2^Rlin)
    Acont_lin = lin_interp(omcont_lin)
    @show size(Acont)
    qttlin, _, _ = quanticscrossinterpolate(Acont_lin; tolerance=tolerance)
    qttfat = TCI4Keldysh.qinterleaved_fattensor_to_regular(
        TCI4Keldysh.qtt_to_fattensor(qttlin.tci.sitetensors),
        Rlin
    )
    @show TCI4Keldysh.rank(qttlin)

    # plot
    omcont_window = abs.(omcont) .< wsize
    plot(omcont[omcont_window], Acont[omcont_window]; linewidth=2, color="red", label="non-TCI")
    plot!(omcont_lin, qttfat; linewidth=2, color="violet", linestyle=:dot, label="TCI")
    xlabel!("ϵ")
    title!("Broadened PSF, linear grid")
    savefig("Acontlin.pdf")
end