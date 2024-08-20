using Combinatorics
using HDF5
using Plots
using LinearAlgebra
using Printf

"""
β=2000.0 in Seung-Sup's data
"""
function default_T()
    return 1.0/(2000.0)
end


function load_AcontAdisc(γ::Float64, sigmak::Float64; T=5*1.e-4)
    # create full correlator to get Acont
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    npt = 2
    D = npt-1
    Ops = ["F1", "F1dag"]

    R = 9
    ωmax = 2.5
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = TCI4Keldysh.ωconvMat_K1()
    flavor_idx = 1
    Acont_folder = "Acontdata"
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; write_Aconts=Acont_folder, T=T, ωs_ext=ωs_ext, flavor_idx=flavor_idx, ωconvMat=ωconvMat, sigmak=[sigmak], γ=γ, name="Kentucky fried chicken")

    # read Adisc
    perm_idx = 1
    perms = collect(permutations(1:npt))
    Adiscs = [TCI4Keldysh.load_Adisc(PSFpath, Ops[p], flavor_idx) for (_,p) in enumerate(perms)]
    Adisc = Adiscs[perm_idx]
    ωdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)
    @assert length(ωdisc)==size(Adisc,1)

    # read Acont
    Acont = h5read(TCI4Keldysh.Acont_h5fname(perm_idx, D; Acont_folder=Acont_folder), "Acont$perm_idx") 
    ωcont = only(KFC.Gps[perm_idx].tucker.ωs_legs)

    return (Adisc, ωdisc, Acont, ωcont)
end

"""
Plot broadening of 2pt PSF for different parameters.
"""
function broaden_2pt(γ::Float64, sigmak::Float64; T=5*1.e-4, outsuffix::String="", save=true)
    
    (Adisc, ωdisc, Acont, ωcont) = load_AcontAdisc(γ, sigmak; T=T)

    printstyled("  #ωdisc points: $(length(ωdisc))\n"; color=:blue)
    printstyled("  #ωcont points: $(length(ωcont))\n"; color=:blue)
    printstyled("Adisc norm: $(sum(Adisc))\n"; color=:blue)
    dωcont = diff(ωcont)
    push!(dωcont, dωcont[end])
    printstyled("Acont norm: $(dot(dωcont, Acont))\n"; color=:blue)

    # plot Adisc, Acont
    scfun = real
    p = TCI4Keldysh.default_plot()
    title!(p, "Broadening, γ=$(@sprintf("%.2e",γ)), σk=$sigmak, T=$(@sprintf("%.2e", T))")
    xlabel!(p, "ω")
    ptwin = twinx(p)
    plot!(ptwin, ωdisc, scfun.(Adisc[:]); label="Adisc", linewidth=1, color=:red)
    plot!(p, ωcont, scfun.(Acont[:]); label="Acont", linewidth=2, color=:blue)
    ylabel!(p, "Acont")
    ylabel!(ptwin, "Adisc")
    xlims!(p, minimum(ωcont)*0.5, maximum(ωcont)*1.05)
    if save
        savefig(p, "AcontvsAdisc"*outsuffix*".png")
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
            (Adisc, ωdisc, Acont, ωcont) = load_AcontAdisc(γ, σ; T=T)
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