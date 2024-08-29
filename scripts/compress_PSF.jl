using TCI4Keldysh
using JSON
using ITensors
using Plots
using LaTeXStrings
using QuanticsTCI
import TensorCrossInterpolation as TCI
using Combinatorics
using Measures

#=
Plot and analyze compression of PSF data.
=#

function PSF_2pt(;beta=2000.0, R=12)

    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=2
    Ops = ["F1", "F1dag"]
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)
    GF = only(GFs)

    perm_idx = 1
    
    # plot Adisc
    Adisc = GF.Gps[perm_idx].tucker.center
    ωdisc = GF.Gps[perm_idx].tucker.ωs_center
    # scatter(only(ωdisc), real.(Adisc); markersize=4, marker=:diamond, xscale=:identity)
    p = histogram(only(ωdisc); weights=real.(Adisc), color=:gray, bins=200)
    xlabel!(p,"ϵ")
    ylabel!(p,"Spectral weight")
    title!(p,"Sample 2point-PSF")
    savefig(p, "PSF_sample1D.png")

    # compress Adisc
end

function PSF_4pt_magnitude()
    # PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    PSFpath = "data/siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/"
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    beta = TCI4Keldysh.dir_to_beta(PSFpath)

    
    # histogram of spectral weights
    Adisc = TCI4Keldysh.load_Adisc(joinpath(PSFpath, "4pt"), Ops, 1)
    ωdisc = TCI4Keldysh.load_ωdisc(joinpath(PSFpath, "4pt"), Ops; nested_ωdisc=false)
    _, ωdiscs, Adisc = TCI4Keldysh.compactAdisc(ωdisc, Adisc)
    max_Adisc = maximum(abs.(Adisc))
    @show max_Adisc
    npeaks = prod(size(Adisc))
    @show npeaks
    p = histogram(reshape(-1 .* log10.(abs.(Adisc) ./ max_Adisc), npeaks); label="Adisc")
    xlabel!("-log10(Adisc)")
    savefig(p, "Adisc_hist.png")

    corrbound = copy(Adisc)
    min_ωfer_sq = (π/beta)^2
    has_zero_eps = [minimum(abs.(omdisc)) <= 1.e-10 for omdisc in ωdiscs]
    for ic in CartesianIndices(corrbound)
        # bosonic
        if has_zero_eps[1]
            corrbound[ic] = 0.0
        else
            corrbound[ic] /= abs(ωdiscs[1][ic[1]])
        end
        # fermionic
        for i in 2:3
            corrbound[ic] /= sqrt(min_ωfer_sq + ωdiscs[i][ic[i]]^2)
        end
    end
    max_corrbound = maximum(abs.(corrbound))
    @show max_corrbound
    p = histogram(reshape(-1 .* log10.(abs.(corrbound) ./ max_corrbound), npeaks); label="GFbound")
    xlabel!("-log10(GFbound)")
    savefig(p, "corrbound_hist.png")
end

function PSF_4pt(;beta=2000.0, R=7, tolerance=1.e-6)

    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=4
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)
    GF = GFs[1]

    perm_idx = 1
    
    # plot Adisc
    Adisc = GF.Gps[perm_idx].tucker.center
    ωdisc = GF.Gps[perm_idx].tucker.ωs_center
    @show size.(ωdisc)
    slice_idx = argmax(abs.(Adisc))[3]
    max_weight = maximum(abs.(Adisc))
    @show argmax(abs.(Adisc))
    @show slice_idx
    # p = histogram2d(ωdisc[1], ωdisc[2]; weights=real.(Adisc[:,:,slice_idx]), color=:gray, bins=200)
    # p = heatmap(ωdisc[1], ωdisc[2], real.(Adisc[:,:,slice_idx]))
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    p = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    p = heatmap!(p, Adisc[:,:,slice_idx] .|> abs .|> (x -> max(1.e-8, x)) .|> (x -> x/max_weight) .|> log10)
    # xlabel!(p, L"\epsilon_2")
    # ylabel!(p, L"\epsilon_1")
    xlabel!(p, L"n_2")
    ylabel!(p, L"n_1")
    title!(p, L"\log_{10}\left|\!\! S_p\right|")
    savefig(p, "PSF_sample3D.png")

    # compress Adisc
    Adisc_size = size(Adisc)
    tucker = GF.Gps[perm_idx].tucker
    function evalAdisc(i...)
        if all(i .<= Adisc_size)        
            return tucker.center[i...]
        else
            return zero(Float64)
        end
    end
    padded_size = ntuple(i -> 2^R, ndims(tucker.center))
    println("  Interpolating...")
    qtt_Adisc, _, _ = quanticscrossinterpolate(
        eltype(tucker.center),
        evalAdisc,
        padded_size;
        tolerance=tolerance)
    mps_Adisc = TCI4Keldysh.QTCItoMPS(qtt_Adisc, ntuple(i->"eps$i", npt-1))

    worst = TCI4Keldysh.worstcase_bonddim(fill(2, length(mps_Adisc)))
    ld = ITensors.linkdims(mps_Adisc)
    rk = maximum(ld)
    printstyled("\n  ---- Bonddims (quantics): $ld\n"; color=:blue)
    printstyled("\n  ---- Rank (quantics): $rk\n"; color=:blue)
    printstyled("  ---- No. of elements in largest array: $(2*rk^2)\n"; color=:blue)

    tfont = 12
    gfont = 16
    lfont = 12
    chiplot = plot(;guidefontsize=gfont, tickfontsize=tfont, legendfontsize=lfont)
    msize = 6
    plot!(chiplot, 1:length(mps_Adisc)-1, ld; label="4pt-PSF", marker=:diamond, markersize=msize, color=:black, yscale=:log10)
    plot!(chiplot, 1:length(mps_Adisc)-1, worst; label="worst case", linestyle=:dash, color=:gray, yscale=:log10)
    yticks!(chiplot, 10.0 .^ (0:3))
    xlabel!(chiplot, "bond index")
    ylabel!(chiplot, "χ")
    savefig(chiplot, "PSF_bonddim_4pt.png")
end

"""
Compress PSF in natural representation
x--x--x
|  |  |
n1 n2 n3
"""
function compress_PSF_natural(; tolerance=1.e-8)

    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/"
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    perm_idx = 1
    p = collect(permutations(collect(1:4)))[perm_idx]
    Adisc = TCI4Keldysh.load_Adisc(PSFpath, Ops[p], 1)

    initpivot = collect(Tuple(argmax(abs.(Adisc))))
    localdims = collect(size(Adisc))
    tt, _, _ = TCI.crossinterpolate2(ComplexF64, i -> Adisc[i...], localdims, [initpivot]; tolerance=tolerance)

    ld = TCI.linkdims(tt)
    printstyled("\n  ---- Worst case: $(size(Adisc))\n"; color=:blue)
    printstyled("  ---- Bond dims: $ld\n"; color=:blue)
    printstyled("  ---- No. of elements in largest array: $(size(Adisc)[2] * prod(ld))\n"; color=:blue)
end

function PSF_3pt(;beta=2000.0, R=7)

    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=3
    Ops = ["F1", "F1dag", "Q34"]
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)
    GF = GFs[1]

    perm_idx = 1
    
    # plot Adisc
    Adisc = GF.Gps[perm_idx].tucker.center
    ωdisc = GF.Gps[perm_idx].tucker.ωs_center
    @show size.(ωdisc)
    max_weight = maximum(abs.(Adisc))
    tfont = 18
    titfont = 30
    gfont = 34
    lfont = 18
    p = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    cbar = false
    p = heatmap!(p, Adisc .|> abs .|> (x -> max(1.e-8, x)) .|> (x -> x/max_weight) .|> log; colorbar=cbar)
    # xlabel!(p,L"n_2", margin=5mm)
    # ylabel!(p,L"n_1", margin=5mm)
    xlabel!(p,L"n_2")
    ylabel!(p,L"n_1")
    plot!(p; xformatter=:none)
    plot!(p; yformatter=:none)
    # title!(p, L"\log_{10}|\!\!S_{n_1n_2}|")
    # title!(p, L"S_{n_1n_2}", margin=5mm)
    savefig(p,"PSF_sample2D_label.png")

end

function default_plot()
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    p = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    return p
end