using TCI4Keldysh
using JSON
using ITensors
using Plots
using LaTeXStrings
using QuanticsTCI

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
    xlabel!(p, L"\epsilon_1")
    ylabel!(p, L"\epsilon_2")
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
    p = heatmap(Adisc .|> real .|> abs .|> (x -> max(1.e-8, x)) .|> (x -> x/max_weight) .|> log)
    xlabel!(p,"ϵ1")
    ylabel!(p,"ϵ2")
    title!(p,"Sample 3-point-PSF")
    savefig(p,"PSF_sample2D.png")

end
