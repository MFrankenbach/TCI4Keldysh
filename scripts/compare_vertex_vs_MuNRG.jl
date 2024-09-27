using MAT
using Plots
using LinearAlgebra
using QuanticsTCI
using JSON
using HDF5
using Combinatorics
using LaTeXStrings
import TensorCrossInterpolation as TCI

#=
Compare our MF/KF vertices with MuNRG results.
Results in principle depend on which estimators (symmetric, left-asymmetric, right-asymmetric) are used for the self-energies,
but the difference is minor.
=#

function check_V_MF_CFdat()
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/V_MF_pp")

    for file in readdir(Vpath; join=true)
        matopen(file, "r") do f
            try
                keys(f)
            catch
                keys(f)
            end
            CFdat = read(f, "CFdat")
            println("For file: $file")
            @show size(CFdat["Ggrid"][1])
            @show size.(CFdat["ogrid"])
            println("")
        end
    end
end

function precompute_K1r_explicit(PSFpath::String, flavor_idx::Int, formalism="MF"; ωs_ext::Vector{Float64}, channel="t")

    T = TCI4Keldysh.dir_to_T(PSFpath)
    ops = if channel=="t"
        # G(12,34)
        ["Q12", "Q34"]
    elseif channel in ["p", "pNRG"]
        # -ζ G(13,24)
        ["Q13", "Q24"]
    elseif channel=="a"
        # -G(14,23)
        ["Q14", "Q23"]
    else
        error("Invalid channel")
    end
    G = TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T=T, flavor_idx=flavor_idx, ωs_ext=(ωs_ext,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    sign = if channel=="t"
        1
    elseif channel in ["p", "pNRG"]
        -1
    elseif channel=="a"
        -1
    else
        error("Invalid channel")
    end

    res = zeros(ComplexF64, length(ωs_ext))
    resdiv = zeros(ComplexF64, length(ωs_ext))
    zero_id = div(length(ωs_ext), 2) + 1
    for Gp in G.Gps
        # regular
        Adisc = Gp.tucker.center
        omdisc = only(Gp.tucker.ωs_center)

        kmul = if only(Gp.ωconvMat)==1
            1 ./ (im * ωs_ext .- omdisc')
        else
            1 ./ (im * reverse(ωs_ext) .- omdisc')
        end
        isdivergent = .!isfinite.(kmul)
        kmul[isdivergent] .= zero(ComplexF64)
        zero_omdisc = (omdisc .!= 0.0)
        kmul[:,zero_omdisc] .*= omdisc[zero_omdisc]'
        Adisc_div = copy(Adisc)
        Adisc_div[zero_omdisc] ./= omdisc[zero_omdisc]
        resdiv += kmul * Adisc_div


        k = if only(Gp.ωconvMat)==1
            1 ./ (im * ωs_ext .- omdisc')
        else
            1 ./ (im * reverse(ωs_ext) .- omdisc')
        end
        isdivergent = .!isfinite.(k)
        k[isdivergent] .= zero(ComplexF64)
        res += k * Adisc

        # # explicit multiplication, ordered by magnitudes
        # # DOES NOT make a difference
        # for w in axes(k,1)
        #     k_act = vec(k[w,:])
        #     res_act = k_act .* Adisc
        #     perm = sortperm(abs.(res_act); rev=true)
        #     for ip in perm
        #         res2[w] += res_act[ip]
        #     end
        # end

        # anomalous
        ano_id = findfirst(o -> abs(o)<1.e-12, omdisc)
        if !isnothing(ano_id)
            ano_term = 0.5 * Adisc[ano_id] / T
            @show ano_term
            res[zero_id] -= ano_term
            resdiv[zero_id] -= ano_term
        end


        println("== Res comparison")
        @show norm(res .- resdiv)

    end
    return sign * res
end

"""
K1 for each channel on 1D frequency grid
"""
function precompute_K1r(PSFpath::String, flavor_idx::Int, formalism="MF"; ωs_ext::Vector{Float64}, channel="t")
    if formalism!="MF"
        error("NYI")
    end

    T = TCI4Keldysh.dir_to_T(PSFpath)
    ops = if channel=="t"
        # G(12,34)
        ["Q12", "Q34"]
    elseif channel in ["p", "pNRG"]
        # -ζ G(13,24)
        ["Q13", "Q24"]
    elseif channel=="a"
        # -G(14,23)
        ["Q14", "Q23"]
    else
        error("Invalid channel")
    end
    G = TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T=T, flavor_idx=flavor_idx, ωs_ext=(ωs_ext,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    sign = if channel=="t"
        1
    elseif channel in ["p", "pNRG"]
        -1
    elseif channel=="a"
        -1
    else
        error("Invalid channel")
    end

    return sign * TCI4Keldysh.precompute_all_values(G)
end

"""
Comparison to MuNRG
K1 is independent of the bosonic frequency in all channels.
"""
function check_K1_MF(;channel="t")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50", "V_MF_" * TCI4Keldysh.channel_translate(channel))
    
    # load K1
    flavor = 1
    K1 = nothing
    grid = nothing
    channel_id = if channel=="t"
        1
    elseif channel=="pNRG"
        2
    elseif channel=="a"
        3
    else
        error("Invalid channel $channel")
    end
    matopen(joinpath(Vpath, "V_MF_U2_$(channel_id).mat")) do f
        CFdat = read(f, "CFdat")
        K1 = CFdat["Ggrid"][flavor]
        grid = CFdat["ogrid"]
        @show size(K1)
        @show size(grid)
    end

    # extract 1D
    K1D = K1[1,1,:]
    @show maximum(abs.(K1))
    # check that K1 slices are actually constant
    for i in axes(K1,3)
        K1ref = K1[1,1,i]
        @assert all(abs.(K1[:,:,i] .- K1ref) .<= 1.e-16) "K1 slice no. $i nonconstant"
    end


    # TCI4Keldysh
    ωs_ext = imag.(vec(grid[end]))
    @assert isodd(length(ωs_ext)) "Grid is not bosonic"
    K1_test = precompute_K1r(PSFpath, flavor; channel=channel, ωs_ext=ωs_ext)

    # explicitly by hand
    K1_expl = precompute_K1r_explicit(PSFpath, flavor; channel=channel, ωs_ext=ωs_ext)
    @show maximum(abs.(K1_expl .- K1_test))
    plot(ωs_ext, real.(K1_expl); label="Re")
    plot!(ωs_ext, imag.(K1_expl); label="Im")
    savefig("K1_expl.png")

    # plot
    p = TCI4Keldysh.default_plot()
    plot!(p, real.(K1D); label="Re, MuNRG")
    plot!(p, imag.(K1D); label="Im, MuNRG")
    plot!(p, real.(K1_test); label="Re, Julia", linestyle=:dash)
    plot!(p, imag.(K1_test); label="Im, Julia", linestyle=:dash)
    title!(p, "K1@$(channel)-channel: MuNRG vs Julia")
    savefig("K1_comparison.pdf")

    # diff
    diff = K1D .- K1_test
    diff_shift = K1D[2:end] .- K1_test[1:end-1]
    @show maximum(abs.(diff))
    p = TCI4Keldysh.default_plot()
    plot!(p, real.(diff))
    title!(p, "K1@$(channel)-channel: real(MuNRG-Julia)";label="")
    savefig("K1_diff.pdf")
end

"""
Comparison to MuNRG
"""
function check_Σ_MF(; channel="t", use_ΣaIE=true)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50", "V_MF_" * TCI4Keldysh.channel_translate(channel))

    # load self-energies
    flavor = 1
    Σnames = ["SE_MF_$i.mat" for i in 1:4]
    Σs = Vector{ComplexF64}[]
    grids = Vector{Float64}[]
    for Σname in Σnames
        matopen(joinpath(Vpath, Σname), "r") do f
            CFdat = read(f, "CFdat")
            Σ_act = CFdat["Ggrid"][flavor]
            grid_act = CFdat["ogrid"][flavor]
            push!(Σs, vec(Σ_act))
            push!(grids, vec(imag.(grid_act)))
        end
    end
    # check whether all frequency grids are the same
    @assert all(length.(grids) .== length(first(grids)))
    for g in grids[2:4]
        @assert maximum(abs.(g .- grids[1])) < 1.e-12
    end

    # TCI4Keldysh
    T = TCI4Keldysh.dir_to_T(PSFpath)
    Nhalf = div(length(grids[1]),2)
    ω_fer = TCI4Keldysh.MF_grid(T, Nhalf, true)
    (ΣL, ΣR) = TCI4Keldysh.calc_Σ_MF_aIE(PSFpath, ω_fer; flavor_idx=flavor, T=T)

    println("==== ΣL vs. ΣR @ TCI4Keldysh")
    @show (length(ΣL), length(ΣR))
    @show maximum(abs.(ΣL .- ΣR))
    println("==== Σ@MuNRG vs. Σ@TCI4Keldysh")
    for sig in Σs[1:1]
        @show maximum(abs.(ΣL .- sig))
        @show maximum(abs.(real.(ΣL .- sig)))
        @show maximum(abs.(imag.(ΣL .- sig)))
        amax = argmax(abs.(ΣL .- sig))
        display(diff(sig[amax-5:amax+5]))
    end
    @show maximum(abs.(ω_fer .- grids[1]))
    println("====")

    # plot
    p = TCI4Keldysh.default_plot()
    for i in eachindex(grids)
        plot!(p, grids[i], real.(Σs[i]); label="Re,i=$i")
        plot!(p, grids[i], imag.(Σs[i]); label="Im,i=$i", legend=:topright)
    end
    plot!(p, ω_fer, real.(ΣL); linestyle=:dash, label="Re(ΣL)")
    plot!(p, ω_fer, imag.(ΣL); linestyle=:dash, label="Im(ΣL)")
    plot!(p, ω_fer, real.(ΣR); linestyle=:dash, label="Re(ΣR)")
    plot!(p, ω_fer, imag.(ΣR); linestyle=:dash, label="Im(ΣR)")
    savefig("SE_comparison.png")
    p = TCI4Keldysh.default_plot()
    plot!(p, grids[1], imag.(ΣL .- Σs[1]); linestyle=:dash, label="Im(ΣMuNRG - ΣJulia)_∞")
    println("==== Examine diff")
    display((ΣL .- Σs[1])[Nhalf-2:Nhalf+3])
    display((ω_fer)[Nhalf-2:Nhalf+3])
    println("====")
    savefig("SE_diff.png")
end

"""
Compare MuNRG Matsubara vertices with TCI4Keldysh.
CAREFUL: Need channel="pNRG" for p-channel to get a consistent frequency convention
"""
function check_V_MF(Nhalf=2^4;channel="t", use_ΣaIE=true)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50", "V_MF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_MF_U4.mat"
    CF = nothing
    Γcore_ref = nothing
    ωs_ext = nothing
    spin = 1
    matopen(joinpath(Vpath, core_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
        # bosonic grid comes last in the data
        ωs_ext = ntuple(i -> imag.(vec(vec(CFdat["ogrid"])[4-i])), 3)
    end
    # bosonic grid comes last in the data
    Γcore_ref = permutedims(Γcore_ref, (3,2,1))
    @show size.(ωs_ext)
    @show size(Γcore_ref)

    # Σ data
    ωs_Σ = nothing
    Σ_file = "SE_MF_1.mat"
    matopen(joinpath(Vpath, Σ_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        ωs_Σ_ = vec(vec(CFdat["ogrid"])[1])
        @assert norm(real.(ωs_Σ_)) <= 1.e-10
        ωs_Σ = imag.(ωs_Σ_)
    end

    @show size(ωs_Σ)
    @show typeof(ωs_Σ)

    # TCI4Keldysh calculation

    T = TCI4Keldysh.dir_to_T(PSFpath)
    om_small = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)
    om_sig = TCI4Keldysh.MF_grid(T, 2*Nhalf, true)

    # Γ core
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    @time testval = if use_ΣaIE
        G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(om_sig,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_auxL   = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(om_sig,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_auxR   = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(om_sig,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");

        G_data = TCI4Keldysh.precompute_all_values(G)
        G_auxL_data = TCI4Keldysh.precompute_all_values(G_auxL)
        G_auxR_data = TCI4Keldysh.precompute_all_values(G_auxR)

        Σ_calcR = TCI4Keldysh.calc_Σ_MF_aIE(G_auxR_data, G_data)
        Σ_calcL = TCI4Keldysh.calc_Σ_MF_aIE(G_auxL_data, G_data)

        TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "MF", PSFpath*"4pt/", Σ_calcR;
            Σ_calcL=Σ_calcL, ωs_ext=om_small, T=T, ωconvMat=ωconvMat, flavor_idx=spin
            )
    else # use sIE for self-energy
        Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(PSFpath, om_sig; flavor_idx=spin, T=T)
        TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "MF", PSFpath*"4pt/", Σ_calc_sIE;
            ωs_ext=om_small, T=T, ωconvMat=ωconvMat, flavor_idx=spin
            )
    end
    
    # calulation DONE

    scfun = x -> real(x)

    slice = [div(length(om_small[1]), 2)+1, :, :]
    @show om_small[1][slice[1]]
    heatmap(scfun.(testval[slice...]); right_margin=10Plots.mm)
    title!("Γcore TCI4Keldysh")
    savefig("gam.png")


    window_half = div(length(om_small[2]), 2)
    data_half = div(length(ωs_ext[2]), 2)
    window_slice = data_half-window_half+1:data_half+window_half
    slice_ref = [div(length(ωs_ext[1]), 2)+1, window_slice, window_slice]
    @show ωs_ext[1][slice_ref[1]]
    heatmap(scfun.(-Γcore_ref[slice_ref...]); right_margin=10Plots.mm)
    title!("Γcore reference")
    savefig("ref.png")

    # compare quantitatively
    window = (data_half-window_half+1:data_half+window_half+1, data_half-window_half+1:data_half+window_half, data_half-window_half+1:data_half+window_half)
    diff = if channel in ["t", "a"]
        Γcore_ref[window...] .- testval
    else # p channel, NRG convention; somehow signs don't agree
        Γcore_ref[window...] .+ testval
    end
    maxdiff = maximum(abs.(diff)) 
    amaxdiff = argmax(abs.(diff)) 
    @show amaxdiff
    @show diff[amaxdiff]
    @show testval[amaxdiff]
    @show -Γcore_ref[window...][amaxdiff]
    printstyled("---- Max. abs. deviation: $(maxdiff) (Γcore value: $(testval[amaxdiff]))\n"; color=:blue)
    # difference comes from real part
    scfun = x -> real(x)
    heatmap(scfun.(Γcore_ref[slice_ref...] .+ testval[slice...]); right_margin=10Plots.mm)
    savefig("diff.png")

    reldiff = real.(testval) ./ real.(Γcore_ref)[window...]
    reldiff = map(x -> ifelse(!isnan(x) && (1. / 1.1 < abs(x)<1.1), x, 1.0), reldiff)
    @show maximum(abs.(reldiff))
    mean = sum(reldiff) / length(reldiff)
    @show mean
    heatmap(abs.(reldiff[slice...]); right_margin=10Plots.mm)
    savefig("reldiff.png")
    return maxdiff
end

function check_V_MF_all()
   check_V_MF(2^4;channel="t") 
   check_V_MF(2^4;channel="a") 
   check_V_MF(2^4;channel="pNRG") 
end

"""
Check Keldysh vertex of TCI4Keldysh against MuNRG results.
MuNRG results have frequency grids of size 2n+1 symmetric around 0.0
"""
function check_V_KF(Nhalf=2^3; iK::Int=2, channel="t")
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), base_path, "V_KF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_KF_U4.mat"
    CF = nothing
    Γcore_ref = nothing
    ωs_ext = nothing
    spin = 1
    matopen(joinpath(Vpath, core_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
        ωs_ext = ntuple(i -> real.(vec(vec(CFdat["ogrid"])[i])), 3)
    end
    iK_tuple = TCI4Keldysh.KF_idx(iK, 3)
    Γcore_ref = permutedims(Γcore_ref, (3,1,2, 4,5,6,7))
    @show size.(ωs_ext)
    @show size(Γcore_ref)

    # Σ data
    ωs_Σ = nothing
    Σ_file = "SE_KF_1.mat"
    matopen(joinpath(Vpath, Σ_file), "r") do f
        CFdat = read(f, "CFdat")
        ωs_Σ = vec(vec(CFdat["ogrid"])[1])
    end
    @show size(ωs_Σ)

    # test
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(base_path; channel=channel)
    @show (γ, only(sigmak))
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)

    ωs_cen = [div(length(om), 2)+1 for om in ωs_ext]
    @show [ωs_ext[i][ωs_cen[i]] for i in eachindex(ωs_ext)]
    om_small = ntuple(i -> ωs_ext[i][ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf], 3)
    ω_Σ_cen = div(length(ωs_Σ), 2) + 1
    om_sig = ωs_Σ[ω_Σ_cen - 2*Nhalf : ω_Σ_cen + 2*Nhalf]

    Σ_ref = TCI4Keldysh.calc_Σ_KF_sIE_viaR(PSFpath, om_sig; T=T, flavor_idx=spin, sigmak, γ)
    testval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "KF",
        PSFpath*"4pt/",
        Σ_ref
        ;
        T,
        flavor_idx = spin,
        ωs_ext = om_small,
        ωconvMat=ωconvMat,
        sigmak, γ
    )

    # plot
    window = ntuple(i -> ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf, 2)
    slice_ref = [div(length(ωs_ext[1]), 2)+1, window..., iK_tuple...]
    @show ωs_ext[1][slice_ref[1]]
    @show ωs_ext[2][slice_ref[2][1:2]]
    @show ωs_ext[3][slice_ref[3][1:2]]
    heatmap(abs.(Γcore_ref[slice_ref...]); clim=(0.0, 0.0042))
    title!("Γcore reference")
    savefig("ref.png")

    # # scan all Keldysh components
    # for iK_ in 1:15
    #     heatmap(abs.(Γcore_ref[vcat(slice_ref[1:3], collect(TCI4Keldysh.KF_idx(iK_,3)))...]))
    #     savefig("ref_iK=$(iK_).png")
    # end

    slice = [div(length(om_small[1]), 2)+1, :, :, iK_tuple...]
    @show om_small[1][slice[1]]
    @show om_small[2][slice[2]]
    @show om_small[3][slice[3]]
    heatmap(abs.(testval[slice...]); clim=(0.0, 0.0042))
    title!("Γcore TCI4Keldysh")
    savefig("gam.png")

    # # scan through permutations of frequency arguments (permuting MuNRG data by p=[3,1,2] seems fine)
    # for p in permutations(1:3)
    #     heatmap(abs.(testval[slice[vcat(p, collect(4:7))]...]))
    #     savefig("gam_p=$p.png")
    # end

    # diff = testval .- Γcore_ref[vcat(fill(window,3), collect(iK_tuple))...]
    diff_slice = testval[slice...] .- Γcore_ref[slice_ref...]
    # Nqu = div(Nhalf, 2)
    # @show testval[slice...][Nqu+2,Nqu]
    # @show Γcore_ref[slice_ref...][Nqu+2,Nqu]
    diff_slice_p = testval[slice...] .+ Γcore_ref[slice_ref...]
    maxref_slice = maximum(abs.(Γcore_ref))
    @show maxref_slice
    heatmap(log10.(abs.(diff_slice) ./ maxref_slice))
    savefig("diff.png")
    heatmap(log10.(abs.(diff_slice_p) ./ maxref_slice))
    savefig("diff_p.png")
end

function load_Γcore_KF(base_path::String = "SIAM_u=0.50"; channel="t")
    Vpath = joinpath(TCI4Keldysh.datadir(), base_path, "V_KF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_KF_U4.mat"
    Γcore_ref = nothing
    spin = 1
    matopen(joinpath(Vpath, core_file), "r") do f
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
    end
    return Γcore_ref
end

function precomp_compr_filename(iK::Int, tolerance::Float64)
    return "vertex_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).h5"    
end

function compress_precomputed_V_KF(Γcore_ref::Array{ComplexF64,7}, iK::Int, channel="t"; store=false, tcikwargs...)

    iK_tuple = TCI4Keldysh.KF_idx(iK, 3)
    Γcore_ref = permutedims(Γcore_ref, (1,2,3, 4,5,6,7))
    R5slice = 100-63:100+64
    to_tci = Γcore_ref[R5slice, R5slice, R5slice, iK_tuple...]
    # to_tci = TCI4Keldysh.zeropad_array(Γcore_ref[:,:,:,iK_tuple...])
    @show size(Γcore_ref)

    qtt, _, _ = quanticscrossinterpolate(to_tci; tcikwargs...)

    println("Compression done")

    tolerance = Dict(tcikwargs)[:tolerance]
    if store
        # qtt_fat = zeros(ComplexF64, size(to_tci))
        # Threads.@threads for ic in CartesianIndices(to_tci)
        #     qtt_fat[ic] = qtt(Tuple(ic)...)
        # end
        qtt_fat = TCI4Keldysh.qtt_to_fattensor(qtt.tci.sitetensors)
        @show size(qtt_fat)
        qtt_fat = TCI4Keldysh.qinterleaved_fattensor_to_regular(qtt_fat, round(Int, log2(length(R5slice))))
        @show size(qtt_fat)
        h5open(joinpath(precompressed_datadir(), precomp_compr_filename(iK, tolerance)), "w") do fid
            fid["qttdata"] = qtt_fat
            fid["reference"] = to_tci
            fid["diff"] = qtt_fat .- to_tci
        end
    end

    @show TCI4Keldysh.rank(qtt)
    return TCI.linkdims(qtt.tci)
end

"""
plot reference - TCI - error
"""
function triptych_vertex(iK::Int, tolerance::Float64)
    fname = joinpath(precompressed_datadir(), precomp_compr_filename(iK, tolerance))
    ref = h5read(fname, "reference")
    tcival = h5read(fname, "qttdata")
    diff = h5read(fname, "diff")

    @assert size(ref)==size(tcival)==size(diff) "incompatible sizes"
    Nhalf = div(size(ref)[1], 2)

    maxref = maximum(abs.(ref))
    slice = (Nhalf, Colon(), Colon())

    scfun(x) = abs(x)
    p = TCI4Keldysh.default_plot()
    heatmap!(p, scfun.(ref[slice...]); right_margin=10Plots.mm)
    title!(p, L"$\Gamma_{\mathrm{core}}$: Reference")
    savefig("V_KFref_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).png")

    p = TCI4Keldysh.default_plot()
    heatmap!(p, scfun.(tcival[slice...]); right_margin=10Plots.mm)
    title!(p, L"$\Gamma_{\mathrm{core}}$: QTCI")
    savefig("V_KFtci_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).png")

    p = TCI4Keldysh.default_plot()
    @show maxref
    heatmap!(p, log10.(abs.(diff[slice...] ./ maxref)); right_margin=10Plots.mm)
    title!(p, L"Error: $|\Gamma_{\mathrm{core}}^{\mathrm{ref}} - \Gamma_{\mathrm{core}}^{\mathrm{QTCI}}| / \max(\Gamma_{\mathrm{core}})$")
    savefig("V_KFdiff_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).png")
end

function V_KF_compressed_name(iK::Int, R::Int=7)
    return "V_KF_bonddims_vs_tol_iK=$(iK)_R=$(R)"
end

function precompressed_datadir()
    return "keldysh_seungsup_results"    
end

function compress_precomputed_V_KF_tolsweep(iK::Int, channel="t", tcikwargs...)
    if haskey(Dict(tcikwargs), "tolerance")
        error("You should not provide a tolerance here")
    end
    d = Dict{Int, Vector{Int}}()
    Γcore_ref = load_Γcore_KF(;channel=channel)
    for tol in reverse(10.0 .^ (-6:-2))
        ld = compress_precomputed_V_KF(Γcore_ref, iK, channel; tolerance=tol, tcikwargs...)
        d[round(Int, log10(tol))] = ld
    end

    TCI4Keldysh.logJSON(d, V_KF_compressed_name(iK, 7), precompressed_datadir())
end

function plot_precomputed_V_KF_ranks(iK::Int, R::Int=7; p=TCI4Keldysh.default_plot(), save=false)
    data = TCI4Keldysh.readJSON(V_KF_compressed_name(iK, R), precompressed_datadir())   

    tols = []
    bonddims = []
    for (tolstr, bd) in pairs(data)
        tol = 10.0 ^ parse(Int, tolstr)
        push!(tols, tol)
        push!(bonddims, bd)
    end
    perm = sortperm(tols; rev=false)
    iKtuple = TCI4Keldysh.KF_idx(iK, 3)
    iKstr = "(" * prod(ntuple(i -> "$(iKtuple[i]),", 4)) * ")"
    # label = L"\mathbf{k}=" * iKstr
    label = ""
    plot!(p, tols[perm], maximum.(bonddims[perm]); xscale=:log10, marker=:circle, xflip=true, legend=:bottomright, label=label)
    # ylabel!(L"rank($\Gamma^{\mathrm{core}}$)")
    if save
        ylabel!(p, L"\chi")
        xlabel!(p, "tolerance")
        title!(p, L"Keldysh $\Gamma_{\mathrm{core}}$: ranks vs. tolerance")
        savefig(joinpath(precompressed_datadir(),"V_KF_ranks_iK=$(iK)_R=$(R).png"))
    end
end

function plot_precomputed_V_KF_ranks_all(R::Int=7)
    p = plot(;guidefontsize=16, titlefontsize=16, tickfontsize=12, legendfontsize=9)

    for iK in 1:15
        plot_precomputed_V_KF_ranks(iK, R; p=p)
    end

    ylabel!(p, L"\chi")
    xlabel!(p, "tolerance")
    title!(p, L"Keldysh components of $\Gamma_{\mathrm{core}}$: QTCI-ranks")
    worstcase = 2^(div(3*R, 2)) 
    ylims!(0, worstcase+30)
    hline!(p, [worstcase]; color=:black, linestyle=:dash, label="worstcase")
    savefig(p, joinpath(precompressed_datadir(), "V_KF_ranks.png"))
end

# for iK in 12:15
#     compress_precomputed_V_KF_tolsweep(iK)
# end