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

"""
* PSFpath: Path to spectral functions
* ωs_ext: Bosonic Matsubara grid read from V_MF_U2_*.mat
"""
function precompute_K1r_explicit_SeungSup(PSFpath::String, flavor_idx::Int; ωs_ext::Vector{Float64}, channel="t")

    # get right temperature
    T = TCI4Keldysh.dir_to_T(PSFpath)
    # get operators for K1_r
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

    # get sign for K1_r
    sign = if channel=="t"
        1
    elseif channel in ["p", "pNRG"]
        -1
    elseif channel=="a"
        -1
    else
        error("Invalid channel")
    end

    # compute K1
    result = zeros(ComplexF64, length(ωs_ext))
    ps = [[1,2], [2,1]]
    # index of zero bosonic frequency
    zero_id = div(length(ωs_ext), 2) + 1
    omtrafo = [1,-1]
    for p in ps
        # no sign because all operators are bosonic
        Adisc = TCI4Keldysh.load_Adisc(PSFpath, ops[p], flavor_idx)
        omdisc = TCI4Keldysh.load_ωdisc(PSFpath, ops[p])

        # regular kernel
        kernel = if omtrafo[p][1]==1
            # first partial correlator
            1 ./ (im * ωs_ext .- omdisc')
        else
            # second partial correlator, has frequencies reversed
            1 ./ (im * reverse(ωs_ext) .- omdisc')
        end
        # set divergent element to 0 (replaced by anomalous part below)
        isdivergent = .!isfinite.(kernel)
        kernel[isdivergent] .= zero(ComplexF64)
        result += kernel * Adisc

        # anomalous part
        ano_id = findfirst(o -> abs(o)<1.e-14, omdisc)
        if !isnothing(ano_id)
            result[zero_id] -= 0.5 * Adisc[ano_id] / T 
        end
    end
    return sign * result
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

# """
# K1 for each channel on 1D frequency grid
# """
# function precompute_K1r(PSFpath::String, flavor_idx::Int, formalism="MF"; ωs_ext::Vector{Float64}, channel="t", broadening_kwargs...)

#     T = TCI4Keldysh.dir_to_T(PSFpath)
#     ops = if channel=="t"
#         # G(12,34)
#         ["Q12", "Q34"]
#     elseif channel in ["p", "pNRG"]
#         # -ζ G(13,24)
#         ["Q13", "Q24"]
#     elseif channel=="a"
#         # -G(14,23)
#         ["Q14", "Q23"]
#     else
#         error("Invalid channel")
#     end
#     G = if formalism=="MF"
#             TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T=T, flavor_idx=flavor_idx, ωs_ext=(ωs_ext,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
#         elseif formalism=="KF"
#             basepath = join(split(rstrip(PSFpath, '/'), "/")[1:end-1], "/")
#             (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
#             # (γ, sigmak) = (5.0*1.e-7, [0.01])
#             println("-- Broadening parameters: γ=$γ, σk=$(only(sigmak))")
#             TCI4Keldysh.FullCorrelator_KF(
#                 PSFpath, ops;
#                 T=T, flavor_idx=flavor_idx, ωs_ext=(ωs_ext,), ωconvMat=reshape([ 1; -1], (2,1)), γ=γ, sigmak=sigmak, name="SIAM 2pG",
#                 broadening_kwargs...);
#         end
#     println("-- Adisc weight K1:")
#     println("  ($(sum(G.Gps[1].tucker.center)), $(sum(G.Gps[2].tucker.center)))")
#     sign = TCI4Keldysh.channel_K1_sign(channel)
#     return sign * TCI4Keldysh.precompute_all_values(G)
# end

function translate_K1_iK(ik::NTuple{4,Int}, channel::String)::NTuple{2,Int}
    _conv(k1::Int,k2::Int) = ifelse(isodd(k1+k2), 2, 1)

    if channel=="t"
        (_conv(ik[2],ik[3]), _conv(ik[1],ik[4]))
    elseif channel in ["p", "pNRG"]
        (_conv(ik[1],ik[3]), _conv(ik[2],ik[4]))
    elseif channel=="a"    
        (_conv(ik[1],ik[2]), _conv(ik[3],ik[4]))
    else
        error("Invalid channel")
    end
end

"""
Read:
emin
emax
TODO: what about estep, estep_fac, epsd and all the other settings in mpNRG_*.mat?
"""
function read_broadening_settings(path=joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50"); channel="t")
    d = Dict{Symbol, Any}()
    matopen(joinpath(path, "mpNRG_$(TCI4Keldysh.channel_translate(channel)).mat")) do f
        d[:emin] = read(f, "emin")
        d[:emax] = read(f, "emax")
    end
    return d
end

function check_K1_KF_all(;channel="t")
    for iK in Iterators.product(ntuple(_->[1,2],4)...)    
        check_K1_KF(iK;channel=channel)
    end
end

"""
Comparison to MuNRG
K1 is independent of the bosonic frequency in all channels.

NOTE: In MuNRG, G^K=G^(22) is computed via the bosonic fluctuation-dissipation relation (cf. Util/Acont2GKel.m, Util/KKi2r.m).
The latter is therefore satisfied to high accuracy (<1.e-8). In Julia, we just compute each component of the
2-point correlator independently.

Apart from that, we have a discrepancy on the order of 1.e-7, most likely due to slightly different broadening settings.
"""
function check_K1_KF(iKtuple=(1,2,1,2);channel="t")
    basepath = "SIAM_u=0.50/"
    mpNRGpath = joinpath(TCI4Keldysh.datadir(), basepath)
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_KF_" * TCI4Keldysh.channel_translate(channel))
    
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
    matopen(joinpath(Vpath, "V_KF_U2_$(channel_id).mat")) do f
        CFdat = read(f, "CFdat")
        K1 = CFdat["Ggrid"][flavor]
        grid = CFdat["ogrid"]
        @show size(K1)
        @show size(grid)
    end

    # extract 1D
    for ik in Iterators.product(ntuple(_->[1,2], 4)...)
        @show (ik, norm(K1[:,:,:,ik...]))
    end
    @show maximum(abs.(K1))
    K13D = K1[:,:,:,iKtuple...]
    K1D = K13D[1,1,:]
    @show maximum(abs.(K13D))
    # check that K13D slices are actually constant
    for i in axes(K13D,3)
        K13Dref = K13D[1,1,i]
        @assert all(abs.(K13D[:,:,i] .- K13Dref) .<= 1.e-16) "K13D slice no. $i nonconstant"
    end
    reverse!(K1D)

    broadening_kwargs = read_broadening_settings(mpNRGpath ;channel=channel)
    # broadening_kwargs[:emax] += 20.0
    # broadening_kwargs[:emin] /= 100.0
    if !haskey(broadening_kwargs, :estep)
        broadening_kwargs[:estep] = 200
    end


    # TCI4Keldysh
    ωs_ext = vec(grid[end])
    K1_test_alliK = TCI4Keldysh.precompute_K1r(PSFpath, flavor, "KF"; mode=:normal, channel=channel, ωs_ext=ωs_ext, broadening_kwargs...)

    ik = TCI4Keldysh.merge_iK_K1(iKtuple, channel)
    K1_test = K1_test_alliK[:,ik...]

    # REASON FOR PREFACTOR: Eq. (100) SIE paper (Lihm et. al), factors P^k1k2k12 and P^k3k4k34 introduce two times 1/√2
    fac = 1/2

    if ik==(2,2)
        println("-- Weights:")
        K1_test_weight = TCI4Keldysh.quadtrapz(imag.(fac * K1_test), vec(grid[1])) / pi
        K1D_weight = TCI4Keldysh.quadtrapz(imag.(K1D), vec(grid[1])) / pi
        println("  Julia: $K1_test_weight")
        println("  MuNRG: $K1D_weight")
        println("--")
    end

    # plot
    omgrid = vec(grid[1])
    mid_id = div(length(K1D),2) + 1
    p = TCI4Keldysh.default_plot()
    plot!(p, omgrid, real.(K1D); label="Re, MuNRG")
    plot!(p, omgrid, imag.(K1D); label="Im, MuNRG")
    # for ik in Iterators.product(ntuple(_->[1,2], 2)...)
    #     plot!(p, real.(fac * K1_test_alliK[:,ik...]); label="Re$(ik), Julia", linestyle=:dash)
    #     plot!(p, imag.(fac * K1_test_alliK[:,ik...]); label="Im$(ik), Julia", linestyle=:dot)
    # end
    plot!(p, omgrid, real.(fac * K1_test); label="Re$(ik), Julia", linestyle=:dash)
    plot!(p, omgrid, imag.(fac * K1_test); label="Im$(ik), Julia", linestyle=:dot)
    title!(p, "K1@$(channel)-channel: MuNRG vs Julia")
    savefig("K1_comparison.pdf")

    # diff
    maxref = maximum(abs.(K1D))
    diff = (K1D .- K1_test * fac) ./ maxref
    p = TCI4Keldysh.default_plot()
    plot!(p, omgrid, abs.(real.((diff))) .+ 1.e-12; label="Re(diff)", yscale=:log10)
    plot!(p, omgrid, abs.(imag.((diff))) .+ 1.e-12; label="Im(diff)", linestyle=:dot, yscale=:log10)
    @show maximum(abs.(real.(diff)))
    @show maximum(abs.(imag.(diff)))
    # ratio = K1D ./ (fac * K1_test)
    # plot!(p, abs.(ratio); label="ratio")
    title!(p, "K1@$(channel)-channel: abs(MuNRG-Julia)")
    savefig("K1_diff.pdf")
    amaxdiff = argmax(abs.(diff))
    printstyled("-- Largest errors iK=$(iKtuple): $(sort(abs.(diff); rev=true)[1:5])\n"; color=:blue)
    printstyled("   Max error at frequency: $(ωs_ext[amaxdiff])\n\n"; color=:blue)
    @show K1D[mid_id-2:mid_id+2]
    @show fac*K1_test[mid_id-2:mid_id+2]

    # reldiff
    reldiff = (K1D .- K1_test * fac) ./ abs.(K1D)
    p = TCI4Keldysh.default_plot()
    plot!(p, omgrid, abs.(reldiff); label="reldiff", yscale=:log10)
    title!(p, "K1@$(channel)-channel: abs(MuNRG-Julia)/abs(MuNRG)")
    savefig("K1_reldiff.pdf")

    # plot FDT for Julia result
    p = TCI4Keldysh.default_plot()
    plot!(p, real.(K1_test_alliK[:,2,2]); label=L"\Re(G^K)")
    plot!(p, imag.(K1_test_alliK[:,2,2]); label=L"\Im(G^K)")
    T = TCI4Keldysh.dir_to_T(PSFpath)
    fdt = (1.0 .+ 2*[1/(exp(w/T)-1) for w in ωs_ext]) .* (K1_test_alliK[:,2,1] .- K1_test_alliK[:,1,2])
    fdt_matlab = (1.0 .+ 2*[1/(exp(w/T)-1) for w in ωs_ext]) .* (K1[1,1,:,1,1,2,1] .- K1[1,1,:,2,1,1,1])
    # @show maximum(abs.(filter(a -> !isnan(a), fac*fdt .- fdt_matlab)))
    diff_fdt = K1_test_alliK[:,2,2] .- fdt
    diff_fdt2 = K1D .- fdt_matlab
    printstyled("-- Largest errors fdt: $(sort(abs.(diff_fdt); rev=true)[1:5])\n"; color=:blue)
    plot!(p, real.(fdt); label=L"\Re(FDT)", linestyle=:dot)
    plot!(p, imag.(fdt); label=L"\Im(FDT)", linestyle=:dot)
    savefig("fdt.pdf")
    p = TCI4Keldysh.default_plot()
    plot!(p, omgrid, imag.(diff_fdt); label=L"\Im(\mathrm{Julia-FDT@Julia})")
    # in MuNRG, G(22) is computed using the FDT, so the error should vanish
    plot!(p, omgrid, imag.(diff_fdt2); label=L"\Im(\mathrm{Matlab-FDT@Matlab})")
    @show maximum(abs.(filter(a -> !isnan(a), diff_fdt2)))
    savefig("fdt_diff.pdf")
end


function test_K1_TCI(formalism="MF"; ωmax::Float64=1.0, channel="t", flavor=1)
    basepath = "SIAM_u=0.50"
    # basepath = "SIAM_u=1.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=2_conn_zavg/")
    R = 12
    T = TCI4Keldysh.dir_to_T(PSFpath)

    # TCI4Keldysh
    ωs_ext = if formalism=="MF"
            TCI4Keldysh.MF_grid(T, 2^(R-1), false)
        else
            TCI4Keldysh.KF_grid_bos(ωmax, R)
        end
    K1_slice = formalism=="MF" ? (1:2^R,) : (1:2^R,:,:)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath;channel=channel)
    K1_test = TCI4Keldysh.precompute_K1r(PSFpath, flavor, formalism; channel=channel, ωs_ext=ωs_ext, broadening_kwargs...)[K1_slice...]

    ncomponents = formalism=="MF" ? 1 : 4
    # TCI4Keldysh: TCI
    K1_tci = TCI4Keldysh.K1_TCI(
        PSFpath,
        R;
        ωmax=ωmax,
        formalism=formalism,
        channel=channel,
        T=T,
        flavor_idx=flavor,
        unfoldingscheme=:interleaved,
        tolerance=1.e-8
        )

    K1_tcivals_ = fill(zeros(ComplexF64, ntuple(_->2,R)), Int(sqrt(ncomponents)), Int(sqrt(ncomponents)))
    for i in eachindex(K1_tcivals_)
        if !isnothing(K1_tci[i])
            K1_tcivals_[i] = TCI4Keldysh.qtt_to_fattensor(K1_tci[i].tci.sitetensors)
        end
    end
    K1_tcivals = [TCI4Keldysh.qinterleaved_fattensor_to_regular(k1, R) for k1 in K1_tcivals_]
    K1_tcivals_block = reshape(vcat(K1_tcivals...), (2^R, ncomponents))

    # plot TCI
    if formalism=="MF"
        p = TCI4Keldysh.default_plot()
        ωs_ext_TCI = ωs_ext[1:2^R]
        plot!(p, ωs_ext_TCI, real.(K1_test); label="Re, Julia", linestyle=:dash)
        plot!(p, ωs_ext_TCI, imag.(K1_test); label="Im, Julia", linestyle=:dash)
        plot!(p, ωs_ext_TCI, real.(K1_tcivals[1]); label="Re, Julia@TCI", linestyle=:dot)
        plot!(p, ωs_ext_TCI, imag.(K1_tcivals[1]); label="Im, Julia@TCI", linestyle=:dot)
        title!(p, "K1@$(channel)-channel: TCI vs convenctional")
        savefig("K1_comparison_tci.pdf")
    end

    K1_test = reshape(K1_test, 2^R, ncomponents)
    diff = abs.(K1_tcivals_block .- K1_test)
    @show maximum(diff) / maximum(abs.(K1_test))
end

"""
Comparison to MuNRG
K1 is independent of the fermionic frequencies in all channels.
"""
function check_K1_MF(;channel="t", flavor=1)
    basepath = "SIAM_u=0.50"
    # basepath = "SIAM_u=1.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_MF_" * TCI4Keldysh.channel_translate(channel))
    
    # load K1
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
    K1_test = TCI4Keldysh.precompute_K1r(PSFpath, flavor; channel=channel, ωs_ext=ωs_ext)

    do_tci = false
    if do_tci
    # TCI4Keldysh: TCI
    R = Int(floor(log2(length(ωs_ext))))
    T = TCI4Keldysh.dir_to_T(PSFpath)
    K1_tci = TCI4Keldysh.K1_TCI(
        PSFpath,
        R;
        formalism="MF",
        channel=channel,
        T=T,
        flavor_idx=flavor,
        unfoldingscheme=:interleaved,
        tolerance=1.e-8
        )[1]

    K1_tcivals_ = TCI4Keldysh.qtt_to_fattensor(K1_tci.tci.sitetensors)
    K1_tcivals = TCI4Keldysh.qinterleaved_fattensor_to_regular(K1_tcivals_, R)
    ωs_ext_TCI = TCI4Keldysh.MF_grid(T, 2^(R-1), false)[1:2^R]
    end

    # plot
    p = TCI4Keldysh.default_plot()
    plot!(p, ωs_ext, real.(K1D); label="Re, MuNRG")
    plot!(p, ωs_ext, imag.(K1D); label="Im, MuNRG")
    plot!(p, ωs_ext, real.(K1_test); label="Re, Julia", linestyle=:dash)
    plot!(p, ωs_ext, imag.(K1_test); label="Im, Julia", linestyle=:dash)
    title!(p, "K1@$(channel)-channel: MuNRG vs Julia")
    savefig("K1_comparison.pdf")

    if do_tci
    # plot TCI
    p = TCI4Keldysh.default_plot()
    iom = findfirst(w -> w==first(ωs_ext_TCI), ωs_ext)
    @show iom
    omslice = iom:iom+2^R-1
    @show omslice
    K1_test
    plot!(p, ωs_ext_TCI, real.(K1_test)[omslice]; label="Re, Julia", linestyle=:dash)
    plot!(p, ωs_ext_TCI, imag.(K1_test)[omslice]; label="Im, Julia", linestyle=:dash)
    plot!(p, ωs_ext_TCI, real.(K1_tcivals); label="Re, Julia@TCI", linestyle=:dot)
    plot!(p, ωs_ext_TCI, imag.(K1_tcivals); label="Im, Julia@TCI", linestyle=:dot)
    title!(p, "K1@$(channel)-channel: TCI vs convenctional")
    savefig("K1_comparison_tci.pdf")
    end


    # diff
    diff = K1D .- K1_test
    @show maximum(abs.(diff))
    p = TCI4Keldysh.default_plot()
    plot!(p, real.(diff))
    title!(p, "K1@$(channel)-channel: real(MuNRG-Julia)";label="")
    savefig("K1_diff.pdf")
end

function check_K2_MF(;channel="t", prime=false)

    basepath = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_MF_" * TCI4Keldysh.channel_translate(channel))
    
    # load K2
    flavor = 1
    K2 = nothing
    grid = nothing
    channel_id = if channel=="t"
        ifelse(prime, 1, 6)
    elseif channel=="pNRG"
        ifelse(prime, 2, 5)
    elseif channel=="a"
        ifelse(prime, 3, 4)
    else
        error("Invalid channel $channel")
    end
    matopen(joinpath(Vpath, "V_MF_U3_$(channel_id).mat")) do f
        CFdat = read(f, "CFdat")
        K2 = CFdat["Ggrid"][flavor]
        grid = vec(CFdat["ogrid"])
        @show size(K2)
        @show size(grid)
    end

    # Σ data
    ωs_Σ = nothing
    Σ_file = "SE_MF_1.mat"
    matopen(joinpath(Vpath, Σ_file), "r") do f
        CFdat = read(f, "CFdat")
        ωs_Σ_ = vec(vec(CFdat["ogrid"])[1])
        @assert norm(real.(ωs_Σ_)) <= 1.e-10
        ωs_Σ = imag.(ωs_Σ_)
    end

    # extract 2D; frequencies in order (ν,ν',ω)
    slice = ifelse(prime, (1,:,:), (:,1,:))
    K2D = K2[slice...]
    # move bosonic frequency to first argument
    K2D = permutedims(K2D, [2,1])

    # TCI4Keldysh
    T = TCI4Keldysh.dir_to_T(PSFpath)
    permute!(grid, [3,1,2])
    ωconvMat = TCI4Keldysh.channel_trafo_K2(channel,prime)
    ωs_ext = ntuple(i -> imag.(vec(grid[i])), 2)
    op_labels = TCI4Keldysh.oplabels_K2(channel,prime)
    (ΣL, ΣR) = TCI4Keldysh.calc_Σ_MF_aIE(PSFpath, ωs_Σ; flavor_idx=flavor,T=T)
    # ΣR = TCI4Keldysh.calc_Σ_MF_sIE(PSFpath, ωs_Σ; flavor_idx=flavor,T=T)
    K2julia = TCI4Keldysh.compute_K2r_symmetric_estimator(
        "MF",
        PSFpath,
        Tuple(op_labels),
        ΣR;
        Σ_calcL=ΣL,
        # Σ_calcL=nothing,
        T=T,
        flavor_idx=flavor,
        ωs_ext=ωs_ext,
        ωconvMat=ωconvMat
    )
    @show size(K2julia)
    @show size(K2D)

    # t-channel: no sign
    # p-channel: 0
    # a-channel: minus sign if prime===false
    sign = ifelse(channel=="a" && !prime, -1, 1)
    K2julia *= sign
    reverse!(K2julia)
    heatmap(real.(K2julia))
    savefig("K2.pdf")
    heatmap(real.(K2D))
    savefig("K2_ref.pdf")
    @show maximum(abs.(real.(K2D .- K2julia)))
    @show maximum(abs.(imag.(K2D .- K2julia)))
    @show maximum(abs.(K2D .- K2julia))
end

function test_K2_TCI_precomputed(;formalism="MF", channel="t", prime=false, flavor_idx=1)
    basepath = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")

    R = 5
    Nhalf = 2^(R-1)
    ωmax = 1.0

    # TCI4Keldysh: Reference
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωs_ext = if formalism=="MF"
            TCI4Keldysh.MF_npoint_grid(T, Nhalf, 2)
        else
            TCI4Keldysh.KF_grid(ωmax, R, 2)
        end
    K2 = TCI4Keldysh.precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext, channel=channel, prime=prime)
    K2slice = if formalism=="MF"
            (1:2^R,Colon())
        else
            (1:2^R,Colon(),:,:,:)
        end
    K2 = K2[K2slice...]

    # TCI4Keldysh: TCI way
    K2tci = TCI4Keldysh.K2_TCI_precomputed(
        PSFpath,
        R;
        formalism=formalism,
        flavor_idx=flavor_idx,
        channel=channel,
        prime=prime,
        T=T,
        ωmax=ωmax,
        tolerance=1.e-8
    )

    ncomponents = formalism=="MF" ? 1 : 8
    nkeldysh = formalism=="MF" ? 1 : 2
    K2tcivals = fill(zeros(ComplexF64, ntuple(_->2, 2*R)), nkeldysh,nkeldysh,nkeldysh)
    for i in eachindex(K2tci)
        if !isnothing(K2tci[i])
            K2tcivals[i] = TCI4Keldysh.qtt_to_fattensor(K2tci[i].tci.sitetensors)
        end
    end
    K2tcivals = [TCI4Keldysh.qinterleaved_fattensor_to_regular(k2, R) for k2 in K2tcivals]
    @show size.(K2tcivals)

    K2_test = reshape(K2, 2^R, 2^R, ncomponents)
    @show size(K2_test)
    for i in 1:ncomponents
        diff = abs.(K2_test[:,:,i] .- K2tcivals[i]) ./ maximum(abs.(K2_test[:,:,i]))
        @show maximum(diff)
    end

    component = 1
    scfun(x) = log10(abs(x))
    # heatmap(scfun.(K2tcivals_block[:,:,component]))
    heatmap(scfun.(K2tcivals[component][:,:]))
    savefig("K2.pdf")
    heatmap(scfun.(K2_test[:,:,component]))
    savefig("K2_ref.pdf")
end


function check_K2_KF(;channel="t", prime=false)
    basepath = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_KF_" * TCI4Keldysh.channel_translate(channel))
    
    # load K2
    flavor = 2
    K2 = nothing
    grid = nothing
    channel_id = if channel=="t"
        ifelse(prime, 1, 6)
    elseif channel=="pNRG"
        ifelse(prime, 2, 5)
    elseif channel=="a"
        ifelse(prime, 3, 4)
    else
        error("Invalid channel $channel")
    end
    matopen(joinpath(Vpath, "V_KF_U3_$(channel_id).mat")) do f
        CFdat = read(f, "CFdat")
        K2 = CFdat["Ggrid"][flavor]
        grid = vec(CFdat["ogrid"])
        @show size(K2)
        @show size(grid)
    end

    # Σ data
    ωs_Σ = nothing
    Σ_file = "SE_KF_1.mat"
    matopen(joinpath(Vpath, Σ_file), "r") do f
        CFdat = read(f, "CFdat")
        ωs_Σ_ = vec(vec(CFdat["ogrid"])[1])
        @assert norm(imag.(ωs_Σ_)) <= 1.e-10
        ωs_Σ = real.(ωs_Σ_)
    end

    # extract 2D; frequencies in order (ν,ν',ω)
    # K2 still carries 4 Keldysh indices
    slice = ifelse(prime, (1,:,:, :,:,:,:), (:,1,:, :,:,:,:))
    K2D = K2[slice...]
    # move bosonic frequency to first argument
    K2D = permutedims(K2D, [2,1,3,4,5,6])
    reverse!(K2D; dims=(1,2))

    # TCI4Keldysh
    permute!(grid, [3,1,2])
    ωs_ext = ntuple(i -> real.(vec(grid[i])), 2)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath)
    broadening_kwargs[:estep] = 20
    # T = TCI4Keldysh.dir_to_T(PSFpath)
    # ωconvMat = TCI4Keldysh.channel_trafo_K2(channel,prime)
    # op_labels = TCI4Keldysh.oplabels_K2(channel,prime)
    # (ΣL, ΣR) = TCI4Keldysh.calc_Σ_KF_aIE(PSFpath, ωs_Σ; flavor_idx=flavor,T=T, γ=γ, sigmak=sigmak, broadening_kwargs...)
    # K2julia = TCI4Keldysh.compute_K2r_symmetric_estimator(
    #     "KF",
    #     PSFpath,
    #     op_labels,
    #     ΣR;
    #     Σ_calcL=ΣL,
    #     # Σ_calcL=nothing,
    #     T=T,
    #     flavor_idx=flavor,
    #     ωs_ext=ωs_ext,
    #     ωconvMat=ωconvMat,
    #     γ=γ,
    #     sigmak=sigmak,
    #     broadening_kwargs...
    # )
    K2julia = TCI4Keldysh.precompute_K2r(
        PSFpath,
        flavor,
        "KF";
        ωs_ext=ωs_ext,
        channel=channel,
        prime=prime,
        γ=γ,
        sigmak=sigmak,
        broadening_kwargs...
    )
    @show size(K2julia)
    @show size(K2D)

    # REASON FOR PREFACTOR: Eq. (100) SIE paper (Lihm et. al), factor P introduces 1/√2
    fac = 1.0/sqrt(2)
    K2julia .*= fac

    # check all Keldysh components
    for ik in TCI4Keldysh.ids_KF(4)
        maxref = maximum(abs.(K2D[:,:,ik...]))
        ik3 = TCI4Keldysh.merge_iK_K2(ik, channel, prime; merged_idx=1)
        diff = K2julia[:,:,ik3...] .- K2D[:,:,ik...]
        println("Norm REF: $(norm(K2D[:,:,ik...]))")
        println("Norm JULIA: $(norm(K2julia[:,:,ik3...]))")
        println("Error for ik=$(ik)∼$(ik3): $(maximum(abs.(diff)) / maxref)\n") 
    end
end


"""
Comparison to MuNRG

In the MuNRG data, the Keldysh components of Σ are stored on a 4-fold duplicated 1D grid,
in the order: (11) (21) (12) (22)

MuNRG code: Multipoint/get_V_4pIE.m; uses asymmetricIE
"""
function check_Σ_KF(; channel="t")
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50", "V_KF_" * TCI4Keldysh.channel_translate(channel))

    # load self-energies
    flavor = 1
    Σnames = ["SE_KF_$i.mat" for i in 1:4]
    Σs = Vector{ComplexF64}[]
    grids = Vector{Float64}[]
    for Σname in Σnames
        matopen(joinpath(Vpath, Σname), "r") do f
            CFdat = read(f, "CFdat")
            Σ_act = CFdat["Ggrid"][flavor]
            grid_act = CFdat["ogrid"][flavor]
            push!(Σs, vec(Σ_act))
            push!(grids, vec(real.(grid_act)))
        end
    end
    @show size.(Σs)
    @show size.(grids)
    # check whether all frequency grids are the same
    @assert all(length.(grids) .== length(first(grids)))
    for g in grids[2:4]
        @assert maximum(abs.(g .- grids[1])) < 1.e-12
    end

    # # compare incoming/outgoing legs
    # for i in 1:3
    #     @show maximum(abs.(Σs[i] .- Σs[i+1]))
    #     amax = argmax(abs.(Σs[i] .- Σs[i+1]))
    #     @show Σs[i][amax]
    #     @show Σs[i+1][amax]
    # end


    # TCI4Keldysh
    T = TCI4Keldysh.dir_to_T(PSFpath)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(base_path; channel=channel)
    broadening_kwargs = read_broadening_settings(joinpath(TCI4Keldysh.datadir(), base_path); channel=channel)
    broadening_kwargs[:estep] = 20
    omsig = grids[1]

    # asymmetric estimators, should be closer to MuNRG
    (Σ_L, Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE(PSFpath, omsig; mode=:normal, flavor_idx=flavor, sigmak, γ, broadening_kwargs...)
    # (Σ_L, Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, omsig; T=T, flavor_idx=flavor, sigmak, γ, broadening_kwargs...)
     
    # inverting matrices vs. computing aIE via retarded component does make a difference
    # @show maximum(abs.(Σ_L_ .- Σ_L))
    # @show maximum(abs.(Σ_R_ .- Σ_R))

    # Σ_test = TCI4Keldysh.calc_Σ_KF_sIE_viaR(PSFpath, omsig; T=T, flavor_idx=flavor, sigmak, γ, broadening_kwargs...)
    sig_id = 1
    Σ_test_aIE = isodd(sig_id) ? Σ_L : Σ_R

    # check out in plots
    p = TCI4Keldysh.default_plot()
    for ik in [(1,1),(2,1),(1,2),(2,2)]
        # plot!(p, omsig, real.(Σ_test[:,ik...]); label=L"\Re(Σ^{%$ik})")
        # plot!(p, omsig, imag.(Σ_test[:,ik...]); label=L"\Im(Σ^{%$ik})", linestyle=:dot)

        # plot!(p, omsig, real.(Σ_test_aIE[:,ik...]); label=L"\Re(Σ^{%$ik}) (aIE)")
        plot!(p, omsig, imag.(Σ_test_aIE[:,ik...]); label=L"\Im(Σ^{%$ik}) (aIE)")
    end
    d = Dict([(1,(1,1)), (2,(2,1)), (3,(1,2)), (4,(2,2))])
    for i in 1:4
        Σ_act = Σs[sig_id][(i-1)*length(omsig)+1 : i*length(omsig)]
        ik = d[i]
        # plot!(p, omsig, real.(Σ_act); label=L"\Re(Σ^{%$ik}_{\mathrm{MuNRG}})", linestyle=:dash)
        plot!(p, omsig, imag.(Σ_act); label=L"\Im(Σ^{%$ik}_{\mathrm{MuNRG}})", linestyle=:dash)
    end
    savefig("sigma_KF.pdf")

    # plot diff
    p = TCI4Keldysh.default_plot()
    for i in 1:4
        Σ_ref = Σs[sig_id][(i-1)*length(omsig)+1 : i*length(omsig)]
        ik = d[i]
        maxref = maximum(abs.(Σ_ref))
        diff = (Σ_test_aIE[:,ik...] .- Σ_ref) ./ maxref
        plot!(p, omsig, abs.(real.(diff)); label=L"|\Re(\Delta\Sigma)|%$ik")
        plot!(p, omsig, abs.(imag.(diff)); label=L"|\Im(\Delta\Sigma)|%$ik", linestyle=:dot)
    end
    savefig("sigma_diff.pdf")

    # to see how Keldysh-components of MuNRG self-energies are ordered
    # colors = [:red, :blue, :green, :cyan]
    # p = TCI4Keldysh.default_plot()
    # for i in eachindex(Σs)[sig_id]
    #     plot!(p, real.(Σs[i]); label="Re,$i", color=colors[i])
    # end
    # for i in eachindex(Σs)[sig_id]
    #     plot!(p, imag.(Σs[i]); label="Im,$i", linestyle=:dot, color=colors[i])
    # end
    # savefig("foo.pdf")

end


"""
Comparison to MuNRG
"""
function check_Σ_MF(; channel="t")
    basepath = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_MF_" * TCI4Keldysh.channel_translate(channel))

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
    savefig("SE_comparison.pdf")
    p = TCI4Keldysh.default_plot()
    plot!(p, grids[1], imag.(ΣL .- Σs[1]); linestyle=:dash, label="Im(ΣMuNRG - ΣJulia)_∞")
    println("==== Examine diff")
    display((ΣL .- Σs[1])[Nhalf-2:Nhalf+3])
    display((ω_fer)[Nhalf-2:Nhalf+3])
    println("====")
    savefig("SE_diff.pdf")
end

"""
Compare MuNRG Matsubara vertices with TCI4Keldysh.
CAREFUL: Need channel="pNRG" for p-channel to get a consistent frequency convention
"""
function check_V_MF(Nhalf=2^4;channel="t", use_ΣaIE=true, spin::Int=1)
    basepath = "SIAM_u=0.50"
    # basepath = "siam05_U0.05_T0.005_Delta0.0318"
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_MF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = "V_MF_U4.mat"
    CF = nothing
    Γcore_ref = nothing
    ωs_ext = nothing
    matopen(joinpath(Vpath, core_file), "r") do f
        CF = read(f, "CF")
        CFdat = read(f, "CFdat")
        Γcore_ref = CFdat["Ggrid"][spin]
        # bosonic grid comes last in the data
        ωs_ext = ntuple(i -> imag.(vec(vec(CFdat["ogrid"])[4-i])), 3)
    end
    # bosonic grid comes last in the data
    Γcore_ref = permutedims(Γcore_ref, (3,1,2))
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
    savefig("gam.pdf")


    window_half = div(length(om_small[2]), 2)
    data_half = div(length(ωs_ext[2]), 2)
    window_slice = data_half-window_half+1:data_half+window_half
    slice_ref = [div(length(ωs_ext[1]), 2)+1, window_slice, window_slice]
    @show ωs_ext[1][slice_ref[1]]
    heatmap(scfun.(-Γcore_ref[slice_ref...]); right_margin=10Plots.mm)
    title!("Γcore reference")
    savefig("ref.pdf")

    # compare quantitatively
    window = (data_half-window_half+1:data_half+window_half+1, data_half-window_half+1:data_half+window_half, data_half-window_half+1:data_half+window_half)
    # Γ(ω,ν,ν')=Γ*(-ω,-ν,-ν')
    diff = Γcore_ref[window...] .- reverse(testval)
    maxdiff = maximum(abs.(diff)) 
    amaxdiff = argmax(abs.(diff)) 
    @show amaxdiff
    @show diff[amaxdiff]
    @show testval[amaxdiff]
    @show Γcore_ref[window...][amaxdiff]
    printstyled("---- Max. abs. deviation: $(maxdiff) (Γcore value: $(testval[amaxdiff]))\n"; color=:blue)
    # difference comes from real part
    scfun = x -> abs(x)
    heatmap(scfun.(Γcore_ref[slice_ref...] .+ testval[slice...]); right_margin=10Plots.mm)
    savefig("diff.pdf")

    reldiff = real.(testval) ./ real.(Γcore_ref)[window...]
    reldiff = map(x -> ifelse(!isnan(x) && (1. / 1.1 < abs(x)<1.1), x, 1.0), reldiff)
    @show maximum(abs.(reldiff))
    mean = sum(reldiff) / length(reldiff)
    @show mean
    heatmap(abs.(reldiff[slice...]); right_margin=10Plots.mm)
    savefig("reldiff.pdf")
    return maxdiff
end

function check_V_MF_all()
   check_V_MF(2^4;channel="t") 
   check_V_MF(2^4;channel="a") 
   check_V_MF(2^4;channel="pNRG") 
end

function load_omgrid_gamcore(base_path="SIAM_u=0.50"; channel="t")
    core_file = "V_KF_U4.mat"
    ωs_ext = nothing
    Vpath = joinpath(TCI4Keldysh.datadir(), base_path, "V_KF_" * TCI4Keldysh.channel_translate(channel))
    matopen(joinpath(Vpath, core_file), "r") do f
        CFdat = read(f, "CFdat")
        ωs_ext = ntuple(i -> real.(vec(vec(CFdat["ogrid"])[i])), 3)
    end
    return ωs_ext
end

"""
Check Keldysh vertex of TCI4Keldysh against MuNRG results.
MuNRG results have frequency grids of size 2n+1 symmetric around 0.0
"""
function check_V_KF(Nhalf=2^3; iK::Int=2, channel="t")
    base_path = "SIAM_u=0.50"
    joinpath(TCI4Keldysh.datadir(), base_path, "PSF_nz=4_conn_zavg/")
    PSFpath = joinpath(TCI4Keldysh.datadir(), base_path, "PSF_nz=4_conn_zavg/")
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
    Γcore_ref = permutedims(Γcore_ref, (3,1,2,4,5,6,7))
    Γcore_ref = reverse(Γcore_ref; dims=(1,2,3))
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

    broadening_kwargs = read_broadening_settings(joinpath(TCI4Keldysh.datadir(), base_path); channel=channel)
    if !haskey(broadening_kwargs, "estep")
        broadening_kwargs[:estep] = 100
    end
    # Σ_ref = TCI4Keldysh.calc_Σ_KF_sIE_viaR(PSFpath, om_sig; T=T, flavor_idx=spin, sigmak, γ, broadening_kwargs...)
    (Σ_L, Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, om_sig; T=T, flavor_idx=spin, sigmak, γ, broadening_kwargs...)
    testval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "KF",
        PSFpath*"4pt/",
        Σ_R
        # Σ_ref
        ;
        Σ_calcL=Σ_L,
        T,
        flavor_idx = spin,
        ωs_ext = om_small,
        ωconvMat=ωconvMat,
        sigmak, γ,
        broadening_kwargs...
    )

    # plot
    scfun(x) = imag(x)
    window1D(i) = ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf
    window = ntuple(i -> window1D(i), 2)
    slice_ref = [window..., div(length(ωs_ext[1]), 2)+1, iK_tuple...]
    # heatmap(abs.(Γcore_ref[slice_ref...]); clim=(0.0, 0.0042))
    heatmap(real.(Γcore_ref[slice_ref...]);right_margin=10Plots.mm)
    title!("Re(Γcore)@MuNRG")
    savefig("refreal.pdf")
    heatmap(imag.(Γcore_ref[slice_ref...]);right_margin=10Plots.mm)
    title!("Im(Γcore)@MuNRG")
    savefig("refimag.pdf")

    slice = [:, :, div(length(om_small[1]), 2)+1, iK_tuple...]
    # heatmap(abs.(testval[slice...]); clim=(0.0, 0.0042))
    heatmap(real.(testval[slice...]); right_margin=10Plots.mm)
    title!("Re(Γcore)@Julia")
    savefig("gamreal.pdf")
    heatmap(imag.(testval[slice...]); right_margin=10Plots.mm)
    title!("Im(Γcore)@Julia")
    savefig("gamimag.pdf")

    block = ntuple(i->ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf, 3)
    mindevs = fill(Inf, 4)
    check_index_ordering = false
    if check_index_ordering
        #=
        It seems that the frequency grids are reversed and the frequencies are permuted as (3,1,2) compared to the Julia code, i.e.
        one has to compare
            permutedims(Γcore_Matlab, (3,1,2,4,5,6,7))[:,:,:,iK...]
            vs.
            reverse(Γcore_Julia)
        =#
        for p in permutations(4:7)
            # for pω in permutations(1:3)
            for pω in [[1,2,3]]
                # printstyled("\n-- Permutation: $p (iK_p=$(iK_tuple_p))\n"; color=:blue)
                Γcore_ref_act = permutedims(Γcore_ref, (pω..., p...))
                diff = testval[:,:,:,iK_tuple...] .- Γcore_ref_act[block...,iK_tuple...]
                diff_p = testval[:,:,:,iK_tuple...] .+ Γcore_ref_act[block...,iK_tuple...]
                # diff_c = testval[:,:,:,iK_tuple...] .- conj.(Γcore_ref_act[block...,iK_tuple...])
                diff_r = reverse(testval[:,:,:,iK_tuple...]) .- (Γcore_ref_act[block...,iK_tuple...])

                mindev_act_re = maximum(abs.(real.(diff)))
                min_realdev = min(mindevs[1], mindev_act_re)
                mindev_act_im = maximum(abs.(imag.(diff)))
                min_imagdev = min(mindevs[2], mindev_act_im)

                mindev_act_rep = maximum(abs.(real.(diff_p)))
                min_realdev_p = min(mindevs[3], mindev_act_rep)
                mindev_act_imp = maximum(abs.(imag.(diff_p)))
                min_imagdev_p = min(mindevs[4], mindev_act_imp)

                mindev_act_re_c = maximum(abs.(real.(diff_r)))
                min_realdev_c = min(mindevs[3], mindev_act_re_c)
                mindev_act_im_c = maximum(abs.(imag.(diff_r)))
                min_imagdev_c = min(mindevs[4], mindev_act_im_c)

                mindevs_act = [mindev_act_re, mindev_act_im, mindev_act_rep, mindev_act_imp, mindev_act_re_c, mindev_act_im_c]
                mindevs = [min_realdev, min_imagdev, min_realdev_p, min_imagdev_p, min_realdev_c, min_imagdev_c]
                if all(mindevs_act[1:2] .< 1.e-3) || all(mindevs_act[3:4] .< 1.e-3) || all(mindevs_act[5:6] .< 1.e-3)
                    printstyled("-- For permutations pω=$(pω), p=$(p): mindevs_act=$(mindevs_act)\n"; color=:blue)
                end
                Γcore_ref_act=nothing
            end
        end
    end

    diff = testval[:,:,:,iK_tuple...] .- Γcore_ref[block..., iK_tuple...]
    @show maximum(abs.(diff) ./ maximum(abs.(Γcore_ref[:,:,:,iK_tuple...])))
    diff_slice = testval[slice...] .- Γcore_ref[slice_ref...]
    maxref_slice = maximum(abs.(Γcore_ref))
    @show maxref_slice
    heatmap(log10.(abs.(diff_slice) ./ maxref_slice))
    savefig("diff.pdf")
end

function load_Γ_KF(base_path::String = "SIAM_u=0.50"; fullvertex=true, channel="t", flavor_idx=1)
    Vpath = joinpath(TCI4Keldysh.datadir(), base_path, "V_KF_" * TCI4Keldysh.channel_translate(channel))

    # Γcore data
    core_file = fullvertex ? "V_KF_sym.mat" : "V_KF_U4.mat"
    Γ_ref = nothing
    matopen(joinpath(Vpath, core_file), "r") do f
        CFdat = read(f, "CFdat")
        Γ_ref = CFdat["Ggrid"][flavor_idx]
    end
    return Γ_ref
end

function precomp_compr_filename(iK::Int, tolerance::Float64)
    return "vertex_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).h5"    
end

function compression_slice()
    return 100-63:100+64
end

function compress_precomputed_V_KF(channel::String, iK::Int, flavor_idx::Int=1; fullvertex=false, qtcikwargs...)
    basepath = "SIAM_u=0.50/"
    gamcore = load_Γ_KF(basepath; fullvertex=fullvertex, channel=channel, flavor_idx=flavor_idx)
    return compress_precomputed_V_KF(gamcore, iK, channel; qtcikwargs...)
end

function compress_precomputed_V_KF(Γcore_ref::Array{ComplexF64,7}, iK::Int, channel="t"; store=false, qtcikwargs...)

    iK_tuple = TCI4Keldysh.KF_idx(iK, 3)
    Γcore_ref = permutedims(Γcore_ref, (1,2,3, 4,5,6,7))
    R7slice = compression_slice()
    to_tci = Γcore_ref[R7slice, R7slice, R7slice, iK_tuple...]
    # to_tci = TCI4Keldysh.zeropad_array(Γcore_ref[:,:,:,iK_tuple...])
    @show size(Γcore_ref)

    qtt, _, _ = quanticscrossinterpolate(to_tci; qtcikwargs...)

    println("Compression done")

    tolerance = Dict(qtcikwargs)[:tolerance]
    if store
        # qtt_fat = zeros(ComplexF64, size(to_tci))
        # Threads.@threads for ic in CartesianIndices(to_tci)
        #     qtt_fat[ic] = qtt(Tuple(ic)...)
        # end
        qtt_fat = TCI4Keldysh.qtt_to_fattensor(qtt.tci.sitetensors)
        @show size(qtt_fat)
        qtt_fat = TCI4Keldysh.qinterleaved_fattensor_to_regular(qtt_fat, round(Int, log2(length(R7slice))))
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

function plot_tci_triptych(ref::Array{T,3}, tcival::Array{T,3}, err::Array{Float64,3}, tolerance, slice_dim::Int, slice_idx::Int) where {T}
    maxval = maximum(abs.(ref))

    slice = ntuple(i -> ifelse(i==slice_dim, slice_idx, Colon()), 3)
    scfun(x) = log10(abs(x))
    heatmap(
        scfun.(ref[slice...]);
        clim=(log10(maxval) + log10(tolerance), log10(maxval))
    )
    savefig("ref_$(slice_dim)$(slice_idx)fix.pdf")
    heatmap(
        scfun.(tcival[slice...]);
        clim=(log10(maxval) + log10(tolerance), log10(maxval))
    )
    savefig("tci_$(slice_dim)$(slice_idx)fix.pdf")
    heatmap(
        log10.(abs.(err[slice...]));
        clim=(log10(tolerance), -1)
    )
    savefig("diff_$(slice_dim)$(slice_idx)fix.pdf")
end

"""
Reference - TCI - error for precomputed MuNRG Keldysh vertex
"""
function triptych_precomputed_V_KF(iK::Int, tolerance::Float64)
    tci = h5read(joinpath(precompressed_datadir(), precomp_compr_filename(iK, tolerance)), "qttdata")
    ref = h5read(joinpath(precompressed_datadir(), precomp_compr_filename(iK, tolerance)), "reference")
    err = abs.(h5read(joinpath(precompressed_datadir(), precomp_compr_filename(iK, tolerance)), "diff")) ./ maximum(abs.(ref))

    @show argmax(err)
    @show maximum(err)
    plot_tci_triptych(ref, tci, err, tolerance, 1, 95)
end

function compress_precomputed_V_MF(channel::String, flavor_idx::Int=1; fullvertex=false, qtcikwargs...)
    basepath = "SIAM_u=0.50/"
    Vpath = joinpath(TCI4Keldysh.datadir(), basepath, "V_MF_" * TCI4Keldysh.channel_translate(channel))
    
    # load core vertex
    Γcore = nothing
    gamname = fullvertex ? "V_MF_sym.mat" : "V_MF_U4.mat"
    matopen(joinpath(Vpath, gamname)) do f
        CFdat = read(f, "CFdat")
        Γcore = CFdat["Ggrid"][flavor_idx]
    end

    return compress_precomputed_V_MF(Γcore; qtcikwargs...)
end

function compress_precomputed_V_MF(Γcore_ref::Array{ComplexF64,3}; qtcikwargs...)
    R7slice = compression_slice()
    to_tci = Γcore_ref[R7slice,R7slice,R7slice]
    @show size(to_tci)

    qtt, _, _ = quanticscrossinterpolate(to_tci; qtcikwargs...)

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
    iKtuple = TCI4Keldysh.KF_idx(iK,3)


    @assert size(ref)==size(tcival)==size(diff) "incompatible sizes"
    Nhalf = div(size(ref)[1], 2)

    maxref = maximum(abs.(ref))
    transfer_offset = 0 
    # transfer frequency comes last in reference data!
    slice = (Colon(), Colon(), Nhalf+1 + transfer_offset)
    # slice = (Nhalf + transfer_offset, Colon(), Colon())
    axis_ids = [1,2]
    axis_labels = (L"\omega", L"\omega'", L"\nu")
    # load frequency grid
    omgrid = load_omgrid_gamcore()
    omgrid_part = [om[compression_slice()] for om in omgrid]

    scfun(x) = abs(x)
    p = TCI4Keldysh.default_plot()
    heatmap!(p, omgrid_part[axis_ids[1]], omgrid_part[axis_ids[2]], scfun.(ref[slice...]); right_margin=10Plots.mm)
    ylabel!(axis_labels[axis_ids[1]])
    xlabel!(axis_labels[axis_ids[2]])
    title!(p, L"$\Gamma_{\mathrm{core}}^{%$iKtuple}$: Reference")
    savefig("V_KFref_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).pdf")

    p = TCI4Keldysh.default_plot()
    heatmap!(p, omgrid_part[axis_ids[1]], omgrid_part[axis_ids[2]], scfun.(tcival[slice...]); right_margin=10Plots.mm)
    ylabel!(axis_labels[axis_ids[1]])
    xlabel!(axis_labels[axis_ids[2]])
    title!(p, L"$\Gamma_{\mathrm{core}}^{%$iKtuple}$: QTCI")
    savefig("V_KFtci_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).pdf")

    p = TCI4Keldysh.default_plot()
    heatmap!(p, omgrid_part[axis_ids[1]], omgrid_part[axis_ids[2]], log10.(abs.(diff[slice...] ./ maxref)); right_margin=10Plots.mm)
    ylabel!(axis_labels[axis_ids[1]])
    xlabel!(axis_labels[axis_ids[2]])
    title!(p, L"$\log_{10}|\Gamma_{\mathrm{core}}^{\mathrm{ref}} - \Gamma_{\mathrm{core}}^{\mathrm{QTCI}}|_\infty / |\Gamma_{\mathrm{core}}^{\mathrm{ref}}|_\infty$")
    savefig("V_KFdiff_iK=$(iK)_tol=$(TCI4Keldysh.tolstr(tolerance)).pdf")
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
    Γcore_ref = load_Γ_KF(;channel=channel)
    for tol in reverse(10.0 .^ (-2:-6))
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
        savefig(joinpath(precompressed_datadir(),"V_KF_ranks_iK=$(iK)_R=$(R).pdf"))
    end
end

function plot_precomputed_V_KF_ranks_all(R::Int=7)
    p = plot(;guidefontsize=16, titlefontsize=16, tickfontsize=12, legendfontsize=9)

    for iK in 1:15
        plot_precomputed_V_KF_ranks(iK, R; p=p)
    end

    ylabel!(p, L"\chi")
    xlabel!(p, "tolerance")
    title!(p, L"All Keldysh components of $\Gamma_{\mathrm{core}}$: QTCI-ranks")
    worstcase = 2^(div(3*R, 2)) 
    ylims!(0, worstcase+30)
    hline!(p, [worstcase]; color=:black, linestyle=:dash, label="worstcase")
    savefig(p, joinpath(precompressed_datadir(), "V_KF_ranks.pdf"))
end

# for channel in ["t","a","pNRG"]
#     printstyled("\n-- CHANNEL $channel\n"; color=:blue)
#     for tol in 10.0 .^ [-2,-3,-4,-5,-6,-7]
#         compress_precomputed_V_MF(channel; unfoldingscheme=:interleaved, tolerance=tol)
#     end
# end

# iK_ranks = []
# for iK in 1:15
#     iK_bds = compress_precomputed_V_KF("p", iK; fullvertex=true, tolerance=1.e-3)
#     push!(iK_ranks, maximum(iK_bds))
# end
# println("Ranks for different Keldysh components: $iK_ranks")