using TCI4Keldysh
using JSON
using ITensors
using Plots
using LaTeXStrings

TCI4Keldysh.VERBOSE() = false
TCI4Keldysh.DEBUG() = false
TCI4Keldysh.TIME() = false

"""
Monitor mean and max error of all permutations for 3/4-point functions before frequency rotation
"""
function check_all_permutations_reg()
    # compile
    TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=3, perm_idx=1)
    npt = 4
    file_out = "tci_$(npt)-pt_errors.txt"
    open(file_out, "a") do f 
        write(f, "perm_idx  mean_err     max_err\n")
    end
    for i in 1:factorial(npt)
        printstyled(" ---- $i-th permutation\n"; color=:blue)
        mean_err, max_err = TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=npt, perm_idx=i)
        open(file_out, "a") do f 
            write(f, "$i        $mean_err, $max_err\n")
        end
    end
end

function plot_kernel_ranks_beta_2pt(;R=12, imagtime=false, tolerance=1.e-8, do_bigR=true)
    # collect data
    fermikernel = false
    npt = 2
    tauR = R+7
    bigR = R+2
    betas = 10.0 .^ (-1:4)
    ranks_freq = Int[]
    ranks_tau = Int[]
    # without singval shifts
    for beta in betas
        rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=R, tolerance=tolerance, beta=beta, fermikernel=fermikernel, singvalshift=false)
        if imagtime
            rt = TCI4Keldysh.imagtime_kernel_ranks(;npt=npt, R=tauR, tolerance=tolerance, beta=beta)
            rtr = fermikernel ? maximum(rt[1:npt-1]) : maximum(rt[npt:end])
            push!(ranks_tau, rtr)
        end
        push!(ranks_freq, rf)
    end
    # without singval shifts, for bigger R
    ranks_freq_bigR = Int[]
    if do_bigR
        for beta in betas
            rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=bigR, tolerance=tolerance, beta=beta, fermikernel=fermikernel, singvalshift=false)
            push!(ranks_freq_bigR, rf)
        end
    end

    # plot
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    rankplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont,
                legend=:topleft,
                xscale=:log10)
    omlabel = imagtime ? "ω, R=$R" : "ω-kernel, R=$R"
    omlabelbig = imagtime ? "ω, R=$bigR" : "ω-kernel, R=$bigR"
    msize = 6
    plot!(rankplot, betas, ranks_freq; label=omlabel, marker=:diamond, markersize=msize, color=:blue)
    if imagtime
        plot!(rankplot, betas, ranks_tau; label="τ, R=$tauR", marker=:circle, markersize=msize, color=:red)
    end
    if do_bigR
        plot!(rankplot, betas, ranks_freq_bigR; label=omlabelbig, marker=:diamond, markersize=msize, color=:blue, linestyle=:dash)
    end

    kernelname = fermikernel ? "fermionic" : "bosonic"
    titl = imagtime ? "Ranks of $kernelname kernels" : "Ranks of $kernelname ω-kernels"
    # title!(rankplot, titl)
    xlabel!(rankplot, "β")
    ylabel!(rankplot, "Kernel rank")
    xticks!(rankplot, betas)
    (_, xmax) = ylims(rankplot)
    ylims!(rankplot, 0, xmax)
    fermistr = fermikernel ? "fer" : "bos"
    prefix = imagtime ? "" : "freq_"
    savefig(rankplot, prefix * "kernelranks_beta_$(npt)pt_$(fermistr)_tol=$(round(Int,log10(tolerance))).png")
end

function compare_1D_3D_kernel(;R=7, perm_idx=1, beta=10.0, tolerance=1.e-8)

    # get PSF
    npt = 4
    perms = [p for p in permutations(collect(1:npt))]
    perm = perms[perm_idx]
    path = npt<=3 ? "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/" : joinpath("data/SIAM_u=0.50/PSF_nz=2_conn_zavg/", "4pt")
    Ops = if npt==2
            ["F1", "F1dag"][perm]
        elseif npt==3
            ["F1", "F1dag", "Q34"][perm]
        else
            ["F1", "F1dag", "F3", "F3dag"][perm]
        end
    spin = 1

    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    ωconvMat_sum = cumsum(ωconvMat[perm[1:(npt-1)],:]; dims=1)
    psf = TCI4Keldysh.PartialCorrelator_reg(npt-1, 1/beta, R, path, Ops, ωconvMat_sum; flavor_idx=spin, nested_ωdisc=false)

    rf = TCI4Keldysh.compress_frequencykernels_light(R, psf.tucker; tolerance=tolerance)
    printstyled("1D ranks: $(TCI4Keldysh.rank.(rf))\n")
    fullkernel = TCI4Keldysh.compress_full_frequencykernel(R, psf.tucker; tolerance=tolerance)
    printstyled("$(npt-1)D rank: $(TCI4Keldysh.rank(fullkernel))\n")
end

function plot_kernel_ranks_beta(;npt=2, R=12, imagtime=false, tolerance=1.e-8, fermikernel=true, do_bigR=true)
    # collect data
    tauR = R+7
    bigR = R+3
    betas = 10.0 .^ (-1:4)
    ranks_freq = Int[]
    ranks_tau = Int[]
    # without singval shifts
    for beta in betas
        rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=R, tolerance=tolerance, beta=beta, fermikernel=fermikernel, singvalshift=false)
        if imagtime
            rt = TCI4Keldysh.imagtime_kernel_ranks(;npt=npt, R=tauR, tolerance=tolerance, beta=beta)
            rtr = fermikernel ? maximum(rt[1:npt-1]) : maximum(rt[npt:end])
            push!(ranks_tau, rtr)
        end
        push!(ranks_freq, rf)
    end
    # without singval shifts, for bigger R
    ranks_freq_bigR = Int[]
    if do_bigR
        for beta in betas
            rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=bigR, tolerance=tolerance, beta=beta, fermikernel=fermikernel, singvalshift=false)
            push!(ranks_freq_bigR, rf)
        end
    end

    ranks_singval = Int[]
    for beta in betas
        rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=R, tolerance=tolerance, beta=beta, fermikernel=fermikernel, singvalshift=true)
        push!(ranks_singval, rf)
    end
    # for bigger R
    if do_bigR
        ranks_singval_bigR = Int[]
        for beta in betas
            rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=bigR, tolerance=tolerance, beta=beta, fermikernel=fermikernel, singvalshift=true)
            push!(ranks_singval_bigR, rf)
        end
    end

    # plot
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    lwidth = 3
    rankplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont,
                legend=:left,
                xscale=:log10)
    omlabel = imagtime ? "ω, R=$R" : "ω-kernel, R=$R"
    omlabelbig = imagtime ? "ω, R=$bigR" : "ω-kernel, R=$bigR"
    msize = 6
    plot!(rankplot, betas, ranks_freq; label=omlabel, marker=:diamond, markersize=msize, color=:blue, linewidth=lwidth)
    if imagtime
        plot!(rankplot, betas, ranks_tau; label="τ, R=$tauR", marker=:circle, markersize=msize, color=:red, linewidth=lwidth)
    end
    if do_bigR
        plot!(rankplot, betas, ranks_freq_bigR; label=omlabelbig, marker=:diamond, markersize=msize, markercolor=:blue, linecolor=:lightgreen, linestyle=:dot, linewidth=lwidth)
    end

    singvalcolor = RGBA(0.0,136/256,58/256,1.0)
    # add ranks with singular value shift
    plot!(rankplot, betas, ranks_singval; label="U(ω,α), R=$R", color=singvalcolor, marker=:diamond, markersize=msize, linewidth=lwidth)
    if do_bigR
        plot!(rankplot, betas, ranks_singval_bigR; label="U(ω,α), R=$bigR", color=singvalcolor, marker=:diamond, markersize=msize, linestyle=:dot, linewidth=lwidth)
    end

    kernelname = fermikernel ? "fermionic" : "bosonic"
    titl = imagtime ? "Ranks of $kernelname kernels" : "Ranks of $kernelname ω-kernels"
    # title!(rankplot, titl)
    xlabel!(rankplot, "β")
    ylabel!(rankplot, "Kernel rank")
    xticks!(rankplot, betas)
    (_, xmax) = ylims(rankplot)
    ylims!(rankplot, 0, xmax)
    fermistr = fermikernel ? "fer" : "bos"
    prefix = imagtime ? "" : "freq_"
    savefig(rankplot, prefix * "kernelranks_beta_$(npt)pt_$(fermistr)_tol=$(round(Int,log10(tolerance))).png")
end

"""
Plot ranks of fermionic kernels against R and beta.
Optionally with imagtime kernels.
"""
function plot_kernel_ranks(;imagtime=true, tolerance=1.e-8, fermikernel=true)

    npt = 4

    Rs = 7:13
    betas = [1.e2, 1.e3, 1.e4]
    colors = [:red, :blue, :green]
    rankplot = plot()
    # bos_idx = 2 # for perm_idx=1, npt=4
    for (i,beta) in enumerate(betas)
        ranks_freq = Int[]
        ranks_tau = Int[]
        for R in Rs
            rf = TCI4Keldysh.frequency_kernel_ranks(;npt=npt, R=R, tolerance=tolerance, beta=beta, fermikernel=fermikernel)
            if imagtime
                rt = TCI4Keldysh.imagtime_kernel_ranks(;npt=npt, R=R, tolerance=tolerance, beta=beta)
                push!(ranks_tau, rt[npt])
            end
            push!(ranks_freq, rf)
        end
        omlabel = imagtime ? "ω, β=$beta" : "ω-kernel, β=$beta"
        plot!(rankplot, Rs, ranks_freq; label=omlabel, marker=:diamond, color=colors[i], legend=:topleft)
        if imagtime
            plot!(rankplot, Rs, ranks_tau; label="τ, β=$beta", marker=:circle, linestyle=:dash, color=colors[i], legend=:left)
        end
    end
    # worst = [2^r for r in Rs]
    # plot!(rankplot, Rs, worst; label="worst case", color=:gray, linestyle=:dot)
    kernelname = fermikernel ? "fermionic" : "bosonic"
    titl = imagtime ? "Ranks of $kernelname kernels" : "Ranks of $kernelname ω-kernels"
    title!(rankplot, titl)
    xlabel!(rankplot, "R (# bits)")
    ylabel!(rankplot, "Kernel rank")
    fermistr = fermikernel ? "fer" : "bos"
    prefix = imagtime ? "" : "freq_"
    savefig(rankplot, prefix * "kernelranks$(npt)pt_$(fermistr)_tol=$(round(Int,log10(tolerance))).png")
end

"""
Reveal dependence of 2pt G(ω) computed via imagtime-TCI and exactly on β and R_τ.
"""
function fit_2pt_imagtime_error()
    betas = 10.0 .^ (-1:4)    
    Rtaus = collect(14:20)

    maxdiffs = fill(0.0, (length(betas), length(Rtaus)))
    stds = fill(0.0, (length(betas), length(Rtaus)))
    for (i, beta) in enumerate(betas)
        for (j, R) in enumerate(Rtaus)
            (m, s) = TCI4Keldysh.test_1peak_imagtime_PartialCorrelator(R, beta)
            maxdiffs[i,j] = m
            stds[i,j] = s
        end
    end

    display(maxdiffs)

    println("\n ---- Fit beta dependence for fixed Rτ")
    for j in eachindex(Rtaus)
        (a,b) = TCI4Keldysh.linear_fit(betas, maxdiffs[:,j])
        println(  "δ = $a * β + $b")
    end
    println("\n ---- Fit Rτ dependence for fixed β")
    for i in eachindex(betas)
        (a,b) = TCI4Keldysh.linear_fit(convert.(Float64, Rtaus), log2.(maxdiffs[i,:]))
        println(  "δ = $a * log2(Rτ) + $b")
    end
end

function time_TCI_precompute_reg_values(;npt=3)
    
    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    R = 7
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; nested_ωdisc=false)

    # pick PSF
    spin = 1
    perm_idx = 1
    Gp = GFs[spin].Gps[perm_idx]

    # TCI computation
    cutoffs = 10.0 .^ (-2:-1:-5)

    times = []
    dummy = Vector{MPS}(undef, 1)
    # for some reason, the second loop with npt=4 takes ages here, also for large cutoffs
    for c in cutoffs
        t = @elapsed begin 
                dummy[1] = TCI4Keldysh.TD_to_MPS_via_TTworld(Gp.tucker; tolerance=1e-12, cutoff=c)
            end
        @show t
        push!(times, t)
        GC.gc()
    end

    # write to file
    d = Dict("cutoffs" => cutoffs, "times" => times, "npt" => npt, "perm_idx" => perm_idx, "ops" => Ops)
    logJSON(d, "reg_val_times_$(npt)pt")
end

# utilities ========== 
function logJSON(data, filename::String)
    fullname = filename*".json"
    open(fullname, "w") do file
        JSON.print(file, data)
    end
    printstyled("File $filename.json written!\n", color=:green)
end

function readJSON(filename::String)
    data = open(filename) do file
        JSON.parse(file)
    end 
    return data
end
# utilities END ========== 

include("../test/utils4tests.jl")

function test_full_correlator(npt::Int; include_ano=true)

    ITensors.disable_warn_order()

    R = npt==2 ? 13 : 6
    beta = 10.0
    tolerance = 1.e-8
    cutoff = npt < 4 ? 1.e-25 : 1.e-20
    # GF = TCI4Keldysh.dummy_correlator(npt, R; beta=beta)[1]
    GF = multipeak_correlator_MF(npt, R; beta=beta, peakdens=1.0, nωdisc=4)

    # Gfull =  TCI4Keldysh.TCI_precompute_reg_values_rotated(GF.Gps[1];
    #             tolerance=tolerance, cutoff=cutoff, include_ano=include_ano)
    # printstyled(" ---- Rank p=1: $(TCI4Keldysh.rank(Gfull))\n"; color=:green)
    Gps_out = Vector{MPS}(undef, factorial(npt))
    for perm_idx in 1:factorial(npt)

        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GF.Gps[perm_idx];
                                tolerance=tolerance, cutoff=cutoff, include_ano=include_ano)

        # for i in 1:npt-1
        #     TCI4Keldysh._adoptinds_by_tags!(Gp_mps, Gfull, "ω$i", "ω$i", R)
        # end
        # Gfull = add(Gfull, Gp_mps; cutoff=1.e-35, use_absolute_cutoff=false)

        Gps_out[perm_idx] = Gp_mps
        printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
        
        # # plot diff to exact
        # Gpref = TCI4Keldysh.precompute_all_values_MF(GF.Gps[perm_idx])
        # Gpref_noano = TCI4Keldysh.precompute_all_values_MF_noano(GF.Gps[perm_idx])
        # diffslice = ntuple(i -> (i==1 ? (2:2^R) : 1:2^R), npt-1)
        # Gpval = TCI4Keldysh.MPS_to_fatTensor(Gp_mps; tags=ntuple(i -> "ω$i", npt-1))

        # scfun = x -> log(abs(x))
        # heatmap(scfun.(Gpref[diffslice...]))
        # savefig("Gpref$perm_idx.png")
        # heatmap(scfun.(Gpval[diffslice...]))
        # savefig("Gpval$perm_idx.png")
        # diff = Gpval[diffslice...] - Gpref[diffslice...]
        # heatmap(scfun.(diff))
        # savefig("diff$perm_idx.png")
        # anoref = Gpref - Gpref_noano
        # heatmap(scfun.(anoref[diffslice...]))
        # savefig("ano$perm_idx.png")

        GC.gc()
    end

    Gfull = TCI4Keldysh.FullCorrelator_add(Gps_out; cutoff=1.e-20, use_absolute_cutoff=false)
    @show TCI4Keldysh.rank(Gfull)

    Gfull_ref = if include_ano
                    TCI4Keldysh.precompute_all_values(GF)
                else
                    TCI4Keldysh.precompute_all_values_MF_noano(GF)
                end
    Gfull_fat = TCI4Keldysh.MPS_to_fatTensor(Gfull; tags=ntuple(i -> "ω$i", npt-1))

    # reference
    diffslice = ntuple(i -> (i==1 ? (2:2^R) : 1:2^R), npt-1)
    diff = abs.(Gfull_fat[diffslice...] - Gfull_ref[diffslice...]) / maximum(abs.(Gfull_ref))
    maxdiff = maximum(diff)
    test_tol = npt < 4 ? 1.e3 * tolerance : 1.e2*tolerance
    printstyled(" ---- Max error $maxdiff for tol=$tolerance, cut=$cutoff, npt=$npt\n"; color=:blue)
    return maxdiff < test_tol
end

"""
Return slice where D+1 correlator computed via TCI is expected to match the conventionally computed one.
"""
function diffslice(D::Int, N::Int)
    return ntuple(i -> (i==1 ? (2:N) : (1:N)), D)
end