using TCI4Keldysh
using JSON
using ITensors
using Plots

TCI4Keldysh.VERBOSE() = false
TCI4Keldysh.DEBUG() = false
TCI4Keldysh.TIME() = false

# for i in 1:1
#     TCI4Keldysh.test_TCI_precompute_anomalous_values(;npt=4, perm_idx=i, tolerance=1e-1)
# end

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
    rankplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont,
                legend=:left,
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

    # add ranks with singular value shift
    plot!(rankplot, betas, ranks_singval; label="sing. val. shift, R=$R", color=:black, marker=:diamond, markersize=msize)
    if do_bigR
        plot!(rankplot, betas, ranks_singval_bigR; label="sing. val. shift, R=$bigR", color=:black, marker=:diamond, markersize=msize, linestyle=:dash)
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

TCI4Keldysh.TIME() = true
# yields max. error ≤1.e-5
# TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(;npt=4, R=7, perm_idx=1, cutoff=1.e-8, tolerance=1.e-8)
TCI4Keldysh.test_TCI_frequency_rotation_reg_values(;npt=4, full_tci=true, perm_idx=3, cutoff=1e-3, tolerance=1e-3)
# TCI4Keldysh.test_imagtime_PartialCorrelator(;npt=4, R=12, perm_idx=1, cutoff=1.e-8, tolerance=1.e-8)