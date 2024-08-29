using BenchmarkTools
using Plots
using ITensors
using Profile
using QuanticsTCI
using StatProfilerHTML
import TensorCrossInterpolation as TCI

"""
Benchmark single-point evaluation of partial 4-point correlator
"""
function benchmark_Gp()
    R = 5
    GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=100.0, nωdisc=100)
    Gp = GF.Gps[1]
    # compile
    x = Gp(1,1,1)
    @show x
    # benchmark
    @btime x = $Gp(rand(1:2^$R, 3)...)
end

"""
Compare times of block-wise vs. pointwise evaluation of full correlator.
"""
function compare_block_vs_pointwise_eval(R::Int, beta::Float64=2000.0, tolerance=1.e-6)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)[1]
    t = @elapsed begin 
            data = TCI4Keldysh.precompute_all_values(GF)
        end
    GFev = TCI4Keldysh.FullCorrEvaluator_MF(GF, true; cutoff=tolerance*1.e-2)
    data_pt = zeros(ComplexF64, size(data))
    t_pt = @elapsed begin
            for ic in CartesianIndices(data_pt)
                data_pt[ic] = GFev(Tuple(ic)...)
            end
        end
    open("compare_block_pointwise_eval.log", "a") do io
        TCI4Keldysh.log_message(io, "Parameters: R=$R, β=$beta, tol=$tolerance")
        TCI4Keldysh.log_message(io, "Time for block evaluation: $t [s]")
        TCI4Keldysh.log_message(io, "Time for pointwise evaluation: $t_pt [s]")
        TCI4Keldysh.log_message(io, "Ratio (point/block): $(t_pt/t)")
        TCI4Keldysh.log_message(io, "")
    end
end

"""
Benchmark single-point evaluation of full 4-point correlator (TT vs kernel convolution)
For R=6, beta=100, TT has rank ≈300 and is slower
"""
function benchmark_GF(R::Int, beta::Float64)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)[1]
    GF2 = deepcopy(GF)
    qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF, true; tolerance=1.e-8, unfoldingscheme=:interleaved)
    fev = TCI4Keldysh.FullCorrEvaluator_MF(GF2, true; cutoff=1.e-10)
    # compile
    fev(rand(1:2^R,3)...)
    qtt(rand(1:2^R,3)...)
    # benchmark
    printstyled("Benchmark pointwise eval...\n"; color=:blue)
    bmp = @benchmark x = $fev(rand(1:2^$R, 3)...)
    printstyled("    Mean time: $(mean(bmp).time/1.e3) μs \n")
    printstyled("Benchmark TT eval...\n"; color=:blue)
    @show TCI4Keldysh.rank(qtt)
    bmtt = @benchmark x = $qtt(rand(1:2^$R, 3)...)
    printstyled("    Mean time: $(mean(bmtt).time/1.e3) μs \n")
    return (bmp, bmtt)
end

function plot_benchmark_GF()
    Rs = [5,6,7]
    colors = [:blue, :red, :green]
    # Rs = [5]
    betas = 10.0 .^ (1:3)
    p = plot(; xscale=:log10)
    for (i,R) in enumerate(Rs)
        times_p = []
        times_tt = []
        for beta in betas
            (bmp, bmtt) = benchmark_GF(R, beta)
            push!(times_p, mean(bmp).time / 1.e3)
            push!(times_tt, mean(bmtt).time / 1.e3)
        end
        plot!(p, betas, times_p; color=colors[i], marker=:diamond, label="R=$R, pw")
        plot!(p, betas, times_tt; color=colors[i], marker=:circle,label="R=$R, TT")
    end
    xlabel!(p, "β")
    ylabel!(p, "time[μs]")
    savefig(p,"GF_eval_timing.png")
end

function time_PartialCorrelator(perm_idx ;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=1000.0, nomdisc=10)
    # GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=beta, nωdisc=nomdisc)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)
    Gp = GF[1].Gps[perm_idx]
    t = @elapsed begin
        qtt = TCI4Keldysh.compress_PartialCorrelator_pointwise(Gp, true; tolerance=tolerance, unfoldingscheme=:interleaved) 
    end
    @show TCI4Keldysh.rank(qtt)
    printstyled("==== Time: $t for tolerance=$tolerance, R=$R\n"; color=:blue)
    return t
end

"""
Compare evaluation of FullCorrelator with and without pruning the tucker center
"""
function time_tucker_cut()
    R = 5 
    npt = 4
    GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=2000.0, nωdisc=20)
    # GF = TCI4Keldysh.dummy_correlator(npt, R; beta=2000.0, is_compactAdisc=false)[1]
    exact_data = TCI4Keldysh.precompute_all_values(GF)

    cutoff = 1.e-8
    tucker_cutoff = 1.e-7
    fev = TCI4Keldysh.FullCorrEvaluator_MF(GF, true; cutoff=cutoff, tucker_cutoff=tucker_cutoff)

    GFmax = maximum(abs.(exact_data))
    @show GFmax
    errors = zeros(Float64, 2^(R*(npt-1)))
    cc = 1
    for w in Iterators.product(fill(1:2^R, npt-1)...)
        error = abs(fev(w...) - fev(Val{:nocut}(),w...)) / GFmax
        errors[cc] = error
        cc += 1
    end
    printstyled("Maximum error: $(maximum(errors))\n"; color=:red)
    @assert maximum(errors) < tucker_cutoff

    @btime $fev(rand(1:2^$R, $npt-1)...)
    @btime $fev(Val{:nocut}(), rand(1:2^$R, $npt-1)...)
end

"""
Check correlator values at the fringes of the grid and compare to tolerance.
"""
function max_correlator_tail(;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=10.0)
    # GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=beta, nωdisc=nomdisc)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)
    GF_center = TCI4Keldysh.dummy_correlator(4, 6; beta=beta)[1]
    data_center = TCI4Keldysh.precompute_all_values(GF_center)
    cenval = maximum(abs.(data_center))
    @time qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF[1], true; tolerance=tolerance, unfoldingscheme=:interleaved) 
    frval = abs(qtt(1,2^(R-1),2^(R-1)))
    printstyled("Fringe value: $frval\n"; color=:blue)
    printstyled("Center max value: $cenval\n"; color=:blue)
    printstyled("Ratio: $(frval/cenval) (tol=$tolerance)\n"; color=:blue)


    # plot line section
    Nhalf = 2^(R-1)
    yvals = [qtt(i, Nhalf, Nhalf) for i in 1:2^R]
    plot(1:2^R, abs.(yvals) ./ cenval; yscale=:log10, label=nothing, linewidth=2)
    ylabel!("|G(ω)|/max|G(ω)|")
    xlabel!("ω1")
    hline!(tolerance; label="tol", color=:red, linestyle=:dashed)
    savefig("corr_tail.png")
    return nothing
end

function time_FullCorrelator(;R::Int=5, tolerance::Float64=1.e-6)
    PSFpath = "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/"
    # PSFpath = "SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    T = TCI4Keldysh.dir_to_T(PSFpath)
    beta = 1.0/T
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta, is_compactAdisc=false, PSFpath=PSFpath)
    t = @elapsed begin
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF[1], true; tolerance=tolerance, unfoldingscheme=:interleaved, cut_tucker=true) 
    end
    @show TCI4Keldysh.rank(qtt)
    printstyled("==== Time: $t for tolerance=$tolerance, R=$R\n"; color=:blue)
    return t
end

function time_FullCorrelator_batch(;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=10.0)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)
    t = @elapsed begin
        tt, _ = TCI4Keldysh.compress_FullCorrelator_batched(GF[1], true; tolerance=tolerance) 
    end
    @show TCI.linkdims(tt)
    printstyled("==== Time: $t for tolerance=$tolerance, R=$R\n"; color=:blue)
    return t
end

function time_FullCorrelator_evalcount(;R::Int=5, tolerance::Float64=1.e-6, beta::Float64=2000.0)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)
    t = @elapsed begin
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise_evalcount(GF[1], true; tolerance=tolerance, unfoldingscheme=:interleaved) 
    end
    @show TCI4Keldysh.rank(qtt)
    printstyled("==== Time: $t for tolerance=$tolerance, R=$R\n"; color=:blue)
    return t
end

# yields worst case at least up to R=7
function time_FullCorrelator_natural(;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=100.0, nomdisc=10)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)
    t = @elapsed begin
        tt = TCI4Keldysh.compress_FullCorrelator_natural(GF[1], true; tolerance=tolerance) 
    end
    @show TCI.rank(tt)
    printstyled("==== Time: $t for tolerance=$tolerance, R=$R\n"; color=:blue)
    return t
end


function compare_PartialFullCorrelator(;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=100.0, nomdisc=10)
    tfull = time_FullCorrelator(;R=R, tolerance=tolerance, beta=beta)
    ts_partial = []
    for i in 1:24
        tp = time_PartialCorrelator(i ;R=R, tolerance=tolerance, beta=beta, nomdisc=nomdisc)
        push!(ts_partial, tp)
    end
    printstyled("\n==== FullCorrelator time: $tfull for tolerance=$tolerance, R=$R\n"; color=:blue)
    printstyled("==== All PartialCorrelators time: $(sum(ts_partial)) for tolerance=$tolerance, R=$R\n"; color=:blue)
end

"""
Compare two ways of computing ∑_i v_i * w_i
"""
function benchmark_dot()
    N = 1000

    function expldot(v::Vector{T}, w::Vector{T}) where {T}
        ret = zero(T)
        l = length(v)
        @inbounds for i in 1:l
            ret += v[i] * w[i]
        end
        return ret
    end

    function sumdot(v::Vector{T}, w::Vector{T}) where {T}
        @views ret = sum(v .* w)
        return ret
    end

    println("--Explicit:")
    @btime $expldot(rand(Float64, $N), rand(Float64, $N))
    println("--Sum:")
    @btime $sumdot(rand(Float64, $N), rand(Float64, $N))
end

"""
Profile pointwise TCI of full correlator.
"""
function profile_FullCorrelator(npt=3)
    Profile.clear()
    R = 6
    tolerance = 1.e-6
    # GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=10.0, nωdisc=10)
    GF = TCI4Keldysh.dummy_correlator(npt, R; beta=2000.0)[1]
    # compile
    @time TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved)
    # profile
    Profile.@profile TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved)
    statprofilehtml()
end

function GFfilename(mode::String, xmin::Int, xmax::Int, tolerance, beta)
    return "timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$beta"
end

"""
Compute error introduced in regular partial correlators when the kernels are truncated with via SVD.
"""
function svd_error_MF(R::Int, beta::Float64, cutoff::Union{Float64, Nothing}=nothing, tolerance=1.e-8)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)[1]

    # truncate
    for Gp in GF.Gps[1:1]
        println(" -- Before")
        ref = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
        @show size(Gp.tucker.center)
        @show size.(Gp.tucker.legs)
        if isnothing(cutoff)
            cutoff = TCI4Keldysh.auto_svd_cutoff(Gp.tucker, tolerance, true)
            printstyled("Automatic cutoff: $cutoff\n"; color=:blue)
        end
        TCI4Keldysh.svd_kernels!(Gp.tucker; cutoff=cutoff)
        println(" -- After")
        @show size(Gp.tucker.center)
        @show size.(Gp.tucker.legs)
        cutval = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
        # compute error
        rel_err =  maximum(abs.(cutval - ref) / maximum(abs.(ref)))
        abs_err =  maximum(abs.(cutval - ref))
        norm_err =  norm(abs.(cutval - ref))
        printstyled("---- Errors: max(rel.)=$rel_err, max(abs.)=$abs_err, norm=$norm_err\n"; color=:green)
    end

end

"""
Check rank of Matsubara kernel
"""
function svd_rank_kernel(R::Int, beta::Float64, cutoff::Float64=1.e-15)
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)
    kernels = GF[1].Gps[1].tucker.legs
    rank_reds = fill(0.0, length(kernels))
    cc = 1
    for k in kernels
        @show size(k)
        _, S, _ = svd(k)
        Scut = [s for s in S if s>cutoff]
        rank_reds[cc] = length(Scut)/size(k,2)
        cc += 1
    end
    return rank_reds
end

"""
Store ranks and timings for computation of full 4-point correlators.
Can vary:
* beta
* nωdisc
* R
* tolerance
"""
function time_FullCorrelator_sweep(mode::String="R"; beta=10.0, Rs=nothing)
    folder = "pwtcidata"
    tolerance = 1.e-8
    npt = 4
    times = []
    qttranks = []
    svd_kernel = true
    if mode=="R"
        nωdisc = 35
        Rs = isnothing(Rs) ? (5:10) : Rs
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["Rs"] = Rs
        d["nomdisc"] = nωdisc
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        outname = GFfilename(mode, first(Rs), last(Rs), tolerance, beta)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            # GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=beta, nωdisc=nωdisc)
            GF = TCI4Keldysh.dummy_correlator(npt, R; beta=beta)[1]
            nωdisc = div(maximum(size(GF.Gps[1].tucker.center)), 2)
            TCI4Keldysh.updateJSON(outname, "nomdisc", nωdisc, folder)
            t = @elapsed begin
                qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)
            end
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
            println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        end
    elseif mode=="nomdisc"
        R = 5
        nωdiscs = 10:50
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["nomdiscs"] = nωdiscs
        d["R"] = R
        d["tolerance"] = tolerance
        outname = GFfilename(mode, first(nωdiscs), last(nωdiscs), tolerance, beta)
        TCI4Keldysh.logJSON(d, outname, folder)

        for n in nωdiscs
            GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=beta, nωdisc=n)
            t = @elapsed begin
                qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)
            end
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
            println(" ===== nωdisc=$n: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        end
    else
        error("Invalid mode $mode")
    end
end

"""
How much RAM would a densely stored correlator need in GB
"""
function RAM_usage_3D(R::Int)
    return 16 * 2^(3*R) / 1.e9
end

function plot_FullCorrelator_timing(param_range, mode="R"; beta=10.0, tolerance=1.e-8, plot_mem=false)
    folder = "pwtcidata"    
    filename = GFfilename(mode, minimum(param_range), maximum(param_range), tolerance, beta)
    data = TCI4Keldysh.readJSON(filename, folder)

    if mode=="R"
        Rs = convert.(Int, data["Rs"])
        RAM_usage = RAM_usage_3D.(Rs)
        times = convert.(Float64, data["times"])
        p = TCI4Keldysh.default_plot()

        plot!(p, Rs, times; marker=:diamond, color=:blue, label="F1F1dagF3F3dag")
        xlabel!(p, "R")
        ylabel!(p, "Wall time [s]")
        title!(p, "Timings full correlator, β=$beta, tol=$tolerance")

        if plot_mem
            ptwin = twinx(p)
            plot!(ptwin, Rs, RAM_usage; marker=:circle, color=:black, linestyle=:dash, yscale=:log10, label=nothing)
            yticks!(ptwin, 10.0 .^ (round(Int, log10(minimum(RAM_usage))) : round(Int, log10(maximum(RAM_usage)))))
            ylabel!(ptwin, "Memory for dense corr. [GB]")
        end

        savefig(p, "corrtiming_beta=$(beta)_tol=$(round(Int,log10(tolerance))).png")
    end
end

function test_reduce_Gps!()
    R = 5
    GF = TCI4Keldysh.dummy_correlator(4, R; beta=100.0, is_compactAdisc=false)[1]
    data = TCI4Keldysh.precompute_all_values(GF)

    TCI4Keldysh.reduce_Gps!(GF)
    data_red = TCI4Keldysh.precompute_all_values(GF)

    @assert maximum(abs.(data .- data_red)) < 1.e-11
end