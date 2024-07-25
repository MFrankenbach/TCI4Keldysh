using BenchmarkTools
using Plots
using ITensors
using Profile
using StatProfilerHTML

"""
Benchmark single-point evaluation of 4-point correlator
"""
function benchmark_Gp()
    R = 8
    GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=100.0, nωdisc=100)
    Gp = GF.Gps[1]
    # compile
    x = Gp(rand(1:2^R,3)...)
    # benchmark
    @btime x = $Gp(rand(1:2^$R, 3)...)
end

function time_FullCorrelator(;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=100.0, nomdisc=10)
    GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=beta, nωdisc=nomdisc)
    t = @elapsed begin
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved) 
    end
    @show rank(qtt)
    printstyled("==== Time: $t for tolerance=$tolerance, R=$R\n"; color=:blue)
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
    R = 5
    tolerance = 1.e-8
    GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=100.0, nωdisc=30)
    # compile
    @time TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved)
    # profile
    Profile.@profile TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved)
    statprofilehtml()
end

function GFfilename(mode::String, xmin::Int, xmax::Int, tolerance, beta)
    return "timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(tolstr(tolerance))_beta=$beta"
end

function tolstr(tolerance)
    return "$(round(Int, log10(tolerance)))"
end

"""
Store ranks and timings for computation of full 4-point correlators.
Can vary:
* beta
* nωdisc
* R
* tolerance
"""
function time_FullCorrelator_sweep(mode::String="R")
    folder = "pwtcidata"
    beta = 10.0
    tolerance = 1.e-8
    npt = 4
    times = []
    qttranks = []
    if mode=="R"
        nωdisc = 35
        Rs = 5:10
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["Rs"] = Rs
        d["nomdisc"] = nωdisc
        d["tolerance"] = tolerance
        outname = GFfilename(mode, first(Rs), last(Rs), tolerance, beta)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            # GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=beta, nωdisc=nωdisc)
            GF = TCI4Keldysh.dummy_correlator(npt, R; beta=beta)[1]
            nωdisc = div(maximum(size(GF.Gps[1].tucker.center)), 2)
            TCI4Keldysh.updateJSON(outname, "nomdisc", nωdisc, folder)
            t = @elapsed begin
                qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved)
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
                qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved)
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

time_FullCorrelator_sweep("R")
time_FullCorrelator_sweep("nomdisc")