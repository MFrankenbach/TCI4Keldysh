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

# TCI4Keldysh.test_compress_PartialCorrelator_pointwise3D(;perm_idx=3, R=5)
# TCI4Keldysh.test_compress_PartialCorrelator_pointwise3D(;perm_idx=3, R=7, nomdisc=30)
# TCI4Keldysh.test_compress_PartialCorrelator_pointwise3D(;perm_idx=3, R=7, nomdisc=50)
# TCI4Keldysh.test_compress_PartialCorrelator_pointwise3D(;perm_idx=3, R=9, nomdisc=20)
# TCI4Keldysh.test_compress_PartialCorrelator_pointwise3D(;perm_idx=3, R=9, nomdisc=50)