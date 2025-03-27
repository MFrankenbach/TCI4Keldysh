using TCI4Keldysh
using JET
using BenchmarkTools
using Profile
using StatProfilerHTML
using Random
using ProfileCanvas
using Serialization

function allocations_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    # trigger compilation
    x = gev(1, 1, 1);
    println("-- Allocations: ")
    println("gev: $((@allocated gev(101, 100, 99)) / 1e6)")
    println("gev.GFevs: ")
    for i in eachindex(gev.GFevs)
        println("$i: ", (@allocated gev.GFevs[i](101, 100, 99)) / 1.e6)
    end
    println("---------------")
end

function benchmark_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    # b = @benchmark $gev(rand(1:2^$R, 3)...)
    b = @benchmark $gev(127, 128, 126)
    display(b)
    println("--------------------------------")
end

function singlethreaded_eval(N::Int=10^4)
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    res = gev(1,1,1)
    TCI4Keldysh.report_mem()
    @time begin for _ in 1:N
        res += gev(rand(1:2^R,3)...)
        end
    end
    TCI4Keldysh.report_mem()
end

function batched_eval(n::Int)
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    d = 8
    gbev = TCI4Keldysh.ΓcoreBatchEvaluator_KF(gev; unfoldingscheme=:fused)
    leftindexset = Vector{Vector{Int}}(undef, n)
    rightindexset = Vector{Vector{Int}}(undef, n)
    # mimic two-site update
    leftsite = 4
    cc = 1
    for il in Iterators.product(ntuple(_->1:d,leftsite-1)...)
        leftindexset[cc] = collect(il)
        if cc<n
            cc += 1
        else
            break
        end
    end
    cc = 1
    for ir in Iterators.product(ntuple(_->1:d,R-leftsite-1)...)
        rightindexset[cc] = collect(ir)
        if cc<n
            cc += 1
        else
            break
        end
    end
    TCI4Keldysh.report_mem()
    @time begin
        out = gbev(leftindexset, rightindexset, Val(2))
        @show (length(gbev.qf.cache))
    end
end

function multithreaded_eval(N::Int=10^4)
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    res = zeros(ComplexF64, N+1)
    res[1] = gev(1,1,1)
    TCI4Keldysh.report_mem()
    @time begin
        Threads.@threads for i in 1:N
            res[i+1] = gev(rand(1:2^R,3)...)
        end
    end
    TCI4Keldysh.report_mem()
    return res
end


function profile_allocations_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    # trigger compilation
    x = gev(1, 1, 1);
    R = 8
    # Profile.Allocs.clear()
    # Profile.Allocs.@profile begin
    @profview gev(101, 100, 99)
    # PProf.Allocs.pprof()
end

function profile_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    Profile.clear()
    @profile begin
        for _ in 1:10000
            gev(rand(1:2^R, 3)...)
        end
    end
    statprofilehtml()
end

function eval_tucker_btime()
    println("3-dimensional")
    N = 20
    A = randn(ComplexF64, (N,N+2,N+3))
    legs = ntuple(i -> randn(ComplexF64, size(A,i)), ndims(A))
    b1 = @benchmark TCI4Keldysh.eval_tucker($A, $legs)
    display(b1)
    b2 = @benchmark TCI4Keldysh.eval_tucker_mat($A, $legs)
    display(b2)

    println("2-dimensional")
    N = 10
    A = randn(ComplexF64, (N,N+2))
    legs = ntuple(i -> randn(ComplexF64, size(A,i)), ndims(A))
    b1 = @benchmark TCI4Keldysh.eval_tucker($A, $legs)
    display(b1)
    b2 = @benchmark TCI4Keldysh.eval_tucker_mat($A, $legs)
    display(b2)
end

function type_instability_gev()
    # force compilation
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    println("==== ΓcoreEvaluator_KF")
    x = gev(1,1,1)
    # print(@report_opt gev(50,53,47))
    print(@code_warntype gev(50,53,47))
    println("\n\n==== MultipoleKFCEvaluator")
    # print(@report_opt gev.GFevs[1](50,53,47))
    print(@code_warntype gev.GFevs[1](50,53,47))
    println("\n\n==== HierarchicalTucker")
    print(@code_warntype gev.GFevs[1].Gps[1,1](50,53,47))
end