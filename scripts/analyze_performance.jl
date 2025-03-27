using TCI4Keldysh
using JET
using BenchmarkTools
using Profile
using StatProfilerHTML
using PProf
using Random
using Serialization

function benchmark_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    # b = @benchmark $gev(rand(1:2^$R, 3)...)
    b = @benchmark $gev(127, 128, 126)
    display(b)
    println("--------------------------------")
    @allocated gev(51,53,46)
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
end


function profile_allocations_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    R = 8
    Profile.Allocs.clear()
    Profile.Allocs.@profile begin
        gev(rand(1:2^R, 3)...)
    end
    PProf.Allocs.pprof()
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
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev.serialized"))
    @report_opt gev(50,53,47)
end