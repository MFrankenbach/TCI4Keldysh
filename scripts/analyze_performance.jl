using TCI4Keldysh
using JET
using BenchmarkTools
using Profile
using StatProfilerHTML
using Random
using ProfileCanvas
using Serialization

function allocations_ht()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"))
    # trigger compilation
    x = TCI4Keldysh.eval_buff!(gev, 1, 1, 1);
    println("-- Allocations HierarchicalTucker: ")
    for i in 1:1
        println("$i: ", (@allocated gev.GFevs[i].Gps[1,1](50,53,47)) / 1.e6)
    end
end

function test_eval_buff()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"))
    x = gev(50,51,12)
    y = TCI4Keldysh.eval_buff!(gev, 50,51,12)
    @show abs(x-y)
    return nothing
end


function allocations_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"))
    # trigger compilation
    x = TCI4Keldysh.eval_buff!(gev, 1, 1, 1);
    x = gev(1,1,1);
    println("-- Allocations: ")
    println("gev       : $((@allocated gev(50,53,47)) / 1e6)")
    println("gev (buff): $((@allocated TCI4Keldysh.eval_buff!(gev, 50,53,47)) / 1e6)")
    println("gev.GFevs: ")
    ret_buff = MVector{16,ComplexF64}(zeros(ComplexF64, 16))
    retarded_buff = MVector{4,ComplexF64}(zeros(ComplexF64, 4))
    idx_int = MVector{3,Int}(0,0,0)
    for i in 1:1
        println("$i:        ", (@allocated gev.GFevs[i](50,53,47)) / 1.e6)
        println("$i (buff): ", (@allocated TCI4Keldysh.eval_buff!(gev.GFevs[i], ret_buff, retarded_buff, idx_int, 50,53,47)) / 1.e6)
    end
    println("---------------")
    println("HierarchicalTucker: ")
    for i in 1:1
        println("$i: ", (@allocated gev.GFevs[i].Gps[1,1](50,53,47)) / 1.e6)
    end
end

function benchmark_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"))
    R = 8
    # b = @benchmark $gev(rand(1:2^$R, 3)...)
    b = @benchmark $gev(127, 128, 126)
    display(b)
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
end


function profile_allocations_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"))
    # trigger compilation
    x = gev(1, 1, 1);
    # Profile.Allocs.clear()
    # Profile.Allocs.@profile begin
    @profview gev(101, 100, 99)
    # PProf.Allocs.pprof()
end

function profile_gev()
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gev2.serialized"))
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
    @time gev = deserialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"))
    println("==== ΓcoreEvaluator_KF")
    x = gev(1,1,1)
    x = TCI4Keldysh.eval_buff!(gev,1,1,1)
    # print(@report_opt gev(50,53,47))
    print(@code_warntype TCI4Keldysh.eval_buff!(gev,50,53,47))
    println("\n\n==== MultipoleKFCEvaluator")
    # print(@report_opt gev.GFevs[1](50,53,47))
    print(@code_warntype gev.GFevs[1](50,53,47))
    println("\n\n==== HierarchicalTucker")
    print(@code_warntype gev.GFevs[1].Gps[1,1](50,53,47))
end

function gen_ΓcoreEvaluator_KF()

    basepath = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    iK = 6
    R = 8
    ommax = 0.3
    channel = "p"
    flavor = 1
    ωs_ext = ntuple(_->TCI4Keldysh.KF_grid_bos(ommax, R), 3)
    broadening_kwargs = TCI4Keldysh.read_all_broadening_params(basepath; channel=channel)
    broadening_kwargs[:estep] = 10
    gev =  TCI4Keldysh.ΓcoreEvaluator_KF(
        PSFpath,
        iK,
        ωs_ext,
        TCI4Keldysh.MultipoleKFCEvaluator{3},
        ;
        channel=channel,
        flavor_idx=flavor,
        KEV_kwargs=Dict(:cutoff => 1.e-5, :nlevel => 4),
        useFDR=false,
        broadening_kwargs...)
    serialize(joinpath(TCI4Keldysh.pdatadir(), "scripts", "gevR8.serialized"), gev)
end