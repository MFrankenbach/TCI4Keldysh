#=
Test accuracy of various evaluators
=#

using TCI4Keldysh
using LinearAlgebra
using MAT
using HDF5
using QuanticsTCI
using Serialization
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

function check_gev_accuracy(R::Int, N::Int=10^5)
    println("==== Load data...")
    Vpath = joinpath(TCI4Keldysh.pdatadir(), "cluster_output/V_KF_conventional_ommax0.65_bosgrid/V_KF_p_R=8.h5")
    core = h5read(Vpath, "V_KF")
    # grid spacing for ωmax=1.0
    Rd = 8
    dw = 2/2^R
    cen = 2^(R-1)+1
    dw_d = 2/2^Rd
    cen_d = 2^(Rd-1)+1
    dwrat = 2^(R-Rd)
    gpath = joinpath(TCI4Keldysh.pdatadir(), "cluster_output", "gencoreR$(R)", "gevR$(R)p.serialized")
    gev = deserialize(gpath)
    println("==== Evaluate...")
    shift = div(2^R, 2^Rd)
    maxrefs = [maximum(abs.(core[:,:,:,TCI4Keldysh.KF_idx(iK,3)...])) for iK in 1:16]
    maxrefs[end] = 1.0
    maxrefs = reshape(maxrefs, 2,2,2,2)
    pts = Array{Int,2}(undef, 3,N)
    err = Array{Float64,5}(undef, 2,2,2,2,N)
    nerr = Array{Float64,5}(undef, 2,2,2,2,N)
    Threads.@threads for n in 1:N
        idx = rand(1:2^Rd, 3)
        idx_gev = ntuple(i -> round(Int, dwrat*(idx[i]-cen_d) + cen), 3)
        d = abs.(core[idx..., :,:,:,:] .- TCI4Keldysh.evaluate_all_iK(gev, idx_gev...))
        err[:,:,:,:,n] .= d
        nerr[:,:,:,:,n] .= d ./ maxrefs
        pts[:,n] .= idx
    end
    h5write("gev_accuracyR$(R)_alliK_N$(N).h5", "err", err)
    h5write("gev_accuracyR$(R)_alliK_N$(N).h5", "nerr", nerr)
    h5write("gev_accuracyR$(R)_alliK_N$(N).h5", "pts", pts)
end

"""
Test evaluation accuracy of Keldysh vertex
"""
function V_KF_eval_accuracy(
    refpath::String,
    R::Int,
    iK::Int;
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg"),
    KEV::Type=TCI4Keldysh.MultipoleKFCEvaluator,
    coreEvaluator_kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}(:cutoff=>1.e-6, :nlevel=>4)
    )

    outfile = joinpath(refpath, "errs_R=$(R)_iK=$iK.h5")
    if isfile(outfile)
        @warn "File $outfile already exists! Aborting."
        return 1
    end

    # load settings
    refjson = only(
        filter(f -> endswith(f, ".json"), readdir(refpath))
    )
    refsettings = TCI4Keldysh.readJSON(refjson, refpath)
    ωmax = refsettings["ommax"]
    broadening_kwargs_ = refsettings["broadening_kwargs"]
    broadening_kwargs = Dict{Symbol,Any}()
    for (key,val) in pairs(broadening_kwargs_)
        broadening_kwargs[Symbol(key)] = val
    end
    γ = refsettings["gamma"]
    sigmak = Vector{Float64}(refsettings["sigmak"])
    channel = refsettings["channel"]
    flavor_idx = refsettings["flavor_idx"]
    if !(R in refsettings["Rs"])
        error("Requested grid size is not available")
    end

    T = TCI4Keldysh.dir_to_T(PSFpath)

    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    # make frequency grid
    D = size(ωconvMat, 2)
    @assert D==3
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)

    # all 16 4-point correlators
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    is_incoming = (false, true, false, true)

    # create correlator objects
    Ncorrs = length(letter_combinations)
    GFs = Vector{TCI4Keldysh.FullCorrelator_KF{D}}(undef, Ncorrs)
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for l in 1:Ncorrs
        letts = letter_combinations[l]
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(TCI4Keldysh.parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, sigmak=sigmak, γ=γ, broadening_kwargs...)
    end

    # evaluate self-energy
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"
    ωstep = abs(ωs_ext[1][1] - ωs_ext[1][2])
    Σω_grid = TCI4Keldysh.KF_grid_fer(2*ωmax, R+1)
    (Σ_L,Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ, broadening_kwargs...)

    # frequency grid offset for self-energy
    ΣωconvMat = incoming_trafo * ωconvMat
    corner_low = [first(ωs_ext[i]) for i in 1:D]
    corner_idx = ones(Int, D)
    corner_image = ΣωconvMat * corner_low
    idx_image = ΣωconvMat * corner_idx
    desired_idx = [findfirst(w -> abs(w-corner_image[i])<ωstep*0.1, Σω_grid) for i in eachindex(corner_image)]
    ωconvOff = desired_idx .- idx_image

    sev = TCI4Keldysh.SigmaEvaluator_KF(Σ_R, Σ_L, ΣωconvMat, ωconvOff)

    gev = TCI4Keldysh.ΓcoreEvaluator_KF(GFs, iK, sev, KEV; coreEvaluator_kwargs...)

    # load reference
    refdatafile = joinpath(refpath, "V_KF_$(channel)_R=$(R).h5")
    refdata = h5read(refdatafile, "V_KF")[:,:,:,TCI4Keldysh.KF_idx(iK,3)...]
    # SETUP DONE

    # check
    println("Evaluate...")
    t = @elapsed begin
        N = 5 * 10^5
        errs = Vector{Float64}(undef, N)
        vals = Vector{ComplexF64}(undef, N)
        idx_range = Base.OneTo.(size(refdata))
        Threads.@threads for n in 1:N
            idx = ntuple(i -> rand(idx_range[i]), 3)
            val = gev(idx...)
            errs[n] = abs(val - refdata[idx...])
            vals[n] = val
        end
    end
    println("Evaluation done in $t seconds ($(Threads.nthreads()) threads)")

    h5write(outfile, "vals", vals)
    h5write(outfile, "errs", errs)
    h5write(outfile, "Neval", N)
    if haskey(coreEvaluator_kwargs, :nlevel)
        h5write(outfile, "nlevel", coreEvaluator_kwargs[:nlevel])
    end
    if haskey(coreEvaluator_kwargs, :cutoff)
        h5write(outfile, "cutoff", coreEvaluator_kwargs[:cutoff])
    end
end

function check_accuracy(refpath::AbstractString, iK::Int, R::Int=7)
    outfile = joinpath(refpath, "errs_R=$(R)_iK=$iK.h5")
    errs = h5read(outfile, "errs")
    ref = h5read(joinpath(refpath, "V_KF_p_R=$(R).h5"), "V_KF")[:,:,:, TCI4Keldysh.KF_idx(iK,3)...]
    maxref = maximum(abs.(ref))
    @show maximum(abs.(errs))
    @show maximum(abs.(errs)) / maxref
    try
        cutoff = h5read(outfile, "cutoff")
        @info "Cutoff was $cutoff"
    catch
        @info "No cutoff value found"
    end
    ref = nothing
    errs = nothing
end

function check_accuracy_ΓcoreEvaluator_MF(reffile::AbstractString; flavor_idx::Int=1, key="V_KF")
    Vref = if endswith(reffile, ".mat")
        open(reffile) do f
            Vref_ = f["CFdat"][flavor_idx]
        end
        reverse(permutedims(Vref_, (3,1,2)))
    elseif endswith(reffile, ".h5")
        h5read(f, key)
    else
        error("File format not recognized")
    end

    Nbos = size(Vref,1)

    # create Evaluator
    error("NYI")
end

function test_ΓcoreEvaluator(;R::Int, tolerance=1.e-8)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    channel = "p"
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)

    # numerically exact evaluator
    gev_ref = ΓcoreEvaluator_MF(PSFpath, R; ωconvMat=ωconvMat, T=T, flavor_idx=1, cutoff=1.e-20)

    cutoff = 0.01 * tolerance
    gev = ΓcoreEvaluator_MF(PSFpath, R; ωconvMat=ωconvMat, T=T, flavor_idx=1, cutoff=cutoff)

    D = 3
    N = 2^R
    Neval = 2^5
    eval_step = max(div(N, Neval), 1)
    Ncenter_h = 4
    gridslice = vcat(collect(1:eval_step:N), collect(div(N,2)-Ncenter_h+1 : div(N,2)+Ncenter_h))
    Neval += 2*Ncenter_h
    @assert length(gridslice)==Neval

    printstyled("Memory usage [GB] of ΓcoreEvaluator_MF (ref): $(Base.summarysize(gev_ref) / (1024^3))\n"; color=:blue)
    printstyled("Memory usage [GB] of ΓcoreEvaluator_MF (approx): $(Base.summarysize(gev) / (1024^3))\n"; color=:blue)

    # reference values
    println("Computing reference...")
    flush(stdout)
    refval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        # if rand(Float64)<=1.e-2
        #     @time gev_ref(w...)
        # end
        refval[idx...] = gev_ref(w...)
    end

    # approximate values
    println("Computing test values...")
    flush(stdout)
    testval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        testval[idx...] = gev(w...)
    end

    maxref = maximum(abs.(refval))
    diff = abs.(testval - refval) ./ maxref
    printstyled("Matsubara GammaCoreEvaluator error for tol=$tolerance\n"; color=:blue)
    @show maxref
    @show maximum(diff)
    open("MFgamcoreEvaluator.txt", "a") do f
        write(f, "R=$R, tol=$tolerance, maxref=$maxref, MAXERR=$(maximum(diff)) ($(length(testval)) evals)\n")
    end
end

GC.gc()
check_gev_accuracy(8,2*10^6)
GC.gc(true)
check_gev_accuracy(9,2*10^6)
GC.gc(true)
check_gev_accuracy(10,2*10^6)
GC.gc(true)
check_gev_accuracy(11,2*10^6)
GC.gc(true)
check_gev_accuracy(12,2*10^6)
