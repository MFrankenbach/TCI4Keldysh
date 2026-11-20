# Compress local density and magnetic QMC vertices (F_d, F_m)
# onto an interleaved quantics Matsubara grid and store the tensor trains
# in the same JLD2 layout as matsubarafull.
#
#   julia --project scripts/compress_QMC_vertices.jl --R 7 --tolerance 1e-4
#
# Optional: --pad zero|asymptotic (default asymptotic), --k 3,
#           --infile PATH, --outdir PATH

using TCI4Keldysh
using HDF5
using JLD2
using QuanticsTCI
import TensorCrossInterpolation as TCI

const T_QMC = 0.25
const QMC_NBOS = 61
const QMC_NFER = 120

# Asymptotic pad is +U (F_d) and -U (F_m). The eight corners of the finite
# QMC box are not those asymptotes, so they are not checked.
const QMC_CHANNELS = (
    (dataset="F_d", sign=+1, outname="QMC_F_d_beta4_U4"),
    (dataset="F_m", sign=-1, outname="QMC_F_m_beta4_U4"),
)

const DEFAULT_INFILE = "data/QMC_vertices/beta4_U4/vertices_n1_u4_b4.hdf5"
const DEFAULT_OUTDIR = "data/QMC_vertices/beta4_U4"

"""
Matsubara index window of a QMC axis with `n` points.
Bosonic axes are odd and symmetric (`n = -30:30` for 61 points).
Fermionic axes are even (`n = -60:59` for 120 points).
"""
function qmc_frequency_indices(n::Int, fermi::Bool)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    if fermi
        iseven(n) || throw(ArgumentError("fermionic length must be even, got $n"))
        return (-n ÷ 2):(n ÷ 2 - 1)
    else
        isodd(n) || throw(ArgumentError("bosonic length must be odd, got $n"))
        return ((1 - n) ÷ 2):((n - 1) ÷ 2)
    end
end

"""
Integer frequency indices stored on one quantics axis with `R` bits.
Both parities use `n = -2^(R-1) : 2^(R-1)-1`. The bosonic frequency
`n = +2^(R-1)` is omitted so the axis length is `2^R`.
"""
function quantics_frequency_indices(R::Int, fermi::Bool)
    R >= 1 || throw(ArgumentError("R must be positive, got $R"))
    Nhalf = 2^(R - 1)
    n = (-Nhalf):(Nhalf - 1)
    return fermi ? n : n
end

"""
Index ranges of the Matsubara indices that lie on both `n_src` and `n_tgt`.
"""
function overlap_index_ranges(n_src::AbstractUnitRange{Int}, n_tgt::AbstractUnitRange{Int})
    nlo = max(first(n_src), first(n_tgt))
    nhi = min(last(n_src), last(n_tgt))
    nlo > nhi && throw(ArgumentError("No Matsubara overlap between $n_src and $n_tgt"))
    src = (nlo - first(n_src) + 1):(nhi - first(n_src) + 1)
    tgt = (nlo - first(n_tgt) + 1):(nhi - first(n_tgt) + 1)
    return src, tgt
end

function _info_truncated(axis::AbstractString, n_src::AbstractUnitRange{Int}, n_tgt::AbstractUnitRange{Int})
    lo = first(n_src):(first(n_tgt) - 1)
    hi = (last(n_tgt) + 1):last(n_src)
    parts = String[]
    !isempty(lo) && push!(parts, string(lo))
    !isempty(hi) && push!(parts, string(hi))
    if !isempty(parts)
        @info "Truncating $axis Matsubara indices outside the quantics window: $(join(parts, " and "))"
    end
    return nothing
end

"""
Copy `V` (bosonic × fermionic × fermionic, already reversed on axis 1)
into a `(2^R, 2^R, 2^R)` array. Sites with no QMC sample stay at `pad`.
"""
function embed_qmc_vertex(V::AbstractArray{<:Complex,3}, R::Int, pad::ComplexF64)
    @assert size(V) == (QMC_NBOS, QMC_NFER, QMC_NFER) "QMC vertex size $(size(V)) is not ($QMC_NBOS, $QMC_NFER, $QMC_NFER)"
    n_b_src = qmc_frequency_indices(QMC_NBOS, false)
    n_f_src = qmc_frequency_indices(QMC_NFER, true)
    n_b_tgt = quantics_frequency_indices(R, false)
    n_f_tgt = quantics_frequency_indices(R, true)
    _info_truncated("bosonic", n_b_src, n_b_tgt)
    _info_truncated("fermionic", n_f_src, n_f_tgt)

    N = 2^R
    out = fill(pad, N, N, N)
    sb, tb = overlap_index_ranges(n_b_src, n_b_tgt)
    sf, tf = overlap_index_ranges(n_f_src, n_f_tgt)
    @views out[tb, tf, tf] .= V[sb, sf, sf]
    return out
end

"""
Filled `k × k × k` block of 1-based quantics indices around the frequency origin.
Index `2^(R-1)+1` is bosonic `n = 0` and fermionic `n = 0` (`ω = +πT`).
"""
function origin_pivots(R::Int, k::Int)
    (R >= 1 && k >= 1) || throw(ArgumentError("Need R >= 1 and k >= 1, got R=$R, k=$k"))
    c = 2^(R - 1) + 1
    lo = c - k ÷ 2
    hi = lo + k - 1
    if !(1 <= lo && hi <= 2^R)
        throw(ArgumentError("pivot window $lo:$hi outside 1:$(2^R) for R=$R, k=$k"))
    end
    idxs = lo:hi
    pivots = Vector{Vector{Int}}()
    for t in Iterators.product(ntuple(_ -> idxs, 3)...)
        push!(pivots, collect(t))
    end
    return pivots
end

"""
`grid_origin` and `grid_step` of the full Matsubara grid used by matsubarafull,
including the bosonic point `n = +2^(R-1)` that the tensor train does not sample.
"""
function qmc_grid_metadata(R::Int)
    grid = TCI4Keldysh.MF_npoint_grid(T_QMC, 2^(R - 1), 3)
    return (TCI4Keldysh.grid_origin(grid), TCI4Keldysh.grid_step(grid))
end

"""
Reverse the bosonic axis so that index 1 is the most negative Matsubara frequency.
Corners of `F_d` / `F_m` are not the bare interaction and are not checked.
"""
function prepare_qmc_vertex(V::AbstractArray{<:Complex,3})
    @assert size(V) == (QMC_NBOS, QMC_NFER, QMC_NFER) "QMC vertex size $(size(V)) is not ($QMC_NBOS, $QMC_NFER, $QMC_NFER)"
    @info "REVERSING AXIS 1 (bosonic) of the input before compression: reverse(V; dims=1)."
    return reverse(V; dims=1)
end

function qmc_pad_value(pad::Symbol, sign::Integer, U::Real)
    if pad == :zero
        return ComplexF64(0)
    elseif pad == :asymptotic
        return ComplexF64(sign * U)
    else
        error("pad must be :zero or :asymptotic, got $pad")
    end
end

function hdf5_scalar(file::AbstractString, name::AbstractString)
    raw = h5read(file, name)
    val = raw isa Number ? raw : only(vec(raw))
    return Float64(val)
end

"""
Cross-interpolate `Vpad` and write `folder/outname_R=\$(R)_qtt.jld2`
with the six keys stored by `save_qtt_function`.
"""
function compress_qmc_vertex(
    Vpad::AbstractArray{<:Complex,3},
    R::Int,
    tolerance::Float64,
    outname::AbstractString,
    folder::AbstractString;
    k::Int=3,
)
    N = 2^R
    @assert size(Vpad) == (N, N, N) "Padded vertex size $(size(Vpad)) is not ($N, $N, $N)"
    pivots = origin_pivots(R, k)
    qtt, _, _ = quanticscrossinterpolate(
        Array{ComplexF64}(Vpad), pivots;
        unfoldingscheme=:interleaved,
        tolerance=tolerance,
        nrandominitpivot=0,
    )

    Nhalf = 2^(R - 1)
    gridmin = max(1, Nhalf - 2^5)
    gridmax = min(N, Nhalf + 2^5)
    grid1D = gridmin:2:gridmax
    checkgrid = collect(Iterators.product(ntuple(_ -> grid1D, 3)...))
    f = (i, j, l) -> Vpad[i, j, l]
    maxerr = TCI4Keldysh.check_interpolation(qtt, f, checkgrid)
    println(" Maximum interpolation error: $maxerr (tol=$tolerance)")
    println(" Bonddims: $(TCI.linkdims(qtt.tci))")
    println(" Rank: $(TCI4Keldysh.rank(qtt))")

    grid_origin, grid_step = qmc_grid_metadata(R)
    mkpath(folder)
    fname = joinpath(folder, outname * "_R=$(R)_qtt.jld2")
    JLD2.save(
        fname,
        Dict(
            "qtt" => qtt.tci,
            "grid_origin" => grid_origin,
            "grid_step" => grid_step,
            "varnames" => TCI4Keldysh.default_frequency_varnames(3),
            "nbits" => R,
            "unfoldingscheme" => qtt.grid.unfoldingscheme,
        ),
    )
    println(" Stored tensor train under $fname.\n")
    return fname
end

function parse_cli(args::Vector{String}=ARGS)
    R = nothing
    tolerance = nothing
    pad = :asymptotic
    k = 3
    infile = DEFAULT_INFILE
    outdir = DEFAULT_OUTDIR
    i = 1
    while i <= length(args)
        a = args[i]
        if i == length(args) && startswith(a, "--")
            error("Missing value for $a")
        end
        if a == "--R"
            R = parse(Int, args[i + 1]); i += 2
        elseif a == "--tolerance"
            tolerance = parse(Float64, args[i + 1]); i += 2
        elseif a == "--pad"
            pad = Symbol(args[i + 1]); i += 2
        elseif a == "--k"
            k = parse(Int, args[i + 1]); i += 2
        elseif a == "--infile"
            infile = args[i + 1]; i += 2
        elseif a == "--outdir"
            outdir = args[i + 1]; i += 2
        else
            error("Unknown argument $a")
        end
    end
    R === nothing && error("--R is required")
    tolerance === nothing && error("--tolerance is required")
    pad in (:zero, :asymptotic) || error("--pad must be zero or asymptotic, got $pad")
    return (; R, tolerance, pad, k, infile, outdir)
end

function julia_main(args::Vector{String}=ARGS)
    opts = parse_cli(args)
    U = hdf5_scalar(opts.infile, "U")
    beta = hdf5_scalar(opts.infile, "beta")
    @assert U == 4.0 "U=$U, expected 4"
    @assert beta == 4.0 "beta=$beta, expected 4"
    @assert beta == 1 / T_QMC

    for channel in QMC_CHANNELS
        V = Array{ComplexF64}(h5read(opts.infile, channel.dataset))
        V = prepare_qmc_vertex(V)
        pad = qmc_pad_value(opts.pad, channel.sign, U)
        Vpad = embed_qmc_vertex(V, opts.R, pad)
        outname = "$(channel.outname)_pad=$(opts.pad)"
        compress_qmc_vertex(Vpad, opts.R, opts.tolerance, outname, opts.outdir; k=opts.k)
    end
    return nothing
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(@__FILE__)
    julia_main()
end
