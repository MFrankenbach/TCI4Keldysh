using MAT
using Random
using LinearAlgebra
using Combinatorics
#=
From actual PSFs, generate smaller PSFs for testing.
=#

oneflavor = [
    "PSF_((F1,F1dag)).mat"
    "PSF_((F1,Q1dag)).mat"
    "PSF_((F1dag,F1)).mat"
    "PSF_((F1dag,Q1)).mat"
    "PSF_((Q1,F1dag)).mat"
    "PSF_((Q1,Q1dag)).mat"
    "PSF_((Q12)).mat"
    "PSF_((Q14)).mat"
    "PSF_((Q1dag,F1)).mat"
    "PSF_((Q1dag,Q1)).mat"
    "PSF_((Q23)).mat"
    "PSF_((Q34)).mat"
]

"""
Determine slice such that the pruned PSF contains relevant spectral weight.
"""
function determine_slice(inputfile::AbstractString, margin=1)

    slices = []
    nom = 0
    mid = 0
    matopen(inputfile) do file
        try
            keys(file)
        catch
            keys(file)
        end
        odisc = read(file, "odisc")
        nom = length(odisc)
        mid = div(nom,2)+1
        _mirror(x::Int) = 2*mid - x
        # for two flavors
        for A in read(file, "Adisc")
            singleton = findall(size(A).==1)
            amax = argmax(abs.(A))
            amax = [amax[i] for i in 1:ndims(A) if !(i in singleton)]
            _s = [collect(max(amax[i]-margin, 1):min(amax[i]+margin, nom)) for i in eachindex(Tuple(amax)) ]
            _smirror = [collect(_mirror(last(ss)):_mirror(first(ss))) for ss in _s]
            slice = unique(sort(vcat(_s..., _smirror...)))
            push!(slices, slice)
        end
    end
    ret = unique(sort(vcat(slices...)))
    if isempty(ret)
        ret = [mid]
    end
    return ret
end

function extract_Ops(fname::AbstractString)
    m = match(r"PSF_\(\((.*?)\)\)\.mat", fname)    
    if m!==nothing
        return convert.(String, split(m.captures[1], ","))
    else
        @warn "No match found in fname"
        return nothing
    end
end

function extract_all_Ops(indir::AbstractString)
    all_Ops = []
    for infile in readdir(indir; join=false)
        if isdir(infile) || !(endswith(infile, ".mat"))
            continue
        else
            Ops = extract_Ops(infile)
            if !(reverse(Ops) in all_Ops)
                push!(all_Ops, Ops)
            end
        end
    end
    return all_Ops
end

function extract_all_OpSets(indir::AbstractString)
    all_Ops = []
    for infile in readdir(indir; join=false)
        if isdir(infile) || !(endswith(infile, ".mat"))
            continue
        else
            Ops = extract_Ops(infile)
            if !any([Ops[p] in all_Ops for p in permutations(collect(1:length(Ops)))])
                push!(all_Ops, Ops)
            end
        end
    end
    return all_Ops
end

"""
sweep through input directory and prune PSFs, write to output directory
ensures PSFs that belong to the same operator combination have the same odisc...
"""
function prune_PSFs(indir, outdir)
    all_Ops = extract_all_OpSets(indir)
    for Ops in all_Ops
        infile1 = joinpath(indir, TCI4Keldysh.parse_Ops_to_filename(Ops))
        slice = determine_slice(infile1)
        D = length(Ops)
        for p in permutations(collect(1:D))
            infile = joinpath(indir, TCI4Keldysh.parse_Ops_to_filename(Ops[p]))
            TCI4Keldysh.prune_PSF(infile, slice, outdir)
        end
    end
end

"""
sweep through input directory and create PSFs in output directory
need to make sure PSFs that belong to the same operator string have the same odisc...
"""
function gen_dummyPSF(indir, outdir, odisc::Vector{Float64})
    all_Ops = extract_all_Ops(indir)
    # generate PSFs
    for Ops in all_Ops
        outname = TCI4Keldysh.parse_Ops_to_filename(Ops)
        nflavor = (outname in oneflavor) ? 1 : 2
        Adisc = TCI4Keldysh.artificial_PSF(joinpath(outdir, outname), odisc, length(Ops)-1; scale=0.1, nflavor=nflavor)
        # omit 0pt functions
        println("WRITE to: $outname")
        if length(Ops)>1
            outname = TCI4Keldysh.parse_Ops_to_filename(reverse(Ops))
            println("    COPY to: $outname")
            TCI4Keldysh.write_PSF(joinpath(outdir, outname), odisc, Adisc)
        end
    end
end

function check_sizes(dir)
    odisc_szs = Int[]
    for file in readdir(dir, join=true)
        if isdir(file) || !(endswith(file, ".mat"))
            continue
        else
            matopen(file) do f
                try
                    keys(f)
                catch
                    keys(f)
                end
                Adisc = read(f, "Adisc")
                odisc = read(f, "odisc")
                push!(odisc_szs, length(odisc))
                for A in Adisc
                    sz = filter(i -> i>1, size(A))
                    n0 = length(sz)
                    if !all(sz .== (length(odisc) * ones(Int, n0)))
                        @warn "In file $(Base.splitdir(file)[2]): Adisc $sz vs odisc $(length(odisc))"
                    end
                end
            end
        end
    end
    return odisc_szs
end

indir = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
outdir = joinpath(TCI4Keldysh.datadir(), "unittest_PSF/PSF/4pt")
# indir = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg")
# outdir = joinpath(TCI4Keldysh.datadir(), "unittest_PSF/PSF")