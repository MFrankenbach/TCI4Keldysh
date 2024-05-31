using QuanticsTCI
using TCI4Keldysh
using Plots
using MAT
using JSON
using Combinatorics
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

#=
TCI-compress 1,2 and 3-pt contributions to Green's functions obtained from NRG partial spectral functions (PSF)
=#

"""
Carries information on a Greens function represented as a tensor train (bonddims etc)
"""
struct GFTTdata
    ops::Vector{String}
    sitedims::Vector{Int}
    bonddims::Vector{Int}
end

# ========== utilities
function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end

function logJSON(gfdata, filename::String)
    fullname = filename*".json"
    open(joinpath("tci_data", fullname), "w") do file
        JSON.print(file, gfdata)
    end
    printstyled("File $filename.json written!\n", color=:green)
end

function readJSON(filename::String, folder::String="tci_data")
    path = joinpath(folder, filename)
    data = open(path) do file
        JSON.parse(file)
    end 
    return data
end

function extract_u_dirname(dirname::String) :: Float64
    eqpos = findfirst(i -> i=='=', dirname)
    return parse(Float64, dirname[eqpos+1:end])
end
# ========== utilities END

# ========== frequency conventions
begin
    
    ωconvMat_t = [
        0 -1  0;
        1  1  0;
        -1  0 -1;
        0  0  1;
    ]
    #ωconvMat_p = [   # NRG convention
    #    0 -1  0;
    #    -1  0 -1;
    #    1  1  0;
    #    0  0  1;
    #]
    ωconvMat_p = [    # MBEsolver convention
        0 -1  0;
        1  0 -1;
    -1  1  0;
        0  0  1;
    ]
    ωconvMat_a = [
        0 -1  0;
        0  0  1;
        -1  0 -1;
        1  1  0;
    ]

    ### deduce frequency conventions for 2p and 3p vertex contributions:

    # K1t = ["Q12", "Q34"]
    # K1p = ["Q13", "Q24"]
    # K1a = ["Q14", "Q23"])
    ωconvMat_K1t = reshape([
        sum(view(ωconvMat_t, [1,2], 1), dims=1);
        sum(view(ωconvMat_t, [3,4], 1), dims=1);
    ], (2,1))
    ωconvMat_K1p = reshape([
        sum(view(ωconvMat_p, [1,3], 1), dims=1);
        sum(view(ωconvMat_p, [2,4], 1), dims=1);
    ], (2,1))
    ωconvMat_K1a =  reshape([
        sum(view(ωconvMat_a, [1,4], 1), dims=1);
        sum(view(ωconvMat_a, [2,3], 1), dims=1);
    ], (2,1))

    # K2t = ("Q34", "1", "1dag")
    # K2p = ("Q24", "1", "3")
    # K2a = ("Q23", "1", "3dag")

    ωconvMat_K2t = [
        sum(view(ωconvMat_t, [3,4], [1,2]), dims=1);
        view(ωconvMat_t, [1,2], [1,2])
    ]

    ωconvMat_K2p = [
        sum(view(ωconvMat_p, [2,4], [1,2]), dims=1);
        view(ωconvMat_p, [1,3], [1,2])
    ]


    ωconvMat_K2a = [
        sum(view(ωconvMat_a, [2,3], [1,2]), dims=1);
        view(ωconvMat_a, [1,4], [1,2])
    ]


    # K2′t = ("Q12", "3", "3dag")
    # K2′p = ("Q13", "1dag", "3dag")
    # K2′a = ("Q14", "3", "1dag")

    ωconvMat_K2′t = [
        1  0;
        0  1;
        -1 -1;
    ]
    ωconvMat_K2′t = [
        sum(view(ωconvMat_t, [1,2], [1,3]), dims=1);
        view(ωconvMat_t, [3,4], [1,3])
    ]

    ωconvMat_K2′p = [
        sum(view(ωconvMat_p, [1,3], [1,3]), dims=1);
        view(ωconvMat_p, [2,4], [1,3])
    ]


    ωconvMat_K2′a = [
        sum(view(ωconvMat_a, [1,4], [1,3]), dims=1);
        view(ωconvMat_a, [2,3], [1,3])
    ]
end
# ========== frequency conventions END

# ========== compression functions
""" 
compress a 1,2, and/or 3-pt object.
"""
function compress_npoint(PSFpath::String; max_npoint=1, R=8,
    op_dict::Dict{Int, Vector{String}}=Dict(1=>["Q12","Q34"], 2=>["Q12", "F3", "F3dag"], 3=>["F1","F1dag", "F3", "F3dag"])
    , nested_ωdisc::Bool=false) :: Tuple{Vector{Float64}, Vector{Float64}, Dict{Int, Array}}

    # ========== parameters
    ## Keldysh paper:    u=0.5 OR u=1.0

    D = 1.0
    u = 0.5
    #u = 1.0
    U = 0.05
    Δ = U / (π * u)
    T = 0.01*U

    Rpos = R-1
    Nωcont_pos = 2^Rpos
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    # ωconts=ntuple(i->ωcont, ndims(Adisc))
    
    # define grids
    # cannot exclude end point for bosonic part, requires equidistant symmetric grid
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    # ωconv = [
    #      1  0;
    #      -1  -1
    
    #      ]
    # ========== parameters END

    ret = Dict{Int, Array}()

    # obtain evaluators, then evaluate on grid (MF)

    if haskey(op_dict, 1)
        K1ts    = [TCI4Keldysh.FullCorrelator_MF(PSFpath, op_dict[1]; T, flavor_idx=i, ωs_ext=(ωbos,), ωconvMat=ωconvMat_K1t, name="SIAM 2pG", nested_ωdisc=nested_ωdisc) for i in 1:2]
        K1ts_values = Vector{Array{ComplexF64, 1}}(undef, 2)
        for spin in 1:2
            K1ts_values[spin] = TCI4Keldysh.precompute_all_values(K1ts[spin])
            ret[1] = K1ts_values
        end
        @show size.(K1ts_values)
    end


    # two-point
    if max_npoint >= 2 && haskey(op_dict, 2)
        Gs      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, op_dict[2]; T, flavor_idx=i, ωs_ext=(ωbos,ωfer), ωconvMat=ωconvMat_K2′t, name="SIAM 3pG", is_compactAdisc=false, nested_ωdisc=nested_ωdisc) for i in 1:2]
        Gs_values = [TCI4Keldysh.precompute_all_values(Gs[spin]) for spin in 1:2]
        println("Size values:")
        @show size.(Gs_values)
        println("Size bonsonic/fermionic grid:")
        @show (size(ωbos), size(ωfer))
        ret[2] = Gs_values
    end

    # three-point
    if max_npoint >= 3 && haskey(op_dict, 3)
        G3D     = TCI4Keldysh.FullCorrelator_MF(joinpath(PSFpath, "4pt"), op_dict[3]; T, flavor_idx=1, ωs_ext=(ωbos,ωfer,ωfer), ωconvMat=ωconvMat_t, name="SIAM 4pG", is_compactAdisc=false, nested_ωdisc=nested_ωdisc)
        G3D_values = [TCI4Keldysh.precompute_all_values(G3D)]
        println("Size values:")
        @show size.(G3D_values)
        println("Size bonsonic/fermionic grid:")
        @show (size(ωbos), size(ωfer))
        ret[3] = G3D_values
    end

    return (ωbos, ωfer, ret)
end

function compress_greens_array(values::Array{ComplexF64,D}, ombos::Vector{Float64}, omfer::Vector{Float64}; tolerance=1e-7) where{D}
    @assert isinteger(log2(length(omfer))) 
    @assert isinteger(log2(length(ombos) - 1)) 
    # bosonic first
    @assert size(values, 1) == length(ombos)
    @assert all([size(values, i) for i in 2:D] .== length(omfer))

    # check for zero arrays
    if all(abs.(values) .<= tolerance)
        printstyled("  Data is smaller than tolerance $tolerance\n"; color=:red)
        return nothing
    end

    # build grid

    R = Int(log2(length(omfer)))
    grid = QG.InherentDiscreteGrid{D}(R, ntuple(i -> 1, D))

    function val_func(ids::Vararg{Int, D})
        return values[ids...]
    end

    GF_qtt, _, _ = quanticscrossinterpolate(ComplexF64, val_func, grid; tolerance=tolerance)

    @show length(GF_qtt.tci)
    @show TCI.linkdims(GF_qtt.tci)
    @show TCI4Keldysh.worstcase_bonddim(GF_qtt.tci.localdims)

    return GF_qtt
end

function compress_all_1pt(R::Int, tolerance::Float64; dir="data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    op_lists = [["Q12","Q34"],["Q13","Q24"],["Q14","Q23"]]
    op_lists = vcat(op_lists, reverse.(op_lists))

    gf_vec = Vector{GFTTdata}()

    nested_ωdisc = !(dir == "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")

    for opl in op_lists
        printstyled("---- Compress $(opl[1]*opl[2])\n"; color=:blue)
        op_dict = Dict(1=>opl)
        ombos, omfer, data_dict = compress_npoint(
            dir; max_npoint=1, op_dict=op_dict, R=R, nested_ωdisc=nested_ωdisc
        )
        gf_qtt = compress_greens_array(first(data_dict[1]), ombos, omfer; tolerance=tolerance)
        if !isnothing(gf_qtt)
            push!(gf_vec, GFTTdata(opl, gf_qtt.tci.localdims, TCI.linkdims(gf_qtt.tci)))
        end
        print("\n")
    end

    # store to json
    data_dir = splitpath(dir)[2]
    logJSON(gf_vec, joinpath(data_dir, "all_1pt_GF_R=$(R)_logtol=$(Int(round(-log10(tolerance))))"))
end

function compress_all_2pt(R::Int, tolerance::Float64; dir="data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    op_lists = []
    f_dict = Dict(1=>"F1",2=>"F1dag",3=>"F3",4=>"F3dag")
    for q in ["12","13","14","23","24","34"] 
        q1, q2 = divrem(parse(Int, q), 10)
        f1i, f2i = sort(setdiff([1,2,3,4], [q1,q2]))[1:2]
        # strings for operator names
        f1 = f_dict[f1i]
        f2 = f_dict[f2i]
        # add all permutations
        op_vec = ["Q"*q, f1, f2]
        for perm in [[1,2,3],[2,1,3],[1,3,2],[3,2,1],[3,1,2],[2,3,1]]
            push!(op_lists, op_vec[perm])
        end
    end
    # for opl in op_lists
    #     println(opl)
    # end
    # check uniqueness
    @assert length(unique(op_lists)) == length(op_lists)

    gf_vec = Vector{GFTTdata}()

    nested_ωdisc = !(dir == "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")

    for opl in op_lists
        printstyled("---- Compress $(reduce(*, opl))\n"; color=:blue)
        op_dict = Dict(2=>opl)
        ombos, omfer, data_dict = compress_npoint(
            dir; max_npoint=2, op_dict=op_dict, R=R, nested_ωdisc=nested_ωdisc
        )
        gf_qtt = compress_greens_array(first(data_dict[2]), ombos, omfer; tolerance=tolerance)
        if !isnothing(gf_qtt)
            push!(gf_vec, GFTTdata(opl, gf_qtt.tci.localdims, TCI.linkdims(gf_qtt.tci)))
        end
        print("\n")
    end

    # store to json
    data_dir = splitpath(dir)[2]
    logJSON(gf_vec, joinpath(data_dir, "all_2pt_GF_R=$(R)_logtol=$(Int(round(-log10(tolerance))))"))
end

function compress_all_3pt(R::Int, tolerance::Float64; dir="data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    names = ["F1","F1dag","F3","F3dag"]
    # generate all 24 permutations
    op_lists = []
    for i in 1:factorial(4)
        push!(op_lists, names[nthperm([1,2,3,4], i)])
    end
    @assert length(op_lists)==factorial(4)
    @assert length(unique(op_lists))==length(op_lists)

    gf_vec = Vector{GFTTdata}()

    nested_ωdisc = !(dir == "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")

    for opl in op_lists
        printstyled("---- Compress $(reduce(*, opl))\n"; color=:blue)
        op_dict = Dict(3=>opl)
        ombos, omfer, data_dict = compress_npoint(
            dir; max_npoint=3, op_dict=op_dict, R=R, nested_ωdisc=nested_ωdisc
        )
        gf_qtt = compress_greens_array(first(data_dict[3]), ombos, omfer; tolerance=tolerance)
        if !isnothing(gf_qtt)
            push!(gf_vec, GFTTdata(opl, gf_qtt.tci.localdims, TCI.linkdims(gf_qtt.tci)))
        end
        print("\n")
    end

    # store to json
    data_dir = splitpath(dir)[2]
    logJSON(gf_vec, joinpath(data_dir, "all_3pt_GF_R=$(R)_logtol=$(Int(round(-log10(tolerance))))"))
end

"""
Compress 2,3,4-pt arrays
"""
function compress_all_greens_array(op_dict::Dict{Int, Vector{String}})
    ombos, omfer, data_dict = compress_npoint(
            # load_ωdisc works
        "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"; max_npoint=3
            # load_ωdisc fails: does not recognize variables, yielding a 'plain' HDF5 file object
        # "data/SIAM_u=1.50/PSF_nz=4_conn_zavg/"; max_npoint=2
        # "data/SIAM_u=1.00/PSF_nz=4_conn_zavg/"; max_npoint=2
        ,op_dict=op_dict
    )
    # 2-pt
    if haskey(data_dict, 1)
        compress_greens_array(first(data_dict[1]), ombos, omfer)
        plot_1pt(first(data_dict[1]))
    end
    # 3-pt
    if haskey(data_dict, 2)
        compress_greens_array(first(data_dict[2]), ombos, omfer)
        plot_2pt(first(data_dict[2]))
    end
    # 4-pt
    if haskey(data_dict, 3)
        compress_greens_array(first(data_dict[3]), ombos, omfer)
    end
end

"""
2D heatmap for 2pt object
"""
function plot_2pt(values::Matrix{ComplexF64})
    func = abs
    heatmap(func.(values); color=:viridis)
    xlabel!("bosonic")
    ylabel!("fermionic")
    title!("$(nameof(func)) of 3pt Green's function")
    savefig("3pt_heatmap.png")
end

"""
1D plot of 1pt object
"""
function plot_1pt(values::Vector{ComplexF64})
    func = abs
    plot(collect(eachindex(values)), func.(values); marker=:cross, line=true, color=:blue)
    xlabel!("index")
    ylabel!("value")
    title!("$(nameof(func)) of 2pt Green's function")
    savefig("2pt_GF.png")
end

function try_matopen(filepath::String)
    println("  File $filepath:")
    matopen(filepath) do file
        varnames = nothing
        try
            varnames = keys(file)
        catch
            varnames = keys(file)
        end
        @show varnames
        try
            PSF = read(file, "PSF")
            @show typeof(PSF)
            @show keys(PSF)
            # info on frequency grid
            # @show PSF["odisc_info"]
            # the frequency grid
            @show keys(PSF["odisc_info"])
        catch
            println("PSF not found")
        end
    end
end

"""
Compress all available n-point functions for different Rs
"""
function compress_all_npoint_Rsweep(npt::Int, Rrange; tolerance=1e-6, dir="data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    for R in Rrange
        if npt==1
            compress_all_1pt(R, tolerance; dir=dir)
        elseif npt==2
            compress_all_2pt(R, tolerance; dir=dir)
        elseif npt==3
            compress_all_3pt(R, tolerance; dir=dir)
        else
            error("Invalid input npt=$npt")
        end
    end
end

"""
Compress all available n-point functions for different tolerances
"""
function compress_all_npoint_tolsweep(npt::Int, tolrange; R=8, dir="data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    for tolerance in tolrange
        if npt==1
            compress_all_1pt(R, tolerance; dir=dir)
        elseif npt==2
            compress_all_2pt(R, tolerance; dir=dir)
        elseif npt==3
            compress_all_3pt(R, tolerance; dir=dir)
        else
            error("Invalid input npt=$npt")
        end
    end
end
# ========== compression functions END

# ========== data analysis functions
function plot_ranks_vs_R(npt::Int, Rrange, tolerance::Float64, tci_dir::String="SIAM_u=0.50")
    # read in ranks
    worstcases = Int[]
    ranks = Vector{Int}[]
    for R in Rrange
        filename = joinpath(tci_dir, "all_$(npt)pt_GF_R=$(R)_logtol=$(Int(round(-log10(tolerance)))).json")
        # list of dicts
        data = readJSON(filename)
        worst = maximum(TCI4Keldysh.worstcase_bonddim(convert.(Int, data[begin]["sitedims"])))
        push!(worstcases, worst)
        ranks_act = Int[]
        for data_dict in data
            push!(ranks_act,  maximum(data_dict["bonddims"]))
        end
        push!(ranks, ranks_act)
    end

    # plot R against rank distributions
    xcoords = vcat([fill(Rrange[i], length(ranks[i])) for i in eachindex(Rrange)]...)
    ycoords = vcat(ranks...)
    for (i,R) in enumerate(Rrange)
        println("  R=$R: mean rank is $(sum(ranks[i])/length(ranks[i]))")
    end
    scatter(xcoords, ycoords; color=:blue, label="TT rank", yscale=:log10, marker=:cross, xtickfontsize=14, ytickfontsize=14, legendfontsize=12)
    scatter!(collect(Rrange), worstcases; color=:red, label="worst case", yscale=:log10)
    xlabel!("TT length", labelfontsize=14)
    ylabel!("TT rank", labelfontsize=14)
    u = extract_u_dirname(tci_dir)
    title!("TCI ranks $(npt+1)pt-functions, tol=1e$(Int(round(log10(tolerance)))), u=$u")
    outname = "ranks_vs_R_$(npt)pt_logtol=$(Int(round(-log10(tolerance))))_u=$(u).png"
    savefig(outname)
    println("-- Saved: " * outname)
end


function plot_ranks_vs_tol(npt::Int, tolrange, R::Int, tci_dir::String="SIAM_u=0.50")
    # read in ranks
    worstcases = Int[]
    ranks = Vector{Int}[]
    for tol in tolrange
        filename = joinpath(tci_dir, "all_$(npt)pt_GF_R=$(R)_logtol=$(Int(round(-log10(tol)))).json" )
        # list of dicts
        data = readJSON(filename)
        worst = maximum(TCI4Keldysh.worstcase_bonddim(convert.(Int, data[begin]["sitedims"])))
        push!(worstcases, worst)
        ranks_act = Int[]
        for data_dict in data
            push!(ranks_act,  maximum(data_dict["bonddims"]))
        end
        push!(ranks, ranks_act)
    end

    # plot R against rank distributions
    xcoords = vcat([fill(tolrange[i], length(ranks[i])) for i in eachindex(tolrange)]...)
    ycoords = vcat(ranks...)
    for (i,tol) in enumerate(tolrange)
        println("  tol=$tol: mean rank is $(sum(ranks[i])/length(ranks[i]))")
    end
    scatter(xcoords, ycoords; color=:blue, label="TT rank", yscale=:log10, xscale=:log10, marker=:cross, xtickfontsize=14, ytickfontsize=14, legendfontsize=12)
    scatter!(collect(tolrange), worstcases; color=:red, label="worst case", yscale=:log10)
    xlabel!("TCI tolerance", labelfontsize=14)
    ylabel!("TT rank", labelfontsize=14)
    u = extract_u_dirname(tci_dir)
    title!("TCI ranks $(npt+1)pt-functions, R=$R, u=$u")
    outname = "ranks_vs_tol_$(npt)pt_R=$(R)_u=$(u).png"
    savefig(outname)
end
# ========== data analysis functions END

dirs = ["data/SIAM_u=0.50/PSF_nz=2_conn_zavg/", "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/", "data/SIAM_u=1.50/PSF_nz=4_conn_zavg_new/"]
tci_dirs = ["SIAM_u=0.50", "SIAM_u=1.00", "SIAM_u=1.50"]

# compress_all_greens_array(Dict(1=>["Q12","Q34"], 2=>["Q12", "F3", "F3dag"], 3=>["F1","F1dag", "F3", "F3dag"]))
# compress_all_greens_array(Dict(1=>["Q14","Q23"], 2=>["F1", "F1dag", "Q34"], 3=>["F1","F3", "F1dag", "F3dag"]))

# try_matopen("/Users/M.Frankenbach/tci4keldysh/data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/PSF_((Q13,Q24)).mat")
# try_matopen("/Users/M.Frankenbach/tci4keldysh/data/SIAM_u=0.50/PSF_nz=2_conn_zavg/PSF_((Q13,Q24)).mat")

# # 1 and 2pt
# compress_all_npoint_Rsweep(1, 4:12; tolerance=1e-6, dir=dirs[2])
# compress_all_npoint_Rsweep(1, 4:12; tolerance=1e-6, dir=dirs[3])
# compress_all_npoint_tolsweep(1, 10.0 .^ (-8.0:-3.0); R=8, dir=dirs[2])
# compress_all_npoint_tolsweep(1, 10.0 .^ (-8.0:-3.0); R=8, dir=dirs[3])

# compress_all_npoint_Rsweep(2, 4:12; tolerance=1e-6, dir=dirs[2])
# compress_all_npoint_Rsweep(2, 4:12; tolerance=1e-6, dir=dirs[3])
# compress_all_npoint_tolsweep(2, 10.0 .^ (-8.0:-3.0); R=8, dir=dirs[2])
# compress_all_npoint_tolsweep(2, 10.0 .^ (-8.0:-3.0); R=8, dir=dirs[3])


# 3pt
# compress_all_npoint_Rsweep(3, 4:8; tolerance=1e-5, dir=dirs[2])
# compress_all_npoint_tolsweep(3, 10.0 .^ (-7.0:-3.0); R=8)

# # plot
# for d in tci_dirs
#     for np in 1:2
#         plot_ranks_vs_R(np, 4:12, 1e-6, d)
#         plot_ranks_vs_tol(np, 10.0 .^ (-8.0:-3.0), 8, d)
#     end
# end
plot_ranks_vs_tol(3, 10.0 .^ (-7.0:-3.0), 8, tci_dirs[1])
plot_ranks_vs_R(3, 4:8, 1e-5, tci_dirs[1])