using Plots

"""
Represents a partial correlator with kernels and Adisc stored in tensor train format.
NOT NEEDED at the moment
"""
# mutable struct PartialCorrelator_TCI{D} <: AbstractTuckerDecomp{D}
#     T           ::Float64
#     formalism:: String                          # "MF" or "KF"
#     Adisc # tensor train
#     Adisc_anoβ::Array{ComplexF64,D}             # anomalous part of the discrete PSF data (only used for computing the anomalous contribution ∝ β)
#     #ωdiscs  ::  Vector{Vector{Float64}}         # discrete frequencies for 
#     #Kernels ::  Vector{Matrix{ComplexF64}}      # regular kernels
#     ωs_ext  ::  NTuple{D,Vector{Float64}}    # external complex frequencies
#     #ωs_int  ::  NTuple{D,Vector{ComplexF64}}    # internal complex frequencies
#     ωconvMat::  SMatrix{D,D,Int}                # matrix encoding frequency conversion in terms of indices ~ i_ωs_int = ωconvMat * i_ωs_ext + ωconvOff
#     ωconvOff::  SVector{D,Int}                  # Offset encoding frequency conversion in terms of (one-based!) indices
#     isFermi ::  SVector{D,Bool}                 # encodes whether the i-th dimension of tucker encodes a bosonic or fermionic frequency (only relevant in MF)

#     ### Constructors ###
#     function PartialCorrelator_TCI(T::Float64, formalism::String, Adisc::Array{Float64,D}, ωdisc::Vector{Float64}, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}; need_compactAdisc::Bool=true) where {D}
#         if !(formalism == "MF" || formalism == "KF")
#             throw(ArgumentError("formalism must be MF when unbroadened Adisc is input."))
#         end
#         if DEBUG()
#             println("Constructing PartialCorrelator_reg (WITHOUT broadening).")
#         end

#         ωs_int, ωconvOff, isFermi = _trafo_ω_args(ωs_ext, ωconvMat)
#         if need_compactAdisc 
#         # Delete rows/columns that contain only zeros
#         _, ωdiscs, Adisc = compactAdisc(ωdisc, Adisc)
#         else
#             ωdiscs, Adisc = [ωdisc for _ in 1:D], Adisc
#         end

#         @TIME Kernels = [get_regular_1D_MF_Kernel(ωs_int[i], ωdiscs[i]) for i in 1:D] "Precomputing 1D kernels (for MF)."

#         if formalism == "MF" && !all(isFermi)
#             i_ωbos = argmax(.!isFermi)
#             # all entries where bosonic frequency is (almost) zero
#             Adisc_anoβ = Adisc[[Colon() for _ in 1:i_ωbos-1]..., abs.(ωdiscs[i_ωbos]) .< 1e-8, [Colon() for _ in i_ωbos+1:D]...]
#         else 
#             Adisc_anoβ = Array{ComplexF64,D}(undef, zeros(Int, D)...)
#         end

#         # tucker = TuckerDecomposition(Adisc, Kernels; ωs_center=ωdiscs, ωs_legs=[ωs_int...])
        
#         return new{D}(T, formalism, tucker, Adisc_anoβ, ωs_ext, ωconvMat, ωconvOff, isFermi)
#     end

function TCI_precompute_reg_values_MF_without_ωconv(
    Gp::PartialCorrelator_reg{D}
)::MPS where {D}

    Gp_mps = TD_to_MPS_via_TTworld(Gp)
    if TCI4Keldysh.VERBOSE()
        TCI4Keldysh.mps_idx_info(Gp_mps)
    end

    return Gp_mps
end

TCI4Keldysh.VERBOSE() = true
TCI4Keldysh.TIME() = true
TCI4Keldysh.DEBUG() = true

function test_TCI_precompute_reg_values_MF_without_ωconv()


    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt = 3
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = dummy_frequency_convention(npt)
    R = 7
    GFs = load_npoint(PSFpath, Ops, npt, R, ωconvMat)

    # pick PSF
    spin = 1
    perm_idx = 1
    Gp = GFs[spin].Gps[perm_idx]
    @show Gp.ωconvOff
    display(cumsum(ωconvMat[GFs[spin].ps[perm_idx][1:npt-1],:], dims=1))

    # compute reference data for selected PSF
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    # TCI computation
    Gp_mps = TD_to_MPS_via_TTworld(Gp.tucker)

    # compare
    tags = npt==3 ? ("ω1", "ω2") : ("ω1", "ω2", "ω3")
    fatGpTCI = TCI4Keldysh.MPS_to_fatTensor(Gp_mps; tags=tags)

    @show size(data_unrotated)
    @show typeof(data_unrotated)
    @show size(fatGpTCI)
    @show typeof(fatGpTCI)

    bosidx = findfirst(i -> !isinteger(log2(i)), collect(size(data_unrotated)))
    off_unrot = 1
    # bosonic frequency range might have been extended to 2^(R+1)
    if isnothing(bosidx)
        bosidx = findfirst(i -> i>2^R, collect(size(data_unrotated)))
        off_unrot = 0
    end
    unrotated_slice = [[Colon() for _ in 1:(bosidx-1)]..., 1:size(data_unrotated,bosidx)-off_unrot, [Colon() for _ in bosidx+1:ndims(data_unrotated)]...]
    TCIslice = [1:2^grid_R(size(leg,1)) for leg in Gp.tucker.legs]
    @show TCIslice
    diff = fatGpTCI[TCIslice...] - data_unrotated[unrotated_slice...]

    @show norm(fatGpTCI[:,end])
    @show argmax(abs.(data_unrotated[:,end]))
    @show norm(data_unrotated[:,end])
    for i in 0:2
        @show i
        @show maximum(abs.(diff[:,1:end-i]))
        @show sum(abs.(diff[:,1:end-i]))/reduce(*,size(diff[:,1:end-i]))
    end

    if npt==3
        scalefun = log
        heatmap(scalefun.(abs.(fatGpTCI[TCIslice...])))
        savefig("TT_heatmap_2D.png")
        heatmap(scalefun.(abs.(data_unrotated[unrotated_slice...])))
        savefig("data_unrotated_2D.png")
        heatmap(scalefun.(abs.(diff[:,:])))
        savefig("logdiff_2D.png")
    end
end

"""
Load spin up and down of a given 2 or 3 point function.
"""
function load_npoint(PSFpath::String, ops::Vector{String}, npt::Int, R, ωconvMat, nested_ωdisc=false)
    # parameters
    U = 0.05
    T = 0.01*U

    # define grids
    Rpos = R-1
    Nωcont_pos = 2^Rpos
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)

    if npt==3
        K1ts = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T, flavor_idx=i, ωs_ext=(ωbos,ωfer), ωconvMat=ωconvMat, name="SIAM $(npt)pG", nested_ωdisc=nested_ωdisc) for i in 1:2]
        return K1ts
    elseif npt==4
        K1ts = [TCI4Keldysh.FullCorrelator_MF(joinpath(PSFpath, "4pt"), ops; T, flavor_idx=i, ωs_ext=(ωbos,ωfer,ωfer), ωconvMat=ωconvMat, name="SIAM $(npt)pG", nested_ωdisc=nested_ωdisc) for i in 1:2]
        return K1ts
    else
        error("npt=$npt invalid")
    end
end

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
    U = 0.05
    Δ = U / (π * u)
    T = 0.01*U

    Rpos = R-1
    Nωcont_pos = 2^Rpos
    
    # define grids
    # cannot exclude end point for bosonic part, requires equidistant symmetric grid
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    # ========== parameters END

    # define some frequency convention
    # 2pt
    ωconvMat_K2′t = [
        1  0;
        0  1;
        -1 -1;
    ]
    ωconvMat_K2′t = [
        sum(view(ωconvMat_t, [1,2], [1,3]), dims=1);
        view(ωconvMat_t, [3,4], [1,3])
    ]

    # 3pt
    ωconvMat_t = [
        0 -1  0;
        1  1  0;
        -1  0 -1;
        0  0  1;
    ]
    ωconvMat_K1t = reshape([
        sum(view(ωconvMat_t, [1,2], 1), dims=1);
        sum(view(ωconvMat_t, [3,4], 1), dims=1);
    ], (2,1))

    ret = Dict{Int, Array}()

    # one-point
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

"""
Get some dummy frequency convention matrix
"""
function dummy_frequency_convention(npt::Int)
    ωconvMat_t = [
        0 -1  0;
        1  1  0;
        -1  0 -1;
        0  0  1;
    ]
    if npt==2
        ωconvMat_K1t = reshape([
            sum(view(ωconvMat_t, [1,2], 1), dims=1);
            sum(view(ωconvMat_t, [3,4], 1), dims=1);
        ], (2,1))
        return ωconvMat_K1t
    elseif npt==3
        ωconvMat_K2prime_t = [
            sum(view(ωconvMat_t, [1,2], [1,3]), dims=1);
            view(ωconvMat_t, [3,4], [1,3])
        ]
        return ωconvMat_K2prime_t
    elseif npt==4
        return ωconvMat_t
    end
end