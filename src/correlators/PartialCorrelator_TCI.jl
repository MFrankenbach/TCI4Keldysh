using Plots

function TCI_precompute_reg_values_MF_without_ωconv(
    Gp::PartialCorrelator_reg{D}
)::MPS where {D}

    Gp_mps = TD_to_MPS_via_TTworld(Gp)
    if TCI4Keldysh.VERBOSE()
        TCI4Keldysh.mps_idx_info(Gp_mps)
    end

    return Gp_mps
end

"""
Test TCI-convolution with regular kernels;
the function in question is `TD_to_MPS_via_TTworld`
"""
function test_TCI_precompute_reg_values_MF_without_ωconv(;npt=3, perm_idx=1, cutoff=1e-5)

    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = dummy_frequency_convention(npt)
    R = 7
    GFs = load_npoint(PSFpath, Ops, npt, R, ωconvMat; nested_ωdisc=false)

    # pick PSF
    spin = 1
    Gp = GFs[spin].Gps[perm_idx]

    # compute reference data for selected PSF
    display(cumsum(ωconvMat[GFs[spin].ps[perm_idx][1:npt-1],:], dims=1))
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    # TCI computation
    Gp_mps = TD_to_MPS_via_TTworld(Gp.tucker; cutoff=cutoff)

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
    TCIslice = Base.OneTo.(ntuple(i -> min(2^grid_R(size(Gp.tucker.legs[i],1)), size(data_unrotated[unrotated_slice...],i)), ndims(data_unrotated)))
    @show TCIslice
    diff = fatGpTCI[TCIslice...] - data_unrotated[unrotated_slice...]

    max_err = maximum(abs.(diff))
    mean_err = sum(abs.(diff))/reduce(*,size(diff))
    @show max_err
    @show mean_err

    if npt==3
        scalefun = log10
        heatmap(scalefun.(abs.(fatGpTCI)))
        savefig("TT_heatmap_2D.png")
        heatmap(scalefun.(abs.(data_unrotated)))
        savefig("data_unrotated_2D.png")
        heatmap(scalefun.(abs.(diff[:,:])))
        savefig("logdiff_2D.png")
    elseif npt==4
        scalefun = log10
        complexfun = abs
        slice_idx = 65
        heatmap(scalefun.(complexfun.(fatGpTCI[TCIslice[1], slice_idx, TCIslice[3]])))
        savefig("TT_heatmap_3D.png")
        heatmap(scalefun.(complexfun.(data_unrotated[unrotated_slice[1], slice_idx, unrotated_slice[3]])))
        savefig("data_unrotated_3D.png")
        heatmap(scalefun.(complexfun.(diff[:,slice_idx,:])))
        savefig("logdiff_3D.png")
    end
    return (mean_err, max_err)
end

"""
Test frequency rotation with TCI.
For full_tci=true, compute 4-pt function on TCI-level as well
"""
function test_TCI_frequency_rotation_reg_values(;npt=3, full_tci=false)

    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = dummy_frequency_convention(npt)
    R_ext = 7
    GFs = load_npoint(PSFpath, Ops, npt, R_ext, ωconvMat)

    # pick PSF
    spin = 1
    perm_idx = 3

    # printstyled("---- Frequency rotation matrices:\n"; color=:blue)
    # for p in 1:factorial(npt)
    #     display(cumsum(ωconvMat[GFs[spin].ps[p][1:npt-1],:], dims=1))
    # end
    # printstyled("----\n"; color=:blue)

    Gp = GFs[spin].Gps[perm_idx]
    display(cumsum(ωconvMat[GFs[spin].ps[perm_idx][1:npt-1],:], dims=1))

    # compute reference data for selected PSF
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    @show size(data_unrotated)

    # TCI frequency kernel convolution
    Gp_mps = if npt==3 || full_tci 
                TD_to_MPS_via_TTworld(Gp.tucker)
            else
                R = grid_R(Gp.tucker)
                compress_padded_array(data_unrotated, R)
            end

    tags = npt==3 ? ("ω1", "ω2") : ("ω1", "ω2", "ω3")
    fatUnrotGp = TCI4Keldysh.MPS_to_fatTensor(Gp_mps; tags=tags)

    if npt==3
        heatmap(log.(abs.(fatUnrotGp)))
        savefig("TT_unrotated_rot.png")
    end

    # ===== frequency rotation @ TCI =====
    tags = npt==3 ? ("ω1", "ω2") : ("ω1", "ω2", "ω3")
    is_ferm_new = vcat([0], fill(1, npt-2))
    old_gridsizes = [min(2^grid_R(size(leg, 1)), size(leg,1)) for leg in Gp.tucker.legs]
    @show old_gridsizes
    @TIME Gp_mps_rot = freq_transform(Gp_mps; tags=tags, ωconvMat=convert(Matrix{Int}, Gp.ωconvMat), isferm_ωnew=is_ferm_new,
                            ext_gridsizes=collect(length.(Gp.ωs_ext))) "Frequency rotation:"
    # ===== frequency rotation END

    # comparison
    D = npt-1
    strides_internal = [stride(data_unrotated, i) for i in 1:D]'
    strides4rot = ((strides_internal * Gp.ωconvMat)...,)
    offset4rot = sum(strides4rot) - sum(strides_internal) + strides_internal * Gp.ωconvOff
    sv = StridedView(data_unrotated, (length.(Gp.ωs_ext)...,), strides4rot, offset4rot)
    compare_values = Array{ComplexF64,D}(sv[[Colon() for _ in 1:D]...])
    @show size(data_unrotated)
    @show size(compare_values)
    @show (length.(Gp.ωs_ext))

    tags = npt==3 ? ("ω1", "ω2") : ("ω1", "ω2", "ω3")
    fatGpTCI = TCI4Keldysh.MPS_to_fatTensor(Gp_mps_rot; tags=tags)
    @show size(fatGpTCI)
    # bosonic grid is not symmetric for tci, last point is missing to get a power of two as grid size
    # this means that the frequency transform is shifted by one in the bosonic direction compared to the reference
    # diff1 = fatGpTCI[2:129, 1:128] - compare_values[2:129, :]
    # @show norm(diff1)

    # plot
    scalefun=log
    if npt==3
        heatmap(scalefun.(abs.(compare_values)))
        savefig("rotated_reference2D.png")
        heatmap(scalefun.(abs.(fatGpTCI)))
        savefig("TT_rotated2D.png")
    elseif npt==4
        slice_idx = 70
        ref_slice = [1:128, slice_idx, 1:128]
        tci_slice = [1:128, slice_idx, 1:128]
        heatmap(scalefun.(abs.(compare_values[:,slice_idx,:])))
        savefig("rotated_reference3D.png")
        heatmap(scalefun.(abs.(fatGpTCI[:,slice_idx,:])))
        savefig("TT_rotated3D.png")
        plotdiff = abs.(compare_values[ref_slice...] - fatGpTCI[tci_slice...])
        heatmap(scalefun.(plotdiff))
        savefig("diff_3D.png")
        testdiff = abs.(compare_values[2:128,1:128,1:128] - fatGpTCI[2:128,1:128,1:128])
        printstyled("  Maximum, mean error 3D rotation:\n"; color=:green)
        @show maximum(testdiff)
        @show sum(testdiff)/prod(size(testdiff))
        @show argmax(testdiff)
    end
    println("========== DONE")
end

"""
Test TCI-computation of anomalous term in the presence of a bosonic grid.\\
*CAREFUL*: So far, only tested for 4-point functions, since our current 3-point functions do not require
an anomalous term. Also, on the local machine, only cutoffs until 1e-5 can be done.
"""
function test_TCI_precompute_anomalous_values(;npt=4, perm_idx=1, tolerance=1e-6)
    
    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = dummy_frequency_convention(npt)
    R = 7
    GFs = load_npoint(PSFpath, Ops, npt, R, ωconvMat; nested_ωdisc=false)

    # pick PSF
    spin = 1
    Gp = GFs[spin].Gps[perm_idx]

    if prod(size(Gp.Adisc_anoβ))==0
        @info "ωdisc does not contain vanishing bosonic frequency"
    end

    # get reference values
    reg_values = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
    full_values = precompute_all_values_MF_without_ωconv(Gp)
    ano_values = full_values .- reg_values

    if npt==3
        heatmap(abs.(ano_values))
        savefig("ano_values_2D.png")
        @warn "No test for 3-pt anomalous term present!"
    end

    if npt==4
        # find bosonic 0 frequency
        bos_idx = findfirst(.!Gp.isFermi)
        ωbos = Gp.tucker.ωs_legs[bos_idx] 
        zero_inds = [i for i in eachindex(ωbos) if abs(ωbos[i]) < 1.e-10]
        slice = [[Colon() for _ in 1:bos_idx-1]..., only(zero_inds), [Colon() for _ in bos_idx+1:ndims(ano_values)]...]
        scalefun = x -> x
        heatmap(scalefun.(abs.(ano_values[slice...])))
        savefig("ano_values_3D_tol=1e$(round(Int, log10(tolerance))).png")

        # perform TCI computation
        println("---- TCI-compress anomalous kernel")
        Gp_ano_mps = anomalous_TD_to_MPS(Gp; tolerance=1e-3)

        grid_ids = [i for i in 1:npt-1 if i!=bos_idx]
        fat_ano = TCI4Keldysh.MPS_to_fatTensor(Gp_ano_mps; tags=ntuple(i -> "ω$(grid_ids[i])", npt-2))

        @show size(fat_ano)

        heatmap(scalefun.(abs.(fat_ano)))
        savefig("ano_values_3D_tci_tol=1e$(round(Int, log10(tolerance))).png")

        diff_slice = [slice[i] for i in grid_ids]
        diff = abs.(fat_ano[diff_slice...] - ano_values[slice...])
        heatmap(diff)
        savefig("ano_values_3D_diff_tol=1e$(round(Int, log10(tolerance))).png")
        @show sum(diff)/prod(size(diff))
        @show maximum(diff)
    end
end

function test_TCI_precompute_anomalous_values_patched(;npt=4, perm_idx=1)
    GFs = load_dummy_correlator(;npt=npt)

    Gp = GFs[1].Gps[perm_idx]

    R = 7
    # even a tolerance of 100 yields patches with bond dimensions of up to 100...
    patch_res = compress_anomalous_kernel_patched(Gp, R; tolerance=1e-2)

    # compress_Adisc_ano_patched(Gp; rtol=1e-5, maxbonddim=50)
end

"""
Load spin up and down of a given 2 or 3 point function.
"""
function load_npoint(PSFpath::String, ops::Vector{String}, npt::Int, R, ωconvMat; nested_ωdisc=false)
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

"""
Load sample matsubara full correlator
"""
function load_dummy_correlator(;npt=3)
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = dummy_frequency_convention(npt)
    R = 7
    GFs = load_npoint(PSFpath, Ops, npt, R, ωconvMat; nested_ωdisc=false)
    return GFs
end