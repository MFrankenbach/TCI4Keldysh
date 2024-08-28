using Plots

#=
Compute PartialCorrelators & FullCorrelators (Matsubara) by convolution with frequency kernels, all in interleaved quantics representation.
Performance: 🤯😢
=#

"""
Compute MPS of PartialCorrelator parametrized in external frequencies, i.e., after frequency rotation.
TODO: Write generic test
"""
function TCI_precompute_reg_values_rotated(
    Gp::PartialCorrelator_reg{D}; tag="ω", tolerance=1e-8, cutoff=1e-20, include_ano=true
)::MPS where {D}
    
    Gp_mps = TD_to_MPS_via_TTworld(Gp.tucker; tolerance=tolerance, cutoff=cutoff)

    println("  Regular corr. rank before rotation: $(rank(Gp_mps))")

    # anomalous term if required
    if include_ano
        if ano_term_required(Gp) && maximum(abs.(Gp.Adisc_anoβ))>tolerance
            @info "Adding anomalous term"
            Gano_mps = anomalous_TD_to_MPS_full(Gp; tolerance=tolerance, cutoff=cutoff)
            println("  Anomalous corr. rank before rotation: $(rank(Gp_mps))")
            R = div(length(Gp_mps), D)
            for i in 1:D
                _adoptinds_by_tags!(Gano_mps, Gp_mps, "ω$i", "ω$i", R)
            end
            Gp_mps = add(Gp_mps, Gano_mps; alg="densitymatrix", cutoff=cutoff, use_absolute_cutoff=true)
        end
    end

    println("  Corr. rank before rotation: $(rank(Gp_mps))")

    # frequency rotation
    freq_transform_kwargs = Dict(:cutoff=>cutoff, :use_absolute_cutoff=>true)
    tags = ntuple(i -> "ω$i", D)
    is_ferm_new = vcat([0], fill(1, D-1))
    Gp_mps_rot = if D>1
                    freq_transform(Gp_mps; tags=tags, ωconvMat=convert(Matrix{Int}, Gp.ωconvMat), isferm_ωnew=is_ferm_new,
                            ext_gridsizes=collect(length.(Gp.ωs_ext)), freq_transform_kwargs...)
                elseif D==1
                    if only(Gp.ωconvMat) == -1
                        Quantics.reverseaxis(Gp_mps; tag="ω1", cutoff=cutoff, use_absolute_cutoff=true)
                    else
                        Gp_mps
                    end
                end

    # saturate extra legs with 1s
    R_ext = grid_R(length(Gp.ωs_ext[end]))
    R_sat = length(Gp_mps_rot) - D*R_ext
    @assert R_sat >= 0
    Gp_sites = siteinds(Gp_mps_rot)
    start_block = ITensor(one(eltype(Gp_mps_rot[1])))
    for i in 1:R_sat
        h = onehot(Gp_sites[i] => 1)
        start_block *= h*Gp_mps_rot[i]
    end
    start_block *= Gp_mps_rot[R_sat+1]

    ret = MPS(vcat([start_block], Gp_mps_rot[R_sat+2:end]))
    # fix tags
    for i in eachindex(ret)
        d = mod(i,D)!=0 ? mod(i,D) : D
        bit = div(i-1,D) + 1
        tag_old = "$(tag)$(d)=$(bit+div(R_sat,D))"
        tag_new = "$(tag)$(d)=$(bit)"
        replacetags!(ret[i], tag_old, tag_new)
    end
    ret = truncate!(ret; cutoff=cutoff, use_absolute_cutoff=true)
    println("  Corr. rank after rotation: $(rank(ret))")
    return ret
end

"""
Determine correlator dimension from number of permutatins
"""
function get_corrdim(nperm::Int)
    D = if nperm==2
            1;
        elseif nperm==6
            2;
        elseif nperm==24
            3;
        else
            # error("Invalid number of PartialCorrelators")
            @warn "Got $nperm PartialCorrelators - some are missing"
            if nperm < 2
                1;
            elseif (nperm < 6) && (nperm > 2)
                2;
            elseif (nperm < 24) && (nperm > 6)
                3;
            end
        end
    return D
end

"""
Compute FullCorrelator as sum of PartialCorrelators by TCI compression.
"""
function FullCorrelator_recompress(Gps::Vector{MPS}; tcikwargs...)
    nperm = length(Gps)
    D = get_corrdim(nperm) 
    @assert all(length(Gps[1]) .== length.(Gps)) "All partial correlator MPS must have the same length!"

    localdims = fill(2, length(Gps[1]))

    function eval_FullCorrelator(idx::Vector{Int})
        res = zero(ComplexF64)
        for Gp in Gps
            res += eval(Gp, idx)
        end
        return res
    end

    eval_FullCorrelator_cache = TCI.CachedFunction{ComplexF64}(eval_FullCorrelator, localdims)
    full_tci, _, _ = TCI.crossinterpolate2(ComplexF64, eval_FullCorrelator_cache, localdims; tcikwargs...)
    R = div(length(Gps[1]), D)
    tags = ntuple(i -> "ω$i", D)
    fullsites = [Index(localdims[R*(d-1)+r], "Qubit, $(tags[d])=$r") for r in 1:R for d in 1:D]
    full_mps = TCI4Keldysh.TCItoMPS(full_tci; sites=fullsites)

    return full_mps
end

"""
Compute full correlator by MPS addition.
Replaces indices of Gps!
"""
function FullCorrelator_add(Gps::Vector{MPS}; addkwargs...)
    @assert all(length(Gps[1]) .== length.(Gps)) "All partial correlator MPS must have the same length!"
    D = get_corrdim(length(Gps))
    R = div(length(Gps[1]), D)
    localdims = dim.(siteinds(Gps[1]))
    fullsites = [Index(localdims[R*(d-1)+r], "Qubit, ω$(d)=$r") for r in 1:R for d in 1:D]
    for Gp in Gps
        _replaceinds!(Gp, siteinds(Gp), fullsites)
    end
    # add
    Gfull = add(Gps...; alg="densitymatrix", addkwargs...)
    # truncate!(Gfull; cutoff=addkwargs[:cutoff], use_absolute_cutoff=true)
    return Gfull
end

"""
Test and visualize TCI-convolution with regular kernels.
The function in question is `TD_to_MPS_via_TTworld`
"""
function test_TCI_precompute_reg_values_MF_without_ωconv(;npt=3, R=7, perm_idx=1, cutoff=1e-5, tolerance=1e-12)

    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = dummy_operators(npt)
    ωconvMat = dummy_frequency_convention(npt)
    GFs = load_npoint(PSFpath, Ops, npt, R, ωconvMat; nested_ωdisc=false)

    # pick PSF
    spin = 1
    Gp = GFs[spin].Gps[perm_idx]

    # compute reference data for selected PSF
    display(cumsum(ωconvMat[GFs[spin].ps[perm_idx][1:npt-1],:], dims=1))
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    # TCI computation
    Gp_mps = TD_to_MPS_via_TTworld(Gp.tucker; tolerance=tolerance, cutoff=cutoff)

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
    @show unrotated_slice
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
Test and visualize frequency rotation with TCI.
For full_tci=true, compute 4-pt function on TCI-level as well
"""
function test_TCI_frequency_rotation_reg_values(;npt=3, full_tci=false, perm_idx::Int=1, cutoff::Float64=1e-20, tolerance::Float64=1e-12, beta::Float64=1.e3)

    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = dummy_frequency_convention(npt)
    R_ext = 7
    GFs = load_npoint(PSFpath, Ops, npt, R_ext, ωconvMat; beta=beta)

    # pick PSF
    spin = 1

    # printstyled("---- Frequency rotation matrices:\n"; color=:blue)
    # for p in 1:factorial(npt)
    #     display(cumsum(ωconvMat[GFs[spin].ps[p][1:npt-1],:], dims=1))
    # end
    # printstyled("----\n"; color=:blue)

    Gp = GFs[spin].Gps[perm_idx]
    display(cumsum(ωconvMat[GFs[spin].ps[perm_idx][1:npt-1],:], dims=1))

    # compute reference data for selected PSF
    data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

    # TCI frequency kernel convolution
    printstyled("  ---- Compressing unrotated data\n"; color=:magenta)
    Gp_mps = if npt==3 || full_tci 
                TD_to_MPS_via_TTworld(Gp.tucker; cutoff=cutoff, tolerance=tolerance)
            else
                R = grid_R(size(data_unrotated))
                compress_padded_array(data_unrotated, R)
            end
    @show length(Gp_mps)

    tags = npt==3 ? ("ω1", "ω2") : ("ω1", "ω2", "ω3")
    fatUnrotGp = TCI4Keldysh.MPS_to_fatTensor(Gp_mps; tags=tags)

    if npt==3
        heatmap(log.(abs.(fatUnrotGp)))
        savefig("TT_unrotated_rot.png")
    end

    # ===== frequency rotation @ TCI =====
    printstyled("  ---- Rotating\n"; color=:magenta)
    freq_transform_kwargs = Dict(:cutoff=>cutoff, :use_absolute_cutoff=>true)
    tags = npt==3 ? ("ω1", "ω2") : ("ω1", "ω2", "ω3")
    is_ferm_new = vcat([0], fill(1, npt-2))
    old_gridsizes = [min(2^grid_R(size(leg, 1)), size(leg,1)) for leg in Gp.tucker.legs]
    @show old_gridsizes
    @TIME Gp_mps_rot = freq_transform(Gp_mps; tags=tags, ωconvMat=convert(Matrix{Int}, Gp.ωconvMat), isferm_ωnew=is_ferm_new,
                            ext_gridsizes=collect(length.(Gp.ωs_ext)), freq_transform_kwargs...) "Frequency rotation:"
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

    # plot
    scalefun=log
    if npt==3
        heatmap(scalefun.(abs.(compare_values)))
        savefig("rotated_reference2D.png")
        heatmap(scalefun.(abs.(fatGpTCI)))
        savefig("TT_rotated2D.png")

        # check difference
        # leave out first bosonic frequency because when bosonic grid is mirrored in at least one PartialCorrelator,
        # we have to discard this frequency anyways.
        diffslice = (2:2^R_ext, 1:2^R_ext)
        diff = fatGpTCI[diffslice...] - compare_values[diffslice...]
        @show norm(diff)
        @show maximum(abs.(diff))

    elseif npt==4

        testdiff = abs.(compare_values[2:2^R_ext,1:2^R_ext,1:2^R_ext] - fatGpTCI[2:2^R_ext,1:2^R_ext,1:2^R_ext])
        printstyled("  Maximum, mean error 3D rotation:\n"; color=:green)
        @show maximum(testdiff)
        @show sum(testdiff)/prod(size(testdiff))
        amax =  argmax(testdiff)

        # slice_idx = 2^(R_ext-1)
        slice_idx = amax[2]
        ref_slice = [1:2^R_ext, slice_idx, 1:2^R_ext]
        tci_slice = [1:2^R_ext, slice_idx, 1:2^R_ext]
        heatmap(scalefun.(abs.(compare_values[:,slice_idx,:])))
        savefig("rotated_reference3D.png")
        heatmap(scalefun.(abs.(fatGpTCI[1:2^R_ext,slice_idx,1:2^R_ext])))
        savefig("TT_rotated3D.png")
        plotdiff = abs.(compare_values[ref_slice...] - fatGpTCI[tci_slice...])
        heatmap(scalefun.(plotdiff))
        savefig("diff_3D.png")
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

function test_TCI_precompute_reg_values_rotated_1D()
    
    R = 12
    beta = 1000.0
    Ops = ["F1", "F1dag"]
    tolerance = 1.e-12
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=2
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)
    GF = only(GFs)

    # TCI calculation
    Gps_out = Vector{MPS}(undef, 2)
    for perm_idx in 1:2
        display(GF.Gps[perm_idx].ωconvMat)
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GF.Gps[perm_idx]; tolerance=tolerance, cutoff=1e-20)
        Gps_out[perm_idx] = Gp_mps
    end
    Gps_fat = [TCI4Keldysh.MPS_to_fatTensor(Gps_out[i]; tags=("ω1",)) for i in 1:2]
    @show size.(Gps_fat)

    # reference
    printstyled(" ---- Check partial 2pt correlators\n"; color=:green)
    Gps_ref = []
    for perm_idx in 1:2
        Gp_ref = precompute_all_values_MF(GF.Gps[perm_idx]) 
        @show size(Gp_ref)
        # disregard first frequency
        slice = 2^(R-1)-20 : 2^(R-1)+20
        diff = abs.(Gps_fat[perm_idx][2:2^R] - Gp_ref[2:2^R])
        @show maximum(diff)
        @show argmax(diff)
        push!(Gps_ref, Gp_ref)
        Gpplot = plot()
        plot!(Gpplot, GF.ωs_ext[1][slice], real.(Gps_fat[perm_idx][slice]); label="TCI Re")
        plot!(Gpplot, GF.ωs_ext[1][slice], imag.(Gps_fat[perm_idx][slice]); label="TCI Im")
        plot!(Gpplot, GF.ωs_ext[1][slice], real.(Gp_ref[slice]); label="ref. Re")
        plot!(Gpplot, GF.ωs_ext[1][slice], imag.(Gp_ref[slice]); label="ref. Im")
        savefig(Gpplot, "1DGp_p=$perm_idx.png")
    end

    # check FullCorrelator
    printstyled(" ---- Check full 2pt correlator\n"; color=:green)
    Gfull_ref = Gps_ref[1] + Gps_ref[2]
    Gfull_mps = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
    Gfull_fat = TCI4Keldysh.MPS_to_fatTensor(Gfull_mps; tags=("ω1",))
    @show size(Gfull_fat)
    slice = 2:2^R
    diff = abs.(Gfull_fat[slice] - Gfull_ref[slice])
    @show maximum(diff)

    # plot
    GFplot = plot()
    plot!(GFplot, GF.ωs_ext[1][slice], real.(Gfull_fat[slice]); label="TCI Re")
    plot!(GFplot, GF.ωs_ext[1][slice], imag.(Gfull_fat[slice]); label="TCI Im")
    plot!(GFplot, GF.ωs_ext[1][slice], real.(Gfull_ref[slice]); label="ref. Re")
    plot!(GFplot, GF.ωs_ext[1][slice], imag.(Gfull_ref[slice]); label="ref. Im")
    savefig(GFplot, "1DGfull.png")
end

function test_TCI_precompute_anomalous_values_patched(;npt=4, perm_idx=1)
    error("Not yet implemented")

    GFs = dummy_correlator(npt, 7)

    Gp = GFs[1].Gps[perm_idx]

    R = 7
    # even a tolerance of 100 yields patches with bond dimensions of up to 100...
    patch_res = compress_anomalous_kernel_patched(Gp, R; tolerance=1e-2)

    # compress_Adisc_ano_patched(Gp; rtol=1e-5, maxbonddim=50)
end

"""
Load spin up and down of a given 2-4 point function.
"""
function load_npoint(PSFpath::String, ops::Vector{String}, npt::Int, R, ωconvMat; beta::Float64=2000.0, nested_ωdisc=false, kwargs...)

    # define grids
    T = 1.0/beta
    Rpos = R-1
    Nωcont_pos = 2^Rpos
    # TODO: call grid functions to generate this
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)

    if npt==2
        K1ts = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T, flavor_idx=1, ωs_ext=(ωbos,), ωconvMat=ωconvMat, name="SIAM $(npt)pG", nested_ωdisc=nested_ωdisc, kwargs...)]
        return K1ts
    elseif npt==3
        K1ts = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T, flavor_idx=i, ωs_ext=(ωbos,ωfer), ωconvMat=ωconvMat, name="SIAM $(npt)pG", nested_ωdisc=nested_ωdisc, kwargs...) for i in 1:2]
        return K1ts
    elseif npt==4
        K1ts = [TCI4Keldysh.FullCorrelator_MF(joinpath(PSFpath, "4pt"), ops; T, flavor_idx=i, ωs_ext=(ωbos,ωfer,ωfer), ωconvMat=ωconvMat, name="SIAM $(npt)pG", nested_ωdisc=nested_ωdisc, kwargs...) for i in 1:2]
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
function dummy_frequency_convention(npt::Int; channel::String="t")
    ωconvMat = channel_trafo(channel)
    if npt==2
        ωconvMat_K1t = reshape([
            sum(view(ωconvMat, [1,2], 1), dims=1);
            sum(view(ωconvMat, [3,4], 1), dims=1);
        ], (2,1))
        return ωconvMat_K1t
    elseif npt==3
        ωconvMat_K2prime_t = [
            sum(view(ωconvMat, [1,2], [1,3]), dims=1);
            view(ωconvMat, [3,4], [1,3])
        ]
        return ωconvMat_K2prime_t
    elseif npt==4
        return ωconvMat
    end
end

"""
Get 2-pt / 3-pt / 4pt operator combination for testing
"""
function dummy_operators(npt::Int)
    if npt==2 
        return ["F1", "F1dag"]
    elseif npt==3
        return ["F1", "F1dag", "Q34"]
    elseif npt==4
        return ["F1", "F1dag", "F3", "F3dag"]
    else 
        error("Invalid: npt=$npt")
    end
end

"""
Load sample Matsubara full correlator.
"""
function dummy_correlator(
    npt::Int, R::Int; 
    beta::Float64=1.e3, channel::String="t", Ops::Union{Nothing,Vector{String}}=nothing,
    PSFpath="SIAM_u=0.50/PSF_nz=2_conn_zavg/",
    kwargs...
    ) :: Vector{FullCorrelator_MF}
    PSFpath = joinpath(datadir(), PSFpath)
    Ops_loc = isnothing(Ops) ?  dummy_operators(npt) : Ops
    ωconvMat = dummy_frequency_convention(npt; channel=channel)
    GFs = load_npoint(PSFpath, Ops_loc, npt, R, ωconvMat; nested_ωdisc=false, beta=beta, kwargs...)
    return GFs
end