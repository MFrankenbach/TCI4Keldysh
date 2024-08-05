"""
    FullCorrelator_MF{D}

The struct FullCorrelator_MF{D} represents a full Matsubara correlator via the sum of PartialCorrelator_reg objects.
This representation is described in Eq.(39) in [Kugler et al. PHYS. REV. X 11, 041006 (2021)],
where the *partial* correlators Gp are the contributions from individual permutations p.

# Members:
* name    ::Vector{String}                    : name to distinguish the objects (e.g.: list of operators)
* Gps     ::Vector{PartialCorrelator_reg}     : list of partial correlators
* Gp_to_G ::Vector{Float64}                   : prefactors for Gp's (currently this coefficient is applied to Adisc => no need to apply it during evaluation.)
* ωs_ext  ::NTuple{D,Vector{ComplexF64}}      : external complex frequencies in the chosen parametrization
* ωconvMat::Matrix{Int}                       : Matrix of size (D+1, D) encoding conversion of external frequencies to the frequencies of the D+1 legs ~ ω_ferm = ωconvMat * ω_ext.
                                                Columns must add up to zero.
                                                Allowed matrix entries: -1, 0, 1
* isBos   ::BitVector                         : BitVector indicating which legs are bosonic

# Evaluation
Objects of type FullCorrelator_MF{D} can be evaluated *pointwisely* at indices, e.g. for D=2
```julia-repl
julia> GM(1, 2)
2.0
```

To evaluate it at *all* available indices:
```julia-repl
julia> precompute_all_values(GM)
MxN Matrix{Float64}
 ...
```
"""
mutable struct FullCorrelator_MF{D}
    name    ::Vector{String}                    
    Gps     ::Vector{PartialCorrelator_reg}     
    Gp_to_G ::Vector{Float64}                   
    ps      ::Vector{Vector{Int}}

    ωs_ext  ::NTuple{D,Vector{Float64}}      
    ωconvMat::Matrix{Int}                       
    isBos   ::BitVector                         

    function FullCorrelator_MF(path::String, Ops::Vector{String}; flavor_idx::Int, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}, T::Float64, name::String="", is_compactAdisc::Bool=true, nested_ωdisc::Bool=false) where{D}
        ##########################################################################
        ############################## check inputs ##############################
        if length(Ops) != D+1
            throw(ArgumentError("Ops must contain "*string(D+1)*" elements."))
        end
        ###################################################################
        
        perms = permutations(collect(1:D+1))
        isBos = (o -> length(o) == 3).(Ops)
        ωdisc = load_ωdisc(path, Ops; nested_ωdisc)
        Adiscs = [load_Adisc(path, Ops[p], flavor_idx) for (i,p) in enumerate(perms)]

        return FullCorrelator_MF(Adiscs, ωdisc; T, isBos, ωs_ext, ωconvMat, name=[Ops; name], is_compactAdisc)
    end


    function FullCorrelator_MF(Adiscs::Vector{Array{Float64,D}}, ωdisc::Vector{Float64}; T::Float64, isBos::BitVector, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}, name::Vector{String}=[], is_compactAdisc::Bool=true) where{D}
        if DEBUG()
            println("Constructing FullCorrelator_MF.")
        end
        ##########################################################################
        ############################## check inputs ##############################
        if size(ωconvMat) != (D+1, D)
            throw(ArgumentError("ωconvMat must have size ("*string(D+1)*", "*string(D)*") but is of size "*string(size(ωconvMat))*"."))
        end
        if maximum(abs.(sum(ωconvMat, dims=1))) != 0
            throw(ArgumentError("The columns in ωconvMat must add up to zero."))
        end
        if length(Adiscs) != factorial(D+1)
            throw(ArgumentError("Adiscs must contain all "*string(D+1)*"! permutations."))
        end
        if (D+1-sum(isBos))%2 != 0
            throw(ArgumentError("isBos must contain an even number of 'false' (fermions)."))
        end
        ##########################################################################


        perms = permutations(collect(1:D+1))
        Gp_to_G = _get_Gp_to_G(D, isBos)
        Gps = [PartialCorrelator_reg(T, "MF", Gp_to_G[i].*Adiscs[i], ωdisc, ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1); is_compactAdisc) for (i,p) in enumerate(perms)]        

        return new{D}(name, Gps, Gp_to_G, [perms...], ωs_ext, ωconvMat)
    end
end


"""
Maps a permutation to its parity, neglecting bosonic operators.
"""
function _get_Gp_to_G(D::Int, isBos::BitVector) ::Vector{Float64}
    N_fermions = D + 1 - sum(isBos)
    i_Fer = zeros(Int, D+1)
    i_Fer[.!isBos] .= 1:N_fermions
    
    Gp_to_G = zeros(factorial(D+1))
    perms = permutations(collect(1:D+1))
    for (i,p) in enumerate(perms)
        order_of_fermions = i_Fer[p][i_Fer[p].>0]
        ζ = (-1)^parity(order_of_fermions)
        Gp_to_G[i] = ζ
    end
    return Gp_to_G
end

function evaluate(G::FullCorrelator_MF{D}, idx::Vararg{Int,D}) where{D}
    eval_gps(gp) = gp(idx...)
    #Gp_values = eval_gps.(G.Gps)
    #return G.Gp_to_G' * Gp_values
    return sum(eval_gps, G.Gps)
end

function (G::FullCorrelator_MF{D})(idx::Vararg{Int,D}) where{D}
    return evaluate(G, idx...)#[1]
end

"""
To evaluate FullCorrelator_MF pointwise.
"""
struct FullCorrEvaluator_MF{T,D,N}

    GF::FullCorrelator_MF{D}
    anevs::Vector{AnomalousEvaluator{T,D,N}}
    ano_terms_required::Vector{Bool}
    anoid_to_Gpid::Vector{Int}

    function FullCorrEvaluator_MF(GF::FullCorrelator_MF{D}, svd_kernel::Bool=false; cutoff::Float64=1.e-12, cachesize=64) where {D}

        @assert intact(GF)
        ano_terms_required = ano_term_required.(GF.Gps)
        ano_ids = [i for i in eachindex(GF.Gps) if ano_term_required(GF.Gps[i])]

        T = eltype(GF.Gps[1].tucker.center)
        # # populate cache
        # cache = zeros(T, fill(cachesize, D))
        # cache_ranges = []
        # for i in 1:D
        #     cenid = div(length(GF.ωs_ext[i]), 2)
        #     lowid = cenid - div(cachesize, 2)
        #     upid = cenid + div(cachesize, 2)
        #     push!(cache_ranges, max(1, lowid):min(length(GF.ωs_ext[i]), upid))
        # end

        # create anomalous term evaluators
        anevs = Vector{AnomalousEvaluator{T,D,D-1}}(undef, length(ano_ids))
        for i in eachindex(ano_ids)
            anevs[i] = AnomalousEvaluator(GF.Gps[ano_ids[i]])
        end
        anoid_to_Gpid = zeros(Int, length(GF.Gps))
        for i in eachindex(ano_ids)
            anoid_to_Gpid[ano_ids[i]] = i
        end

        # svd kernels if requested
        if svd_kernel
            println("  SVD-decompose kernels with cut=$cutoff...")
            for Gp in GF.Gps
                if any(size(Gp.tucker.center) .> 1000) @warn "SVD-ing legs of sizes $(size.(Gp.tucker.legs))" end
                size_old = size(Gp.tucker.center)
                svd_kernels!(Gp.tucker; cutoff=cutoff)
                size_new = size(Gp.tucker.center)
                println(" Reduced tucker center from $size_old to $size_new")
            end
        end
        return new{T,D,D-1}(GF, anevs, ano_terms_required, anoid_to_Gpid)
    end
end

"""
Evaluate full Matsubara correlator, including anomalous terms.
"""
function (fev::FullCorrEvaluator_MF{T,D,N})(w::Vararg{Int,D}) where {T,D,N}
    ret = zero(ComplexF64)
    for i in eachindex(fev.GF.Gps)
        if fev.ano_terms_required[i]
            anev_act = fev.anevs[fev.anoid_to_Gpid[i]]
            ret += fev.GF.Gps[i](w...) + anev_act(w...)
        else
            ret += fev.GF.Gps[i](w...)
        end
    end
    return ret
end

"""
Try to evaluate FullCorrelator_MF batchwise.
Should be compressed with crossinterpolate2, not quanticscrossinterpolate.
NOT YET IMPLEMENTED (not clear whether batch evaluation makes sense here)
"""
struct FullCorrBatchEvaluator_MF{T,D,N} <: TCI.BatchEvaluator{T}

    GF::FullCorrelator_MF{D}
    anevs::Vector{AnomalousEvaluator{T,D,N}}
    ano_terms_required::Vector{Bool}
    anoid_to_Gpid::Vector{Int}
    grid::QuanticsGrids.InherentDiscreteGrid{D}
    localdims::Vector{Int}

    function FullCorrEvaluator_MF(GF::FullCorrelator_MF{D}, localdims::Vector{Int}, svd_kernel::Bool=false; cutoff::Float64=1.e-12, unfoldingscheme=:interleaved) where {D}

        @assert intact(GF)
        ano_terms_required = ano_term_required.(GF.Gps)
        ano_ids = [i for i in eachindex(GF.Gps) if ano_term_required(GF.Gps[i])]

        # create anomalous term evaluators
        T = eltype(GF.Gps[1].tucker.center)
        anevs = Vector{AnomalousEvaluator{T,D,D-1}}(undef, length(ano_ids))
        for i in eachindex(ano_ids)
            anevs[i] = AnomalousEvaluator(GF.Gps[ano_ids[i]])
        end
        anoid_to_Gpid = zeros(Int, length(GF.Gps))
        for i in eachindex(ano_ids)
            anoid_to_Gpid[ano_ids[i]] = i
        end

        # svd kernels if requested
        if svd_kernel
            println("  SVD-decompose kernels with cut=$cutoff...")
            for Gp in GF.Gps
                if any(size(Gp.tucker.center) .> 300) @warn "SVD-ing legs of sizes $(size.(Gp.tucker.legs))" end
                size_old = size(Gp.tucker.center)
                svd_kernels!(Gp.tucker; cutoff=cutoff)
                size_new = size(Gp.tucker.center)
                println(" Reduced tucker center from $size_old to $size_new")
            end
        end

        # grid
        R = grid_R(GF)
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        return new{T,D,D-1}(GF, anevs, ano_terms_required, anoid_to_Gpid, grid, localdims)
    end
end

function (fbev::FullCorrBatchEvaluator_MF{T,D,N})(
    leftindexsset::Vector{Vector{Int}}, rightindexsset::Vector{Vector{Int}}, ::Val{M}
    ) where {T,D,N,M}
    error("NYI")
end


function intact(GF::FullCorrelator_MF{D}) where {D}
    return all(intact.(GF.Gps))
end

function precompute_all_values(G :: FullCorrelator_MF{D}) ::Array{ComplexF64,D} where{D}
    @assert intact(G) "TuckerDecomposition has been modified"
    return  sum(gp -> precompute_all_values_MF(gp), G.Gps)
end

"""
Precompute values, including only regular kernel convolutions and frequency rotations
TODO: TEST
"""
function precompute_all_values_MF_noano(
    G::FullCorrelator_MF{D}
    )::Array{ComplexF64,D} where {D}

    @assert all(intact.(G.Gps)) "TuckerDecomposition has been modified"
    return sum(gp -> precompute_all_values_MF_noano(gp), G.Gps)
end

"""
    reduce_Gps!(G_in :: FullCorrelator_MF{D})

Summarize the Gps that essentially have the same kernels, namely p and reverse(p), e.g. (1,2,3,4) and (4,3,2,1).
"""
function reduce_Gps!(G_in :: FullCorrelator_MF{D}) where{D}
    function idx_of_p(p::Vector{Int}, ps::Vector{Vector{Int}}) ::Int
        for (i,x) in enumerate(ps)
            if x == p
                return i
            end
        end
        return -1
    end
    
    function findindepps(ps::Vector{Vector{Int}}) ::Vector{Int}
        N = length(ps)
        is_indep = ones(Int, N) .== 1
        for (i,p) in enumerate(ps)
            ipr = idx_of_p(reverse(p), ps)
            if is_indep[i] && ipr!=-1
                is_indep[ipr] = false
            end
        end
        return collect(1:N)[is_indep]
    end
    
    #using Combinatorics
    #G_in = deepcopy(Gs[1])
    #G_new = deepcopy(G_in)
    ps = G_in.ps
    i_indepps = findindepps(ps)
    Adiscs_new = [G_in.Gps[ip].tucker.center for ip in i_indepps]
    Adiscs_ano_new = [G_in.Gps[ip].Adisc_anoβ for ip in i_indepps]
    for (i,ip) in enumerate(i_indepps)
        ipr = idx_of_p(reverse(ps[ip]), ps)
        if ipr != -1
            Adiscs_new[i] += reverse(permutedims(G_in.Gps[ipr].tucker.center, (reverse(collect(1:D))))) .* (-1)^D
            Adiscs_ano_new[i] += reverse(permutedims(G_in.Gps[ipr].Adisc_anoβ, (reverse(collect(1:D))))) .* (-1)^(D-1)
            @DEBUG all([maxabs(G_in.Gps[ip].ωdiscs[d] + reverse(G_in.Gps[ipr].ωdiscs[D-d+1])) < 1e-10 for d in 1:D]) "Kernels for p and reverse(p) are not equivalent."
            @DEBUG all([maxabs(G_in.Gps[ip].Kernels[d] - G_in.Gps[ipr].Kernels[D-d+1]) < 1e-10 for d in 1:D]) "Kernels for p and reverse(p) are not equivalent."
        end
    end
    G_in.ps = ps[i_indepps]
    G_in.Gp_to_G = G_in.Gp_to_G[i_indepps]
    G_in.Gps = G_in.Gps[i_indepps]
    for i in 1:length(i_indepps)
        G_in.Gps[i].tucker.center = Adiscs_new[i]
        G_in.Gps[i].Adisc_anoβ = Adiscs_ano_new[i]
    end
    return nothing
end



"""
    FullCorrelator_KF{D}

The struct FullCorrelator_KF{D} represents a full Keldysh correlator via the sum of PartialCorrelator_reg objects.
This representation is described in Eqs.(67) in [Kugler et al. PHYS. REV. X 11, 041006 (2021)],
where the *partial* correlators Gp are the contributions from individual permutations p.

# Members:
* name    ::Vector{String}                    : name to distinguish the objects (e.g.: list of operators)
* Gps     ::Vector{PartialCorrelator_reg}     : list of partial correlators
* Gp_to_G ::Vector{Float64}                   : prefactors for Gp's (currently this coefficient is applied to Adisc => no need to apply it during evaluation.)
* GR_to_GK::Array{Float64,3}                  : Matrix of size (D+1, 2^{D+1}) mapping fully-retarded Gp to Keldysh Gp
* ωs_ext  ::NTuple{D,Vector{ComplexF64}}      : external complex frequencies in the chosen parametrization
* ωconvMat::Matrix{Int}                       : Matrix of size (D+1, D) encoding conversion of external frequencies to the frequencies of the D+1 legs ~ ω_ferm = ωconvMat * ω_ext.
                                                Columns must add up to zero.
                                                Allowed matrix entries: -1, 0, 1
* isBos   ::BitVector                         : BitVector indicating which legs are bosonic

# Evaluation
Objects of type FullCorrelator_KF{D} can be evaluated *pointwisely* at frequency indices and Keldysh indices, e.g. for D=2
```julia-repl
julia> evaluate_all_iK(GK, 1, 2)
N-element Vector{ComplexF64}
 ...

julia> evaluate(GK, 1, 2; iK=1)
 0.0
```

To evaluate it at *all* available indices:
```julia-repl
julia> precompute_all_values(GK)
MxNx8 Matrix{ComplexF64}
 ...
```
"""
struct FullCorrelator_KF{D}
    name    ::Vector{String}                    # list of operators to distinguish the objects
    Gps     ::Vector{PartialCorrelator_reg}     # list of partial correlators
    Gp_to_G ::Vector{Float64}                   # prefactors for Gp's (currently this coefficient is applied to Adisc => no need to apply it during evaluation.)
    GR_to_GK::Array{Float64,3}                  # Matrix of size (D+1, 2^{D+1}) mapping fully-retarded Gp to Keldysh Gp

    ωs_ext  ::NTuple{D,Vector{Float64}}      # external complex frequencies in the chosen parametrization
    ωconvMat::Matrix{Int}                       # matrix of size (D+1, D) encoding conversion of external frequencies to the frequencies of the D+1 legs
                                                # columns must add up to zero
    isBos   ::BitVector                         # BitVector indicating which legs are bosonic

    function FullCorrelator_KF(
        path::String, 
        Ops::Vector{String}
        ; 
        T::Float64,
        flavor_idx::Int, 
        ωs_ext::NTuple{D,Vector{Float64}}, 
        ωconvMat::Matrix{Int}, 
        name::String="", 
        sigmak  ::Vector{Float64},              # Sensitivity of logarithmic position of spectral
                                                # weight to z-shift, i.e. |d[log(|omega|)]/dz|. These values will
                                                # be used to broaden discrete data. (\sigma_{ij} or \sigma_k in
                                                # Lee2016.)
        γ       ::Float64,                      # Parameter for secondary linear broadening kernel. (\gamma
                                                # in Lee2016.)
        broadening_kwargs...                    # other broadening kwargs (see documentation for BroadenedPSF)
        ) where{D}
        ##########################################################################
        ############################## check inputs ##############################
        if length(Ops) != D+1
            throw(ArgumentError("Ops must contain "*string(D+1)*" elements."))
        end
        ##########################################################################
        
        print("Loading stuff: ")
        @time begin
        perms = permutations(collect(1:D+1))
        isBos = (o -> o[1] == 'Q' && length(o) == 3).(Ops)
        ωdisc = load_ωdisc(path, Ops)
        Adiscs = [load_Adisc(path, Ops[p], flavor_idx) for (i,p) in enumerate(perms)]
        end
        print("Creating Broadened PSFs: ")
        function get_Acont_p(i, p)
            ωs_int, _, _ = _trafo_ω_args(ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1))
            return BroadenedPSF(ωdisc, Adiscs[i], sigmak, γ; ωconts=(ωs_int...,), broadening_kwargs...)
        end
        @time Aconts = [get_Acont_p(i, p) for (i,p) in enumerate(perms)]

        return FullCorrelator_KF(Aconts; T, isBos, ωs_ext, ωconvMat, name=[Ops; name])
    end


    function FullCorrelator_KF(Aconts::Vector{<:AbstractTuckerDecomp{D}}; T::Float64, isBos::BitVector, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}, name::Vector{String}=[]) where{D}
        ##########################################################################
        ############################## check inputs ##############################
        if size(ωconvMat) != (D+1, D)
            throw(ArgumentError("ωconvMat must have size ("*string(D+1)*", "*string(D)*") but is of size "*string(size(ωconvMat))*"."))
        end
        if maximum(abs.(sum(ωconvMat, dims=1))) != 0
            throw(ArgumentError("The columns in ωconvMat must add up to zero."))
        end
        if length(Aconts) != factorial(D+1)
            throw(ArgumentError("Aconts must contain all "*string(D+1)*"! permutations."))
        end
        if (D+1-sum(isBos))%2 != 0
            throw(ArgumentError("isBos must contain an even number of 'false' (fermions)."))
        end
        ##########################################################################
        function get_GR_to_GK(D) ::Array{Float64,3}
            perms = permutations(collect(1:D+1))
            GR_to_GK = zeros(D+1, (ones(Int, D+1).*2)..., factorial(D+1))
            for (ip, p) in enumerate(perms)
                for idxs in Base.Iterators.product(collect.(axes(GR_to_GK)[2:end-1])...)
                    pidxs = idxs[p]
                    GR_to_GK[:, idxs..., ip] .= (-1).^(1 .+ cumsum(pidxs.-1)) .* (pidxs.-1)
                end
            end
            GR_to_GK = reshape(GR_to_GK, (D+1, 2^(D+1), factorial(D+1)))
            GR_to_GK .*= 2^(-D/2+0.5)
            return GR_to_GK
        end

        print("All the rest: ")
        @time begin
        perms = permutations(collect(1:D+1))
        Gp_to_G = _get_Gp_to_G(D, isBos)
        for (i,sp) in enumerate(Aconts)
            sp.center .*= Gp_to_G[i]
        end
        Gps = [PartialCorrelator_reg(T, "KF", Aconts[i], ntuple(i->ωs_ext[i], D), cumsum(ωconvMat[p[1:D],:], dims=1)) for (i,p) in enumerate(perms)]        
        GR_to_GK = get_GR_to_GK(D)
        end
        return new{D}(name, Gps, Gp_to_G, GR_to_GK, ωs_ext, ωconvMat, isBos)
    end
end


"""
    evaluate_all_iK(G::FullCorrelator_KF{D}, idx::Vararg{Int,D})

Evaluate Keldysh correlator at indices idx. Returns all Keldysh components in a vector.
iK ∈ 1:2^D is the linear index for the 2x...x2 Keldysh structure.

"""
function evaluate_all_iK(G::FullCorrelator_KF{D}, idx::Vararg{Int,D}) where{D}
    #eval_gps(gp) = evaluate_with_ωconversion_KF(gp, idx...)
    #Gp_values = eval_gps.(G.Gps)
    result = transpose(evaluate_with_ωconversion_KF(G.Gps[1], idx...)) * view(G.GR_to_GK, :, :, 1)# .* G.Gp_to_G[1]
    for i in 2:length(G.Gps)
        println("i: ", i)
        result .+= transpose(evaluate_with_ωconversion_KF(G.Gps[i], idx...)) * view(G.GR_to_GK, :, :, i)# .* G.Gp_to_G[i]
    end
    return result
    #return mapreduce(gp -> evaluate_with_ωconversion_KF(gp, idx...)' * G.GR_to_GK, +, G.Gps)
end


"""
    evaluate(G::FullCorrelator_KF{D}, idx::Vararg{Int,D}; iK::Int)

Evaluate Keldysh correlator at indices idx and Keldysh component iK.
iK ∈ 1:2^D is the linear index for the 2x...x2 Keldysh structure.
"""
function evaluate(G::FullCorrelator_KF{D},  idx::Vararg{Int,D}; iK::Int) where{D}
    #eval_gps(gp) = evaluate_with_ωconversion_KF(gp, idx...)
    #Gp_values = eval_gps.(G.Gps)
    result = transpose(evaluate_with_ωconversion_KF(G.Gps[1], idx...)) * G.GR_to_GK[:, iK, 1]# .* G.Gp_to_G[1]
    for i in 2:length(G.Gps)
        result += transpose(evaluate_with_ωconversion_KF(G.Gps[i], idx...)) * G.GR_to_GK[:, iK, i]# .* G.Gp_to_G[i]
    end
    return result
    #return mapreduce(gp -> evaluate_with_ωconversion_KF(gp, idx...)' * G.GR_to_GK, +, G.Gps)
end


function precompute_all_values(G :: FullCorrelator_KF{D}) where{D}
    
#    @assert false # Not tested at all!
    p = 1
    temp = precompute_all_values_KF(G.Gps[p])
    result = reshape(temp, (prod(size(temp)[1:end-1]),D+1)) * view(G.GR_to_GK, :, :, p)
    for p in 2:length(G.Gps)
        temp = precompute_all_values_KF(G.Gps[p])
        result += reshape(temp, (prod(size(temp)[1:end-1]),D+1)) * view(G.GR_to_GK, :, :, p)
    end
    return reshape(result, (length.(G.ωs_ext)..., (2*ones(Int, D+1))...))
end


function propagate_ωs_ext!(G::Union{FullCorrelator_MF{D}, FullCorrelator_MF{D}}, ωs_ext_new::NTuple{D,Vector{Float64}}=G.ωs_ext) where{D}
    G.ωs_ext = ωs_ext_new
    for (ip, Gp) in enumerate(G.Gps)
        G.Gps[ip].ωs_ext = ωs_ext_new
        update_frequency_args!(G.Gps[ip])
    end
    return nothing
end




function get_GR(
    path::String, 
    Ops::Vector{String}
    ; 
    T::Float64,
    flavor_idx::Int, 
    ωs_ext::Vector{Float64} ,
    sigmak  ::Vector{Float64},              
    γ       ::Float64,                      
    broadening_kwargs...                    
    ) #where{D}
    ##########################################################################
    ############################## check inputs ##############################
    if length(Ops) != 2
        throw(ArgumentError("Ops must contain "*string(2)*" elements."))
    end
    ##########################################################################
    
    print("Loading stuff: ")
    @time begin
    #perms = permutations(collect(1:D+1))
    isBos = (o -> o[1] == 'Q' && length(o) == 3).(Ops)
    @assert all(isBos) || all(.!isBos)
    isBos = isBos[1]
    ωdisc = load_ωdisc(path, Ops)
    Adisc = load_Adisc(path, Ops, flavor_idx) + reverse(load_Adisc(path, reverse(Ops), flavor_idx)) * (isBos ? -1 : 1)
    end
    #print("Creating Broadened PSFs: ")
    #function get_Acont_p(i, p)
    #    ωs_int, _, _ = _trafo_ω_args(ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1))
    #    return BroadenedPSF(ωdisc, Adiscs[i], sigmak, γ; ωconts=(ωs_int...,), broadening_kwargs...)
    #end
    #@time Aconts = [get_Acont_p(i, p) for (i,p) in enumerate(perms)]
    _, Acont = getAcont(ωdisc, reshape(Adisc, length(Adisc), 1), sigmak, γ; ωcont = ωs_ext, broadening_kwargs...)
    return -im * π * hilbert_fft(Acont; dims=1)[:]
end