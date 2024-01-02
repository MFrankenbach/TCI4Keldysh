struct FullCorrelator_MF{D}
    name    ::Vector{String}                    # list of operators to distinguish the objects
    Gps     ::Vector{PartialCorrelator_reg}     # list of partial correlators
    Gp_to_G ::Vector{Float64}                   # prefactors for Gp's (currently this coefficient is applied to Adisc => no need to apply it during evaluation.)

    ωs_ext  ::NTuple{D,Vector{ComplexF64}}      # external complex frequencies in the chosen parametrization
    ωconvMat::Matrix{Int}                       # matrix of size (D+1, D) encoding conversion of external frequencies to the frequencies of the D+1 legs
                                                # columns must add up to zero
                                                # allowed matrix entries: -1, 0, 1
    isBos   ::BitVector                         # BitVector indicating which legs are bosonic

    function FullCorrelator_MF(path::String, Ops::Vector{String}; flavor_idx::Int, ωs_ext::NTuple{D,Vector{ComplexF64}}, ωconvMat::Matrix{Int}, name::String="") where{D}
        ##########################################################################
        ############################## check inputs ##############################
        if length(Ops) != D+1
            throw(ArgumentError("Ops must contain "*string(D+1)*" elements."))
        end
        ###################################################################
        
        perms = permutations(collect(1:D+1))
        isBos = (o -> length(o) == 3).(Ops)
        ωdisc = load_ωdisc(path, Ops)
        Adiscs = [load_Adisc(path, Ops[p], flavor_idx) for (i,p) in enumerate(perms)]

        return FullCorrelator_MF(Adiscs, ωdisc; isBos, ωs_ext, ωconvMat, name=[Ops; name])
    end


    function FullCorrelator_MF(Adiscs::Vector{Array{Float64,D}}, ωdisc::Vector{Float64}; isBos::BitVector, ωs_ext::NTuple{D,Vector{ComplexF64}}, ωconvMat::Matrix{Int}, name::Vector{String}=[]) where{D}
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
        Gp_to_G = get_Gp_to_G(D, isBos)
        Gps = [PartialCorrelator_reg("MF", Gp_to_G[i].*Adiscs[i], ωdisc, ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1)) for (i,p) in enumerate(perms)]        

        return new{D}(name, Gps, Gp_to_G, ωs_ext, ωconvMat)
    end
end


function get_Gp_to_G(D::Int, isBos::BitVector) ::Vector{Float64}
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


function precompute_all_values(G :: FullCorrelator_MF{D}) ::Array{ComplexF64,D} where{D}
    #result = precompute_all_values(G.Gps[1]) .* G.Gp_to_G[1]
    #for i in 2:length(G.Gps)
    #    result .+= precompute_all_values(G.Gps[i]) .* G.Gp_to_G[i]
    #end
    #return result
    return  sum(gp -> precompute_all_values(gp), G.Gps)
end


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
        isBos = (o -> o[1] == 'Q').(Ops)
        ωdisc = load_ωdisc(path, Ops)
        Adiscs = [load_Adisc(path, Ops[p], flavor_idx) for (i,p) in enumerate(perms)]
        end
        print("Creating Broadened PSFs: ")
        function get_Acont_p(i, p)
            ωs_int, _, _ = trafo_ω_args(ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1))
            return BroadenedPSF(ωdisc, Adiscs[i], sigmak, γ; ωconts=ωs_int, broadening_kwargs...)
        end
        @time Aconts = [get_Acont_p(i, p) for (i,p) in enumerate(perms)]

        return FullCorrelator_KF(Aconts; isBos, ωs_ext, ωconvMat, name=[Ops; name])
    end


    function FullCorrelator_KF(Aconts::Vector{BroadenedPSF{D}}; isBos::BitVector, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}, name::Vector{String}=[]) where{D}
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
        Gp_to_G = get_Gp_to_G(D, isBos)
        for (i,sp) in enumerate(Aconts)
            sp.Adisc .*= Gp_to_G[i]
        end
        Gps = [PartialCorrelator_reg("KF", Aconts[i], ntuple(i->ωs_ext[i] .+0im, D), cumsum(ωconvMat[p[1:D],:], dims=1)) for (i,p) in enumerate(perms)]        
        GR_to_GK = get_GR_to_GK(D)
        end
        return new{D}(name, Gps, Gp_to_G, GR_to_GK, ωs_ext, ωconvMat, isBos)
    end
end


function evaluate(G::FullCorrelator_KF{D}, idx::Vararg{Int,D}) where{D}
    #eval_gps(gp) = evaluate_with_ωconversion_KF(gp, idx...)
    #Gp_values = eval_gps.(G.Gps)
    result = evaluate_with_ωconversion_KF(G.Gps[1], idx...)' * view(G.GR_to_GK, :, :, 1)# .* G.Gp_to_G[1]
    for i in 2:length(G.Gps)
        println("i: ", i)
        result .+= evaluate_with_ωconversion_KF(G.Gps[i], idx...)' * view(G.GR_to_GK, :, :, i)# .* G.Gp_to_G[i]
    end
    return result
    #return mapreduce(gp -> evaluate_with_ωconversion_KF(gp, idx...)' * G.GR_to_GK, +, G.Gps)
end


function evaluate(G::FullCorrelator_KF{D}, iK::Int,  idx::Vararg{Int,D}) where{D}
    #eval_gps(gp) = evaluate_with_ωconversion_KF(gp, idx...)
    #Gp_values = eval_gps.(G.Gps)
    result = evaluate_with_ωconversion_KF(G.Gps[1], idx...)' * G.GR_to_GK[:, iK, 1]# .* G.Gp_to_G[1]
    for i in 2:length(G.Gps)
        result += evaluate_with_ωconversion_KF(G.Gps[i], idx...)' * G.GR_to_GK[:, iK, i]# .* G.Gp_to_G[i]
    end
    return result
    #return mapreduce(gp -> evaluate_with_ωconversion_KF(gp, idx...)' * G.GR_to_GK, +, G.Gps)
end