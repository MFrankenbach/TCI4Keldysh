struct FullCorrelator_MF{D}
    name    ::Vector{String}                    # list of operators to distinguish the objects
    Gps     ::Vector{PartialCorrelator_reg}     # list of partial correlators
    Gp_to_G ::Matrix{Float64}                   # Matrix mapping Gp values to 

    ωs_ext  ::NTuple{D,Vector{ComplexF64}}      # external complex frequencies in the chosen parametrization
    ωconvMat::Matrix{Int}                       # matrix of size (D+1, D) encoding conversion of external frequencies to the frequencies of the D+1 legs
                                                # columns must add up to zero
    isBos   ::BitVector                         # BitVector indicating which legs are bosonic

    function FullCorrelator_MF(path::String, Ops::Vector{String}, flavor_idx::Int, ωs_ext::NTuple{D,Vector{ComplexF64}}, ωconvMat::Matrix{Int}; name::String="") where{D}
        ##########################################################################
        ############################## check inputs ##############################
        if length(Ops) != D+1
            throw(ArgumentError("Ops must contain "*string(D+1)*" elements."))
        end
        ###################################################################
        
        perms = permutations(collect(1:D+1))
        isBos = (o -> o[1] == 'Q').(Ops)
        ωdisc = load_ωdisc(path, Ops)
        Adiscs = [load_Adisc(path, Ops[p], flavor_idx) for (i,p) in enumerate(perms)]

        return FullCorrelator_MF(Adiscs, ωdisc, isBos, ωs_ext, ωconvMat; name=[Ops; name])
    end


    function FullCorrelator_MF(Adiscs::Vector{Array{Float64,D}}, ωdisc::Vector{Float64}, isBos::BitVector, ωs_ext::NTuple{D,Vector{ComplexF64}}, ωconvMat::Matrix{Int}; name::Vector{String}=[]) where{D}
        ##########################################################################
        ############################## check inputs ##############################
        if size(ωconvMat) != (D+1, D)
            throw(ArgumentError("ωconvMat must have size ("*string(D+1)*", "*string(D)*") but is of size "*string(size(ωconvMat))*"."))
        end
        if maximum(abs.(sum(ωconvMat, dims=1))) != 0
            throw(ArgumentError("The columns in ωconvMat must add up to zero."))
        end
        ##########################################################################


        N_fermions = D + 1 - sum(isBos)
        i_Fer = zeros(Int, D+1)
        i_Fer[.!isBos] .= 1:N_fermions
        
        Gp_to_G = zeros(1, factorial(D+1))
        perms = permutations(collect(1:D+1))
        for (i,p) in enumerate(perms)
            order_of_fermions = i_Fer[p][i_Fer[p].>0]
            ζ = (-1)^parity(order_of_fermions)
            Gp_to_G[:,i] .= ζ
        end

        Gps = [PartialCorrelator_reg(Adiscs[i], ωdisc, ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1)) for (i,p) in enumerate(perms)]        

        return new{D}(name, Gps, Gp_to_G, ωs_ext, ωconvMat)
    end
end


function evaluate(G::FullCorrelator_MF{D}, idx::Vararg{Int,D}) where{D}
    eval_gps(gp) = gp(idx...)
    Gp_values = eval_gps.(G.Gps)
    return G.Gp_to_G * Gp_values
end

function (G::FullCorrelator_MF{D})(idx::Vararg{Int,D}) where{D}
    return evaluate(G, idx...)[1]
end


function precompute_all_values(G :: FullCorrelator_MF{D}) ::Array{ComplexF64,D} where{D}
    result = precompute_all_values(G.Gps[1]) .* G.Gp_to_G[1]
    #println("typeof result: ", typeof(result))
    for i in 2:length(G.Gps)
        result .+= precompute_all_values(G.Gps[i]) .* G.Gp_to_G[i]
    end
    return result # mapreduce(gp -> precompute_all_values(gp), +, G.Gps)
end