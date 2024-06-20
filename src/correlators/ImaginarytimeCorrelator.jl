#=
Compute partial and full correlators in imaginary time domain
=#

"""
Θ(τ,ϵ), the Matsubara summed 1D regular kernel, bosonic case
"""
function realtimeKernel1D_bos()
  
end

"""
Θ(τ,ϵ), the Matsubara summed 1D regular kernel, fermionic case
"""
function realtimeKernel1D_fer()
  
end

"""
Get MPS of a PSF (no broadening)
"""
function compute_Adisc_MPS(path::String, Ops::Vector{String}, flavor_idx::Int; tagname="eps", padding_R=nothing, nested_ωdisc=false, kwargs...)
    Adisc = load_Adisc(path, Ops, flavor_idx)
    # load frequencies as well for compatification
    ωdisc = load_ωdisc(path, Ops; nested_ωdisc=nested_ωdisc)
    _, ωdisc, Adisc = compactAdisc(ωdisc, Adisc)
    R = if isnothing(padding_R)
            maximum(trunc.(Int, log2.(collect(size(Adisc)))) .+ 1)
        else
            padding_R
        end
    
    A_qtt, _, _ = quanticscrossinterpolate(TCI4Keldysh.zeropad_array(Adisc, R); kwargs...)  
    Adisc_mps = TCI4Keldysh.QTCItoMPS(A_qtt, ntuple(i->"$(tagname)$i", ndims(Adisc)))

    return Adisc_mps
end