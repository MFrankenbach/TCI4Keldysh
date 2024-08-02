#=
Compute interection vertex with symmetric improved estimators and TCI
=#

"""
Evaluate self-energy pointwise by symmetric improved estimator.
"""
struct SigmaEvaluator_MF{D}
    G_QQ::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    G_QF::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    G_FQ::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    G::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    Σ_H::Float64
    ωconvMat::Matrix{Int}
    ωconvOff::Vector{Int}

    function SigmaEvaluator_MF(
        G_QQ_::FullCorrelator_MF{1},
        G_QF_::FullCorrelator_MF{1},
        G_FQ_::FullCorrelator_MF{1},
        G_::FullCorrelator_MF{1},
        Σ_H::Float64,
        ωconvMat::Matrix{Int};
        )

        G_QQ = FullCorrEvaluator_MF(G_QQ_, true; cutoff=1.e-12)
        G_QF = FullCorrEvaluator_MF(G_QF_, true; cutoff=1.e-12)
        G_FQ = FullCorrEvaluator_MF(G_FQ_, true; cutoff=1.e-12)
        G = FullCorrEvaluator_MF(G_, true; cutoff=1.e-12)

        @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"

        D = size(ωconvMat, 2)
        Nfer = length(only(G_.ωs_ext))
        return new{D}(G_QQ, G_QF, G_FQ, G, Σ_H, ωconvMat, freq_shift_rot(ωconvMat, div(Nfer,2)))
    end
end

function SigmaEvaluator_MF(PSFpath::String, R::Int, T::Float64, ωconvMat::Matrix{Int}; flavor_idx::Int=1)
    
    # need twice the grid size of the external grids
    ω_fer = MF_grid(T, 2^R, true)
    # TODO: Do required operators depend on flavor_idx here
    G = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QF = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_FQ = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");

    # TODO: IS THIS CORRECT
    Adisc_Σ_H = load_Adisc_0pt(PSFpath, "Q12", flavor_idx)
    Σ_H = only(Adisc_Σ_H)

    return SigmaEvaluator_MF(G_QQ, G_QF, G_FQ, G, Σ_H, ωconvMat)
end

"""
Frequency shift for frequency transform with one-based indices.
Assume that first external grid is bosonic, others are fermionic and that internal frequencies are all fermionic.
* Nfer : Number of external fermionic frequencies. Internal grids then have size 2*Nfer.
"""
function freq_shift_rot(ωconvMat::Matrix{Int}, Nfer::Int)
    D = size(ωconvMat, 2)
    @assert size(ωconvMat, 1)==D+1
    # trafo maps new -> old   
    corner_new = ntuple(i -> i==1 ? -Nfer : -Nfer + 1, D)
    corner_old = ωconvMat * collect(corner_new)
    @assert all(abs.(corner_old) .<= 2*Nfer - 1) "invalid frequency transformation"
    corner_old_idx = [div(c+2*Nfer, 2) + 1 for c in corner_old]

    idx_new = ones(Int, D)
    idx_old = ωconvMat * idx_new
    return corner_old_idx .- idx_old
end

function (sev::SigmaEvaluator_MF{D})(row::Int, w::Vararg{Int,D}) where {D}
    w_int = dot(sev.ωconvMat[row,:], SA[w...]) + sev.ωconvOff[row]
    # G_QQ .+ Σ_H .- G_QF ./ G .* G_FQ
    return sev.G_QQ(w_int) + sev.Σ_H - (sev.G_QF(w_int) / sev.G(w_int)) * sev.G_FQ(w_int)
end


function Γ_core_TCI_MF(
    PSFpath::String,
    R::Int;
    ωconvMat::Matrix{Int},
    T::Float64,
    flavor_idx::Int=1, # why not Vector{Int}?
    qtcikwargs...
)

    # make frequency grid
    D = size(ωconvMat, 2)
    Nhalf = 2^(R-1)
    ombos = MF_grid(T, Nhalf, false)
    omfer = MF_grid(T, Nhalf, true)
    ωs_ext = ntuple(i -> i>1 ? omfer : ombos, D)

    # all 16 4-point correlators
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    is_incoming = (false, true, false, true)

    GFs = Vector{FullCorrelator_MF}(undef, length(letter_combinations))

    # create correlator objects
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for (l,letts) in enumerate(letter_combinations)
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_MF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat);
    end

    # create full correlator evaluators
    kwargs_dict = Dict(qtcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    GFevs = [FullCorrEvaluator_MF(GF, true; cutoff=cutoff) for GF in GFs]

    # create self-energy evaluator
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    sev = SigmaEvaluator_MF(PSFpath, R, T, incoming_trafo * ωconvMat; flavor_idx=flavor_idx)

    # function to interpolate
    function eval_Γ_core(w::Vararg{Int,3})
        ret = zero(ComplexF64)
        for (i, letts) in enumerate(letter_combinations)
            addval = GFevs[i](w...)
            for il in eachindex(letts)
                if letts[il]=='F'
                    addval *= -sev(il, w...)
                end
            end
            ret += addval
        end
        return ret
    end

    qtt, _, _ = quanticscrossinterpolate(ComplexF64, eval_Γ_core, ntuple(i -> 2^R, D); qtcikwargs...)

    return qtt
end