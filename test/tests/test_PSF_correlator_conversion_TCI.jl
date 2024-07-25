using ITensors
using QuanticsGrids
"""
Test Partial Spectral Function → Correlator conversion @ TCI
"""

# whether to test 4-point correlators (takes time...)
__DO_FOURPOINT() = false

@testset "PSF -> Partial Correlator" begin

    ITensors.disable_warn_order()
    
    function test_TD_to_MPS_via_TTworld(npt::Int, perm_idx::Int)
        
        # load data
        R = npt==2 ? 13 : 8
        beta = npt<4 ? 1.e3 : 10.0
        GFs = TCI4Keldysh.dummy_correlator(npt, R; beta=beta)

        # pick PSF
        spin = 1
        Gp = GFs[spin].Gps[perm_idx]

        # compute reference data for selected PSF
        data_unrotated = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)

        # TCI computation
        printstyled(" ---- Compressing $npt-point function...\n"; color=:blue)
        tolerance = npt < 4 ? 1.e-10 : 1.e-8
        cutoff = npt < 4 ? 1.e-25 : 1.e-20
        Gp_mps = TCI4Keldysh.TD_to_MPS_via_TTworld(Gp.tucker; tolerance=tolerance, cutoff=cutoff)

        # compare
        tags = ntuple(i -> "ω$i", npt-1)
        fatGpTCI = TCI4Keldysh.MPS_to_fatTensor(Gp_mps; tags=tags)
        unrotated_slice = [Gp.isFermi[i] ? Colon() : 1:size(data_unrotated, i)-1 for i in eachindex(Gp.isFermi)]
        TCIslice = Base.OneTo.(ntuple(i -> min(2^TCI4Keldysh.grid_R(size(Gp.tucker.legs[i],1)), size(data_unrotated[unrotated_slice...],i)), ndims(data_unrotated)))
        diff = (fatGpTCI[TCIslice...] - data_unrotated[unrotated_slice...]) / maximum(abs.(data_unrotated))

        maxdiff = maximum(abs.(diff)) 
        printstyled(" ---- Max error $maxdiff for tol=$tolerance, cut=$cutoff, npt=$npt\n"; color=:blue)

        # we have higher accuracy for higher beta
        test_tol = npt < 4 ? 5*1.e2 * tolerance : 1.e2*tolerance
        return maxdiff < test_tol
    end

    """
    Test non-anomalous rotated partial correlators.
    """
    function test_TCI_precompute_reg_values_rotated(npt::Int, perm_idx::Int)
        
        # load data
        R = npt==2 ? 13 : 8
        beta = npt<4 ? 1.e3 : 10.0
        GFs = TCI4Keldysh.dummy_correlator(npt, R; beta=beta)
        Gp = GFs[1].Gps[perm_idx]

        # TCI computation
        printstyled(" ---- Computing $npt-point function...\n"; color=:blue)
        tolerance = npt < 4 ? 1.e-10 : 1.e-8
        cutoff = npt < 4 ? 1.e-25 : 1.e-20
        Gmps = TCI4Keldysh.TCI_precompute_reg_values_rotated(Gp; tolerance=tolerance, cutoff=cutoff, include_ano=false)
        fatGp = TCI4Keldysh.MPS_to_fatTensor(Gmps; tags=ntuple(i -> "ω$i", npt-1))

        # reference data
        ref_data = TCI4Keldysh.precompute_all_values_MF_noano(Gp)
        diffslice = ntuple(i -> (i==1 ? (2:2^R) : 1:2^R), npt-1)

        diff = (ref_data[diffslice...] - fatGp[diffslice...]) / maximum(abs.(ref_data))
        maxdiff = maximum(abs.(diff))
        printstyled(" ---- Max error $maxdiff for tol=$tolerance, cut=$cutoff, npt=$npt\n"; color=:blue)

        # we have higher accuracy for higher beta
        test_tol = npt < 4 ? 5*1.e2 * tolerance : 1.e3*tolerance
        return maxdiff < test_tol
    end

    """
    2/3D anomalous Matsubara term
    """
    function test_anomalous_TD_to_MPS_23D(npt::Int, perm_idx::Int)
        beta = 1.e1
        R = 8
        GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=beta)

        # TCI computation
        tolerance = 1.e-8
        Gp = GF.Gps[perm_idx]
        cutoff = 1.e-20
        if !TCI4Keldysh.ano_term_required(Gp)
            # nothing to test
            return true
        end
        Gano_mps = TCI4Keldysh.anomalous_TD_to_MPS(Gp; tolerance=tolerance, cutoff=cutoff)

        bos_idx = findfirst(.!Gp.isFermi)
        grid_ids = [i for i in 1:npt-1 if i!=bos_idx]
        Gano_fat = TCI4Keldysh.MPS_to_fatTensor(Gano_mps; tags=ntuple(i -> "ω$(grid_ids[i])", npt-2))

        # reference
        reg_values = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
        full_values = TCI4Keldysh.precompute_all_values_MF_without_ωconv(Gp)
        ωbos = Gp.tucker.ωs_legs[bos_idx] 
        zero_inds = [i for i in eachindex(ωbos) if abs(ωbos[i]) < 1.e-10]
        ano_values = (full_values .- reg_values)[ntuple(i -> i==bos_idx ? only(zero_inds) : Colon(), npt-1)...]

        diff = (ano_values .- Gano_fat[Base.OneTo.(size(ano_values))...]) ./ maximum(abs.(ano_values))
        maxdiff = maximum(abs.(diff))
        test_tol = 50*tolerance
        printstyled(" ---- Max error $maxdiff for tol=$tolerance, cut=$cutoff, npt=$npt\n"; color=:blue)
        return maxdiff < test_tol
    end

    """
    1D anomalous Matsubara term
    """
    function test_anomalous_TD_to_MPS_full_1D()
        R = 8
        GF = TCI4Keldysh.multipeak_correlator_MF(2, R)
        Gp = GF.Gps[1]

        tolerance = 1.e-10
        cutoff = 1.e-20
        if !TCI4Keldysh.ano_term_required(Gp)
            # nothing to test
            return
        end
        Gano_mps = TCI4Keldysh.anomalous_TD_to_MPS_full(Gp; tolerance=tolerance, cutoff=cutoff)

        bos_idx = TCI4Keldysh.get_bosonic_idx(Gp)
        ano_val = only(Gp.Adisc_anoβ)
        zero_idx = findfirst(x -> abs(x)<=1.e-15, Gp.tucker.ωs_legs[bos_idx])
        state = QuanticsGrids.index_to_quantics(zero_idx; numdigits=R, base=2)
        @test length(Gano_mps)==R
        @test isapprox(ano_val, TCI4Keldysh.eval(Gano_mps, state); atol=1.e-14)
    end

    """
    2/3D anomalous term embedded in D dimensions
    """
    function test_anomalous_TD_to_MPS_full_23D(npt::Int, perm_idx::Int)
        @assert npt>2
        R = 7
        GF = TCI4Keldysh.multipeak_correlator_MF(npt, R)
        Gp = GF.Gps[perm_idx]

        tolerance = 1.e-10
        cutoff = 1.e-20
        bos_idx = TCI4Keldysh.get_bosonic_idx(Gp)
        if !TCI4Keldysh.ano_term_required(Gp)
            printstyled("\n  ANOMALOUS: nothing to test\n"; color=:gray)
            return
        end
        Gano_mps = TCI4Keldysh.anomalous_TD_to_MPS_full(Gp; tolerance=tolerance, cutoff=cutoff)

        # check zero outside state
        perm = collect(1:(npt-1))
        perm[1] = bos_idx
        perm[bos_idx] = 1
        R = div(length(Gano_mps), npt-1)
        zero_idx = findfirst(x -> abs(x)<=1.e-15, Gp.tucker.ωs_legs[bos_idx])
        state = QuanticsGrids.index_to_quantics(zero_idx; numdigits=R, base=2)
        for _ in 1:20
            v1 = rand([1,2], R)
            v2 = rand([1,2], R)
            nonstate = rand([1,2], R)
            if nonstate==state
                nonstate[1] = (x -> x==1 ? 2 : 1)(state[1])
            end
            to_interleave = npt==3 ? [nonstate, v1] : [nonstate, v1, v2]
            vtot = QuanticsGrids.interleave_dimensions(to_interleave[perm]...)
            @test isapprox(0.0, TCI4Keldysh.eval(Gano_mps, vtot); atol=1.e-14)
        end

        # check nonzero
        Gano_lowdim = TCI4Keldysh.anomalous_TD_to_MPS(Gp; tolerance=tolerance, cutoff=cutoff)
        # find max value
        biniter = TCI4Keldysh.iterate_binvec((npt-2)*R)
        max_ = 0.0
        for v in biniter
            max_ = max(max_, abs(TCI4Keldysh.eval(Gano_lowdim, collect(v))))
        end
        perm = collect(1:(npt-1))
        perm[1] = bos_idx
        perm[bos_idx] = 1
        for _ in 1:20
            v1 = rand([1,2], R)
            v2 = rand([1,2], R)
            to_interleave = npt==3 ? [state, v1] : [state, v1, v2]
            vtot = QuanticsGrids.interleave_dimensions(to_interleave[perm]...)
            val = TCI4Keldysh.eval(Gano_mps, vtot)
            vref = QuanticsGrids.interleave_dimensions(deleteat!(to_interleave[perm], bos_idx)...)
            refval = TCI4Keldysh.eval(Gano_lowdim, vref)
            @test isapprox(0.0, (refval-val) / max_; atol=1.e1*tolerance)
        end
        printstyled("\n  PASSED\n"; color=:green)
    end

    @test test_TD_to_MPS_via_TTworld(2,1)
    @test test_TD_to_MPS_via_TTworld(2,2)
    @test test_TD_to_MPS_via_TTworld(3,1)
    @test test_TD_to_MPS_via_TTworld(3,3)
    if __DO_FOURPOINT()
        @test test_TD_to_MPS_via_TTworld(4,1)
        @test test_TD_to_MPS_via_TTworld(4,4)
    end

    @test test_TCI_precompute_reg_values_rotated(2,2)
    @test test_TCI_precompute_reg_values_rotated(3,1)
    @test test_TCI_precompute_reg_values_rotated(3,2)
    @test test_TCI_precompute_reg_values_rotated(3,3)
    if __DO_FOURPOINT()
        @test test_TCI_precompute_reg_values_rotated(4,1)
        @test test_TCI_precompute_reg_values_rotated(4,3)
        @test test_TCI_precompute_reg_values_rotated(4,23)
    end

    @test test_anomalous_TD_to_MPS_23D(3, 1)
    @test test_anomalous_TD_to_MPS_23D(3, 2)
    @test test_anomalous_TD_to_MPS_23D(3, 3)
    if __DO_FOURPOINT()
        @test test_anomalous_TD_to_MPS_23D(4, 1)
        @test test_anomalous_TD_to_MPS_23D(4, 15)
        @test test_anomalous_TD_to_MPS_23D(4, 4)
    end

    test_anomalous_TD_to_MPS_full_1D()
    test_anomalous_TD_to_MPS_full_23D(3, 1)
    test_anomalous_TD_to_MPS_full_23D(3, 3)
    if __DO_FOURPOINT()
        test_anomalous_TD_to_MPS_full_23D(4, 6)
        test_anomalous_TD_to_MPS_full_23D(4, 2)
        test_anomalous_TD_to_MPS_full_23D(4, 3)
    end
end

@testset "Partial -> Full Correlator" begin
    
    function test_FullCorrelator_add()
        N = 20
        s1 = siteinds(2, N)
        Gp1 = random_mps(Float64, s1)
        s2 = siteinds(2, N)
        Gp2 = random_mps(Float64, s2)
        s3 = siteinds(2, N)
        Gp3 = random_mps(Float64, s3)
        Gpfull = TCI4Keldysh.FullCorrelator_add([Gp1, Gp2, Gp3]; cutoff=1.e-20)

        for _ in 1:20
            v = rand([1,2], N)
            fval = TCI4Keldysh.eval(Gpfull, v)
            refval = TCI4Keldysh.eval(Gp1, v) + TCI4Keldysh.eval(Gp2, v) + TCI4Keldysh.eval(Gp3, v)
            if !isapprox(fval, refval; atol=1.e-10)
                return false
            end
        end
        return true
    end

    function test_FullCorrelator_recompress()
        N = 20
        s1 = siteinds(2, N)
        Gp1 = random_mps(Float64, s1)
        s2 = siteinds(2, N)
        Gp2 = random_mps(Float64, s2)
        Gpfull = TCI4Keldysh.FullCorrelator_recompress([Gp1, Gp2]; tolerance=1.e-12)

        for _ in 1:20
            v = rand([1,2], N)
            fval = TCI4Keldysh.eval(Gpfull, v)
            refval = TCI4Keldysh.eval(Gp1, v) + TCI4Keldysh.eval(Gp2, v)
            if !isapprox(fval, refval; atol=1.e-9)
                return false
            end
        end
        return true
    end

    function test_full_correlator(npt::Int; include_ano=true)

        ITensors.disable_warn_order()

        R = npt==2 ? 13 : 8
        beta = npt<4 ? 1.e3 : 10.0
        tolerance = npt < 4 ? 1.e-10 : 1.e-8
        cutoff = npt < 4 ? 1.e-25 : 1.e-20
        GFs = TCI4Keldysh.dummy_correlator(npt, R; beta=beta)
        spin = 1

        Gps_out = Vector{MPS}(undef, factorial(npt))
        for perm_idx in 1:factorial(npt)
            Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GFs[spin].Gps[perm_idx];
                                    tolerance=tolerance, cutoff=cutoff, include_ano=include_ano)
            Gps_out[perm_idx] = Gp_mps
            printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
        end

        Gfull = TCI4Keldysh.FullCorrelator_add(Gps_out; cutoff=1.e-18, use_absolute_cutoff=false)
        @show TCI4Keldysh.rank(Gfull)

        Gfull_fat = TCI4Keldysh.MPS_to_fatTensor(Gfull; tags=ntuple(i -> "ω$i", npt-1))

        # reference
        Gfull_ref = if include_ano
                        TCI4Keldysh.precompute_all_values(GFs[spin])
                    else
                        TCI4Keldysh.precompute_all_values_MF_noano(GFs[spin])
                    end
        diffslice = ntuple(i -> (i==1 ? (2:2^R) : 1:2^R), npt-1)
        diff = abs.(Gfull_fat[diffslice...] - Gfull_ref[diffslice...]) / maximum(abs.(Gfull_ref))
        maxdiff = maximum(diff)
        test_tol = npt < 4 ? 1.e3 * tolerance : 50.0*tolerance
        printstyled(" ---- Max error $maxdiff for tol=$tolerance, cut=$cutoff, npt=$npt\n"; color=:blue)
        return maxdiff < test_tol
    end

    @test test_FullCorrelator_recompress()
    @test test_FullCorrelator_add()
    @test test_full_correlator(2)
    @test test_full_correlator(3)
    if __DO_FOURPOINT()
        @test test_full_correlator(4; include_ano=false)
        @test test_full_correlator(4; include_ano=true)
    end
end