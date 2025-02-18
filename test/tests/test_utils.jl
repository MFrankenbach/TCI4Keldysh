using SpecialFunctions
using Interpolations
using StaticArrays

@testset "Hilbert transform" begin
    
    function maxdev_in_hilbertTrafo_Gauss(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        xs_out[end] = x_in_max
        xm_half = x_in_max/10.0
        # Gaussian
        _to_transform = x -> exp(-(x/xm_half)^2)
        ys = _to_transform.(xs)
        res = TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys)
        ht = imag.(res)
        # cf. Wikipedia
        expected = (x -> 2/sqrt(pi) * dawson(x/xm_half)).(xs_out)

        return maximum(abs.(ht .- expected))
    end

        
    function maxdev_in_hilbertTrafo_sinc(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        ys = sin.(xs) ./ xs
        ht = imag.(TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys))
        expected = (x -> (1. - cos(x)) / x).(xs_out)#  * π

        return maximum(abs.(ht - expected))

    end

    function maxdev_in_hilbertTrafo_rat(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        ϵ = 2.
        ys = (x -> 1. / ((x-ϵ)^2 + 1.)).(xs)
        ht = imag.(TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys))
        expected = (x -> (x - ϵ) / ((x-ϵ)^2 + 1.)).(xs_out)#  * π

        return maximum(abs.(ht - expected))

    end


    function maxdev_in_hilbertTrafo_sinc_fft(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = xs
        ys = sin.(xs) ./ xs
        ht = imag.(TCI4Keldysh.hilbert_fft(ys))
        expected = (x -> (1. - cos(x)) / x).(xs_out)

        return maximum(abs.(ht - expected))

    end

    function maxdev_in_hilbertTrafo_rat_fft(N_in, x_in_max)
        #println("maxdev rat fft")
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = xs
        ϵ = 2.
        ys = (x -> 1. / ((x-ϵ)^2 + 1.)).(xs)
        ht = imag.(TCI4Keldysh.hilbert_fft(ys))
        expected = (x -> (x - ϵ) / ((x-ϵ)^2 + 1.)).(xs_out) # * π

        return maximum(abs.(ht - expected))

    end

    ## analyze convergence:
    #maxdev_in_hilbertTrafo_rat.([5000, 10000, 20000], [50., 75., 100.]')
    #maxdev_in_hilbertTrafo_sinc.([5000, 10000, 20000], [50., 75., 100.]')
    @test maxdev_in_hilbertTrafo_rat(20000, 100.) < 1e-3
    @test maxdev_in_hilbertTrafo_sinc(20000,  1000.) < 1e-3
    @test maxdev_in_hilbertTrafo_rat_fft(20000, 1000.) < 2e-3
    @test maxdev_in_hilbertTrafo_sinc_fft(20000,  1000.) < 1e-3
    @test maxdev_in_hilbertTrafo_Gauss(10000, 5.0) < 1.e-6


end

@testset "utils for Tucker decompositions" begin
    filename = joinpath(dirname(@__FILE__), "data_PSF_2D.h5")
    f = h5open(filename, "r")
    Adisc = read(f, "Adisc")
    ωdisc = read(f, "ωdisc")
    close(f)


    D = 1.
    Nωcont_pos = 2^6
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))

    σ = 1.
    sigmab = [σ]
    g = 2.
    tol = 1.e-14
    estep = 160
    emin = 1e-6; emax = 1e2;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    original_data = broadenedPsf[:,:]

    TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf)
    newdata = broadenedPsf[:,:]
    atol = 1e-6
    TCI4Keldysh.svd_trunc_Adisc!(broadenedPsf; atol)
    truncdata = broadenedPsf[:,:]

    @test TCI4Keldysh.maxabs(newdata - original_data) < 1e-12
    @test TCI4Keldysh.maxabs(truncdata - original_data) < atol

    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    TCI4Keldysh.shift_singular_values_to_center_DIRTY!(broadenedPsf)

    newdata_DIRTY = broadenedPsf[:,:]
    
    @test TCI4Keldysh.maxabs(newdata_DIRTY - original_data) < 1e-12
    
    
    function test_tucker_eval(D::Int)
        center = randn(Float64, ntuple(i -> 5+i, D)) .+ 0.0.*im
        Nw = 10
        legs = [randn(ComplexF64, Nw, size(center,i)) for  i in 1:D]

        td = TCI4Keldysh.TuckerDecomposition(center, legs)
        w = rand(1:Nw, D)

        tdval = td(w...)
        evval = TCI4Keldysh.eval_tucker(center, [Matrix(transpose(legs[i])) for i in 1:D], w...)
        evval_vec = TCI4Keldysh.eval_tucker(center, [legs[i][w[i],:] for i in 1:D])
        @test isapprox(tdval, evval; atol=1.e-12)
        @test isapprox(evval, evval_vec; atol=1.e-12)
    end

    test_tucker_eval(2)
    test_tucker_eval(3)
    test_tucker_eval(4)
end

@testset "Hierarchical Tucker decomposition" begin

    function test_multipole_matrix()
        A = TCI4Keldysh.truncatable_matrix((50,30))
        A *= 1.e5
        Us, SVs = TCI4Keldysh.multipole_matrix(A, 1; cutoff=1.e-8)
        Anew = vcat([Us[i] * SVs[i] for i in eachindex(Us)]...) 
        @test norm(A - Anew)/norm(A) < 1.e-8
    end

    function test_hierarchical_tucker()
        center = randn(20,25,21)
        N_oms = (50,30,40)
        kernels = ntuple(i -> TCI4Keldysh.truncatable_matrix((N_oms[i],size(center,i))), ndims(center))

        cutoff = 1.e-5
        ht = TCI4Keldysh.HierarchicalTucker(center, kernels, 4; cutoff=cutoff)

        ref = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(kernels, center)

        for _ in 1:100
            test_id = ntuple(i -> rand(1:N_oms[i]), 3)
            @assert abs(ht(test_id...) - ref[test_id...]) / ref[test_id...] <= cutoff * 1.e3
        end

        # test precompute_all_values
        ht_dense = TCI4Keldysh.precompute_all_values(ht)
        @test maximum(abs.(ht_dense .- ref)) / maximum(abs.(ref)) < cutoff * 1.e3
    end

    function test_MultipoleKFCEvaluator()

        npt = 4
        D = npt-1
        basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
        channel = "p"
        Ops = TCI4Keldysh.dummy_operators(npt)
        (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
        ωconvMat = TCI4Keldysh.channel_trafo(channel)
        ommax = 0.5
        R = 4
        G = TCI4Keldysh.FullCorrelator_KF(
            PSFpath,
            Ops;
            T=TCI4Keldysh.dir_to_T(PSFpath),
            ωconvMat=ωconvMat,
            ωs_ext=TCI4Keldysh.KF_grid(ommax, R, D),
            flavor_idx=1,
            γ=γ,
            sigmak=sigmak,
            emax=max(20.0, 3*ommax),
            emin=2.5*1.e-5,
            estep=50
        )

        Gref = TCI4Keldysh.precompute_all_values(G)
        Gev = TCI4Keldysh.MultipoleKFCEvaluator(G; nlevel=2, cutoff=1.e-8)
        maxref = maximum(abs.(Gref))
        for idx in Iterators.product(fill(1:2^R,D)...)
            gval = reshape(Gev(idx...), ntuple(_->2,D+1))
            refval = Gref[idx...,:,:,:,:]
            @test maximum(abs.(gval .- refval)) / maxref < 1.e-8
        end
    end

    test_multipole_matrix()
    test_hierarchical_tucker()
    test_MultipoleKFCEvaluator()

end

@testset "Convert Tucker decomposition to QTT" begin

 

    begin
        filename = joinpath(dirname(@__FILE__), "data_PSF_2D.h5")
        f = h5open(filename, "r")
        Adisc = read(f, "Adisc")
        ωdisc = read(f, "ωdisc")
        close(f)

        Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

        ### System parameters of SIAM ### 
        D = 1.
        # Keldysh paper:    u=0.5 OR u=1.0
        U = 1. / 20.
        T = 0.01 * U
        #Δ = (U/pi)/0.5

        σ = 0.6
        sigmab = [σ]
        g = T * 1.
        tol = 1.e-14
        estep = 2048
        emin = 1e-6; emax = 1e4;
        Lfun = "FD" 
        is2sum = false
        verbose = false

        R = 6
        Nωcont_pos = 2^R # 512#
        ωcont = get_ωcont(D*0.5, Nωcont_pos)
        ωconts=ntuple(i->ωcont, ndims(Adisc))

        # get functor which can evaluate broadened data pointwisely
        broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    end


    tol = 1e-8
    qtt = TCI4Keldysh.TDtoQTCI(broadenedPsf; method="svd", tolerance=tol)
    qtt2 = TCI4Keldysh.TDtoQTCI(broadenedPsf; method="qtci", tolerance=tol)

    origdat= (broadenedPsf[:,:])[1:end-1, 1:end-1]
    qttdat = qtt[:,:]
    qttdat2= qtt2[:,:]

    @test maximum(abs.(origdat - qttdat)) < tol
    #@test maximum(abs.(origdat - qttdat2))< tol        ## doesn't work reliably somehow...


end

@testset "Misc. utils" begin

    function test_MF_grid()
        T = 0.01
        Nhalf = 2^4
        ombos = TCI4Keldysh.MF_grid(T, Nhalf, false)
        omfer = TCI4Keldysh.MF_grid(T, Nhalf, true)
        @test length(ombos)==2*Nhalf+1
        @test length(omfer)==2*Nhalf
        @test all(ombos .+ reverse(ombos) .<= 1.e-12)
        @test all(omfer .+ reverse(omfer) .<= 1.e-12)

        Dgrid = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)
        @test length(Dgrid) == 3
        @test length.(Dgrid) == (length(ombos), length(omfer), length(omfer))
    end

    function test_KF_idx()
        D = 3
        iK = 3
        K = TCI4Keldysh.KF_idx(iK, D)
        iK_ = TCI4Keldysh.KF_idx(K, D)
        @test iK_==iK
    end

    function test_merge_iK_K2()
        for channel in ["t","a","p"]
            for prime in [true, false]
                for ik in TCI4Keldysh.ids_KF(3)
                    ik4s = TCI4Keldysh.equivalent_iK_K2(ik, channel, prime)
                    ik_ = only(unique(TCI4Keldysh.merge_iK_K2.(Tuple.(ik4s), channel, prime)))
                    @test ik_==ik
                end
            end
        end
    end

    function test_merge_iK_K1()
        for channel in ["t","a","p"]
            for ik in TCI4Keldysh.ids_KF(2)
                ik4s = TCI4Keldysh.equivalent_iK_K1(ik, channel)
                ik_ = only(unique(TCI4Keldysh.merge_iK_K1.(Tuple.(ik4s), channel)))
                @test ik_==ik
            end
        end
    end

    function test_trafo_grids()
        ωs_ext = (0.5 * [1,2,3], 0.5*[-1,0,1,2])
        trafo = [1 -1; 0 1]
        ωs_new = TCI4Keldysh.trafo_grids(ωs_ext, trafo)
        @test ωs_new == ([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], [-0.5, 0.0, 0.5, 1.0])
    end

    function test_idx_trafo_offset()
        ωs_ext = (0.5 * [1,2,3], 0.5*[-1,0,1,2])
        trafo = [1 -1; 0 1]
        ωs_new = TCI4Keldysh.trafo_grids(ωs_ext, trafo)
        # ωs_new == ([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0], [-0.5, 0.0, 0.5, 1.0])
        s = TCI4Keldysh.idx_trafo_offset(ωs_ext, ωs_new, trafo)
        @test s==[4,0]
    end


    test_MF_grid()
    test_KF_idx()
    test_merge_iK_K1()
    test_merge_iK_K2()
    test_trafo_grids()
    test_idx_trafo_offset()
end

@testset "Numerical utils" begin
    function test_lin_interp_array()

        f(i::Int) = [0.01/(1 + i^2/1000) 0.02; 0.01/(1 + abs(i)/1000) 0.02*i/(3.0 + i^2)]

        idomain =  collect(-100:100)
        abstol = 1.e-4
        p = Inf
        (interp_ids, interp_val) = TCI4Keldysh.lin_interp_array(f, idomain; p=p, abstol=abstol)

        # test
        nf = length(f(idomain[1]))
        for id in 1:nf
            flin = linear_interpolation(idomain[interp_ids], [iv[id] for iv in interp_val])
            flinval = flin.(idomain)
            fval = [f(ii)[id] for ii in idomain]
            @test norm(fval .- flinval, p) <= abstol
        end
    end

    function test_interpolate_trilinear()
        vals = zeros(Float64,2,2,2)
        corners = map(t -> SA[t...] * 2.0, Iterators.product(0:1,0:1,0:1))
        vals[2,2,2] = 1.0
        vals[1,1,1] = -1.0
        @test TCI4Keldysh.interpolate_trilinear(vals, corners, SA[1.0,1.0,1.0])==0.0
        @test TCI4Keldysh.interpolate_trilinear(vals, corners, SA[1.5,1.5,1.5])==(1.5^3-0.5^3)/8.0
        @test TCI4Keldysh.interpolate_trilinear(vals, corners, SA[2.0,2.0,2.0])==1.0
    end

    test_lin_interp_array()
    test_interpolate_trilinear()
end

@testset "Convenience utils" begin
    @test isapprox(TCI4Keldysh.bonddims_to_RAM([2,3,3])*1.e6/16, 40.0; atol=1.e-10)
    @test isapprox(TCI4Keldysh.bonddims_to_RAM([2,3,3], 8)*1.e6/16, 160.0; atol=1.e-10)
end