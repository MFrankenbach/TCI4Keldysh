

@testset "calc. Matsubara Correlators from PSFs: " begin

    ### Hubbard atom propagator:
    T = 3.

    N_MF = 100
    ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * π * T
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * π * T
    Nωdisc = 100
    ωdisc_shift = 1#50

    u = 0.5
    ωdisc, Adisc = get_Adisc_δpeak_mp([ωdisc_shift], Nωdisc, 1; ωdisc_min=u)

    #u = ωdisc[Nωdisc + 1 + ωdisc_shift]
    Zinv = 1. / (exp(u/T) + 1) / 2.
    expβu = exp(u/T)
    Adisc = Zinv * (expβu * Adisc + reverse(Adisc))
    Adiscs = [Adisc, Adisc]
    ωs_ext = (ω_fer, )
    ωsconvMat = reshape([1 ; -1], (2,1))
    isBos = [false, false] .== true

    G = TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; T, isBos=isBos, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name=["Hubbard atom propagator"])
    G_data = G.(collect(1:length(ω_fer)))
    G_expec=  HA_exact_corr_F1_F1dag.(ω_fer; u=u)
    @test maximum(abs.(G_data - G_expec)) < 1.e-13



    ### 3-point correlator corresponding to <d_↑^† d_↓ (d_↓^† d_↑)>: 
    D = 2
    ωdisc = get_Adisc_δpeak_mp([0,0], Nωdisc, 2; ωdisc_min=u)[1]
    Adiscs = [
        get_Adisc_δpeak_mp([ ωdisc_shift,           0], Nωdisc, 2)[2] * Zinv * expβu
        ,get_Adisc_δpeak_mp([-ωdisc_shift,-ωdisc_shift], Nωdisc, 2)[2] * Zinv         * (-1)
        ,get_Adisc_δpeak_mp([ ωdisc_shift,           0], Nωdisc, 2)[2] * Zinv * expβu * (-1)
        ,get_Adisc_δpeak_mp([-ωdisc_shift,-ωdisc_shift], Nωdisc, 2)[2] * Zinv
        ,get_Adisc_δpeak_mp([           0, ωdisc_shift], Nωdisc, 2)[2] * Zinv * expβu
        ,get_Adisc_δpeak_mp([           0, ωdisc_shift], Nωdisc, 2)[2] * Zinv * expβu * (-1)
    ]
    ωs_ext = (ω_bos, ω_fer)
    ωsconvMat = [-1 -1; 0 1; 1 0]
    isBos = [false, false, true] .== true
    G = TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; T, isBos=isBos, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name=["Hubbard atom 3pt correlator"])
    data_axes = ntuple(i -> reshape(collect(axes(ωs_ext[i])[1]), (ones(Int,i-1)..., length(ωs_ext[i]))), D)

    #@time G_data = G.(data_axes...);    # this data is generated via on-the-fly computation of correlator values (WITHOUT anomalous terms!)
    func_tmp = (w, v) -> begin w=im*w; v=im*v; - ((u^2 + v *(v+w)) + (abs(w) > 1.e-10 ? 0. : -u/T*expβu*Zinv* (2*u^2 - v^2 - (v+w)^2)) ) / ((u^2 - v^2) * (u^2 - (v+w)^2)) end
    G_expec= func_tmp.(ω_bos, reshape(ω_fer,(1,length(ω_fer))));
    G_prec = TCI4Keldysh.precompute_all_values(G);

    @test maximum(abs.(G_prec - G_expec)) < 1e-10

    ### 4-point correlator corresponding to <d_↑^† d_↓ (d_↓^† d_↑)>: 
    D = 3
    N_MF = 10
    ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * π * T
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * π * T
    Nωdisc = 10
    
    ωdisc = get_Adisc_δpeak_mp([0,0,0], Nωdisc, D; ωdisc_min=u)[1]
    Adiscs = [
         get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 #  1 # [1234] u  u† d  d†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         #  2 # [1243] u  u† d† d
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv         * (-1)  #  3 # [1324] u  d  u† d†
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 #  4 # [1342] u  d  d† u†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu * (-1)  #  5 # [1423] u  d† u† d
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         #  6 # [1432] u  d† d  u†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         #  7 # [2134] u† u  d  d†
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 #  8 # [2143] u† u  d† d
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu * (-1)  #  9 # [2314] u† d  u  d†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         # 10 # [2341] u† d  d† u
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv         * (-1)  # 11 # [2413] u† d† u  d  
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 # 12 # [2431] u† d† u  d  
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 # 13 # [3124] d  u  u† d†
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv         * (-1)  # 14 # [3142] d  u  d† u†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         # 15 # [3214] d  u† u  d†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu * (-1)  # 16 # [3241] d  u† d† u
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 # 17 # [3412] d  d† u  u†
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         # 18 # [3421] d  d† u† u
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         # 19 # [4123] d† u  u† d
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu * (-1)  # 20 # [4132] d† u  d  u†
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 # 21 # [4213] d† u† u  d
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv         * (-1)  # 22 # [4231] d† u† d  u
        ,get_Adisc_δpeak_mp([ ωdisc_shift, 0,  ωdisc_shift], Nωdisc, D)[2] * Zinv * expβu         # 23 # [4312] d† d  u  u†
        ,get_Adisc_δpeak_mp([-ωdisc_shift, 0, -ωdisc_shift], Nωdisc, D)[2] * Zinv                 # 24 # [4321] d† d  u† u
    ]
    ωs_ext = (ω_bos, ω_fer, ω_fer)
    ωsconvMat = [-1 -1  0; 
                  0  1  0; 
                  0  0 -1; 
                  1  0  1]
    isBos = [false, false, false, false] .== true
    G = TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; T, isBos=isBos, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name=["Hubbard atom 4pt correlator"])
    data_axes = ntuple(i -> reshape(collect(axes(ωs_ext[i])[1]), (ones(Int,i-1)..., length(ωs_ext[i]))), D)

    ## missing:
    function HA_4p_conn_correlator(w, v, vp; u, T) 
        w = im*w; v = im*v; vp = im*vp
        ωs_fermionic = [-w-v, v, -vp, w+vp]
        _f(ϵ) = 1 / (1 + exp(ϵ/T))
        _D(x,y) = (-x^2 + u^2) * (-y^2 + u^2) #/ (x*y)^2
        res_reg = (2*u*prod(ωs_fermionic) + u^3 * sum(ωs_fermionic.^2) - 6*u^5) 
        res_a = abs(v-vp) > 1.e-10 ? 0. :  2 * u^2 / T * _D(v,w+vp) * _f(-u)
        res_p = abs(w+v+vp) > 1.e-10 ? 0. : -2 * u^2 / T * _D(v,vp) * _f(u)
        res_t = abs(w) > 1.e-10 ? 0. : - u^2 / T * _D(v,vp) * (_f(u) - _f(-u))
        disco = abs(w) > 1.e-10 ? 0. : -1 /T * v*vp / (-v^2+u^2) / (-vp^2+u^2)
        return disco + (res_reg + res_a + res_p + res_t) / prod(u^2 .- ωs_fermionic.^2)
    end 
    G_expec= HA_4p_conn_correlator.(ω_bos, reshape(ω_fer,(1,length(ω_fer))), reshape(ω_fer,(1,1,length(ω_fer))); u=u, T=T);
    G_prec = TCI4Keldysh.precompute_all_values(G)

    @test TCI4Keldysh.maxabs(imag.(G_prec)) < 1e-14
    @test maximum(abs.(G_prec - G_expec)) < 1e-10

end


@testset "Test DLR stuff on MF 3p correlators" begin
    begin
        # set frequency conventions
        ωconvMat_K2′t = [
            1   0;
            -1  -1;
             0   1
        ]

        ### System parameters of SIAM ### 
        D = 1.
        ## Keldysh paper:    u=0.5 OR u=1.0
        # set physical parameters
        u = 0.5; 
        #u = 1.0
    
        U = 0.05;
        Δ = U / (π * u)
        T = 0.01*U
    
        Rpos = 6
        R = Rpos + 1
        Nωcont_pos = 2^Rpos # 512#
        ωcont = get_ωcont(D*0.5, Nωcont_pos)
        
        # get functor which can evaluate broadened data pointwisely
        #broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
        ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
        ωfer = π * T *(collect(-Nωcont_pos*2:Nωcont_pos*2-1) * 2 .+ 1)
        ωs_ext = (ωbos, ωfer)
        
        PSFpath = joinpath(dirname(@__FILE__), "../../data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
        Gs      = [TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "F3", "F3dag"]; T, flavor_idx=i, ωs_ext=(ωbos,ωfer), ωconvMat=ωconvMat_K2′t, name="SIAM 3pG", is_compactAdisc=false) for i in 1:2];
    end

    G_in = deepcopy(Gs[1])
    TCI4Keldysh.reduce_Gps!(G_in)
    
    
    G_in_data = TCI4Keldysh.precompute_all_values(G_in)
    G_predata = TCI4Keldysh.precompute_all_values(Gs[1])
    
    @test maximum(abs.(G_in_data - G_predata)) < 1e-10
    


end

@testset "Correlator Evaluation" begin
    
    """
    Test pointwise evaluation of FullCorrelator
    """
    function test_FullCorrelator_evaluate(npt::Int=4)
        
        R = 4
        D = npt-1
        GF = TCI4Keldysh.dummy_correlator(npt, R; beta=2000.0, is_compactAdisc=true)[1]
        N = 2^R

        val = zeros(ComplexF64, ntuple(_->N, D))
        Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:N,D)...))
            val[idx...] = TCI4Keldysh.evaluate(GF, idx...)
        end

        refval = TCI4Keldysh.precompute_all_values(GF)

        @test maximum(abs.(refval[1:2^R, ntuple(_->Colon(), D-1)...] .- val)) <= 1.e-10
    end

    test_FullCorrelator_evaluate(2)
    test_FullCorrelator_evaluate(3)
    test_FullCorrelator_evaluate(4)
end

