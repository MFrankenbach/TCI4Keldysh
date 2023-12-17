

@testset "calc. Matsubara Correlators from PSFs: " begin

    ### Hubbard atom propagator:
    T = 3.

    N_MF = 100
    ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * im * π * T
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * π * T
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

    G = TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; isBos=isBos, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name=["Hubbard atom propagator"])
    G_data = G.(collect(1:length(ω_fer)))
    G_expec=  (ω -> 1 / (ω - u^2 / ω)).(ω_fer)
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
    G = TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; isBos=isBos, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name=["Hubbard atom 3pt correlator"])
    data_axes = ntuple(i -> reshape(collect(axes(ωs_ext[i])[1]), (ones(Int,i-1)..., length(ωs_ext[i]))), D)

    #@time G_data = G.(data_axes...);    # this data is generated via on-the-fly computation of correlator values (WITHOUT anomalous terms!)
    func_tmp = (w, v) -> - ((u^2 + v *(v+w)) + (abs(w) > 1.e-10 ? 0. : -u/T*expβu*Zinv* (2*u^2 - v^2 - (v+w)^2)) ) / ((u^2 - v^2) * (u^2 - (v+w)^2))
    G_expec= func_tmp.(ω_bos, reshape(ω_fer,(1,length(ω_fer))));
    G_prec = TCI4Keldysh.precompute_all_values(G);

    @test maximum(abs.(G_prec - G_expec)) < 1e-10

    ### 4-point correlator corresponding to <d_↑^† d_↓ (d_↓^† d_↑)>: 
    D = 3
    N_MF = 10
    ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * im * π * T
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * π * T
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
    G = TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; isBos=isBos, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name=["Hubbard atom 4pt correlator"])
    data_axes = ntuple(i -> reshape(collect(axes(ωs_ext[i])[1]), (ones(Int,i-1)..., length(ωs_ext[i]))), D)

    ## missing:
    function HA_4p_conn_correlator(w, v, vp; u, T) 
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