@testset "calc. Matsubara Correlators from PSFs: " begin

    ### Hubbard atom propagator:
    T = 3.

    N_MF = 100
    ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * im * π * T
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * π * T
    Nωdisc = 100
    ωdisc_shift = 50


    ωdisc, Adisc = get_Adisc_δpeak_mp([ωdisc_shift], Nωdisc, 1)

    u = ωdisc[Nωdisc + 1 + ωdisc_shift]
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
    ωdisc = get_Adisc_δpeak_mp([0,0], Nωdisc, 2)[1]
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

end