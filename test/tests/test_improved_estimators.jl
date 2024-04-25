
@testset "IE formulas for Σ" begin
    N_MF = 100
    u = 0.5
    T = 1.
    #ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * π * T
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * π * T

    Σ_Hartree = u
    G0_HA_inv = im.* ω_fer .+ u
    G_HA = HA_exact_corr_F1_F1dag.(ω_fer, u=u)
    Σ_HA = HA_exact_selfenergy.(ω_fer, u=u)
    G_aux = HA_exact_corr_Q1_F1dag.(ω_fer, u=u)
    G_QQ_aux = HA_exact_corr_Q1_Q1dag.(ω_fer, u=u)

    Σ_calc_dir = TCI4Keldysh.calc_Σ_MF_dir(G_HA, G0_HA_inv)
    Σ_calc_aIE = TCI4Keldysh.calc_Σ_MF_aIE(G_aux, G_HA)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux, G_aux, G_aux, G_HA, Σ_Hartree)

    @test Σ_HA ≈ Σ_calc_dir
    @test Σ_calc_aIE ≈ Σ_HA
    @test Σ_calc_sIE ≈ Σ_HA
end