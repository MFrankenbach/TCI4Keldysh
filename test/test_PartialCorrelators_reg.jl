@testset "PartialCorrelator_reg for formalism==MF" begin

    ### compare with broadened Dirac-δ peak:        2D
    Nωs_pos = 30
    idx_ω′s1 = [10, 12]
    ωdisc, Adisc = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 2)
    argmax(Adisc)
    Adisc[(idx_ω′s1 .+ Nωs_pos .+ 1)...]

    N_MF = 100
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im
    ω_bos = collect(-N_MF:N_MF) * (2.) * im


    G_p_disc = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, (ω_fer, ω_bos), [1 0; 0 1])
    G_p_data = G_p_disc.(collect(1:length(ω_fer)), collect(1:length(ω_bos))')
    expected = ((x, y) -> 1. / (x - ωdisc[idx_ω′s1[1] + Nωs_pos + 1]) / ( y - ωdisc[idx_ω′s1[2] + Nωs_pos + 1])).(ω_fer, reshape(ω_bos,(1, length(ω_bos))))
    @test maximum(abs.(G_p_data - expected)) < 1.e-10



    G_p_disc = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, (ω_fer, ω_bos), [1 0; 1 1])
    G_p_data = G_p_disc.(collect(1:length(ω_fer)), collect(1:length(ω_bos))')
    expected = ((x, y) -> 1. / (x - ωdisc[idx_ω′s1[1] + Nωs_pos + 1]) / ( x + y - ωdisc[idx_ω′s1[2] + Nωs_pos + 1])).(ω_fer, reshape(ω_bos,(1, length(ω_bos))))
    @test maximum(abs.(G_p_data - expected)) < 1.e-10



    G_p_disc = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, (ω_fer, ω_bos), [1 0; 0 -1])
    G_p_data = G_p_disc.(collect(1:length(ω_fer)), collect(1:length(ω_bos))')
    expected = ((x, y) -> 1. / (x - ωdisc[idx_ω′s1[1] + Nωs_pos + 1]) / ( -y - ωdisc[idx_ω′s1[2] + Nωs_pos + 1])).(ω_fer, reshape(ω_bos,(1, length(ω_bos))))
    @test maximum(abs.(G_p_data - expected)) < 1.e-10



    G_p_disc = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, (ω_fer, ω_bos), [1 1; 1 -1])
    G_p_data = G_p_disc.(collect(1:length(ω_fer)), collect(1:length(ω_bos))')
    expected = ((x, y) -> 1. / (x + y - ωdisc[idx_ω′s1[1] + Nωs_pos + 1]) / (x -y - ωdisc[idx_ω′s1[2] + Nωs_pos + 1])).(ω_fer, reshape(ω_bos,(1, length(ω_bos))))
    @test maximum(abs.(G_p_data - expected)) < 1.e-10



    G_p_disc = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, (ω_fer, ω_bos), [-1 1; 0 1])
    G_p_data = G_p_disc.(collect(1:length(ω_fer)), collect(1:length(ω_bos))')
    expected = ((x, y) -> 1. / (-x +y - ωdisc[idx_ω′s1[1] + Nωs_pos + 1]) / ( y - ωdisc[idx_ω′s1[2] + Nωs_pos + 1])).(ω_fer, reshape(ω_bos,(1, length(ω_bos))))
    @test maximum(abs.(G_p_data - expected)) < 1.e-10



    N_MF = 50
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im
    ω_bos = collect(-N_MF:N_MF) * (2.) * im



    ### compare with broadened Dirac-δ peak:        3D
    Nωs_pos = 30
    idx_ω′s1 = [10, 12, 13]
    ωdisc, Adisc = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 3)
    argmax(Adisc)
    Adisc[(idx_ω′s1 .+ Nωs_pos .+ 1)...]


    G_p_disc = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, (ω_bos, ω_fer, ω_fer), [1 0 1; 1 1 1; -1 1 0])
    G_p_data = G_p_disc.(collect(1:length(ω_bos)), collect(1:length(ω_fer))', reshape(collect(1:length(ω_fer)), (1,1,length(ω_fer))));
    #@time G_p_disc[:,:,:];
    expected = ((x, y, z) -> 1. / (x + z - ωdisc[idx_ω′s1[1] + Nωs_pos + 1]) / ( x + y + z - ωdisc[idx_ω′s1[2] + Nωs_pos + 1]) / ( -x + y - ωdisc[idx_ω′s1[3] + Nωs_pos + 1])).(ω_bos, reshape(ω_fer,(1, length(ω_fer))), reshape(ω_fer,(1, 1, length(ω_fer))));
    @test maximum(abs.(G_p_data - expected)) < 1.e-10


end

@testset "PartialCorrelator_reg for formalism==KF" begin

end