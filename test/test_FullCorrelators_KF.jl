@testset "formula G^R[S](ω)" begin
    # test formula for computing a retarded correlator from a spectral function
    # formula used in KF constructor for PartialCorrelator_reg

    N = 1000
    ωdisc = collect( -N:N) / 10

    Adisc = 1 ./ (ωdisc.^2 .+ 1) ./ π       #   Lorentzian
    GR = -im * π * (TCI4Keldysh.hilbert_fft(Adisc))



    @test TCI4Keldysh.maxabs(real.(GR) - ωdisc ./ (ωdisc.^2 .+ 1)) < 1e-2
    @test TCI4Keldysh.maxabs(imag.(GR) +     1 ./ (ωdisc.^2 .+ 1)) < 1e-14
    #plot(ωdisc, [real.(GR), imag.(GR), ωdisc ./ (ωdisc.^2 .+ 1), -1. ./ (ωdisc.^2 .+ 1)], labels=["Re(GR)" "Im(GR)" "ω/(ω^2+1)" "-1/(ω^2+1)"], xlims=[-10, 10])

end