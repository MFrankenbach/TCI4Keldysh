@testset "Hilbert transform" begin
    
        
    function maxdev_in_hilbertTrafo_sinc(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-50., 50., 1000))
        ys = sin.(xs) ./ xs
        ht = TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys)
        expected = (x -> (1. - cos(x)) / x).(xs_out)  * π

        return maximum(abs.(ht - expected))

    end

    function maxdev_in_hilbertTrafo_rat(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-50., 50., 1000))
        ys = (x -> 1. / (x^2 + 1.)).(xs)
        ht = TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys)
        expected = (x -> x / (x^2 + 1.)).(xs_out)  * π

        return maximum(abs.(ht - expected))

    end

    ## analyze convergence:
    #maxdev_in_hilbertTrafo_rat.([5000, 10000, 20000], [50., 75., 100.]')
    #maxdev_in_hilbertTrafo_sinc.([5000, 10000, 20000], [50., 75., 100.]')
    @test maxdev_in_hilbertTrafo_rat(10000, 100.) < 1e-3
    @test maxdev_in_hilbertTrafo_sinc(10000,  100.) < 1e-3

end