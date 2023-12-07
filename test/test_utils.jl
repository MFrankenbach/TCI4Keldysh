@testset "Hilbert transform" begin
    
        
    function maxdev_in_hilbertTrafo_sinc(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        ys = sin.(xs) ./ xs
        ht = TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys)
        expected = (x -> (1. - cos(x)) / x).(xs_out)  * π

        return maximum(abs.(ht - expected))

    end

    function maxdev_in_hilbertTrafo_rat(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        ϵ = 2.
        ys = (x -> 1. / ((x-ϵ)^2 + 1.)).(xs)
        ht = TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys)
        expected = (x -> (x - ϵ) / ((x-ϵ)^2 + 1.)).(xs_out)  * π

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


end
