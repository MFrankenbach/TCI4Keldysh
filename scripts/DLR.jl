##
# Try out compression via Discrete Lehmann Representation
## 



using Lehmann
β = 100.0 # inverse temperature
Euv = 1.0 # ultraviolt energy cutoff of the Green's function
rtol = 1e-8 # accuracy of the representation
isFermi = true
symmetry = :none # :ph if particle-hole symmetric, :pha is antisymmetric, :none if there is no symmetry

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

dlr = DLRGrid(Euv, β, rtol, isFermi, symmetry) #initialize the DLR parameters and basis
# A set of most representative grid points are generated:
 dlr.ω #gives the real-frequency grids
# dlr.τ gives the imaginary-time grids
dlr.ωn #and dlr.n gives the Matsubara-frequency grids. The latter is the integer version.
dlr.n
dlr.kernel_n

println("Prepare the Green's function sample ...")
Nτ, Nωn = 10000, 10000 # many τ and n points are needed because Gτ is quite singular near the boundary
ngrid = collect(-Nωn:Nωn)  # create a set of Matsubara-frequency points
Gn = Sample.SemiCircle(dlr, :n, ngrid)#.^2 # Use semicircle spectral density to generate the sample Green's function in ωn
#Gn = 1 ./ ((2 .*ngrid.+1) .- 0.5 *im)#.^2

println("Compress Green's function into ~20 coefficients ...")
spectral_from_Gω = matfreq2dlr(dlr, Gn, ngrid)

Gn_2D_1 = Gn .* ones(2)'
spectral_from_Gω_2D_1 = matfreq2dlr(dlr, Gn_2D_1, ngrid; axis=1)

Gn_2D_2 = transpose(Gn) .* ones(2)
spectral_from_Gω_2D_2 = matfreq2dlr(dlr, Gn_2D_2, ngrid; axis=2)
maximum(abs.(spectral_from_Gω - spectral_from_Gω_2D_1[:,1]))
maximum(abs.(spectral_from_Gω - spectral_from_Gω_2D_2[1,:]))




dlr.kernel_nc
dlr.isFermi
typeof(dlr)
green = Gn[]
kernel = Lehmann.Spectral.kernelΩ(Float64, Val(dlr.isFermi), Val(symmetry), ngrid, dlr.ω, dlr.β, true)
err = nothing
g, partialsize = Lehmann._tensor2matrix(green, Val(1))
@benchmark coeff = Lehmann._weightedLeastSqureFit(dlr, g, err, kernel, nothing)
C = g
B = kernel
@benchmark coeffs_alt = B \ C

maximum(abs.(C - B*coeffs_alt))

coeff - coeffs_alt

using LinearAlgebra
u, s, v = svd(B'B)
Bqr = qr(B)
Bqr.Q
Bqr.R
s
BBinv = inv(B'*B)
coeffs_linlsq = BBinv * B' * C
RRinv = inv(Bqr.R'Bqr.R)
coeffs_linlsq_QR = RRinv * B' * C
u, s, v = svd(Bqr.R'*Bqr.R)
s


using BenchmarkTools
@benchmark coeffs_qr = Bqr \ C
maximum(abs.(C - B * coeffs_qr))

Blu = lu(B)
coeffs_lu = Blu \ C
maximum(abs.(C - B * coeffs_qr))


coeff - coeffs_linlsq

# You can use the above functions to fit noisy data by providing the named parameter ``error``

println("Prepare the target Green's functions to benchmark with ...")
n = collect(-2Nωn:2Nωn)  # create a set of Matsubara-frequency points
#Gn_target = Sample.SemiCircle(dlr, :n, n)#.^2
Gn_target = 1 ./ ((2 .*n.+1) .- 0.5)#.^2

println("Interpolation benchmark ...")
Gn_interp = dlr2matfreq(dlr, spectral_from_Gω, n)
println("iω → iω accuracy: ", diff(Gn_interp, Gn_target))

println("Fourier transform benchmark...")
Gτ_to_n = dlr2matfreq(dlr, spectral_from_Gτ, n)
println("τ → iω accuracy: ", diff(Gτ_to_n, Gn_target))
Gn_to_τ = dlr2tau(dlr, spectral_from_Gω, τ)
println("iω → τ accuracy: ", diff(Gn_to_τ, Gτ_target))

using Plots
plot(real.(spectral_from_Gω))
plot(imag.(spectral_from_Gω))