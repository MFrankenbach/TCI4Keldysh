##
# Try out compression via Discrete Lehmann Representation
## 



using Lehmann
β = 100.0 # inverse temperature
Euv = 1.0 # ultraviolt energy cutoff of the Green's function
rtol = 1e-8 # accuracy of the representation
isFermi = false
symmetry = :none # :ph if particle-hole symmetric, :pha is antisymmetric, :none if there is no symmetry

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

dlr = DLRGrid(Euv, β, rtol, isFermi, symmetry) #initialize the DLR parameters and basis
# A set of most representative grid points are generated:
# dlr.ω gives the real-frequency grids
# dlr.τ gives the imaginary-time grids
# dlr.ωn and dlr.n gives the Matsubara-frequency grids. The latter is the integer version.

println("Prepare the Green's function sample ...")
Nτ, Nωn = 10000, 10000 # many τ and n points are needed because Gτ is quite singular near the boundary
ngrid = collect(-Nωn:Nωn)  # create a set of Matsubara-frequency points
Gn = Sample.SemiCircle(dlr, :n, ngrid)#.^2 # Use semicircle spectral density to generate the sample Green's function in ωn
Gn = 1 ./ ((2 .*ngrid.+1).*im .- 1.).^2

println("Compress Green's function into ~20 coefficients ...")
spectral_from_Gω = matfreq2dlr(dlr, Gn, ngrid)
# You can use the above functions to fit noisy data by providing the named parameter ``error``

println("Prepare the target Green's functions to benchmark with ...")
n = collect(-2Nωn:2Nωn)  # create a set of Matsubara-frequency points
Gn_target = Sample.SemiCircle(dlr, :n, n)#.^2
Gn_target = 1 ./ ((2 .*n.+1).*im .- 1.).^2

println("Interpolation benchmark ...")
Gn_interp = dlr2matfreq(dlr, spectral_from_Gω, n)
println("iω → iω accuracy: ", diff(Gn_interp, Gn_target))

println("Fourier transform benchmark...")
Gτ_to_n = dlr2matfreq(dlr, spectral_from_Gτ, n)
println("τ → iω accuracy: ", diff(Gτ_to_n, Gn_target))
Gn_to_τ = dlr2tau(dlr, spectral_from_Gω, τ)
println("iω → τ accuracy: ", diff(Gn_to_τ, Gτ_target))

plot(real.(spectral_from_Gω))