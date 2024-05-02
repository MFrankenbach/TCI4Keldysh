##
# Try out compression NRG correlators via Discrete Lehmann Representation
## 
using Revise
using TCI4Keldysh
using Lehmann
begin
    ### System parameters of SIAM ### 
    D = 1.
    ## Keldysh paper:    u=0.5 OR u=1.0
    # set physical parameters
    u = 0.5; 
    #u = 1.0

    U = 0.05;
    Δ = U / (π * u)
    T = 0.01*U
    β = 1/T # 100.0 # inverse temperature
    Euv = 1.0 # ultraviolt energy cutoff of the Green's function

end

rtol = 1e-8 # accuracy of the representation
isFermi = true
symmetry = :none # :ph if particle-hole symmetric, :pha is antisymmetric, :none if there is no symmetry

diff(a, b) = maximum(abs.(a - b)) # return the maximum deviation between a and b

# create DLR grids for bosonic and fermionic MF frequencies
dlr_fer = DLRGrid(Euv, β, rtol, true, symmetry) #initialize the DLR parameters and basis
dlr_bos = DLRGrid(Euv, β, rtol, false, symmetry) #initialize the DLR parameters and basis

Niω_fer = maximum(abs.(dlr_fer.n))
Niω_bos = maximum(abs.(dlr_bos.n))
ngrid_fer = collect(-Niω_fer:Niω_fer-1)
ngrid_bos = collect(-Niω_bos:Niω_bos)
ω_fer = π * T *(ngrid_fer * 2 .+ 1)
ω_bos = π * T *(ngrid_bos * 2)

## check DLR for 1D function:
PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
G     = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
Gdata = TCI4Keldysh.precompute_all_values(G)
spectral_from_Gω = matfreq2dlr(dlr_fer, Gdata, ngrid_fer)
kernel = Lehmann.Spectral.kernelΩ(Float64, Val(dlr_fer.isFermi), Val(symmetry), ngrid_fer, dlr_fer.ω, dlr_fer.β, true)
Gdata_rec = kernel * spectral_from_Gω

maximum(abs.(Gdata - Gdata_rec)) / maximum(abs.(Gdata))

ωconvMat_K2′t = [ 1   0; -1  -1; 0   1]
G2D      = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q12", "F3", "F3dag"]; T, flavor_idx=1, ωs_ext=(ω_bos,ω_fer), ωconvMat=ωconvMat_K2′t, name="SIAM 3pG", is_compactAdisc=false);

Gp2D_1 = G2D.Gps[1]
Gp2D_1_data = Gp2D_1[[1,2,5,6],]

Gp2D_1_data = nothing

using QuanticsGrids
QuanticsGrids.

m = reshape(collect(1:16), (4,4))
m[[1,2], [2,4]]

GC.gc()






using OffsetArrays
ma = OffsetArray(m, -4, -4)
mama = ma*ma

Gp2D_1_view = OffsetArrays.no_offset_view(m)