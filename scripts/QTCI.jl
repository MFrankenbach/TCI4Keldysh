using HDF5
using QuanticsTCI
#import TensorCrossInterpolation as TCI
using TCI4Keldysh
#using MAT
#using JLD
#using Plots
using LinearAlgebra


d = 2^2
########################################
#### TCI compress simple matrices:  ####
########################################

m = zeros(d,d)
m[:,2] .= 1.
qtt = TCI4Keldysh.fatTensortoQTCI(m; tolerance=1e-10, method="qtci")
QuanticsTCI.TensorCrossInterpolation.linkdims(qtt.tt)
size.(qtt.tt.T)
qtt.tt.T
maximum(abs.(qtt[:,:]-m))



m = zeros(d,d)
m[16,16] = 1.
qtt = TCI4Keldysh.fatTensortoQTCI(m; tolerance=1e-10, method="svd")
QuanticsTCI.TensorCrossInterpolation.linkdims(qtt.tt)
size.(qtt.tt.T)
qtt.tt.T
maximum(abs.(qtt[:,:]-m))


m = Matrix(LinearAlgebra.Diagonal(ones(d)))
qtt = TCI4Keldysh.fatTensortoQTCI(m; tolerance=1e-10, method="qtci")
QuanticsTCI.TensorCrossInterpolation.linkdims(qtt.tt)
size.(qtt.tt.T)
qtt.tt.T
maximum(abs.(qtt[:,:]-m))


m = reverse(Matrix(LinearAlgebra.Diagonal(ones(d))), dims=1) + Matrix(LinearAlgebra.Diagonal(ones(d)))
qtt = TCI4Keldysh.fatTensortoQTCI(m; tolerance=1e-10, method="qtci")
QuanticsTCI.TensorCrossInterpolation.linkdims(qtt.tt)
size.(qtt.tt.T)
qtt.tt.T
maximum(abs.(qtt[:,:]-m))