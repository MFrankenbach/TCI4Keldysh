
@testset "frequency conversions for MPS" begin



nbit = 2
d = 2^nbit
tags =("x", "y")

m = reshape(collect(1:d^2)*1., (d,d)) # zeros(ComplexF64, d,d)
qtt = TCI4Keldysh.fatTensortoQTCI(m; tolerance=1e-10, method="qtci")
sites = TCI4Keldysh.getsitesforqtt(qtt; tags)
mps = TCI4Keldysh.TCItoMPS(qtt.tt; sites)
qtt[:,:]


isferm_ωnew = [0, 1] # 
ωconvMat = [1 0 ; 0 -1]
mps_rot = TCI4Keldysh.affine_freq_transform(mps; tags, ωconvMat, isferm_ωnew)

ωconvMat2 = [1 0 ; 1 1]
mps_rot2 = TCI4Keldysh.affine_freq_transform(mps_rot; tags, ωconvMat=ωconvMat2, isferm_ωnew)

ωconvMat_dir = [1 0 ; -1 -1]
mps_rot_dir = TCI4Keldysh.affine_freq_transform(mps; tags, ωconvMat=ωconvMat_dir, isferm_ωnew)



f_vec  = TCI4Keldysh.MPS_to_fatTensor(mps; tags)
g_vec1 = TCI4Keldysh.MPS_to_fatTensor(mps_rot; tags)
g_vec2 = TCI4Keldysh.MPS_to_fatTensor(mps_rot2; tags)
g_vec3 = TCI4Keldysh.MPS_to_fatTensor(mps_rot_dir; tags)

@test maximum(abs.(f_vec .-  [1.0  5.0   9.0  13.0; 
                              2.0  6.0  10.0  14.0; 
                              3.0  7.0  11.0  15.0; 
                              4.0  8.0  12.0  16.0])) < 1e-12
@test maximum(abs.(g_vec1.-  [  13.0   9.0  5.0  1.0;
                                14.0  10.0  6.0  2.0;
                                15.0  11.0  7.0  3.0;
                                16.0  12.0  8.0  4.0])) < 1e-12

@test maximum(abs.(g_vec2.-     [5.0   1.0  13.0   9.0;
                                2.0  14.0  10.0   6.0;
                                15.0  11.0   7.0   3.0;
                                12.0   8.0   4.0  16.0])) < 1e-12
@test maximum(abs.(g_vec2.-  [  5.0   1.0  13.0   9.0;
                                2.0  14.0  10.0   6.0;
                                15.0  11.0   7.0   3.0;
                                12.0   8.0   4.0  16.0])) < 1e-12

@test maximum(abs.(g_vec3.-  [  5.0   1.0  13.0   9.0;
                                2.0  14.0  10.0   6.0;
                                15.0  11.0   7.0   3.0;
                                12.0   8.0   4.0  16.0])) < 1e-12


ωconvMat_flip = [-1 0 ; 0 1]
mps_flip = TCI4Keldysh.affine_freq_transform(mps; tags, ωconvMat=ωconvMat_flip, isferm_ωnew)
g_vec_flip = TCI4Keldysh.MPS_to_fatTensor(mps_flip; tags)
@test maximum(abs.(g_vec_flip.-  [  1.0  5.0   9.0  13.0;
                                    4.0  8.0  12.0  16.0;
                                    3.0  7.0  11.0  15.0;  # <-- corresponds to ω = 0
                                    2.0  6.0  10.0  14.0])) < 1e-12

end