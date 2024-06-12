using TCI4Keldysh

meanerrs = []
maxerrs = []
npt = 4
for i in 1:factorial(npt)
    mean_err, max_err = TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=npt, perm_idx=i)
    @show mean_err
    @show max_err
    push!(meanerrs, mean_err)
    push!(maxerrs, max_err)
end
@show meanerrs
@show maxerrs
# TCI4Keldysh.test_TCI_frequency_rotation_reg_values()