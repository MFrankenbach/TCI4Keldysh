using TCI4Keldysh

data_dir = "./data/PSF_nz=2_conn_zavg/";

Ops = [
    ["F1", "F1dag", "F3", "F3dag"],
    ["F1", "F1dag", "F3dag", "F3"],
    ["F1", "F3", "F1dag", "F3dag"],
    ["F1dag", "F1", "F3", "F3dag"],
    ["F1dag", "F1", "F3dag", "F3"],
    ["F1dag", "F3dag", "F1", "F3"]
]

for o in Ops
    #TCI4Keldysh.symmetry_expand(data_dir, o)
end