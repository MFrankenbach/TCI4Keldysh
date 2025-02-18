using MAT
ddir::String = "/scratch/m/M.Frankenbach/tci4keldysh/data/SIAM_u5_U0.05_T0.0005_Delta0.0031831/PSF_nz=2_conn_zavg/"
failcount = 0
failfiles = String[]
nullfiles = String[]
for f in readdir(ddir)
    fname = joinpath(ddir, f)
    if !endswith(fname, ".mat")
        continue
    end
    f = matopen(fname, "r")
    try
        keys(f)
    catch
        keys(f)
    end
    try
        @show keys(f)
        for k in keys(f)
            val = read(f, k)
            if ismissing(val)
                @info "Key $k missing in file $fname"
            end
        end
        # Adisc = read(f, "Adisc")[1]
        # @show first(Adisc)
    catch
        global failcount += 1
        push!(failfiles, fname)
    finally
        close(f)
    end
end

println("FAILED to read $failcount files:")
for f in failfiles
    println(f)
end
# println("NULL reference files ($(length(nullfiles)))")
# for f in nullfiles
#     println(f)
# end