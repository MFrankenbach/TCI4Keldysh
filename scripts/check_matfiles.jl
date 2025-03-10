using MAT
ddir::String = "/scratch/m/M.Frankenbach/tci4keldysh/data/PRX_jae-mo_PSF/PSF_nz=4_conn_zavg/"
failcount = 0
missingcount = 0
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
                global missingcount += 1
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

println("missing-valued keys: $missingcount")
println("FAILED to read $failcount files:")
for f in failfiles
    println(f)
end
# println("NULL reference files ($(length(nullfiles)))")
# for f in nullfiles
#     println(f)
# end