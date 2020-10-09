#!/usr/bin/env julia

using SparseArrays
using Base.Iterators

function printml(arg)
    show(IOContext(stdout), "text/plain", arg)
    println()
end

function make_rebin_matrix_1d(sizein::T, ntogroup::T) where T
    sizeout = sizein÷ntogroup
    @assert sizein%ntogroup==0

    Kblock = sparse(ones(Int64, (ntogroup,1)))
    # println("Kblock: $(Array(Kblock))")

    blockdiag(repeated(Kblock, sizeout)...)
end

function make_rebin_matrices_2d(shapein::Tuple{T,T}, ntogroup::Tuple{T,T}) where T
    Kleft  = make_rebin_matrix_1d(shapein[1], ntogroup[1])'
    Kright = make_rebin_matrix_1d(shapein[2], ntogroup[2])

    Kleft, Kright
end

function check1d()
    sizein=12::Int64
    ngroup=4::Int64

    K = make_rebin_matrix_1d(sizein, ngroup)
    @time K = make_rebin_matrix_1d(sizein, ngroup)
    #println("Size in: $sizein")
    #println("N to group: $ngroup")
    #println("Shape K: $(size(K))")
    #println("K: $(Array(K))")
    #println()

    arrin1  = ones(Float64, (1,sizein))
    #println("Shape1 in: $(size(arrin1))")

    arrout1 = arrin1*K
    #println("Shape1 out: $(size(arrout1))")
    #println("Arr1 in: $arrin1")
    #println("Arr1 out: $arrout1")
    #println()

    arrin2 = ones(Float64, (3,sizein))
    arrin2[2,:].*=2
    arrin2[3,:].*=3
    #println("Shape2 in: $(size(arrin2))")

    arrout2 = arrin2*K
    #println("Shape2 out: $(size(arrout2))")
    #println("Arr2 in: $arrin2")
    #println("Arr2 out: $arrout2")
    #println()
end

function check2d()
    shapein=(6, 4)
    ngroup=(3, 2)

    Kleft, Kright = make_rebin_matrices_2d(shapein, ngroup)
    @time Kleft, Kright = make_rebin_matrices_2d(shapein, ngroup)

    #println("N to group: $ngroup")
    #println("Shape K left: $(size(Kleft))")
    #println("Shape in: $shapein")
    #println("Shape K right: $(size(Kright))")
    #println("K left:")
    #println(Kleft)
    #println("K right:")
    #printml(Array(Kright))
    #println()

    arrin1 = reshape(Array(range(0, length=prod(shapein))), reverse(shapein))'
    #println("Shape1 in: $(size(arrin1))")

    arrout1 = Kleft*arrin1*Kright
    #println("Shape1 out: $(size(arrout1))")
    #println("Arr1 in:")
    #printml(arrin1)
    #println("Arr1 out:")
    #printml(arrout1)
    #println()

    arrin2 = ones(Float64, shapein)
    arrin2[begin:3, begin:2] .= 1.0
    arrin2[begin:3, 3:end]   .= 2.0
    arrin2[4:end, begin:2]   .= 3.0
    arrin2[4:end, 3:end]     .= 4.0
    #println("Shape2 in: $(size(arrin2))")

    arrout2 = Kleft*arrin2*Kright
    #println("Shape2 out: $(size(arrout2))")
    #println("Arr2 in:")
    #printml(arrin2)
    #println("Arr2 out:")
    #printml(arrout2)
end

function main()
    check1d()
    check2d()
end

main()
