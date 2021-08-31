rdr((r1, r2)::Pair) = (0.5 * (r1 + r2), r2 - r1)

############################################################################################
# Sparse matrix transformations [involves OrbitalStructure's]
# all merged_* functions assume matching structure of sparse matrices
#region

# merge several sparse matrices onto the first using only structural zeros
function merge_sparse(mats)
    mat0 = first(mats)
    O = eltype(mat0)
    nrows, ncols = size(mat0)
    nrows == ncols || throw(ArgumentError("Internal error: matrix not square"))
    nnzguess = sum(mat -> nnz(mat), mats)
    collector = CSC{O}(ncols, nnzguess)
    for col in 1:ncols
        for (n, mat) in enumerate(mats)
            vals = nonzeros(mat)
            rows = rowvals(mat)
            for p in nzrange(mat, col)
                val = n == 1 ? vals[p] : zero(O)
                row = rows[p]
                pushtocolumn!(collector, row, val, false) # skips repeated rows
            end
        end
        finalizecolumn!(collector)
    end
    matrix = sparse(collector, ncols)
    return matrix
end

# flatten and merge several sparse matrices onto first according to OrbitalStructures
function merge_flatten_sparse(mats,
                              os::OrbitalStructure{<:SMatrix},
                              flatos::OrbitalStructure{T} = flatten(os)) where {T<:Number}
    mat0 = first(mats)
    check_orbstruct_consistency(mat0, os)
    norbs = norbitals(os)
    ncolsflatguess = size(mat0, 2) * maximum(norbs)
    nnzflatguess = nnz(mat0) * maximum(norbs)
    collector = CSC{T}(ncolsflatguess, nnzflatguess)
    multiple_matrices = length(mats) > 1
    needs_column_sort = multiple_matrices
    skip_column_dupcheck = !multiple_matrices
    for scol in sublats(os), col in siterange(os, scol)
        ncol = norbs[scol]
        for j in 1:ncol  # block column
            for (n, mat) in enumerate(mats)
                vals = nonzeros(mat)
                rows = rowvals(mat)
                for p in nzrange(mat, col)
                    row = rows[p]
                    rowoffset´, nrow = site_to_flatoffset_norbs(row, os, flatos)
                    for i in 1:nrow
                        val´ = n == 1 ? vals[p][i, j] : zero(T)
                        pushtocolumn!(collector, rowoffset´ + i, val´, skip_column_dupcheck)
                    end
                end
            end
            finalizecolumn!(collector, needs_column_sort)
        end
    end
    flatmat = sparse(collector, nsites(flatos))
    return flatmat
end

check_orbstruct_consistency(mat, os) = nsites(os) == size(mat, 1) == size(mat, 2) ||
    throw(ArgumentError("Matrix size $(size(mat)) inconsistent with number of sites $(nsites(os))"))

function site_to_sublat(siteidx, orbstruct)
    offsets´ = offsets(orbstruct)
    l = length(offsets´)
    for s in 2:l
        @inbounds offsets´[s] + 1 > siteidx && return s - 1
    end
    return l
end

function site_to_flatoffset_norbs(siteidx, orbstruct, flatorbstruct)
    s = site_to_sublat(siteidx, orbstruct)
    N = norbitals(orbstruct)[s]
    offset = offsets(orbstruct)[s]
    offset´ = offsets(flatorbstruct)[s]
    Δi = siteidx - offset
    flatoffset = offset´ + (Δi - 1) * N
    return flatoffset, N
end

flatten(mat::SparseMatrixCSC{O}, os::OrbitalStructure{O}, flatos::OrbitalStructure{<:Number} = flatten(os)) where {O<:SMatrix} =
    merge_flatten_sparse((mat,), os, flatos)

# merged_mul! (mul! specializations assuming target does not need to change structure [is "merged"])

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:Number}, _, A::SparseMatrixCSC{<:Number}, b::Number, α, β) =
    merged_mul!(C, A, b, α, β)

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:SMatrix{N,N}}, _, A::SparseMatrixCSC{<:SMatrix{N,N}}, b::Number, α, β) where {N} =
    merged_mul!(C, A, b, α, β)

maybe_flatten_merged_mul!(C::SparseMatrixCSC{<:Number}, os, A::SparseMatrixCSC{<:SMatrix}, b::Number, α, β) =
    flatten_merged_mul!(C, os, A, b, α, β)

function merged_mul!(C::SparseMatrixCSC, A::SparseMatrixCSC, b::Number, α = 1, β = 0)
    nzA = nonzeros(A)
    nzC = nonzeros(C)
    if length(nzA) == length(nzC)  # assume idential structure (C has merged structure)
        @. nzC = β * nzC + α * b * nzA
    else
        for col in axes(A, 2), p in nzrange(A, col)
            row = rowvals(A)[p]
            for p´ in nzrange(C, col)
                row´ = rowvals(C)[p´]
                if row == row´
                    nzC[p´] = β * nzC[p´] + α * b * nzA[p]
                    break
                end
            end
        end
    end
    return C
end

function flatten_merged_mul!(C::SparseMatrixCSC, (os, flatos), A::SparseMatrixCSC, b::Number, α , β = 0)
    colsA = axes(A, 2)
    rowsA = rowvals(A)
    valsA = nonzeros(A)
    rowsC = rowvals(C)
    valsC = nonzeros(C)
    for col in colsA
        coloffset´, ncol = site_to_flatoffset_norbs(col, os, flatos)
        for p in nzrange(A, col)
            valA = valsA[p]
            row  = rowsA[p]
            rowoffset´, nrow = site_to_flatoffset_norbs(row, os, flatos)
            rowfirst´ = rowoffset´ + 1
            for ocol in 1:ncol
                col´ = coloffset´ + ocol
                for p´ in nzrange(C, col´)
                    if rowsC[p´] == rowfirst´
                        for orow in 1:nrow
                            p´´ = p´ + orow - 1
                            valsC[p´´] = β * valsC[p´´] + α * b * valA[orow, ocol]
                        end
                        break
                    end
                end
            end
        end
    end
    return C
end

#endregion