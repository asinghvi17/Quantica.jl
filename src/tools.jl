toSMatrix() = SMatrix{0,0,Float64}()
toSMatrix(s) = toSMatrix(tuple(s))
toSMatrix(ss::NTuple{M,NTuple{N,Number}}) where {N,M} = toSMatrix(SVector{N}.(ss))
toSMatrix(ss::NTuple{M,SVector{N}}) where {N,M} = hcat(ss...)

toSMatrix(::Type{T}, ss) where {T<:Number} = _toSMatrix(T, toSMatrix(ss))
_toSMatrix(::Type{T}, s::SMatrix{N,M}) where {N,M,T} = convert(SMatrix{N,M,T}, s)
# Dynamic dispatch
toSMatrix(s::AbstractMatrix) = SMatrix{size(s,1), size(s,2)}(s)
toSMatrix(s::AbstractVector) = toSMatrix(Tuple(s))

toSVector(::Tuple{}) = SVector{0,Float64}()
toSVector(v::SVector) = v
toSVector(v::NTuple{N,Any}) where {N} = SVector(v)
toSVector(x::Number) = SVector{1}(x)
toSVector(::Type{T}, v) where {T} = T.(toSVector(v))
toSVector(::Type{T}, ::Tuple{}) where {T} = SVector{0,T}()
# Dynamic dispatch
toSVector(v::AbstractVector) = SVector(Tuple(v))

unitvector(::Type{SVector{L,T}}, i) where {L,T} = SVector{L,T}(unitvector(NTuple{L,T}, i))
unitvector(::Type{CartesianIndex{L}}, i) where {L} = CartesianIndex(unitvector(NTuple{L,Int}, i))
unitvector(::Type{NTuple{L,T}}, i) where {L,T} =ntuple(j -> j == i ? one(T) : zero(T), Val(L))

ensuretuple(s::Tuple) = s
ensuretuple(s) = (s,)

indstopair(s::Tuple) = Pair(last(s), first(s))

filltuple(x, L) = ntuple(_ -> x, L)
filltuple(x, ::NTuple{N,Any}) where {N} = ntuple(_ -> x, Val(N))

# # toSVector can deal with the L=0 edge case, unlike SVector
# frontsvec(x, ::Val{L}) where {L} = toSVector(ntuple(i->x[i], Val(L)))
# tailtuple(x::SVector{N}, ::Val{L}) where {N,L} = ntuple(i->x[N+i-L], Val(L))

@inline tuplejoin() = ()
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

tuplesplice(s::NTuple{N,T}, ind, el) where {N,T} = ntuple(i -> i === ind ? T(el) : s[i], Val(N))

shiftleft(s::NTuple{N,Any}, n = 1) where {N} = ntuple(i -> s[mod1(i+n, N)], Val(N))
shiftright(s, n = 1) = shiftleft(s, -n)

tupleproduct(p1, p2) = tupleproduct(ensuretuple(p1), ensuretuple(p2))
tupleproduct(p1::NTuple{M,Any}, p2::NTuple{N,Any}) where {M,N} =
    ntuple(i -> (p1[1+fld(i-1, N)], p2[1+mod(i-1, N)]), Val(M * N))

tupleswapfront(tup::NTuple{L}, (i, j)) where {L} =
    i < j ? swap(swap(tup, i => 1), j => 2) : swap(swap(tup, j => 2), i => 1)

swap(tup::NTuple{L}, (i, i´)) where {L} =
    ntuple(l -> tup[ifelse(l == i´, i, ifelse(l == i, i´, l))], Val(L))

tuplepairs(::Val{V}) where {V} = tuplepairs((), ntuple(identity, Val(V)))
tuplepairs(c::Tuple, ::Tuple{}) = c

function tuplepairs(c::Tuple, r::NTuple{V}) where {V}
    t = Base.tail(r)
    c´ = (c..., tuple.(first(r), t)...)
    return tuplepairs(c´, t)
end

# Base.tail(t) .- first(t) but avoiding rounding errors in difference
tuple_minus_first(t::Tuple{T,Vararg{T,D}}) where {D,T} =
    ntuple(i -> ifelse(t[i+1] ≈ t[1], zero(T), t[i+1] - t[1]), Val(D))

firsttail(t::Tuple) = first(t), Base.tail(t)

frontSVector(s::SVector) = SVector(Base.front(Tuple(s)))

mergetuples(ts...) = keys(merge(tonamedtuple.(ts)...))

tonamedtuple(ts::Val{T}) where {T} = NamedTuple{T}(filltuple(0,T))

function deletemultiple_nocheck(dn::SVector{N}, axes::NTuple{M,Int}) where {N,M}
    ind = first(axes)
    dn´ = deleteat(dn, ind)
    taxes = Base.tail(axes)
    axes´ = taxes .- (taxes .> ind)
    return deletemultiple_nocheck(dn´, axes´)
end
deletemultiple_nocheck(dn::SVector, axes::Tuple{}) = dn

_rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

# zerotuple(::Type{T}, ::Val{L}) where {T,L} = ntuple(_ -> zero(T), Val(L))

function padright!(v::Vector, x, n::Integer)
    n0 = length(v)
    resize!(v, max(n, n0))
    for i in (n0 + 1):n
        v[i] = x
    end
    return v
end

padright(sv::StaticVector{E,T}, x::T, ::Val{E}) where {E,T} = sv
padright(sv::StaticVector{E1,T1}, x::T2, ::Val{E2}) where {E1,T1,E2,T2} =
    (T = promote_type(T1,T2); SVector{E2,T}(padright(Tuple(sv), x, Val(E2))))
padright(sv::StaticVector{E,T}, ::Val{E2}) where {E,T,E2} = padright(sv, zero(T), Val(E2))
padright(sv::StaticVector{E,T}, ::Val{E}) where {E,T} = sv
padright(t::NTuple{N´,Any}, x, ::Val{N}) where {N´,N} = ntuple(i -> i > N´ ? x : t[i], Val(N))
padright(t::NTuple{N´,Any}, ::Val{N}) where {N´,N} = ntuple(i -> i > N´ ? 0 : t[i], Val(N))

padright(v, ::Type{<:Number}) = first(v)
padright(v, ::Type{S}) where {E,T,S<:SVector{E,T}} = padright(v, zero(T), S)
padright(v, x::T, ::Type{S}) where {E,T,S<:SVector{E,T}} =
    SVector{E,T}(ntuple(i -> i > length(v) ? x : convert(T, v[i]), Val(E)))

# Pad element type to a "larger" type
@inline padtotype(s::SMatrix{E,L}, ::Type{S}) where {E,L,E2,L2,S<:SMatrix{E2,L2}} =
    S(SMatrix{E2,E}(I) * s * SMatrix{L,L2}(I))
@inline padtotype(s::StaticVector, ::Type{S}) where {N,T,S<:SVector{N,T}} =
    padright(T.(s), Val(N))
@inline padtotype(x::Number, ::Type{S}) where {E,L,S<:SMatrix{E,L}} =
    S(x * (SMatrix{E,1}(I) * SMatrix{1,L}(I)))
@inline padtotype(s::Number, ::Type{S}) where {N,T,S<:SVector{N,T}} =
    padright(SA[T(s)], Val(N))
@inline padtotype(x::Number, ::Type{T}) where {T<:Number} = T(x)
@inline padtotype(u::UniformScaling, ::Type{T}) where {T<:Number} = T(u.λ)
@inline padtotype(u::UniformScaling, ::Type{S}) where {S<:SMatrix} = S(u)

## Work around BUG: -SVector{0,Int}() isa SVector{0,Union{}}
negative(s::SVector{L,<:Number}) where {L} = -s
negative(s::SVector{0,<:Number}) = s

display_as_tuple(v, prefix = "") = isempty(v) ? "()" :
    string("(", prefix, join(v, string(", ", prefix)), ")")

displayvectors(mat::SMatrix{E,L,<:AbstractFloat}; kw...) where {E,L} =
    ntuple(l -> round.(Tuple(mat[:,l]); kw...), Val(L))
displayvectors(mat::SMatrix{E,L,<:Integer}; kw...) where {E,L} =
    ntuple(l -> Tuple(mat[:,l]), Val(L))

chop(x::T) where {T} = ifelse(abs2(x) < eps(real(T)), zero(T), x)

# pseudoinverse of supercell s times an integer n, so that it is an integer matrix (for accuracy)
pinvmultiple(s::SMatrix{L,0}) where {L} = (SMatrix{0,0,Int}(), 0)
function pinvmultiple(s::SMatrix{L,L´}) where {L,L´}
    L < L´ && throw(DimensionMismatch("Supercell dimensions $(L´) cannot exceed lattice dimensions $L"))
    qrfact = qr(s)
    # Cannot check det(s) ≈ 0 because s can be non-square
    det(qrfact.R) ≈ 0 && throw(ErrorException("Supercell appears to be singular"))
    pinverse = inv(qrfact.R) * qrfact.Q'
    n = round.(Int, det(s's))
    npinverse = round.(Int, n * pinverse)
    return npinverse, n
end

pinverse(::SMatrix{E,0,T}) where {E,T} = SMatrix{0,E,T}() # BUG: workaround StaticArrays bug SMatrix{E,0,T}()'

function pinverse(m::SMatrix)
    qrm = qr(m)
    return inv(qrm.R) * qrm.Q'
end

issquare(a::AbstractMatrix) = size(a, 1) == size(a, 2)

# normalize_axis_directions(q::SMatrix{M,N}) where {M,N} = hcat(ntuple(i->q[:,i]*sign(q[i,i]), Val(N))...)

padprojector(::Type{S}, ::Val{N}) where {M,N,S<:SMatrix{M,M}} = S(Diagonal(SVector(padright(filltuple(1, Val(N)), Val(M)))))
padprojector(::Type{S}, ::NTuple{N,Any}) where {S,N} = padprojector(S, Val(N))

_blockdiag(s1::SMatrix{E1,L1,T1}, s2::SMatrix{E2,L2,T2}) where {E1,L1,T1,E2,L2,T2} = hcat(
    ntuple(j->vcat(s1[:,j], zero(SVector{E2,T2})), Val(L1))...,
    ntuple(j->vcat(zero(SVector{E1,T1}), s2[:,j]), Val(L2))...)

function isgrowing(vs::AbstractVector, i0 = 1)
    i0 > length(vs) && return true
    vprev = vs[i0]
    for i in i0 + 1:length(vs)
        v = vs[i]
        v <= vprev && return false
        vprev = v
    end
    return true
end

function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

chop(x::T, x0 = one(T)) where {T<:Real} = ifelse(abs2(x) < eps(T(x0)), zero(T), x)
chop(x::C, x0 = one(R)) where {R<:Real,C<:Complex{R}} = chop(real(x), x0) + im*chop(imag(x), x0)

# function chop!(A::AbstractArray{T}, atol = default_tol(T)) where {T}
#     for (i, a) in enumerate(A)
#         # if abs(a) < atol
#         #     A[i] = zero(T)
#         if !iszero(a) #&& (abs(real(a)) < atol || abs(imag(a)) < atol)
#             A[i] = chop(a)
#         elseif abs(a) > 1/atol || isnan(a)
#             A[i] = T(Inf)
#         end
#     end
#     return A
# end

default_tol(::Type{T}) where {T} = sqrt(eps(real(T)))

function unique_sorted_approx!(v::AbstractVector{T}) where {T}
    i = 1
    xprev = first(v)
    for j in 2:length(v)
        if v[j] ≈ xprev
            xprev = v[j]
        else
            i += 1
            xprev = v[i] = v[j]
        end
    end
    resize!(v, i)
    return v
end

normalize_columns!(kmat::AbstractMatrix) = normalize_columns!(kmat, axes(kmat, 2))

function normalize_columns!(kmat::AbstractMatrix, cols)
    for col in cols
        normalize!(view(kmat, :, col))
    end
    return kmat
end

normalize_columns(s::SMatrix{L,0}) where {L} = s
normalize_columns(s::SMatrix{0,L}) where {L} = s
normalize_columns(s) = mapslices(v -> v/norm(v), s, dims = 1)

eltypevec(::AbstractMatrix{T}) where {T<:Number} = T
eltypevec(::AbstractMatrix{S}) where {N,T<:Number,S<:SMatrix{N,N,T}} = SVector{N,T}

tuplesort((a,b)::Tuple{<:Number,<:Number}) = a > b ? (b, a) : (a, b)
tuplesort(t::Tuple) = t
tuplesort(::Missing) = missing

# Gram-Schmidt but with column normalization only when norm^2 >= threshold (otherwise zero!)
function orthonormalize!(m::AbstractMatrix, threshold = 0)
    @inbounds for j in axes(m, 2)
        col = view(m, :, j)
        for j´ in 1:j-1
            col´ = view(m, :, j´)
            norm2´ = dot(col´, col´)
            iszero(norm2´) && continue
            r = dot(col´, col)/norm2´
            col .-= r .* col´
        end
        norm2 = real(dot(col, col))
        factor = ifelse(norm2 < threshold, zero(norm2), 1/sqrt(norm2))
        col .*= factor
    end
    return m
end

# Like copyto! but with potentially different tensor orders (adapted from Base.copyto!, see #33588)
function copyslice!(dest::AbstractArray{T1,N1}, Rdest::CartesianIndices{N1},
                    src::AbstractArray{T2,N2}, Rsrc::CartesianIndices{N2}, by = identity) where {T1,T2,N1,N2}
    isempty(Rdest) && return dest
    if length(Rdest) != length(Rsrc)
        throw(ArgumentError("source and destination must have same length (got $(length(Rsrc)) and $(length(Rdest)))"))
    end
    checkbounds(dest, first(Rdest))
    checkbounds(dest, last(Rdest))
    checkbounds(src, first(Rsrc))
    checkbounds(src, last(Rsrc))
    src′ = Base.unalias(dest, src)
    @inbounds for (Is, Id) in zip(Rsrc, Rdest)
        @inbounds dest[Id] = by(src′[Is])
    end
    return dest
end

function append_slice!(dest::AbstractArray, src::AbstractArray{T,N}, Rsrc::CartesianIndices{N}) where {T,N}
    checkbounds(src, first(Rsrc))
    checkbounds(src, last(Rsrc))
    Rdest = (length(dest) + 1):(length(dest) + length(Rsrc))
    resize!(dest, last(Rdest))
    src′ = Base.unalias(dest, src)
    for (Is, Id) in zip(Rsrc, Rdest)
        @inbounds dest[Id] = src′[Is]
    end
    return dest
end

permutations(ss::NTuple) = permutations!(typeof(ss)[], ss, ())

function permutations!(p, s1, s2)
    for (i, s) in enumerate(s1)
        permutations!(p, delete(s1, i), (s2..., s))
    end
    return p
end

permutations!(p, ::Tuple{}, s2) = push!(p, s2)

delete(t::NTuple{N,Any}, i) where {N} = ntuple(j -> j < i ? t[j] : t[j+1], Val(N-1))

######################################################################
# convert a matrix/number block to a matrix/inlinematrix string
######################################################################

_isreal(x) = all(o -> imag(o) ≈ 0, x)
_isimag(x) = all(o -> real(o) ≈ 0, x)

matrixstring(row, x) = string("Onsite[$row] : ", _matrixstring(x))
matrixstring(row, col, x) = string("Hopping[$row, $col] : ", _matrixstring(x))

matrixstring_inline(row, x, digits) = string("Onsite[$row] : ", _matrixstring_inline(round.(x, sigdigits = digits)))
matrixstring_inline(row, col, x, digits) = string("Hopping[$row, $col] : ", _matrixstring_inline(round.(x, sigdigits = digits)))

_matrixstring(x::Number) = numberstring(x)
_matrixstring_inline(x::Number) = numberstring(x)
function _matrixstring(s::SMatrix)
    ss = repr("text/plain", s)
    pos = findfirst(isequal('\n'), ss)
    return pos === nothing ? ss : ss[pos:end]
end

function _matrixstring_inline(s::SMatrix{N}) where {N}
    stxt = numberstring.(transpose(s))
    stxt´ = vcat(stxt, SMatrix{1,N}(ntuple(_->";", Val(N))))
    return string("[", stxt´[1:end-1]..., "]")
end

numberstring(x) = _isreal(x) ? string(" ", real(x)) : _isimag(x) ? string(" ", imag(x), "im") : string(" ", x)

############################################################################################
######## fast sparse copy #  Revise after #33589 is merged #################################
############################################################################################

# Using broadcast .+= instead allocates unnecesarily
function _plain_muladd!(dst, src, α)
    @boundscheck checkbounds(dst, axes(src)...)
    for i in eachindex(src)
        @inbounds dst[i] += α * src[i]
    end
    return dst
end

# Only needed for dense <- sparse (#33589), copy!(sparse, sparse) is fine in v1.5+
function _fast_sparse_copy!(dst::AbstractMatrix{T}, src::SparseMatrixCSC) where {T}
    @boundscheck checkbounds(dst, axes(src)...)
    fill!(dst, zero(eltype(src)))
    for col in 1:size(src, 1)
        for p in nzrange(src, col)
            @inbounds dst[rowvals(src)[p], col] = nonzeros(src)[p]
        end
    end
    return dst
end

function _fast_sparse_muladd!(dst::AbstractMatrix{T}, src::SparseMatrixCSC, α = I) where {T}
    @boundscheck checkbounds(dst, axes(src)...)
    for col in 1:size(src, 1)
        for p in nzrange(src, col)
            @inbounds dst[rowvals(src)[p], col] += α * nonzeros(src)[p]
        end
    end
    return dst
end

rclamp(r1::UnitRange, r2::UnitRange) = isempty(r1) ? r1 : clamp(minimum(r1), extrema(r2)...):clamp(maximum(r1), extrema(r2)...)

iclamp(minmax, r::Missing) = minmax
iclamp((x1, x2), (xmin, xmax)) = (max(x1, xmin), min(x2, xmax))