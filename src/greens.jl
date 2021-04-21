#######################################################################
# Green's function
#######################################################################
abstract type AbstractGreensSolver end

struct GreensFunction{S<:AbstractGreensSolver,L,B<:NTuple{L,Union{Int,Missing}},H<:Hamiltonian}
    solver::S
    h::H
    boundaries::B
end

"""
    greens(h::Hamiltonian, solveobject; boundaries::NTuple{L,Integer} = missing)

Construct the Green's function `g::GreensFunction` of `L`-dimensional Hamiltonian `h` using
the provided `solveobject`. Currently valid `solveobject`s are

- the `Bandstructure` of `h` (for an unbounded `h` or an `Hamiltonian{<:Superlattice}}`)
- the `Spectrum` of `h` (for a bounded `h`)
- `SingleShot1D(; direct = false)` (single-shot generalized [or direct if `direct = true`] eigenvalue approach for 1D Hamiltonians)

If a `boundaries = (n₁, n₂, ...)` is provided, a reflecting boundary is assumed for each
non-missing `nᵢ` perpendicular to Bravais vector `i` at a cell distance `nᵢ` from the
origin.

    h |> greens(h -> solveobject(h), args...)

Curried form equivalent to the above, giving `greens(h, solveobject(h), args...)`.

    g(ω, cells::Pair)

From a constructed `g::GreensFunction`, obtain the retarded Green's function matrix at
frequency `ω` between unit cells `src` and `dst` by calling `g(ω, src => dst)`, where `src,
dst` are `::NTuple{L,Int}` or `SVector{L,Int}`. If not provided, `cells` default to
`(1, 1, ...) => (1, 1, ...)`.

    g(ω, missing)

If allowed by the used `solveobject`, build an efficient function `cells -> g(ω, cells)`
that can produce the Greens function between different cells at fixed `ω` without repeating
cell-independent parts of the computation.

# Examples

```jldoctest
julia> g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> greens(bandstructure(resolution = 17))
GreensFunction{Bandstructure}: Green's function from a 2D bandstructure
  Matrix size    : 1 × 1
  Element type   : scalar (Complex{Float64})
  Band simplices : 512

julia> g(0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im

julia> m = similarmatrix(g); g(m, 0.2)
1×1 Array{Complex{Float64},2}:
 6.663377810046025 - 24.472789025006396im
```

# See also
    `greens!`, `SingleShot1D`
"""
greens(h::Hamiltonian{<:Any,L}, solverobject; boundaries = filltuple(missing, Val(L))) where {L} =
    GreensFunction(greensolver(solverobject, h), h, boundaries)
greens(solver::Function, args...; kw...) = h -> greens(h, solver(h), args...; kw...)

# solver fallback
greensolver(s::AbstractGreensSolver) = s

# call API fallback
(g::GreensFunction)(ω, cells = default_cells(g)) = greens!(similarmatrix(g), g, ω, cells)

similarmatrix(g::GreensFunction, type = Matrix{blocktype(g.h)}) = similarmatrix(g.h, type)

greens!(matrix, g, ω, cells) = greens!(matrix, g, ω, sanitize_cells(cells, g))

default_cells(::GreensFunction{S,L}) where {S,L} = filltuple(1, Val(L)) => filltuple(1, Val(L))

sanitize_cells((cell0, cell1)::Pair{<:Integer,<:Integer}, ::GreensFunction{S,1}) where {S} =
    SA[cell0] => SA[cell1]
sanitize_cells((cell0, cell1)::Pair{<:NTuple{L,Integer},<:NTuple{L,Integer}}, ::GreensFunction{S,L}) where {S,L} =
    SVector(cell0) => SVector(cell1)
sanitize_cells(cells, g::GreensFunction{S,L}) where {S,L} =
    throw(ArgumentError("Cells should be of the form `cᵢ => cⱼ`, with each `c` an `NTuple{$L,Integer}`"))

const SVectorPair{L} = Pair{SVector{L,Int},SVector{L,Int}}

#######################################################################
# SingleShot1DGreensSolver
#######################################################################
using QuadEig: Linearization

"""
    SingleShot1D(; direct = false)

Return a Greens function solver using the generalized eigenvalue approach, whereby given the
energy `ω`, the eigenmodes of the infinite 1D Hamiltonian, and the corresponding infinite
and semi-infinite Greens function can be computed by solving the generalized eigenvalue
equation

    A⋅φχ = λ B⋅φχ
    A = [0 I; V ω-H0]
    B = [I 0; 0 V']

This is the matrix form of the problem `λ(ω-H0)φ - Vφ - λ²V'φ = 0`, where `φχ = [φ; λφ]`,
and `φ` are `ω`-energy eigenmodes, with (possibly complex) momentum `q`, and eigenvalues are
`λ = exp(-iqa₀)`. The algorithm assumes the Hamiltonian has only `dn = (0,)` and `dn = (±1,
)` Bloch harmonics (`H0`, `V` and `V'`), so its unit cell will be enlarged before applying
the solver if needed. Bound states in the spectrum will yield delta functions in the density
of states that can be resolved by adding a broadening in the form of a small positive
imaginary part to `ω`.

To avoid singular solutions `λ=0,∞`, the nullspace of `V` is projected out of the problem.
This produces a new `A´` and `B´` with reduced dimensions. `B´` can often be inverted,
turning this into a standard eigenvalue problem, which is slightly faster to solve. This is
achieved with `direct = true`. However, `B´` sometimes is still non-invertible for some
values of `ω`. In this case use `direct = false` (the default).

# Examples
```jldoctest
julia> using LinearAlgebra

julia> h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), (10,10)) |> Quantica.wrap(2);

julia> g = greens(h, SingleShot1D(), boundaries = (0,))
GreensFunction{SingleShot1DGreensSolver}: Green's function using the single-shot 1D method
  Matrix size    : 40 × 40
  Reduced size   : 20 × 20
  Element type   : scalar (ComplexF64)
  Boundaries     : (0,)

julia> tr(g(0.3))
-32.193416068730684 - 3.4399800712973074im
```

# See also
    `greens`
"""
struct SingleShot1D{R}
    atol::R  # could be missing for default_tol(T)
end

SingleShot1D(; atol = missing) = SingleShot1D(atol)

struct SingleShot1DGreensSolver{T,M,R<:Real} <: AbstractGreensSolver
    h0::M
    hp::M
    hm::M
    hLR::Matrix{T}
    hRL::Matrix{T}
    lin::Linearization{T,M}
    Aω0::M
    Adense::Matrix{T}
    Bdense::Matrix{T}
    QIV::M
    L::Matrix{T}
    R::Matrix{T}
    m1::Matrix{T}  # prealloc n x n
    m2::Matrix{T}  # prealloc n x n
    m3::Matrix{T}  # prealloc n x n
    atol::R
end

function Base.show(io::IO, g::GreensFunction{<:SingleShot1DGreensSolver})
    print(io, summary(g), "\n",
"  Matrix size    : $(size(g.solver.h0, 1)) × $(size(g.solver.h0, 2))
  Deflated size  : $(size(g.solver.Adense, 1)) × $(size(g.solver.Adense, 2))
  Element type   : $(displayelements(g.h))
  Boundaries     : $(g.boundaries)")
end

Base.summary(g::GreensFunction{<:SingleShot1DGreensSolver}) =
    "GreensFunction{SingleShot1DGreensSolver}: Green's function using the single-shot 1D method"

# hasbulk(gs::SingleShot1DGreensSolver) = !iszero(size(gs.Pb, 1))

function greensolver(s::SingleShot1D, h)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a SingleShot1D Green function solver with an $(latdim(h))-dimensional Hamiltonian"))
    maxdn = max(1, maximum(har -> abs(first(har.dn)), h.harmonics))
    H = flatten(maxdn == 1 ? h : unitcell(h, (maxdn,)))
    T = complex(blockeltype(H))
    A0, A1, A2 = -H[(1,)], -H[(0,)], -H[(-1,)]
    n = size(A1, 1)
    atol   = s.atol === missing ? default_tol(T) : s.atol
    # Fourth companion linearization C4(A0, A1, A2), built as the reverse (A↔B) of the deflated C2(A2, A1, A0)
    lin    = linearize(A2, A1, A0; atol)
    Aω0    = copy(lin.A)                       # store the ω=0 matrix A to be able to update l
    QIV    = lin.Q * [I(n) 0I; 0I 0I] * lin.V  # We shift Aω0 before deflating by -ω*QIV to turn h0 into h0 - ω
    R      = Matrix(last(nullspace_decomposition(A0, atol)))
    L      = Matrix(last(nullspace_decomposition(A2, atol)))
    l, r   = size(L, 2), size(R, 2)
    Adense = Matrix{T}(undef, 2r, 2r)          # Needs to be dense for schur!(Aω, Bω)
    Bdense = Matrix{T}(undef, 2r, 2r)
    h0     = H[(0,)]
    hp     = H[(1,)]
    hm     = H[(-1,)]
    hLR    = -L'*A0*R
    hRL    = -R'*A2*L
    m1     = Matrix{T}(undef, n, n)            # temporary
    m2     = Matrix{T}(undef, n, n)            # temporary
    m3     = Matrix{T}(undef, n, n)            # temporary
    return SingleShot1DGreensSolver(h0, hp, hm, hLR, hRL, lin, Aω0, Adense, Bdense, QIV, L, R, m1, m2, m3, atol)
end

function get_C4_AB(d::Linearization, s::SingleShot1DGreensSolver)
    copy!(s.Adense, d.B)
    copy!(s.Bdense, d.A)
    return s.Adense, s.Bdense
end

function idsLR(s)
    l, r = size(s.L, 2), size(s.R, 2)
    iL, iR = 1:l, l+1:l+r
    return iL, iR
end

function shiftω!(s::SingleShot1DGreensSolver, ω)
    s.lin.A .= s.Aω0 .+ ω * s.QIV
    return s
end

function nullspace_decomposition(mat, atol)
    q´ = QuadEig.pqr(copy(mat'))
    basis = QuadEig.getQ(q´)
    RP´ = QuadEig.getRP´(q´)
    r = QuadEig.nonzero_rows(RP´, atol)
    n = size(basis, 2)
    complement, kernel = view(basis, :, 1:r), view(basis, :, r+1:n)
    return kernel, complement
end

## Solver execution: compute self-energy

function (s::SingleShot1DGreensSolver{T,M})(ω) where {T,M}
    n = size(s.h0, 1)

    if s.atol <= 0
        AA = [ω*I - s.h0 -s.hp; -I 0I]
        BB = [s.hm 0I; 0I -I]

        sch = schur(Matrix(AA), Matrix(BB))
        ordschur!(sch, abs.(sch.α ./ sch.β) .<= 1)
        # @show sort(abs.(sch.α ./ sch.β)[1:n])
        ϕΛR⁻¹ = view(sch.Z, 1:n, 1:n)
        ϕR⁻¹ = view(sch.Z, n+1:2n, 1:n)
        ΣR = s.hm * ϕΛR⁻¹ / ϕR⁻¹
    else
        shiftω!(s, ω)
        d = deflate(s.lin)
        ΣR = selfenergy(d, s, ω)
    end

    GR⁻¹ = ω*I - s.h0 - ΣR
    luGR⁻¹ = lu(GR⁻¹)
    Gh₊ = ldiv!(luGR⁻¹, copy!(s.m1, s.hp))
    Gh₋ = ldiv!(luGR⁻¹, copy!(s.m2, s.hm))

    # @show sum(abs.(ΣR - s.hm * ((ω*I - s.h0 - ΣR) \ Matrix(s.hp))))
    # @show sum(abs.(Σ0 - s.hm * ((ω*I - s.h0 - Σ0) \ Matrix(s.hp))))

    G∞⁻¹ = GR⁻¹
    luG∞⁻¹ = lu(G∞⁻¹)

    return luG∞⁻¹, Gh₊, Gh₋
end

function selfenergy(d::Linearization{T,M}, s::SingleShot1DGreensSolver, ω) where {T,M}
    A, B = get_C4_AB(d, s)
    n = size(d.V, 1) ÷ 2
    ndeflated = size(A, 1) ÷ 2
    V1 = view(d.V, 1:n, :)
    V2 = view(d.V, n+1:2n, :)

    sch = schur!(A, B)
    retarded = s.atol .<= abs.(sch.α ./ sch.β) .<= 1
    nretarded = sum(retarded)
    ordschur!(sch, retarded)
    # @show sort(abs.(sch.α ./ sch.β)[1:nretarded])
    Z11 = V1 * view(sch.Z, :, 1:nretarded)
    Z21 = V2 * view(sch.Z, :, 1:nretarded)

    defect = ndeflated - nretarded
    skip_jordan = false
    if defect == 0 && skip_jordan
        source = Z11
        target = Z21
    else
        ig0, hLR, hRL = effective_model(s, ω)
        φs, h₋Jφs = jordan_chain(ig0, hLR, hRL, s)
        source = [φs Z11]
        target = [h₋Jφs Z21]
    end

    # @show size(s.R'*source), cond(s.R'*source)
    # @show size(s.R'*Z11), cond(s.R'*Z11)

    Σ = M(target * ((s.R'*source) \ s.R'))

    return Σ
end

function effective_model(s, ω)
    L, R = s.L, s.R
    luA1 = lu(ω*I - s.h0)
    iA1R, iA1L = luA1 \ R, luA1 \ L
    ig0 = inv([L'*iA1L L'*iA1R; R'*iA1L R'*iA1R])
    hLR = s.hLR
    hRL = s.hRL
    return ig0, hLR, hRL
end

function jordan_chain(g0⁻¹, h₊, h₋, s)
    local ΣRR
    iL, iR = idsLR(s)
    G0 = inv(g0⁻¹)
    GLL = view(G0, iL, iL)
    GRLh₊ = view(G0, iR, iL)*h₊
    ns, _ = nullspace_decomposition(GRLh₊, s.atol)
    nullity = size(ns, 2)
    while true
        ΣRR = h₋*GLL*h₊
        G0 = inv(g0⁻¹ - [0I 0I; 0I ΣRR])
        GLL = view(G0, iL, iL)
        GRLh₊´ = GRLh₊ * view(G0, iR, iL)
        ns, _ = nullspace_decomposition(GRLh₊´, s.atol)
        nullity == size(ns, 2) && break
        nullity = size(ns, 2)
        GRLh₊ = GRLh₊´
    end
    φs = s.R * ns
    h₋Jφs = s.R * ΣRR * ns
    return φs, h₋Jφs
end

function phi_chain(ψn, ψn´, s)
    r = size(s.Z0, 2)
    n = size(ψn´, 1)
    φZ0 = s.Z0 * view(ψn´, 1:r, :)
    φQ0 = s.Q0 * view(ψn, r+1:n, :)
    φ = [φZ0; φQ0]
    return φ
end

## Greens execution

(g::GreensFunction{<:SingleShot1DGreensSolver})(ω, cells) = g(ω, missing)(sanitize_cells(cells, g))

# Infinite: G∞_{N}  = GVᴺ G∞_{0}
function (g::GreensFunction{<:SingleShot1DGreensSolver,1,Tuple{Missing}})(ω, ::Missing)
    G∞⁻¹, Gh₊, Gh₋ = g.solver(ω)
    return cells -> G_infinite(G∞⁻¹, Gh₊, Gh₋, cells)
end

function G_infinite(G∞⁻¹, Gh₊, Gh₋, (src, dst))
    N = only(dst) - only(src)
    N == 0 && return inv(G∞⁻¹)
    Ghᴺ = GVᴺ(Gh₊, Gh₋, N)
    G∞ = rdiv!(Ghᴺ, G∞⁻¹)
    return G∞
end

# Semiinifinite: G_{N,M} = (GVᴺ⁻ᴹ - GVᴺGV⁻ᴹ)G∞_{0}
function (g::GreensFunction{<:SingleShot1DGreensSolver,1,Tuple{Int}})(ω, ::Missing)
    gs = g.solver
    G∞⁻¹, Gh₊, Gh₋ = gs(ω)
    N0 = only(g.boundaries)
    return cells -> G_semiinfinite!(gs, G∞⁻¹, Gh₊, Gh₋, shift_cells(cells, N0))
end

function G_semiinfinite!(gs::SingleShot1DGreensSolver{T}, G∞⁻¹, Gh₊, Gh₋, (src, dst)) where {T}
    M = only(src)
    N = only(dst)
    if sign(N) != sign(M)
        G∞ = fill!(gs.m3, zero(T))
    else
        GVᴺ⁻ᴹ = GVᴺ(Gh₊, Gh₋, N-M)
        GVᴺ = GVᴺ(Gh₊, Gh₋, N)
        GV⁻ᴹ = GVᴺ(Gh₊, Gh₋, -M)
        mul!(GVᴺ⁻ᴹ, GVᴺ, GV⁻ᴹ, -1, 1) # (GVᴺ⁻ᴹ - GVᴺGV⁻ᴹ)
        G∞ = rdiv!(GVᴺ⁻ᴹ , G∞⁻¹)
    end
    return G∞
end

GVᴺ(Gh₊, Gh₋, N) = N == 0 ? one(Gh₊) : N > 0 ? Gh₊^N : Gh₋^-N

shift_cells((src, dst), N0) = (only(src) - N0, only(dst) - N0)

#######################################################################
# BandGreensSolver
#######################################################################
# struct SimplexData{D,E,T,C<:SMatrix,DD,SA<:SubArray}
#     ε0::T
#     εmin::T
#     εmax::T
#     k0::SVector{D,T}
#     Δks::SMatrix{D,D,T,DD}     # k - k0 = Δks * z
#     volume::T
#     zvelocity::SVector{D,T}
#     edgecoeffs::NTuple{E,Tuple{T,C}} # s*det(Λ)/w.w and Λc for each edge
#     dωzs::NTuple{E,NTuple{2,SVector{D,T}}}
#     defaultdη::SVector{D,T}
#     φ0::SA
#     φs::NTuple{D,SA}
# end

# struct BandGreensSolver{P<:SimplexData,E,H<:Hamiltonian} <: AbstractGreensSolver
#     simplexdata::Vector{P}
#     indsedges::NTuple{E,Tuple{Int,Int}} # all distinct pairs of 1:V, where V=D+1=num verts
#     h::H
# end

# function Base.show(io::IO, g::GreensFunction{<:BandGreensSolver})
#     print(io, summary(g), "\n",
# "  Matrix size    : $(size(g.h, 1)) × $(size(g.h, 2))
#   Element type   : $(displayelements(g.h))
#   Band simplices : $(length(g.solver.simplexdata))")
# end

# Base.summary(g::GreensFunction{<:BandGreensSolver}) =
#     "GreensFunction{Bandstructure}: Green's function from a $(latdim(g.h))D bandstructure"

# function greensolver(b::Bandstructure{D}, h) where {D}
#     indsedges = tuplepairs(Val(D))
#     v = [SimplexData(simplex, band, indsedges) for band in bands(b) for simplex in band.simplices]
#     return BandGreensSolver(v,  indsedges, h)
# end

# edges_per_simplex(L) = binomial(L,2)

# function SimplexData(simplex::NTuple{V}, band, indsedges) where {V}
#     D = V - 1
#     vs = ntuple(i -> vertices(band)[simplex[i]], Val(V))
#     ks = ntuple(i -> SVector(Base.front(Tuple(vs[i]))), Val(V))
#     εs = ntuple(i -> last(vs[i]), Val(V))
#     εmin, εmax = extrema(εs)
#     ε0 = first(εs)
#     k0 = first(ks)
#     Δks = hcat(tuple_minus_first(ks)...)
#     zvelocity = SVector(tuple_minus_first(εs))
#     volume = abs(det(Δks))
#     edgecoeffs = edgecoeff.(indsedges, Ref(zvelocity))
#     dωzs = sectionpoint.(indsedges, Ref(zvelocity))
#     defaultdη = dummydη(zvelocity)
#     φ0 = vertexstate(first(simplex), band)
#     φs = vertexstate.(Base.tail(simplex), Ref(band))
#     return SimplexData(ε0, εmin, εmax, k0, Δks, volume, zvelocity, edgecoeffs, dωzs, defaultdη, φ0, φs)
# end

# function edgecoeff(indsedge, zvelocity::SVector{D}) where {D}
#     basis = edgebasis(indsedge, Val(D))
#     othervecs = Base.tail(basis)
#     edgevec = first(basis)
#     cutvecs = (v -> dot(zvelocity, edgevec) * v - dot(zvelocity, v) * edgevec).(othervecs)
#     Λc = hcat(cutvecs...)
#     Λ = hcat(zvelocity, Λc)
#     s = sign(det(hcat(basis...)))
#     coeff = s * (det(Λ)/dot(zvelocity, zvelocity))
#     return coeff, Λc
# end

# function edgebasis(indsedge, ::Val{D}) where {D}
#     inds = ntuple(identity, Val(D+1))
#     swappedinds = tupleswapfront(inds, indsedge) # places the two edge vertindices first
#     zverts = (i->unitvector(SVector{D,Int}, i-1)).(swappedinds)
#     basis = (z -> z - first(zverts)).(Base.tail(zverts)) # first of basis is edge vector
#     return basis
# end

# function sectionpoint((i, j), zvelocity::SVector{D,T}) where {D,T}
#     z0, z1 = unitvector(SVector{D,Int}, i-1), unitvector(SVector{D,Int}, j-1)
#     z10 = z1 - z0
#     # avoid numerical cancellation errors due to zvelocity perpendicular to edge
#     d = chop(dot(zvelocity, z10), maximum(abs.(zvelocity)))
#     dzdω = z10 / d
#     dz0 = z0 - z10 * dot(zvelocity, z0) / d
#     return dzdω, dz0   # The section z is dω * dzdω + dz0
# end

# # A vector, not parallel to zvelocity, and with all nonzero components and none equal
# function dummydη(zvelocity::SVector{D,T}) where {D,T}
#     (D == 1 || iszero(zvelocity)) && return SVector(ntuple(i->T(i), Val(D)))
#     rng = MersenneTwister(0)
#     while true
#         dη = rand(rng, SVector{D,T})
#         isparallel = dot(dη, zvelocity)^2 ≈ dot(zvelocity, zvelocity) * dot(dη, dη)
#         isvalid = allunique(dη) && !isparallel
#         isvalid && return dη
#     end
#     throw(error("Unexpected error finding dummy dη"))
# end

# function vertexstate(ind, band)
#     ϕind = 1 + band.dimstates*(ind - 1)
#     state = view(band.states, ϕind:(ϕind+band.dimstates-1))
#     return state
# end

# ## Call API

# function greens!(matrix, g::GreensFunction{<:BandGreensSolver,L}, ω::Number, (src, dst)::SVectorPair{L}) where {L}
#     fill!(matrix, zero(eltype(matrix)))
#     dn = dst - src
#     for simplexdata in g.solver.simplexdata
#         g0, gjs = green_simplex(ω, dn, simplexdata, g.solver.indsedges)
#         addsimplex!(matrix, g0, gjs, simplexdata)
#     end
#     return matrix
# end

# function green_simplex(ω, dn, data::SimplexData{L}, indsedges) where {L}
#     dη = data.Δks' * dn
#     phase = cis(dot(dn, data.k0))
#     dω = ω - data.ε0
#     gz = simplexterm.(dω, Ref(dη), Ref(data), data.edgecoeffs, data.dωzs, indsedges)
#     g0z, gjz = first.(gz), last.(gz)
#     g0 = im^(L-1) * phase * sum(g0z)
#     gj = -im^L * phase * sum(gjz)
#     return g0, gj
# end

# function simplexterm(dω, dη::SVector{D,T}, data, coeffs, (dzdω, dz0), (i, j)) where {D,T}
#     bailout = Complex(zero(T)), Complex.(zero(dη))
#     z = dω * dzdω + dz0
#     # Edges with divergent sections do not contribute
#     all(isfinite, z) || return bailout
#     z0 = unitvector(SVector{D,T},i-1)
#     z1 = unitvector(SVector{D,T},j-1)
#     coeff, Λc = coeffs
#     # If dη is zero (DOS) use a precomputed (non-problematic) simplex-constant vector
#     dη´ = iszero(dη) ? data.defaultdη : dη
#     d = dot(dη´, z)
#     d0 = dot(dη´, z0)
#     d1 = dot(dη´, z1)
#     # Skip if singularity in formula
#     (d ≈ d0 || d ≈ d1) && return bailout
#     s = sign(dot(dη´, dzdω))
#     coeff0 = coeff / prod(Λc' * dη´)
#     coeffj = isempty(Λc) ? zero(dη) : (Λc ./ ((dη´)' * Λc)) * sumvec(Λc)
#     params = s, d, d0, d1
#     zs = z, z0, z1
#     g0z = iszero(dη) ? g0z_asymptotic(D, coeff0, params) : g0z_general(coeff0, params)
#     gjz = iszero(dη) ? gjz_asymptotic(D, g0z, coeffj, coeff0, zs, params) :
#                        gjz_general(g0z, coeffj, coeff0, zs, params)
#     return g0z, gjz
# end

# sumvec(::SMatrix{N,M,T}) where {N,M,T} = SVector(ntuple(_->one(T),Val(M)))

# g0z_general(coeff0, (s, d, d0, d1)) =
#     coeff0 * cis(d) * ((cosint_c(-s*(d0-d)) + im*sinint(d0-d)) - (cosint_c(-s*(d1-d)) + im*sinint(d1-d)))

# gjz_general(g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1)) =
#     g0z * (im * z - coeffj) + coeff0 * ((z0-z) * cis(d0) / (d0-d) - (z1-z) * cis(d1) / (d1-d))

# g0z_asymptotic(D, coeff0, (s, d, d0, d1)) =
#     coeff0 * (cosint_a(-s*(d0-d)) - cosint_a(-s*(d1-d))) * (im*d)^(D-1)/factorial(D-1)

# function gjz_asymptotic(D, g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1))
#     g0z´ = g0z
#     for n in 1:(D-1)
#         g0z´ += coeff0 * im^n * (im*d)^(D-1-n)/factorial(D-1-n) *
#                 ((d0-d)^n - (d1-d)^n)/(n*factorial(n))
#     end
#     gjz = g0z´ * (im * z - im * coeffj * d / D) +
#         coeff0 * ((z0-z) * (im*d0)^D / (d0-d) - (z1-z) * (im*d1)^D / (d1-d)) / factorial(D)
#     return gjz
# end

# cosint_c(x::Real) = ifelse(iszero(abs(x)), zero(x), cosint(abs(x))) + im*pi*(x<=0)

# cosint_a(x::Real) = ifelse(iszero(abs(x)), zero(x), log(abs(x))) + im*pi*(x<=0)

# function addsimplex!(matrix, g0, gjs, simplexdata)
#     φ0 = simplexdata.φ0
#     φs = simplexdata.φs
#     vol = simplexdata.volume
#     for c in CartesianIndices(matrix)
#         (row, col) = Tuple(c)
#         x = g0 * (φ0[row] * φ0[col]')
#         for (φ, gj) in zip(φs, gjs)
#             x += (φ[row]*φ[col]' - φ0[row]*φ0[col]') * gj
#         end
#         matrix[row, col] += vol * x
#     end
#     return matrix
# end