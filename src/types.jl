############################################################################################
# Lattice  -  see lattice.jl for methods
#region

struct Sublat{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    name::Symbol
end

struct Unitcell{T<:AbstractFloat,E}
    sites::Vector{SVector{E,T}}
    names::Vector{Symbol}
    offsets::Vector{Int}        # Linear site number offsets for each sublat
end

struct Bravais{T,E,L}
    matrix::Matrix{T}
    function Bravais{T,E,L}(matrix) where {T,E,L}
        (E, L) == size(matrix) || throw(ErrorException("Internal error: unexpected matrix size $((E,L)) != $(size(matrix))"))
        L > E &&
            throw(DimensionMismatch("Number $L of Bravais vectors cannot be greater than embedding dimension $E"))
        return new(matrix)
    end
end

struct Lattice{T<:AbstractFloat,E,L}
    bravais::Bravais{T,E,L}
    unitcell::Unitcell{T,E}
    nranges::Vector{Tuple{Int,T}}  # [(nth_neighbor, min_nth_neighbor_distance)...]
end

#region ## Constructors ##

Bravais(::Type{T}, E, m) where {T} = Bravais(T, Val(E), m)
Bravais(::Type{T}, ::Val{E}, m::Tuple{}) where {T,E} =
    Bravais{T,E,0}(sanitize_Matrix(T, E, ()))
Bravais(::Type{T}, ::Val{E}, m::NTuple{E´,Number}) where {T,E,E´} =
    Bravais{T,E,1}(sanitize_Matrix(T, E, (m,)))
Bravais(::Type{T}, ::Val{E}, m::NTuple{L,Any}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::SMatrix{E,L}) where {T,E,L} =
    Bravais{T,E,L}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::AbstractMatrix) where {T,E} =
    Bravais{T,E,size(m,2)}(sanitize_Matrix(T, E, m))
Bravais(::Type{T}, ::Val{E}, m::AbstractVector) where {T,E} =
    Bravais{T,E,1}(sanitize_Matrix(T, E, hcat(m)))

#endregion

#region ## API ##

bravais(l::Lattice) = l.bravais

unitcell(l::Lattice) = l.unitcell

nranges(l::Lattice) = l.nranges

bravais_vectors(l::Lattice) = bravais_vectors(l.bravais)
bravais_vectors(b::Bravais) = eachcol(b.matrix)

bravais_matrix(l::Lattice) = bravais_matrix(l.bravais)
bravais_matrix(b::Bravais{T,E,L}) where {T,E,L} =
    convert(SMatrix{E,L,T}, ntuple(i -> b.matrix[i], Val(E*L)))

matrix(b::Bravais) = b.matrix

sublatnames(l::Lattice) = l.unitcell.names
sublatname(l::Lattice, s) = sublatname(l.unitcell, s)
sublatname(u::Unitcell, s) = u.names[s]
sublatname(s::Sublat) = s.name

nsublats(l::Lattice) = nsublats(l.unitcell)
nsublats(u::Unitcell) = length(u.names)

sublats(l::Lattice) = sublats(l.unitcell)
sublats(u::Unitcell) = 1:nsublats(u)

nsites(s::Sublat) = length(s.sites)
nsites(lat::Lattice, sublat...) = nsites(lat.unitcell, sublat...)
nsites(u::Unitcell) = length(u.sites)
nsites(u::Unitcell, sublat) = sublatlengths(u)[sublat]

sites(s::Sublat) = s.sites
sites(l::Lattice, sublat...) = sites(l.unitcell, sublat...)
sites(u::Unitcell) = u.sites
sites(u::Unitcell, sublat) = view(u.sites, u.offsets[sublat]+1:u.offsets[sublat+1])

site(l::Lattice, i) = sites(l)[i]
site(l::Lattice, i, dn) = site(l, i) + bravais_matrix(l) * dn

siterange(l::Lattice, sublat) = siterange(l.unitcell, sublat)
siterange(u::Unitcell, sublat) = (1+u.offsets[sublat]):u.offsets[sublat+1]

sitesublat(lat::Lattice, siteidx, ) = sitesublat(lat.unitcell.offsets, siteidx)

function sitesublat(offsets, siteidx)
    l = length(offsets)
    for s in 2:l
        @inbounds offsets[s] + 1 > siteidx && return s - 1
    end
    return l
end

sitesublatname(lat, i) = sublatname(lat, sitesublat(lat, i))

sitesublatiter(l::Lattice) = sitesublatiter(l.unitcell)
sitesublatiter(u::Unitcell) = ((i, s) for s in sublats(u) for i in siterange(u, s))

offsets(l::Lattice) = offsets(l.unitcell)
offsets(u::Unitcell) = u.offsets

sublatlengths(lat::Lattice) = sublatlengths(lat.unitcell)
sublatlengths(u::Unitcell) = diff(u.offsets)

embdim(::Sublat{<:Any,E}) where {E} = E
embdim(::Lattice{<:Any,E}) where {E} = E

latdim(::Lattice{<:Any,<:Any,L}) where {L} = L

numbertype(::Sublat{T}) where {T} = T
numbertype(::Lattice{T}) where {T} = T

zerocell(::Lattice{<:Any,<:Any,L}) where {L} = zero(SVector{L,Int})

Base.copy(l::Lattice) = deepcopy(l)

#endregion
#endregion

############################################################################################
# Selectors  -  see selector.jl for methods
#region

struct SiteSelector{F,S}
    region::F
    sublats::S
end

struct AppliedSiteSelector{T,E,L}
    lat::Lattice{T,E,L}
    region::FunctionWrapper{Bool,Tuple{SVector{E,T}}}
    sublats::Vector{Symbol}
end

struct HopSelector{F,S,D,R}
    region::F
    sublats::S
    dcells::D
    range::R
    adjoint::Bool  # make apply take the "adjoint" of the selector
end

struct AppliedHopSelector{T,E,L}
    lat::Lattice{T,E,L}
    region::FunctionWrapper{Bool,Tuple{SVector{E,T},SVector{E,T}}}
    sublats::Vector{Pair{Symbol,Symbol}}
    dcells::Vector{SVector{L,Int}}
    range::Tuple{T,T}
end

struct Neighbors
    n::Int
end

#region ## Constructors ##

HopSelector(re, su, dc, ra) = HopSelector(re, su, dc, ra, false)

#endregion

#region ## API ##

Base.Int(n::Neighbors) = n.n

region(s::Union{SiteSelector,HopSelector}) = s.region

lattice(ap::AppliedSiteSelector) = ap.lat
lattice(ap::AppliedHopSelector) = ap.lat

dcells(ap::AppliedHopSelector) = ap.dcells

# if isempty(s.dcells) or isempty(s.sublats), none were specified, so we must accept any
inregion(r, s::AppliedSiteSelector) = s.region(r)
inregion((r, dr), s::AppliedHopSelector) = s.region(r, dr)

insublats(n, s::AppliedSiteSelector) = isempty(s.sublats) || n in s.sublats
insublats(npair::Pair, s::AppliedHopSelector) = isempty(s.sublats) || npair in s.sublats

indcells(dcell, s::AppliedHopSelector) = isempty(s.dcells) || dcell in s.dcells

iswithinrange(dr, s::AppliedHopSelector) = iswithinrange(dr, s.range)
iswithinrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  ifelse(rmin^2 <= dr'dr <= rmax^2, true, false)

isbelowrange(dr, s::AppliedHopSelector) = isbelowrange(dr, s.range)
isbelowrange(dr, (rmin, rmax)::Tuple{Real,Real}) =  ifelse(dr'dr < rmin^2, true, false)

Base.adjoint(s::SiteSelector) = s

Base.adjoint(s::HopSelector) = HopSelector(s.region, s.sublats, s.dcells, s.range, !s.adjoint)

#endregion
#endregion

############################################################################################
# Model Terms  -  see model.jl for methods
#region

# Terms #

struct TightbindingModel{T}
    terms::T  # Collection of `TightbindingModelTerm`s
end

struct OnsiteTerm{F,S<:SiteSelector,T<:Number}
    o::F
    selector::S
    coefficient::T
end

struct AppliedOnsiteTerm{T,E,L,B}
    o::FunctionWrapper{B,Tuple{SVector{E,T},Int}}  # o(r, sublat_orbitals)
    selector::AppliedSiteSelector{T,E,L}
end

struct HoppingTerm{F,S<:HopSelector,T<:Number}
    t::F
    selector::S
    coefficient::T
end

struct AppliedHoppingTerm{T,E,L,B}
    t::FunctionWrapper{B,Tuple{SVector{E,T},SVector{E,T},Tuple{Int,Int}}}  # t(r, dr, (orbs1, orbs2))
    selector::AppliedHopSelector{T,E,L}
end

const TightbindingModelTerm = Union{OnsiteTerm,HoppingTerm,AppliedOnsiteTerm,AppliedHoppingTerm}

#region ## Constructors ##

TightbindingModel(ts::TightbindingModelTerm...) = TightbindingModel(ts)

OnsiteTerm(t::OnsiteTerm, os::SiteSelector) = OnsiteTerm(t.o, os, t.coefficient)

HoppingTerm(t::HoppingTerm, os::HopSelector) = HoppingTerm(t.t, os, t.coefficient)

#endregion

#region ## API ##

terms(t::TightbindingModel) = t.terms

selector(t::TightbindingModelTerm) = t.selector

(term::OnsiteTerm{<:Function})(r) = term.coefficient * term.o(r)
(term::OnsiteTerm)(r) = term.coefficient * term.o

(term::AppliedOnsiteTerm)(r, orbs) = term.o(r, orbs)

(term::HoppingTerm{<:Function})(r, dr) = term.coefficient * term.t(r, dr)
(term::HoppingTerm)(r, dr) = term.coefficient * term.t

(term::AppliedHoppingTerm)(r, dr, orbs) = term.t(r, dr, orbs)

# Model term algebra

Base.:*(x::Number, m::TightbindingModel) = TightbindingModel(x .* terms(m))
Base.:*(m::TightbindingModel, x::Number) = x * m
Base.:-(m::TightbindingModel) = (-1) * m

Base.:+(m::TightbindingModel, m´::TightbindingModel) = TightbindingModel((terms(m)..., terms(m´)...))
Base.:-(m::TightbindingModel, m´::TightbindingModel) = m + (-m´)

Base.:*(x::Number, o::OnsiteTerm) = OnsiteTerm(o.o, o.selector, x * o.coefficient)
Base.:*(x::Number, t::HoppingTerm) = HoppingTerm(t.t, t.selector, x * t.coefficient)

Base.adjoint(m::TightbindingModel) = TightbindingModel(adjoint.(terms(m)))
Base.adjoint(t::OnsiteTerm{<:Function}) = OnsiteTerm(r -> t.o(r)', t.selector, t.coefficient')
Base.adjoint(t::OnsiteTerm) = OnsiteTerm(t.o', t.selector, t.coefficient')
Base.adjoint(t::HoppingTerm{<:Function}) = HoppingTerm((r, dr) -> t.t(r, -dr)', t.selector', t.coefficient')
Base.adjoint(t::HoppingTerm) = HoppingTerm(t.t', t.selector', t.coefficient')

#endregion
#endregion

############################################################################################
# Model Modifiers  -  see model.jl for methods
#region

# wrapper of a function f(x1, ... xN; kw...) with N arguments and the kwargs in params
struct ParametricFunction{N,F}
    f::F
    params::Vector{Symbol}
end

struct OnsiteModifier{N,S<:SiteSelector,F<:ParametricFunction{N}}
    f::F
    selector::S
end

struct AppliedOnsiteModifier{N,B,R<:SVector,F<:ParametricFunction{N}}
    blocktype::Type{B}
    f::F
    ptrs::Vector{Tuple{Int,R,Int}}
    # [(ptr, r, norbs)...] for each selected site, dn = 0 harmonic
end

struct HoppingModifier{N,S<:HopSelector,F<:ParametricFunction{N}}
    f::F
    selector::S
end

struct AppliedHoppingModifier{N,B,R<:SVector,F<:ParametricFunction{N}}
    blocktype::Type{B}
    f::F
    ptrs::Vector{Vector{Tuple{Int,R,R,Tuple{Int,Int}}}}
    # [[(ptr, r, dr, (norbs, norbs´)), ...], ...] for each selected hop on each harmonic
end

const Modifier = Union{OnsiteModifier,HoppingModifier}
const AppliedModifier = Union{AppliedOnsiteModifier,AppliedHoppingModifier}

#region ## Constructors ##

ParametricFunction{N}(f::F, params) where {N,F} = ParametricFunction{N,F}(f, params)

#endregion

#region ## API ##

selector(m::Modifier) = m.selector

parameters(m::Union{Modifier,AppliedModifier}) = m.f.params

parametric_function(m::Union{Modifier,AppliedModifier}) = m.f

pointers(m::AppliedModifier) = m.ptrs

(m::AppliedOnsiteModifier{1,B})(o, r, orbs; kw...) where {B} =
    sanitize_block(B, m.f.f(o; kw...), (orbs, orbs))
(m::AppliedOnsiteModifier{2,B})(o, r, orbs; kw...) where {B} =
    sanitize_block(B, m.f.f(o, r; kw...), (orbs, orbs))

(m::AppliedHoppingModifier{1,B})(t, r, dr, orbs; kw...) where {B} =
    sanitize_block(B, m.f.f(t; kw...), orbs)
(m::AppliedHoppingModifier{3,B})(t, r, dr, orbs; kw...) where {B} =
    sanitize_block(B, m.f.f(t, r, dr; kw...), orbs)

#endregion
#endregion

############################################################################################
# OrbitalStructure  -  see hamiltonian.jl for methods
#region

struct OrbitalStructure{B<:Union{Number,SMatrix}}
    blocktype::Type{B}    # Hamiltonian's blocktype
    norbitals::Vector{Int}
    offsets::Vector{Int}  # index offset for each sublattice (== offsets(::Lattice))
end

#region ## Constructors ##

# norbs is a collection of number of orbitals, one per sublattice (or a single one for all)
# B type instability when calling from `hamiltonian` is removed by @inline (const prop)
@inline function OrbitalStructure(lat::Lattice, norbs, T = numbertype(lat))
    B = blocktype(T, norbs)
    return OrbitalStructure{B}(lat, norbs)
end

function OrbitalStructure{B}(lat::Lattice, norbs) where {B}
    norbs´ = sanitize_Vector_of_Type(Int, nsublats(lat), norbs)
    offsets´ = offsets(lat)
    return OrbitalStructure{B}(B, norbs´, offsets´)
end

blocktype(T::Type, norbs) = blocktype(T, val_maximum(norbs))
blocktype(::Type{T}, m::Val{1}) where {T} = Complex{T}
blocktype(::Type{T}, m::Val{N}) where {T,N} = SMatrix{N,N,Complex{T},N*N}

val_maximum(n::Int) = Val(n)
val_maximum(ns::Tuple) = Val(maximum(argval.(ns)))

argval(::Val{N}) where {N} = N
argval(n::Int) = n

#endregion

#region ## API ##

norbitals(o::OrbitalStructure) = o.norbitals

orbtype(::OrbitalStructure{B}) where {B} = orbtype(B)
orbtype(::Type{B}) where {B<:Number} = B
orbtype(::Type{B}) where {N,T,B<:SMatrix{N,N,T}} = SVector{N,T}

blocktype(o::OrbitalStructure) = o.blocktype

offsets(o::OrbitalStructure) = o.offsets

nsites(o::OrbitalStructure) = last(offsets(o))

nsublats(o::OrbitalStructure) = length(norbitals(o))

sublats(o::OrbitalStructure) = 1:nsublats(o)

siterange(o::OrbitalStructure, sublat) = (1+o.offsets[sublat]):o.offsets[sublat+1]

# Equality does not need equal T
Base.:(==)(o1::OrbitalStructure, o2::OrbitalStructure) =
    o1.norbs == o2.norbs && o1.offsets == o2.offsets

#endregion
#endregion

############################################################################################
# Harmonic  -  see hamiltonian.jl for methods
#region

struct Harmonic{L,M<:AbstractArray}
    dn::SVector{L,Int}
    h::M
end

#region ## API ##

matrix(h::Harmonic) = h.h

dcell(h::Harmonic) = h.dn

Base.size(h::Harmonic, i...) = size(matrix(h), i...)

Base.isless(h::Harmonic, h´::Harmonic) = sum(abs2, dcell(h)) < sum(abs2, dcell(h´))

#endregion
#endregion

############################################################################################
# Hamiltonian  -  see hamiltonian.jl for methods
#region

abstract type AbstractHamiltonian{T,E,L,B} end

struct Hamiltonian{T,E,L,B} <: AbstractHamiltonian{T,E,L,B}
    lattice::Lattice{T,E,L}
    orbstruct::OrbitalStructure{B}
    harmonics::Vector{Harmonic{L,SparseMatrixCSC{B,Int}}}
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{T,E,L,B}(lattice, orbstruct, harmonics) where {T,E,L,B}
        n = nsites(lattice)
        all(har -> size(matrix(har)) == (n, n), harmonics) ||
            throw(DimensionMismatch("Harmonic $(size.(matrix.(harmonics), 1)) sizes don't match number of sites $n"))
        sort!(harmonics)
        length(harmonics) > 0 && iszero(dcell(first(harmonics))) || pushfirst!(harmonics,
            Harmonic(zero(SVector{L,Int}), spzeros(B, n, n)))
        return new(lattice, orbstruct, harmonics)
    end
end

#region ## API ##

Hamiltonian(l::Lattice{T,E,L}, o::OrbitalStructure{B}, h::Vector{Harmonic{L,SparseMatrixCSC{B,Int}}}) where {T,E,L,B} =
    Hamiltonian{T,E,L,B}(l, o, h)

hamiltonian(h::Hamiltonian) = h

orbitalstructure(h::Hamiltonian) = h.orbstruct

lattice(h::Hamiltonian) = h.lattice

harmonics(h::Hamiltonian) = h.harmonics

orbtype(h::Hamiltonian) = orbtype(orbitalstructure(h))

blocktype(h::Hamiltonian) = blocktype(orbitalstructure(h))

norbitals(h::Hamiltonian) = norbitals(orbitalstructure(h))

Base.size(h::Hamiltonian, i...) = size(first(harmonics(h)), i...)

copy_harmonics(h::Hamiltonian) = Hamiltonian(lattice(h), orbitalstructure(h), deepcopy(harmonics(h)))

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isassigned(h, -hh.dn) || return false
        hh.h ≈ h[-hh.dn]' || return false
    end
    return true
end

#endregion
#endregion

############################################################################################
# ParametricHamiltonian  -  see hamiltonian.jl for methods
#region

struct ParametricHamiltonian{T,E,L,B,M<:NTuple{<:Any,AppliedModifier}} <: AbstractHamiltonian{T,E,L,B}
    hparent::Hamiltonian{T,E,L,B}
    h::Hamiltonian{T,E,L,B}
    modifiers::M                   # Tuple of AppliedModifier's
    allptrs::Vector{Vector{Int}}   # allptrs are all modified ptrs in each harmonic (needed for reset!)
    allparams::Vector{Symbol}
end

#region ## API ##

Base.parent(h::ParametricHamiltonian) = h.hparent

hamiltonian(h::ParametricHamiltonian) = h.h

parameters(h::ParametricHamiltonian) = h.allparams

modifiers(h::ParametricHamiltonian) = h.modifiers

pointers(h::ParametricHamiltonian) = h.allptrs

harmonics(h::ParametricHamiltonian) = harmonics(parent(h))

orbitalstructure(h::ParametricHamiltonian) = orbitalstructure(parent(h))

orbtype(h::ParametricHamiltonian) = orbtype(parent(h))

blocktype(h::ParametricHamiltonian) = blocktype(parent(h))

lattice(h::ParametricHamiltonian) = lattice(parent(h))

Base.size(h::ParametricHamiltonian, i...) = size(parent(h), i...)

#endregion
#endregion

############################################################################################
# FlatHamiltonian  -  see hamiltonian.jl for methods
#region

struct FlatHamiltonian{T,E,L,B<:Number,H<:AbstractHamiltonian{T,E,L,<:SMatrix}} <: AbstractHamiltonian{T,E,L,B}
    h::H
    flatorbstruct::OrbitalStructure{B}
end

#region ## API ##

orbitalstructure(h::FlatHamiltonian) = h.flatorbstruct

unflatten(h::FlatHamiltonian) = parent(h)

lattice(h::FlatHamiltonian) = lattice(parent(h))

harmonics(h::FlatHamiltonian) = harmonics(parent(h))

orbtype(h::FlatHamiltonian) = orbtype(orbitalstructure(h))

blocktype(h::FlatHamiltonian) = blocktype(orbitalstructure(h))

norbitals(h::FlatHamiltonian) = norbitals(orbitalstructure(h))

Base.size(h::FlatHamiltonian) = nsites(orbitalstructure(h)), nsites(orbitalstructure(h))
Base.size(h::FlatHamiltonian, i) = i <= 0 ? throw(BoundsError()) : ifelse(1 <= i <= 2, nsites(orbitalstructure(h)), 1)

Base.parent(h::FlatHamiltonian) = h.h

#endregion
#endregion

############################################################################################
# Bloch  -  see hamiltonian.jl for methods
#region

abstract type AbstractBloch{L} end

struct Bloch{L,B,M<:AbstractMatrix{B},H<:AbstractHamiltonian{<:Any,<:Any,L}} <: AbstractBloch{L}
    h::H
    output::M       # output has same structure as merged harmonics(h)
end                 # or its flattened version if eltype(M) != blocktype(H)

#region ## API ##

matrix(b::Bloch) = b.output

hamiltonian(b::Bloch) = b.h

blocktype(::Bloch{<:Any,B}) where {B} = B

orbtype(::Bloch{<:Any,B}) where {B} = orbtype(B)

function spectrumtype(b::Bloch)
    E = complex(eltype(blocktype(b)))
    O = orbtype(b)
    return Spectrum{E,O}
end

latdim(b::Bloch) = latdim(lattice(b.h))

Base.size(b::Bloch, dims...) = size(b.output, dims...)

#endregion
#endregion

############################################################################################
# Velocity  -  see hamiltonian.jl for call API
#region

struct Velocity{L,B<:Bloch{L}} <: AbstractBloch{L}
    bloch::B
    axis::Int
    function Velocity{L,B}(b, axis) where {L,B<:Bloch{L}}
        1 <= axis <= L || throw(ArgumentError("Velocity axis for this system should be between 1 and $L"))
        return new(b, axis)
    end
end

#region ## API ##

Velocity(b::B, axis) where {L,B<:Bloch{L}} = Velocity{L,B}(b, axis)

velocity(b, axis) = Velocity(b, axis)

matrix(v::Velocity) = matrix(v.bloch)

hamiltonian(v::Velocity) = hamiltonian(v.bloch)

blocktype(v::Velocity) = blocktype(v.bloch)

orbtype(v::Velocity) = orbtype(v.bloch)

spectrumtype(v::Velocity) = spectrumtype(v.bloch)

latdim(v::Velocity) = latdim(v.bloch)

Base.size(v::Velocity, dims...) = size(v.bloch, dims...)

axis(v::Velocity) = v.axis

#endregion
#endregion

############################################################################################
# Mesh  -  see mesh.jl for methods
#region

abstract type AbstractMesh{V,S} end

struct Mesh{V,S} <: AbstractMesh{V,S}
    verts::Vector{V}
    neighs::Vector{Vector{Int}}          # all neighbors neighs[i][j] of vertex i
    simps::Vector{NTuple{S,Int}}         # list of simplices, each one a group of neighboring vertex indices
end

#region ## Constructors ##

function Mesh{S}(verts, neighs) where {S}
    simps  = build_cliques(neighs, Val(S))
    return Mesh(verts, neighs, simps)
end

#endregion

#region ## API ##

dim(::AbstractMesh{<:Any,S}) where {S} = S - 1

vertices(m::Mesh) = m.verts
vertices(m::Mesh, i) = m.verts[i]

neighbors(m::Mesh) = m.neighs
neighbors(m::Mesh, i::Int) = m.neighs[i]

neighbors_forward(m::Mesh, i::Int) = Iterators.filter(>(i), m.neighs[i])
neighbors_forward(v::Vector, i::Int) = Iterators.filter(>(i), v[i])

simplices(m::Mesh) = m.simps
simplices(m::Mesh, i::Int) = m.simps[i]

Base.copy(m::Mesh) = Mesh(copy(m.verts), deepcopy(m.neighs), copy(m.simps))

#endregion
#endregion

############################################################################################
# Eigensolvers  -  see band.jl for AppliedEigensolver call API and Spectrum constructors
#                  /presets/eigensolvers.jl for solver backends <: AbstractEigensolver
#region

abstract type AbstractEigensolver end

const Spectrum{C<:Complex,O} = Eigen{O,C,Matrix{O},Vector{C}}

struct AppliedEigensolver{T,L,C,O}
    solver::FunctionWrapper{Spectrum{C,O},Tuple{SVector{L,T}}}
end

#region ## Constructors ##

Spectrum(evals, evecs) = Eigen(sorteigs!(evals, evecs)...)
Spectrum(evals::AbstractVector, evecs::AbstractVector{<:AbstractVector}) =
    Spectrum(evals, hcat(evecs...))
Spectrum(evals::AbstractVector{<:Real}, evecs::AbstractMatrix) =
    Spectrum(complex.(evals), evecs)

function sorteigs!(ϵ::AbstractVector, ψ::AbstractMatrix)
    p = Vector{Int}(undef, length(ϵ))
    p´ = similar(p)
    sortperm!(p, ϵ, by = real, alg = Base.DEFAULT_UNSTABLE)
    Base.permute!!(ϵ, copy!(p´, p))
    Base.permutecols!!(ψ, copy!(p´, p))
    return ϵ, ψ
end

#endregion

#region ## API ##

solver(s::AppliedEigensolver) = s.solver

energies(s::Spectrum) = s.values

states(s::Spectrum) = s.vectors

Base.size(s::Spectrum, i...) = size(s.vectors, i...)

#endregion
#endregion

############################################################################################
# Band and friends -  see band.jl for methods
#region

const MatrixView{O} = SubArray{O,2,Matrix{O},Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}

struct BandVertex{T<:AbstractFloat,E,O}
    coordinates::SVector{E,T}
    states::MatrixView{O}
end

# Subband is a type of AbstractMesh with manifold dimension = embedding dimension - 1
# and with interval search trees to allow slicing
struct Subband{T,E,O} <: AbstractMesh{BandVertex{T,E,O},E}  # we restrict S == E
    mesh::Mesh{BandVertex{T,E,O},E}
    trees::NTuple{E,IntervalTree{T,IntervalValue{T,Int}}}
end

struct Band{T,E,L,C,O} # E = L+1
    subbands::Vector{Subband{T,E,O}}
    solvers::Vector{AppliedEigensolver{T,L,C,O}}  # one per Julia thread
end

#region ## Constructors ##

BandVertex(x, s::Matrix) = BandVertex(x, view(s, :, 1:size(s, 2)))
BandVertex(m, e, s::Matrix) = BandVertex(m, e, view(s, :, 1:size(s, 2)))
BandVertex(m, e, s::SubArray) = BandVertex(vcat(m, e), s)

Subband(verts::Vector{<:BandVertex{<:Any,E}}, neighs) where {E} =
    Subband(Mesh{E}(verts, neighs))

function Subband(mesh::Mesh{<:BandVertex{T,E}}) where {T,E}
    verts, simps = vertices(mesh), simplices(mesh)
    order_simplices!(simps, verts)
    trees = ntuple(Val(E)) do i
        list = [IntervalValue(shrinkright(extrema(j->coordinates(verts[j])[i], s))..., n)
                     for (n, s) in enumerate(simps)]
        sort!(list)
        return IntervalTree{T,IntervalValue{T,Int}}(list)
    end
    return Subband(mesh, trees)
end

# Interval is closed, we want semiclosed on the left -> exclude the upper limit
shrinkright((x, y)) = (x, prevfloat(y))

#endregion

#region ## API ##
# BandVertex #

coordinates(v::BandVertex) = v.coordinates

energy(v::BandVertex) = last(v.coordinates)

base_coordinates(v::BandVertex) = SVector(Base.front(Tuple(v.coordinates)))

states(v::BandVertex) = v.states

degeneracy(v::BandVertex) = size(v.states, 2)

parentrows(v::BandVertex) = first(parentindices(v.states))
parentcols(v::BandVertex) = last(parentindices(v.states))

embdim(::AbstractMesh{<:SVector{E}}) where {E} = E

embdim(::AbstractMesh{<:BandVertex{<:Any,E}}) where {E} = E

# Subband #

vertices(s::Subband, i...) = vertices(s.mesh, i...)

neighbors(s::Subband, i...) = neighbors(s.mesh, i...)

neighbors_forward(s::Subband, i) = neighbors_forward(s.mesh, i)

simplices(s::Subband, i...) = simplices(s.mesh, i...)

simplex_edges(s::Subband, i::Int) = simplex_edges(s, simplices(s, i))
simplex_edges(s::Subband, is) =
    ((vertices(s, is[i]), vertices(s, is[j])) for i in 1:length(is) for j in i+1:length(is))

vertex_coordinates(s::Subband) = (coordinates(v) for v in vertices(s))
vertex_coordinates(s::Subband, i::Int) = coordinates(vertices(s)[i])
vertex_coordinates(s::Subband, is) = (vertex_coordinates(s, i) for i in is)

edge_coordinates(s::Subband, i::Int, j::Int) = (vertex_coordinates(s, i), vertex_coordinates(s, j))
edge_coordinates(s::Subband) =
    (edge_coordinates(s, i, j) for i in eachindex(vertices(s)) for j in neighbors_forward(s, i))

edge_indices(s::Subband) =
    ((i, j) for i in eachindex(vertices(s)) for j in neighbors_forward(s, i))

trees(s::Subband) = s.trees
trees(s::Subband, i::Int) = s.trees[i]

# last argument: saxes = ((dim₁, x₁), (dim₂, x₂)...)
function foreach_simplex(f, s::Subband, ((dim, k), xs...))
    for interval in intersect(trees(s, dim), (k, k))
        interval_in_slice!(interval, s, xs...) || continue
        sind = value(interval)
        f(sind)
    end
    return nothing
end

interval_in_slice!(interval, s, (dim, k), xs...) =
    interval in intersect(trees(s, dim), (k, k)) && interval_in_slice!(interval, s, xs...)
interval_in_slice!(interval, s) = true

Base.isempty(s::Subband) = isempty(simplices(s))

# Band #

basemesh(b::Band) = b.basemesh

subbands(b::Band) = b.subbands

subbands(b::Band, i...) = getindex(b.subbands, i...)

#endregion
#endregion