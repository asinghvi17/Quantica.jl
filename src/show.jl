
############################################################################################
# Lattice
#region

Base.summary(::Sublat{T,E}) where {T,E} =
    "Sublat{$E,$T} : sublattice of $T-typed sites in $(E)D space"

Base.show(io::IO, s::Sublat) = print(io, summary(s), "\n",
"  Sites    : $(nsites(s))
  Name     : $(displayname(s))")

###

Base.summary(::Lattice{T,E,L}) where {T,E,L} =
    "Lattice{$T,$E,$L} : $(L)D lattice in $(E)D space"

function Base.show(io::IO, lat::Lattice)
    i = get(io, :indent, "")
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(display_rounded_vectors(bravais_vectors(lat)))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatlengths(lat))) --> $(nsites(lat)) total per unit cell")
end

displaynames(l::Lattice) = display_as_tuple(sublatnames(l), ":")

displayname(s::Sublat) = sublatname(s) == Symbol(:_) ? "pending" : string(":", sublatname(s))

display_as_tuple(v, prefix = "") = isempty(v) ? "()" :
    string("(", prefix, join(v, string(", ", prefix)), ")")

display_rounded_vectors(vs) = isempty(vs) ? "[]" : display_rounded_vector.(vs)
display_rounded_vector(v) = round.(v, digits = 6)

#endregion

############################################################################################
# LatticeSlice
#region

Base.summary(::LatticeSlice{T,E,L}) where {T,E,L} =
    "LatticeSlice{$T,$E,$L} : collection of subcells for a $(L)D lattice in $(E)D space"

function Base.show(io::IO, ls::LatticeSlice)
    i = get(io, :indent, "")
    print(io, i, summary(ls), "\n",
"$i  Cells       : $(length(subcells(ls)))
$i  Cell range  : $(isempty(ls) ? "empty" : boundingbox(ls))
$i  Total sites : $(nsites(ls))")
end

#endregion

############################################################################################
# Model
#region

function Base.show(io::IO, m::TightbindingModel)
    ioindent = IOContext(io, :indent => "  ")
    print(io, "TightbindingModel: model with $(termstext(m))", "\n")
    foreach(t -> print(ioindent, t, "\n"), m.terms)
end

function Base.show(io::IO, m::ParametricModel)
    ioindent = IOContext(io, :indent => "  ")
    print(io, "ParametricModel: model with $(termstext(m))", "\n")
    foreach(t -> print(ioindent, t, "\n"), m.terms)
end

function Base.show(io::IO, o::OnsiteTerm{F,<:SiteSelector}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)OnsiteTerm{$(displayparameter(F))}:
$(i)  Sublattices      : $(o.selector.sublats === missing ? "any" : o.selector.sublats)
$(i)  Coefficient      : $(o.coefficient)")
end

function Base.show(io::IO, h::HoppingTerm{F,<:HopSelector}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)HoppingTerm{$(displayparameter(F))}:
$(i)  Sublattice pairs : $(h.selector.sublats === missing ? "any" : h.selector.sublats)
$(i)  dn cell distance : $(h.selector.dcells === missing ? "any" : h.selector.dcells)
$(i)  Hopping range    : $(displayrange(h.selector.range))
$(i)  Coefficient      : $(h.coefficient)
$(i)  Reverse hops     : $(h.selector.adjoint)")
end

function Base.show(io::IO, o::ParametricOnsiteTerm{N}) where {N}
    i = get(io, :indent, "")
    print(io,
"$(i)ParametricOnsiteTerm{$N}}:
$(i)  Functor arguments : $N
$(i)  Sublattices       : $(o.selector.sublats === missing ? "any" : o.selector.sublats)
$(i)  Coefficient       : $(o.coefficient)
$(i)  Parameters        : $(parameters(o))")
end

function Base.show(io::IO, h::ParametricHoppingTerm{N}) where {N}
    i = get(io, :indent, "")
    print(io,
"$(i)ParametricHoppingTerm{$N}:
$(i)  Functor arguments : $N
$(i)  Sublattice pairs  : $(h.selector.sublats === missing ? "any" : h.selector.sublats)
$(i)  dn cell distance  : $(h.selector.dcells === missing ? "any" : h.selector.dcells)
$(i)  Hopping range     : $(displayrange(h.selector.range))
$(i)  Coefficient       : $(h.coefficient)
$(i)  Reverse hops      : $(h.selector.adjoint)
$(i)  Parameters        : $(parameters(h))")
end

displayparameter(::Type{<:Function}) = "Function"
displayparameter(::Type{T}) where {T} = "$T"

termstext(m) = ifelse(length(terms(m)) == 1, "1 term", "$(length(terms(m))) terms")

displayrange(r::Real) = round(r, digits = 6)
displayrange(::Missing) = "any"
displayrange(nr::Neighbors) = "Neighbors($(Int(nr)))"
displayrange(rs::Tuple) = "($(displayrange(first(rs))), $(displayrange(last(rs))))"

#endregion

############################################################################################
# Hamiltonian
#region

function Base.show(io::IO, h::Union{Hamiltonian,ParametricHamiltonian})
    i = get(io, :indent, "")
    print(io, i, summary(h), "\n",
"$i  Bloch harmonics  : $(length(harmonics(h)))
$i  Harmonic size    : $((n -> "$n × $n")(size(h, 1)))
$i  Orbitals         : $(norbitals(h))
$i  Element type     : $(displaytype(blocktype(h)))
$i  Onsites          : $(nonsites(h))
$i  Hoppings         : $(nhoppings(h))
$i  Coordination     : $(round(coordination(h), digits = 5))")
    showextrainfo(io, i, h)
end

Base.summary(h::Hamiltonian{T,E,L}) where {T,E,L} =
    "Hamiltonian{$T,$E,$L}: Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(h::ParametricHamiltonian{T,E,L}) where {T,E,L} =
    "ParametricHamiltonian{$T,$E,$L}: Parametric Hamiltonian on a $(L)D Lattice in $(E)D space"

displaytype(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displaytype(::Type{S}) where {N,T,S<:SMatrixView{N,N,T}} = "At most $N × $N blocks ($T)"
displaytype(::Type{T}) where {T} = "scalar ($T)"

# fallback
showextrainfo(io, i, h) = nothing

showextrainfo(io, i, h::ParametricHamiltonian) = print(io, i, "\n",
"$i  Parameters       : $(parameters(h))")

#endregion

############################################################################################
# AbstractEigenSolver
#region

function Base.show(io::IO, s::AbstractEigenSolver)
    i = get(io, :indent, "")
    print(io, i, summary(s))
end

Base.summary(s::AbstractEigenSolver) =
    "AbstractEigenSolver ($(Base.nameof(typeof(s))))"

#endregion

############################################################################################
# SpectrumSolver
#region

function Base.show(io::IO, s::SpectrumSolver)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => "  ")
    print(io, i, summary(s), "\n")
end

Base.summary(::SpectrumSolver{T,L}) where {T,L} =
    "SpectrumSolver{$T,$L}: Spectrum solver over an $L-dimensional parameter manifold of type $T"

#endregion

############################################################################################
# AbstractMesh
#region

Base.summary(::Mesh{V}) where {V} =
    "Mesh{$(nameof(V))}: Mesh with vertices of type $(nameof(V))"

Base.summary(::Subband{T,L}) where {T,L} =
    "Subband{$T,$L}: Subband over a $L-dimensional parameter (energy-momentum) space of type $T"

function Base.show(io::IO, m::AbstractMesh)
    i = get(io, :indent, "")
    print(io, i, summary(m), "\n",
"$i  Mesh dim  : $(dim(m))
$i  Space dim : $(embdim(m))
$i  Vertices  : $(length(vertices(m)))
$i  Edges     : $(isempty(neighbors(m)) ? 0 : sum(length, neighbors(m)) ÷ 2)
$i  Simplices : $(length(simplices(m)))")
end

#endregion

############################################################################################
# Bands
#region

function Base.show(io::IO, b::Bands)
    i = get(io, :indent, "")
    print(io, i, summary(b), "\n",
"$i  Subbands  : $(length(subbands(b)))
$i  Vertices  : $(sum(s->length(vertices(s)), subbands(b)))
$i  Edges     : $(sum(s -> sum(length, neighbors(s)), subbands(b)) ÷ 2)
$i  Simplices : $(sum(s->length(simplices(s)), subbands(b)))")
end

Base.summary(::Bands{T,E,L}) where {T,E,L} =
    "Bands{$T,$E,$L}: $(E)D Bandstructure over a $L-dimensional parameter space of type $T"

#endregion

# ############################################################################################
# # Coupler
# #region

# function Base.show(io::IO, ::MIME"text/plain", c::Coupler)
#     i = get(io, :indent, "")
#     ioindent = IOContext(io, :indent => "  ")
#     print(io, summary(c), "\n",
# "$i  Number of blocks  : $(size(c, 2))
# $i  Assigned blocks   : $(showassignedblocks(c))", "\n")
#     print(ioindent, lattice(c))
# end

# showassignedblocks(c) = ifelse.(assignedblocks(c), :assigned, :unassigned)

# Base.summary(io::IO, ::Coupler{T,E,L,B}) where {T,E,L,B} = print(io, "Coupler{$T,$E,$L,$B}: AbstractHamiltonian coupler")

# #endregion