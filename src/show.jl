displaynames(l::AbstractLattice) = display_as_tuple(l.unitcell.names, ":")

displayname(s::Sublat) = s.name == nametype(:_) ? "pending" : string(":", s.name)

display_as_tuple(v, prefix = "") = isempty(v) ? "()" :
    string("(", prefix, join(v, string(", ", prefix)), ")")

displayvectors(mat::SMatrix{E,L,<:AbstractFloat}; kw...) where {E,L} =
    ntuple(l -> round.(Tuple(mat[:,l]); kw...), Val(L))

displayvectors(mat::SMatrix{E,L,<:Integer}; kw...) where {E,L} =
    ntuple(l -> Tuple(mat[:,l]), Val(L))

#######################################################################
# Lattice
#######################################################################
Base.summary(::Sublat{T,E}) where {T,E} =
    "Sublat{$E,$T} : sublattice of $T-typed sites in $(E)D space"

Base.show(io::IO, s::Sublat) = print(io, summary(s),
"  Sites    : $(length(s.sites))
  Name     : $(displayname(s))")

###

Base.summary(::Lattice{T,E,L}) where {T,E,L} =
    "Lattice{$T,$E,$L} : $(L)D lattice in $(E)D space"

function Base.show(io::IO, lat::Lattice)
    i = get(io, :indent, "")
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(bravais(lat); digits = 6))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatlengths(lat))) --> $(nsites(lat)) total per unit cell")
end

###

Base.summary(::Supercell{L,L´}) where {L,L´} =
    "Supercell{$L,$(L´)} for $(L´)D superlattice of the base $(L)D lattice"

function Base.show(io::IO, s::Supercell{L,L´}) where {L,L´}
    i = get(io, :indent, "")
    print(io, i, summary(s),
"$i  Supervectors  : $(displayvectors(s.matrix))
$i  Supersites    : $(nsites(s))")
end

###

Base.summary(::Superlattice{T,E,L,L´}) where {T,E,L,L´} =
    "Superlattice{$T,$E,$L,$L´} : $(L)D lattice in $(E)D space, filling a $(L´)D supercell"

function Base.show(io::IO, lat::Superlattice)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(bravais(lat); digits = 6))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatlengths(lat))) --> $(nsites(lat)) total per unit cell\n")
    print(ioindent, lat.supercell)
end