############################################################################################
# sublat
#region

sublat(sites...; name = :A) =
    Sublat(sanitize_Vector_of_float_SVectors(sites), Symbol(name))

sublat(sites::Vector; name = :A) =
    Sublat(sanitize_Vector_of_float_SVectors(sites), Symbol(name))

#endregion

############################################################################################
# lattice
#region

lattice(s::Sublat, ss::Sublat...; kw...) = _lattice(promote(s, ss...)...; kw...)

# Start with an empty list of nranges, to be filled as they are requested
Lattice(b::Bravais{T}, u::Unitcell{T}) where {T} = Lattice(b, u, Tuple{Int,T}[])

function _lattice(ss::Sublat{T,E}...;
                  bravais = (),
                  dim = Val(E),
                  type::Type{T´} = T,
                  names = sublatname.(ss)) where {T,E,T´}
    u = unitcell(ss, names, postype(dim, type))
    b = Bravais(type, dim, bravais)
    return Lattice(b, u)
end

function lattice(lat::Lattice{T,E};
                 bravais = bravais_matrix(lat),
                 dim = Val(E),
                 type::Type{T´} = T,
                 names = sublatnames(lat)) where {T,E,T´}
    u = unitcell(unitcell(lat), names, postype(dim, type))
    b = Bravais(type, dim, bravais)
    return Lattice(b, u)
end

postype(dim, type) = SVector{dim,type}
postype(::Val{E}, type) where {E} = SVector{E,type}

function unitcell(sublats, names, postype::Type{S}) where {S<:SVector}
    sites´ = S[]
    offsets´ = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sites(sublats[s])
            push!(sites´, sanitize_SVector(S, site))
        end
        push!(offsets´, length(sites´))
    end
    return Unitcell(sites´, names, offsets´)
end

function unitcell(u::Unitcell, names, postype::Type{S}) where {S<:SVector}
    sites´ = sanitize_SVector.(S, sites(u))
    offsets´ = offsets(u)
    Unitcell(sites´, names, offsets´)
end

#endregion

############################################################################################
# indexing Lattice and LatticeSlice - returns a LatticeSlice
#region

Base.getindex(lat::Lattice; kw...) = lat[siteselector(; kw...)]

Base.getindex(lat::Lattice, ss::SiteSelector) = lat[apply(ss, lat)]

function Base.getindex(lat::Lattice, as::AppliedSiteSelector)
    latslice = LatticeSlice(lat)
    sinds = Int[]
    foreach_cell(as) do cell
        scell = Subcell(sinds, cell)
        foreach_site(as, cell) do s, i, r
            push!(scell, i)
        end
        if isempty(scell)
            return false
        else
            push!(latslice, scell)
            sinds = Int[]   #start new site list
            return true
        end
    end
    return latslice
end

Base.getindex(ls::LatticeSlice; kw...) = getindex(ls, siteselector(; kw...))

Base.getindex(ls::LatticeSlice, ss::SiteSelector) = getindex(ls, apply(ss, parent(ls)))

# indexlist is populated with latslice indices of selected sites
function Base.getindex(latslice::LatticeSlice, as::AppliedSiteSelector)
    lat = parent(latslice)
    latslice´ = LatticeSlice(lat)
    sinds = Int[]
    j = 0
    for subcell in subcells(latslice)
        dn = cell(subcell)
        scell = Subcell(sinds, dn)
        for i in siteindices(subcell)
            j += 1
            r = site(lat, i, dn)
            if (i, r, dn) in as
                push!(scell, i)
            end
        end
        if !isempty(scell)
            push!(latslice´, scell)
            sinds = Int[]  #start new site list
        end
    end
    return latslice´
end

#endregion

############################################################################################
# findsubcell(c, ::LatticeSlice) and findsite(i, ::Subcell)
#region

findsubcell(c, l::LatticeSlice{<:Any,<:Any,L}) where {L} =
    findsubcell(SVector{L,Int}(c), l)

# returns (subcell, siteoffset), or nothing if not found
function findsubcell(cell´::SVector, l::LatticeSlice)
    offset = 0
    for sc in subcells(l)
        if cell´ == cell(sc)
            return sc, offset
            return nothing  # since cells are unique
        else
            offset += nsites(sc)
        end
    end
    return nothing
end

findsite(i::Integer, s::Subcell) = findfirst(==(i), siteindices(s))

#endregion

############################################################################################
# merge(lss::LatticeSlice...)
#region

function Base.merge(lss::S...) where {L,S<:LatticeSlice{<:Any,<:Any,L}}
    lat = parent(first(lss))
    all(l -> l === lat, parent.(lss)) ||
        argerror("Cannot merge LatticeBlocks of different lattices")

    allcellinds = Tuple{SVector{L,Int},Int}[]
    for ls in lss, scell in subcells(ls), ind in siteindices(scell)
        push!(allcellinds, (cell(scell), ind))
    end
    sort!(allcellinds)
    unique!(allcellinds)

    currentcell = first(first(allcellinds))
    scell = Subcell(currentcell)
    scells = [scell]
    for (c, i) in allcellinds
        if c == currentcell
            push!(siteindices(scell), i)
        else
            scell = Subcell(c)
            push!(siteindices(scell), i)
            push!(scells, scell)
        end
    end
    latslice = LatticeSlice(lat, scells)
    return latslice
end

#endregion

############################################################################################
# merge lattices - combine sublats if equal name
#region

function combine(lats::Lattice{T,E,L}...) where {T,E,L}
    isapprox_modulo_shuffle(bravais_matrix.(lats)...) ||
        throw(ArgumentError("To merge lattices they must all share the same Bravais matrix"))
    bravais´ = bravais(first(lats))
    unitcell´ = combine(unitcell.(lats)...)
    return Lattice(bravais´, unitcell´)
end

function combine(ucells::Unitcell...)
    names´ = vcat(sublatnames.(ucells)...)
    sites´ = vcat(sites.(ucells)...)
    offsets´ = combined_offsets(offsets.(ucells)...)
    return Unitcell(sites´, names´, offsets´)
end

isapprox_modulo_shuffle() = true

function isapprox_modulo_shuffle(s::AbstractMatrix, ss::AbstractMatrix...)
    for s´ in ss, c´ in eachcol(s´)
        any(c -> c ≈ c´ || c ≈ -c´, eachcol(s)) || return false
    end
    return true
end

function combined_offsets(offsets...)
    offsets´ = cumsum(Iterators.flatten(diff.(offsets)))
    prepend!(offsets´, 0)
    return offsets´
end

#endregion

############################################################################################
# neighbors
#region

function nrange(n, lat)
    for (n´, r) in nranges(lat)
        n == n´ && return r
    end
    r = compute_nrange(n, lat)
    push!(nranges(lat), (n, r))
    return r
end


function compute_nrange(n, lat::Lattice{T}) where {T}
    latsites = sites(lat)
    dns = BoxIterator(zerocell(lat))
    br = bravais_matrix(lat)
    # 128 is a heuristic cutoff for kdtree vs brute-force search
    if length(latsites) <= 128
        dists = fill(T(Inf), n)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for (i, ri) in enumerate(latsites), (j, rj) in enumerate(latsites)
                j <= i && iszero(dn) && continue
                r = ri - rj + br * dn
                update_dists!(dists, r'r)
            end
            isfinite(last(dists)) || acceptcell!(dns, dn)
        end
        dist = sqrt(last(dists))
    else
        tree = KDTree(latsites)
        dist = T(Inf)
        for dn in dns
            iszero(dn) || ispositive(dn) || continue
            for r0 in latsites
                r = r0 + br * dn
                dist = min(dist, compute_nrange(n, tree, r, nsites(lat)))
            end
            isfinite(dist) || acceptcell!(dns, dn)
        end
    end
    return dist
end

function update_dists!(dists, dist)
    len = length(dists)
    for (n, d) in enumerate(dists)
        isapprox(dist, d) && break
        if dist < d
            dists[n+1:len] .= dists[n:len-1]
            dists[n] = dist
            break
        end
    end
    return dists
end

function compute_nrange(n, tree, r::AbstractVector, nmax)
    for m in n:nmax
        _, dists = knn(tree, r, 1 + m, true)
        popfirst!(dists)
        unique_sorted_approx!(dists)
        length(dists) == n && return maximum(dists)
    end
    return convert(eltype(r), Inf)
end

function unique_sorted_approx!(v::AbstractVector)
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

function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

#endregion
