############################################################################################
# Sublat constructors
#region

sublat(sites...; name = :_) =
    Sublat(sanitize_Vector_of_SVectors(sites), Symbol(name))

#endregion

############################################################################################
# Bravais constructors
#region

Bravais{T,E}(v::Vararg{<:Any,L}) where {T,E,L} = Bravais{T,E,L}(sanitize_Matrix(T, E, v...))
Bravais{T,E}(m::AbstractMatrix) where {T,E} = Bravais{T,E,size(m,2)}(sanitize_Matrix(T, E, m))

#endregion

############################################################################################
# Unitcell constructors
#region

function unitcell(sublats, names, ::Type{S}) where {S<:SVector}
    sites´ = S[]
    offsets´ = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sites(sublats[s])
            push!(sites´, sanitize_SVector(S, site))
        end
        push!(offsets´, length(sites´))
    end
    names´ = uniquenames!(sanitize_Vector_of_Symbols(names))
    return Unitcell(sites´, names´, offsets´)
end

function unitcell(u::Unitcell, names, ::Type{S}) where {S<:SVector}
    sites´ = sanitize_SVector.(S, sites(u))
    names´ = uniquenames!(sanitize_Vector_of_Symbols(names))
    offsets´ = offsets(u)
    Unitcell(sites´, names´, offsets´)
end

function uniquenames!(names::Vector{Symbol})
    allnames = Symbol[:_]
    for (i, name) in enumerate(names)
        name in allnames && (names[i] = uniquename(allnames, name, i))
        push!(allnames, name)
    end
    return names
end

function uniquename(allnames, name, i)
    newname = Symbol(Char(64+i)) # Lexicographic, starting from Char(65) = 'A'
    return newname in allnames ? uniquename(allnames, name, i + 1) : newname
end

#endregion

############################################################################################
# Lattice constructors
#region

lattice(s::Sublat, ss::Sublat...; kw...) = _lattice(promote(s, ss...)...; kw...)

function _lattice(ss::Sublat...;
                  bravais = (),
                  dim = embdim(first(ss)),
                  type = numbertype(first(ss)),
                  names = sublatname.(ss))
    u = unitcell(ss, names, SVector{dim,type})
    b = Bravais{type,dim}(bravais)
    return Lattice(b, u)
end

function lattice(l::Lattice;
                 bravais = bravais_mat(lat),
                 dim = embdim(l),
                 type = numbertype(l),
                 names = names(l))
    u = unitcell(unitcell(lat), names, SVector{dim,type})
    b = Bravais{type,dim}(bravais)
    return Lattice(b, u)
end

#endregion