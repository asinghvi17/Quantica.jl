############################################################################################
# greenfunction
#region

greenfunction(s::AbstractGreenSolver) = oh -> greenfunction(oh, s)

greenfunction() = h -> greenfunction(h)

greenfunction(h::AbstractHamiltonian, args...) = greenfunction(OpenHamiltonian(h), args...)

function greenfunction(oh::OpenHamiltonian, s::AbstractGreenSolver = default_green_solver(hamiltonian(oh)))
    cs = Contacts(oh)
    h = hamiltonian(oh)
    as = apply(s, h, cs)
    return GreenFunction(h, as, cs)
end

default_green_solver(::AbstractHamiltonian0D) = GS.SparseLU()
default_green_solver(::AbstractHamiltonian1D) = GS.Schur()
# default_green_solver(::AbstractHamiltonian) = GS.Bands()

#endregion

############################################################################################
# GreenFuntion call! API
#region

## TODO: test copy(g) for aliasing problems
(g::GreenFunction)(; params...) = minimal_callsafe_copy(call!(g; params...))
(g::GreenFunction)(ω; params...) = minimal_callsafe_copy(call!(g, ω; params...))
(g::GreenFunctionSlice)(; params...) = minimal_callsafe_copy(call!(g; params...))
(g::GreenFunctionSlice)(ω; params...) = copy(call!(g, ω; params...))

function call!(g::GreenFunction, ω; params...)
    h = parent(g)
    contacts´ = contacts(g)
    call!(h; params...)
    Σblocks = call!(contacts´, ω; params...)
    cbs = blockstructure(contacts´)
    slicer = solver(g)(ω, Σblocks, cbs)
    return GreenSolution(h, slicer, Σblocks, cbs)
end

call!(g::GreenFunctionSlice; params...) =
    GreenFunctionSlice(call!(greenfunction(g); params...), slicerows(g), slicecols(g))

call!(g::GreenFunctionSlice, ω; params...) =
    call!(greenfunction(g), ω; params...)[slicerows(g), slicecols(g)]

#endregion

############################################################################################
# Contacts call! API
#region

call!(c::Contacts; params...) = Contacts(call!.(c.selfenergies; params...), c.blockstruct)

function call!(c::Contacts, ω; params...)
    Σblocks = selfenergyblocks(c)
    call!.(c.selfenergies, Ref(ω); params...) # updates matrices in Σblocks
    return Σblocks
end

call!_output(c::Contacts) = selfenergyblocks(c)

minimal_callsafe_copy(s::Contacts) =
    Contacts(minimal_callsafe_copy.(s.selfenergies), s.blockstruct)

#endregion

############################################################################################
# GreenSolution indexing
#region

Base.getindex(g::GreenFunction, i, j = i) = GreenFunctionSlice(g, i, j)
Base.getindex(g::GreenFunction; kw...) = g[siteselector(; kw...)]

Base.getindex(g::GreenSolution; kw...) = g[getindex(lattice(g); kw...)]

Base.view(g::GreenSolution, i::Integer, j::Integer = i) = view(slicer(g), i, j)
Base.view(g::GreenSolution, i::Colon, j::Colon = i) = view(slicer(g), i, j)
Base.getindex(g::GreenSolution, i::Integer, j::Integer = i) = copy(view(g, i, j))
Base.getindex(g::GreenSolution, ::Colon, ::Colon = :) = copy(view(g, :, :))

function Base.getindex(g::GreenSolution, i)
    ai = ind_to_slice(i, g)
    return getindex(g, ai, ai)
end

Base.getindex(g::GreenSolution, i, j) = getindex(g, ind_to_slice(i, g), ind_to_slice(j, g))

# fallback for cases where i and j are not *both* contact indices -> convert to OrbitalSlice
function ind_to_slice(c::Integer, g)
    contactbs = blockstructure(g)
    cinds = contactinds(contactbs, c)
    os = orbslice(contactbs)[cinds]
    return os
end

ind_to_slice(c::CellSites, g) = orbslice(c, hamiltonian(g))
ind_to_slice(l::LatticeSlice, g) = orbslice(l, hamiltonian(g))
ind_to_slice(s::SiteSelector, g) = ind_to_slice(lattice(g)[s], g)
ind_to_slice(kw::NamedTuple, g) = ind_to_slice(getindex(lattice(g); kw...), g)
ind_to_slice(cell::Union{SVector,Tuple}, g::GreenSolution{<:Any,<:Any,L}) where {L} =
    ind_to_slice(cellsites(sanitize_SVector(SVector{L,Int}, cell), :), g)
ind_to_slice(c::CellSites{<:Any,Colon}, g) = cellorbs(cell(c), 1:flatsize(parent(g)))
ind_to_slice(c::CellSites{<:Any,Symbol}, g) =
    # uses a UnitRange instead of a Vector
    cellorbs(cell(c), flatrange(hamiltonian(g), siteindices(c)))
ind_to_slice(c::CellOrbitals, g) = c

Base.getindex(g::GreenSolution, i::OrbitalSlice, j::OrbitalSlice) =
    mortar([g[si, sj] for si in subcells(i), sj in subcells(j)])

Base.getindex(g::GreenSolution, i::OrbitalSlice, j::CellOrbitals) =
    mortar([g[si, sj] for si in subcells(i), sj in (j,)])

Base.getindex(g::GreenSolution, i::CellOrbitals, j::OrbitalSlice) =
    mortar([g[si, sj] for si in (i,), sj in subcells(j)])

Base.getindex(g::GreenSolution, i::CellOrbitals, j::CellOrbitals) = slicer(g)[i, j]

# fallback
Base.getindex(s::GreenSlicer, ::CellOrbitals, ::CellOrbitals) =
    internalerror("getindex of $(nameof(typeof(s))): not implemented")

#endregion

############################################################################################
# selfenergy(::GreenSolution[, contactinds::Int...]; onlyΓ = false)
#   if no contactinds are provided, all are returned. Otherwise, a view of Σ over contacts.
#   Note that a smaller, contact-specific similar_contactΣ(g, contact) cannot be used here,
#   since selfenergies are MatrixBlock's over the complete similar_contactΣ(g)
#region

function selfenergy(g::GreenSolution; kw...)
    Σ = similar_contactΣ(g)
    selfenergy!(Σ, g; kw...)
    return Σ
end

function selfenergy(g::GreenSolution, cind::Int, cinds::Int...; kw...)
    Σ = similar_contactΣ(g)
    selfenergy!(Σ, g, cind, cinds...; kw...)
    # view over selected contacts
    inds = copy(contactinds(g, cind))
    foreach(i -> append!(inds, contactinds(g, i)), cinds)
    unique!(sort!(inds))
    Σv = view(Σ, inds, inds)
    return Σv
end

function similar_contactΣ(g::Union{GreenFunction{T},GreenSolution{T}}) where {T}
    contactbs = blockstructure(g)  # ContactBlockStructure
    n = flatsize(contactbs)
    Σ = zeros(Complex{T}, n, n)
    return Σ
end

function selfenergy!(Σ::AbstractMatrix{T}, g::GreenSolution; onlyΓ = false) where {T}
    fill!(Σ, zero(T))
    addselfenergy!.(Ref(Σ), selfenergies(g))
    onlyΓ && extractΓ!(Σ)
    return Σ
end

function selfenergy!(Σ::AbstractMatrix{T}, g::GreenSolution, is...; onlyΓ = false) where {T}
    fill!(Σ, zero(T))
    addselfenergy!(Σ, g, is...)
    onlyΓ && extractΓ!(Σ)
    return Σ
end

addselfenergy!(Σ, g::GreenSolution, i::Int, is...) =
    addselfenergy!(addselfenergy!(Σ, selfenergies(g)[i]), g, is...)
addselfenergy!(Σ, ::GreenSolution) = Σ

# RegularSelfEnergy case
function addselfenergy!(Σ, b::MatrixBlock)
    v = view(Σ, blockrows(b), blockcols(b))
    v .+= blockmat(b)
    return Σ
end

# ExtendedSelfEnergy case
function addselfenergy!(Σ, (V´, g⁻¹, V)::NTuple{<:Any,MatrixBlock})
    v = view(Σ, blockrows(V´), blockcols(V))
    Vd = denseblockmat(V)
    copy!(Vd, blockmat(V))
    ldiv!(lu(blockmat(g⁻¹)), Vd)
    mul!(v, blockmat(V´), Vd, 1, 1)
    return Σ
end

function extractΓ!(Σ)
    Σ .-= Σ'
    Σ .*= im
    return Σ
end

#endregion

############################################################################################
# selfenergyblocks
#    Build MatrixBlocks from contacts, including extended inds for ExtendedSelfEnergySolvers
#region

function selfenergyblocks(contacts::Contacts)
    Σs = selfenergies(contacts)
    solvers = solver.(Σs)
    extoffset = flatsize(blockstructure(contacts))
    cinds = contactinds(contacts)
    Σblocks = selfenergyblocks(extoffset, cinds, 1, (), solvers...)
    return Σblocks
end

# extoffset: current offset where extended indices start
# contactinds: orbital indices for all selfenergies in contacts
# ci: auxiliary index for current selfenergy being processed
# blocks: tuple accumulating all MatrixBlocks from all selfenergies
# solvers: selfenergy solvers that will update the MatrixBlocks
selfenergyblocks(extoffset, contactinds, ci, blocks) = blocks

function selfenergyblocks(extoffset, contactinds, ci, blocks, s::RegularSelfEnergySolver, ss...)
    c = contactinds[ci]
    Σblock = MatrixBlock(call!_output(s), c, c)
    return selfenergyblocks(extoffset, contactinds, ci + 1, (blocks..., -Σblock), ss...)
end

function selfenergyblocks(extoffset, contactinds, ci, blocks, s::ExtendedSelfEnergySolver, ss...)
    Vᵣₑ, gₑₑ⁻¹, Vₑᵣ = shiftedmatblocks(call!_output(s), contactinds[ci], extoffset)
    extoffset += size(gₑₑ⁻¹, 1)
    # there is no minus sign here!
    return selfenergyblocks(extoffset, contactinds, ci + 1, (blocks..., (Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)), ss...)
end

function shiftedmatblocks((Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)::NTuple{3,AbstractArray}, cinds, shift)
    extsize = size(gₑₑ⁻¹, 1)
    Vᵣₑ´ = MatrixBlock(Vᵣₑ, cinds, shift+1:shift+extsize)
    # adds denseblock for selfenergy ldiv
    Vₑᵣ´ = MatrixBlock(Vₑᵣ, shift+1:shift+extsize, cinds, Matrix(Vₑᵣ))
    gₑₑ⁻¹´ = MatrixBlock(gₑₑ⁻¹, shift+1:shift+extsize, shift+1:shift+extsize)
    return Vᵣₑ´, gₑₑ⁻¹´, Vₑᵣ´
end

#endregion

############################################################################################
# contact_blockstructure constructors
#   Build a ContactBlockStructure from a Hamiltonian and a set of latslices
#region

contact_blockstructure(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} =
    ContactBlockStructure{L}()

contact_blockstructure(h::AbstractHamiltonian, ls, lss...) =
    contact_blockstructure(blockstructure(h), ls, lss...)

function contact_blockstructure(bs::OrbitalBlockStructure, lss...)
    lsall = combine(lss...)
    subcelloffsets = Int[]
    siteoffsets = Int[]
    osall = orbslice(lsall, bs, siteoffsets, subcelloffsets)
    contactinds = [contact_indices(lsall, siteoffsets, ls) for ls in lss]
    return ContactBlockStructure(osall, contactinds, siteoffsets, subcelloffsets)
end

# computes the orbital indices of ls sites inside the combined lsall
function contact_indices(lsall::LatticeSlice, siteoffsets, ls::LatticeSlice)
    contactinds = Int[]
    for scell´ in subcells(ls)
        so = findsubcell(cell(scell´), lsall)
        so === nothing && continue
        # here offset is the number of sites in lsall before scell
        (scell, offset) = so
        for i´ in siteindices(scell´), (n, i) in enumerate(siteindices(scell))
            n´ = offset + n
            i == i´ && append!(contactinds, siteoffsets[n´]+1:siteoffsets[n´+1])
        end
    end
    return contactinds
end

#endregion

############################################################################################
# contact_sites_to_orbitals
#  convert a list of sites in a ContactBlockStructure to a list of orbitals
#region

function contact_sites_to_orbitals(siteinds, bs::ContactBlockStructure)
    finds = Int[]
    for iunflat in siteinds
        append!(finds, siterange(bs, iunflat))
    end
    return finds
end

#endregion

############################################################################################
# TMatrixSlicer <: GreenSlicer
#    Given a slicer that works without any contacts, implement slicing with contacts through
#    a T-Matrix equation g(i, j) = g0(i, j) + g0(i, k)T(k,k')g0(k', j), and T = (1-Σ*g0)⁻¹*Σ
#region

struct TMatrixSlicer{C,L,V<:AbstractArray{C},S} <: GreenSlicer{C}
    g0slicer::S
    tmatrix::V
    gcontacts::V
    blockstruct::ContactBlockStructure{L}
end

#region ## Constructors ##

function TMatrixSlicer(g0slicer::GreenSlicer{C}, Σblocks, blockstruct) where {C}
    if isempty(Σblocks)
        zeromat = view(zeros(C, 0, 0), 1:0, 1:0)
        return TMatrixSlicer(g0slicer, zeromat, zeromat, blockstruct)
    else
        Σblocks´ = tupleflatten(Σblocks...)
        os = orbslice(blockstruct)
        nreg = norbs(os)                                        # number of regular orbitals
        n = max(nreg, maxrows(Σblocks´), maxcols(Σblocks´))     # includes extended orbitals
        Σmatext = Matrix{C}(undef, n, n)
        Σbm = BlockMatrix(Σmatext, Σblocks´)
        update!(Σbm)                                            # updates Σmat with Σblocks´
        g0mat = zeros(C, nreg, nreg)
        off = offsets(os)
        for (j, sj) in enumerate(subcells(os)), (i, si) in enumerate(subcells(os))
            irng = off[i]+1:off[i+1]
            jrng = off[j]+1:off[j+1]
            g0view = view(g0mat, irng, jrng)
            copy!(g0view, g0slicer[si, sj])
        end
        Σmatᵣᵣ = view(Σmatext, 1:nreg, 1:nreg)
        Σmatₑᵣ = view(Σmatext, nreg+1:n, 1:nreg)
        Σmatᵣₑ = view(Σmatext, 1:nreg, nreg+1:n)
        Σmatₑₑ = view(Σmatext, nreg+1:n, nreg+1:n)
        Σmat = copy(Σmatᵣᵣ)
        Σmat´ = ldiv!(lu!(Σmatₑₑ), Σmatₑᵣ)
        mul!(Σmat, Σmatᵣₑ, Σmat´, 1, 1)              # Σmat = Σmatᵣᵣ + ΣmatᵣₑΣmatₑₑ⁻¹ Σmatₑᵣ
        den = Matrix{C}(I, nreg, nreg)
        mul!(den, Σmat, g0mat, -1, 1)                          # den = 1-Σ*g0
        luden = lu!(den)
        tmatrix = ldiv!(luden, Σmat)                           # tmatrix = (1 - Σ*g0)⁻¹Σ
        gcontacts = rdiv!(g0mat, luden)                        # gcontacts = g0*(1 - Σ*g0)⁻¹

        return TMatrixSlicer(g0slicer, tmatrix, gcontacts, blockstruct)
    end
end

#endregion

#region ## API ##

Base.view(s::TMatrixSlicer, i::Integer, j::Integer) =
    view(s.gcontacts, contactinds(s.blockstruct, i), contactinds(s.blockstruct, j))

Base.view(s::TMatrixSlicer, ::Colon, ::Colon) = s.gcontacts

function Base.getindex(s::TMatrixSlicer, i::CellOrbitals, j::CellOrbitals)
    g0 = s.g0slicer
    g0ij = g0[i, j]
    tkk´ = s.tmatrix
    isempty(tkk´) && return g0ij
    k = orbslice(s.blockstruct)
    g0ik = mortar([g0[si, sk] for si in (i,), sk in subcells(k)])
    g0k´j = mortar([g0[sk´, sj] for sk´ in subcells(k), sj in (j,)])
    gij = mul!(g0ij, g0ik, tkk´ * g0k´j, 1, 1)  # = g0ij + g0ik * tkk´ * g0k´j
    return gij
end

minimal_callsafe_copy(s::TMatrixSlicer) = TMatrixSlicer(minimal_callsafe_copy(s.g0slicer),
    s.tmatrix, s.gcontacts, s.blockstruct)

#endregion

#endregion