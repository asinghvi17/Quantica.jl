############################################################################################
# Green solvers
#   All new solver::AbstractGreenSolver must live in the GreenSolvers module, and must implement
#     - apply(solver, h::AbstractHamiltonian, c::Contacts) -> AppliedGreenSolver
#   All new s::AppliedGreenSolver must implement (with Σblock a [possibly nested] tuple of MatrixBlock's)
#      - s(ω, Σblocks, ::ContactBlockStructure) -> GreenSlicer
#      - minimal_callsafe_copy(gs)
#   A gs::GreenSlicer's allows to compute G[gi, gi´]::AbstractMatrix for indices gi
#   To do this, it must implement
#      - view(gs, ::Int, ::Int) -> g(ω; kw...) between specific contacts
#      - view(gs, ::Colon, ::Colon) -> g(ω; kw...) between all contacts
#      - gs[i::CellOrbitals, j::CellOrbitals] -> must return a Matrix for type stability
#      - minimal_callsafe_copy(gs)
#   The user-facing indexing API accepts:
#      - i::Integer -> Sites of Contact number i
#      - cellsites(cell::Tuple, sind::Int)::Subcell -> Single site in a cell
#      - cellsites(cell::Tuple, sindcollection)::Subcell -> Site collection in a cell
#      - cellsites(cell::Tuple, slat::Symbol)::Subcell -> Whole sublattice in a cell
#      - cellsites(cell::Tuple, :) ~ cell::Union{NTuple,SVector} -> All sites in a cell
#      - sel::SiteSelector ~ NamedTuple -> forms a LatticeSlice
############################################################################################

############################################################################################
# SelfEnergy solvers
#   All s::AbstractSelfEnergySolver must support the call! API
#     - call!(s::RegularSelfEnergySolver, ω; params...) -> Σreg::AbstractMatrix
#     - call!(s::ExtendedSelfEnergySolver, ω; params...) -> (Vᵣₑ, gₑₑ⁻¹, Vₑᵣ) AbsMats
#         With the extended case, the equivalent Σreg reads Σreg = VᵣₑgₑₑVₑᵣ
#     - call!_output(s::AbstractSelfEnergySolver) -> object returned by call!(s, ω; kw...)
#     - minimal_callsafe_copy(s::AbstractSelfEnergySolver)
#   These AbstractMatrices are flat, defined on the LatticeSlice in parent SelfEnergy
#       Note: `params` are only needed in cases where s adds new parameters that must be
#       applied (e.g. SelfEnergyModelSolver). Otherwise one must assume that any parent
#       ParametricHamiltonian to GreenFunction has already been call!-ed before calling s.
############################################################################################

############################################################################################
# SelfEnergy constructors
#   For each attach(h, sargs...; kw...) syntax we need, we must implement:
#     - SelfEnergy(h::AbstractHamiltonian, sargs...; kw...) -> SelfEnergy
#   SelfEnergy wraps the corresponding SelfEnergySolver, be it Regular or Extended
############################################################################################

module GreenSolvers

using Quantica: Quantica, AbstractGreenSolver, ensureloaded

struct SparseLU <:AbstractGreenSolver end

struct Schur{T<:AbstractFloat} <: AbstractGreenSolver
    shift::T                      # Tunable parameter in algorithm, see Ω in scattering.pdf
    boundary::T                   # Cell index for boundary (float to allow boundary at Inf)
end

Schur(; shift = 1.0, boundary = Inf) = Schur(shift, float(boundary))

struct KPM{B<:Union{Missing,NTuple{2}}} <: AbstractGreenSolver
    order::Int
    bandrange::B
end

# KPM(; order = 100, bandrange = missing) = KPM(order, bandrange)
function KPM(; order = 100, bandrange = missing)
    ensureloaded(:Arpack)
    return KPM(order, bandrange)
end


function bandrange_arnoldi(h::AbstractMatrix{T}) where {T}
    # ensureloaded(:ArnoldiMethod)
    R = real(T)
    decompl, _ = Quantica.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.LR());
    decomps, _ = Quantica.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.SR());
    ϵmax = R(real(decompl.eigenvalues[1]))
    ϵmin = R(real(decomps.eigenvalues[1]))
    return (ϵmin, ϵmax)
end

function bandrange_arpack(h::AbstractMatrix{T}) where {T}
    R = real(T)
    ϵL, _ = Quantica.Arpack.eigs(h, nev=1, tol=1e-4, which = :LR);
    ϵR, _ = Quantica.Arpack.eigs(h, nev=1, tol=1e-4, which = :SR);
    ϵmax = R(real(ϵL[1]))
    ϵmin = R(real(ϵR[1]))
    return (ϵmin, ϵmax)
end

end # module

const GS = GreenSolvers

include("greensolvers/selfenergymodel.jl")
include("greensolvers/sparselu.jl")
include("greensolvers/schur.jl")
include("greensolvers/kpm.jl")
# include("greensolvers/bands.jl")

