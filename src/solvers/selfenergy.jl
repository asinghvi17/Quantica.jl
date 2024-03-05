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
#   Optional: AbstractSelfEnergySolver's can also implement `selfenergy_plottables`
#     - selfenergy_plottables(s::AbstractSelfEnergySolver, parent_latslice, boundary_sel)
#       -> collection of tuples to be passed to plotlattice!(axis, tup...) for visualization
############################################################################################

############################################################################################
# SelfEnergy constructors
#   For each attach(h, sargs...; kw...) syntax we need, we must implement:
#     - SelfEnergy(h::AbstractHamiltonian, sargs...; kw...) -> SelfEnergy
#   SelfEnergy wraps the corresponding SelfEnergySolver, be it Regular or Extended
############################################################################################

selfenergy_plottables(s::SelfEnergy, boundaryselector) =
    selfenergy_plottables(s.solver, orbslice(s), boundaryselector)

# fallback
selfenergy_plottables(s::AbstractSelfEnergySolver, parent_latslice, boundaryselector) =
    (parent_latslice[boundaryselector],)

include("selfenergy/nothing.jl")
include("selfenergy/model.jl")
include("selfenergy/schur.jl")
include("selfenergy/generic.jl")
