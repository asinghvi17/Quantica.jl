module Quantica

const REPOISSUES = "https://github.com/pablosanjose/Quantica.jl/issues"

using Base.Threads: Iterators

using StaticArrays
using NearestNeighbors
using SparseArrays
using SparseArrays: getcolptr, AbstractSparseMatrix, AbstractSparseMatrixCSC
using LinearAlgebra
using Dictionaries
using ProgressMeter
using Random
using SuiteSparse
using FunctionWrappers: FunctionWrapper
using ExprTools
using IntervalTrees
using FrankenTuples
using Statistics: mean
using QuadGK
using SpecialFunctions
using DelimitedFiles

export sublat, bravais_matrix, lattice, sites, supercell, hamiltonian,
       hopping, onsite, @onsite, @hopping, @onsite!, @hopping!, pos, ind, cell,
       position,
       plusadjoint, neighbors, siteselector, hopselector, diagonal,
       unflat, torus, transform, translate, combine,
       spectrum, energies, states, bands, subdiv,
       greenfunction, selfenergy, attach,
       plotlattice, plotlattice!, plotbands, plotbands!, qplot, qplot!, qplotdefaults,
       conductance, josephson, ldos, current, transmission, densitymatrix,
       OrbitalSliceArray, OrbitalSliceVector, OrbitalSliceMatrix, orbaxes

export LatticePresets, LP, RegionPresets, RP, HamiltonianPresets, HP, ExternalPresets, EP
export EigenSolvers, ES, GreenSolvers, GS
export @SMatrix, @SVector, SMatrix, SVector, SA
export ishermitian, tr, I, norm, dot, diag, det
export ftuple

# Types
include("types.jl")

# Preamble
include("iterators.jl")
include("builders.jl")
include("tools.jl")
include("docstrings.jl")

# API
include("specialmatrices.jl")
include("selectors.jl")
include("lattice.jl")
include("slices.jl")
include("models.jl")
include("hamiltonian.jl")
include("supercell.jl")
include("transform.jl")
include("mesh.jl")
include("bands.jl")
include("greenfunction.jl")
include("observables.jl")
# Plumbing
include("apply.jl")
include("show.jl")
include("convert.jl")
include("sanitizers.jl")


# Solvers
include("solvers/eigen.jl")
include("solvers/green.jl")
include("solvers/selfenergy.jl")

# Presets
include("presets/regions.jl")
include("presets/lattices.jl")
include("presets/hamiltonians.jl")
include("presets/external.jl")
include("presets/docstrings.jl")

# include("precompile.jl")

# Extension stubs for QuanticaMakieExt
function plotlattice end
function plotlattice! end
function plotbands end
function plotbands! end
function qplot end
function qplot! end
function qplotdefaults end

qplot(args...; kw...) =
    argerror("No plotting backend found or unexpected argument. Forgot to do e.g. `using GLMakie`?")

end
