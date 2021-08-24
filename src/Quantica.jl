module Quantica

# Use README as the docstring of the module:
using Base.Threads: Iterators
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) Quantica

using Requires

function __init__()
      @require GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("plot_makie.jl")
      @require VegaLite = "112f6efa-9a02-5b7d-90c0-432ed331239a" include("plot_vegalite.jl")
end

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays,
      ProgressMeter, Random, SuiteSparse

using FunctionWrappers: FunctionWrapper

using ExprTools

using SparseArrays: getcolptr, AbstractSparseMatrix, AbstractSparseMatrixCSC

using Statistics: mean

using Compat # for use of argmin/argmax in bandstructure.jl

export sublat, bravais, lattice, dims, supercell,
       hopping, onsite, @onsite!, @hopping!, @block!, parameters, siteselector, hopselector, nrange,
       sitepositions, siteindices, not,
       ket, ketmodel, randomkets, basiskets,
       hamiltonian, parametric, bloch, bloch!, similarmatrix,
       flatten, unflatten, orbitalstructure, wrap, transform!, combine,
       spectrum, bandstructure, diagonalizer, cuboid, isometric, splitbands,
       bands, vertices, minima, maxima, gapedge, gap, isinband,
       energies, states, degeneracy,
       momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM,
       greens, greensolver, Schur1D

export LatticePresets, LP, RegionPresets, RP #, HamiltonianPresets, HP

export LinearAlgebraPackage, ArpackPackage, ArnoldiMethodPackage, KrylovKitPackage

export @SMatrix, @SVector, SMatrix, SVector, SA

export ishermitian, I

export SparseMatrixCSC

include("types.jl")
include("apply.jl")
include("iterators.jl")
include("selector.jl")
include("lattice.jl")
include("model.jl")
include("builders.jl")
include("hamiltonian.jl")
include("supercell.jl")
# include("ket.jl")
# include("parametric.jl")
# include("slice.jl")
# include("mesh.jl")
# include("diagonalizer.jl")
# include("bandstructure.jl")
# include("KPM.jl")
# include("effective.jl")
# include("greens.jl")
include("sanitize.jl")
include("show.jl")
include("tools.jl")
include("convert.jl")

include("presets.jl")

include("precompile.jl")

end