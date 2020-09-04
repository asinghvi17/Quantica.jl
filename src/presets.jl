#######################################################################
# Lattice presets
#######################################################################

module LatticePresets

using Quantica
using Quantica: NameType

linear(; a0 = 1, semibounded = false, kw...) =
    lattice(a0 * bravais((1.,); kw...), sublat((0.,)); kw...)

square(; a0 = 1, kw...) =
    lattice(a0 * bravais((1., 0.), (0., 1.); kw...), sublat((0., 0.)); kw...)

triangular(; a0 = 1, kw...) =
    lattice(a0 * bravais(( cos(pi/3), sin(pi/3)),(-cos(pi/3), sin(pi/3)); kw...),
        sublat((0., 0.)); kw...)

honeycomb(; a0 = 1, kw...) =
    lattice(a0 * bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3)); kw...),
        sublat((0.0, -0.5*a0/sqrt(3.0)), name = :A),
        sublat((0.0,  0.5*a0/sqrt(3.0)), name = :B); kw...)

cubic(; a0 = 1, kw...) =
    lattice(a0 * bravais((1., 0., 0.), (0., 1., 0.), (0., 0., 1.); kw...),
        sublat((0., 0., 0.)); kw...)

fcc(; a0 = 1, kw...) =
    lattice(a0 * bravais(@SMatrix([-1. -1. 0.; 1. -1. 0.; 0. 1. -1.])'/sqrt(2.); kw...),
        sublat((0., 0., 0.)); kw...)

bcc(; a0 = 1, kw...) =
    lattice(a0 * bravais((1., 0., 0.), (0., 1., 0.), (0.5, 0.5, 0.5); kw...),
        sublat((0., 0., 0.)); kw...)

end # module

#######################################################################
# Hamiltonian presets
#######################################################################

module HamiltonianPresets

using Quantica, LinearAlgebra

function graphene(; a0 = 0.246, range = a0/sqrt(3), t0 = 2.7)
    lat = LatticePresets.honeycomb(a0 = a0)
    # h = hamiltonian(lat, hopping((r, dr) -> t0 * exp(-3*(norm(dr)/a0 - 1)), range = 1.01*range), orbitals = size(t0, 1))
    h = hamiltonian(lat, hopping(-I, range = range), orbitals = size(t0, 1))
    return h
end

function twisted_bilayer_graphene(;
    twistindex = 1, twistindices = (twistindex, 1), a0 = 0.246, interlayerdistance = 1.36a0,
    rangeintralayer = a0/sqrt(3), rangeinterlayer = 4a0/sqrt(3), hopintra = 2.70,
    hopinter = 0.48, modelintra = hopping(hopintra, range = rangeintralayer), kw...)

    (m, r) = twistindices
    θ = acos((3m^2 + 3m*r +r^2/2)/(3m^2 + 3m*r + r^2))
    sAbot = sublat((0.0, -0.5a0/sqrt(3.0), - interlayerdistance / 2); name = :Ab)
    sBbot = sublat((0.0,  0.5a0/sqrt(3.0), - interlayerdistance / 2); name = :Bb)
    sAtop = sublat((0.0, -0.5a0/sqrt(3.0),   interlayerdistance / 2); name = :At)
    sBtop = sublat((0.0,  0.5a0/sqrt(3.0),   interlayerdistance / 2); name = :Bt)
    brbot = a0 * bravais(( cos(pi/3), sin(pi/3), 0), (-cos(pi/3), sin(pi/3), 0))
    brtop = a0 * bravais((-cos(pi/3), sin(pi/3), 0), ( cos(pi/3), sin(pi/3), 0))
    # Supercell matrices sc.
    # The one here is a [1 0; -1 1] rotation of the one in Phys. Rev. B 86, 155449 (2012)
    if gcd(r, 3) == 1
        scbot = @SMatrix[m -(m+r); (m+r) 2m+r] * @SMatrix[1 0; -1 1]
        sctop = @SMatrix[m+r -m; m 2m+r] * @SMatrix[1 0; -1 1]
    else
        scbot = @SMatrix[m+r÷3 -r÷3; r÷3 m+2r÷3] * @SMatrix[1 0; -1 1]
        sctop = @SMatrix[m+2r÷3 r÷3; -r÷3 m+r÷3] * @SMatrix[1 0; -1 1]
    end
    latbot = lattice(brbot, sAbot, sBbot)
    lattop = lattice(brtop, sAtop, sBtop)
    htop = hamiltonian(lattop, modelintra; kw...) |> unitcell(sctop)
    hbot = hamiltonian(latbot, modelintra; kw...) |> unitcell(scbot)
    let R = @SMatrix[cos(θ/2) -sin(θ/2) 0; sin(θ/2) cos(θ/2) 0; 0 0 1]
        transform!(r -> R * r, htop)
    end
    let R = @SMatrix[cos(θ/2) sin(θ/2) 0; -sin(θ/2) cos(θ/2) 0; 0 0 1]
        transform!(r -> R * r, hbot)
    end
    modelinter = hopping((r,dr) -> (
        hopintra * exp(-3*(norm(dr)/a0 - 1))  *  dot(dr, SVector(1,1,0))^2/sum(abs2, dr) -
        hopinter * exp(-3*(norm(dr)/a0 - interlayerdistance/a0)) * dr[3]^2/sum(abs2, dr)),
        range = rangeinterlayer)
    return combine(hbot, htop; coupling = modelinter)
end

end # module

#######################################################################
# Region presets
#######################################################################

module RegionPresets

using StaticArrays

struct Region{E,F} <: Function
    f::F
end

Region{E}(f::F) where {E,F<:Function} = Region{E,F}(f)

(region::Region{E})(r::SVector{E2}) where {E,E2} = region.f(r)

Base.show(io::IO, ::Region{E}) where {E} =
    print(io, "Region{$E} : region in $(E)D space")

extended_eps(T = Float64) = sqrt(eps(T))

circle(radius = 10.0, c...) = Region{2}(_region_ellipse((radius, radius), c...))

ellipse(radii = (10.0, 15.0), c...) = Region{2}(_region_ellipse(radii, c...))

square(side = 10.0, c...) = Region{2}(_region_rectangle((side, side), c...))

rectangle(sides = (10.0, 15.0), c...) = Region{2}(_region_rectangle(sides, c...))

sphere(radius = 10.0, c...) = Region{3}(_region_ellipsoid((radius, radius, radius), c...))

spheroid(radii = (10.0, 15.0, 20.0), c...) = Region{3}(_region_ellipsoid(radii, c...))

cube(side = 10.0, c...) = Region{3}(_region_cuboid((side, side, side), c...))

cuboid(sides = (10.0, 15.0, 20.0), c...) = Region{3}(_region_cuboid(sides, c...))

function _region_ellipse((rx, ry), (cx, cy) = (0, 0))
    return r -> ((r[1]-cx)/rx)^2 + ((r[2]-cy)/ry)^2 <= 1 + extended_eps(Float64)
end

function _region_rectangle((lx, ly), (cx, cy) = (0, 0))
    return r -> abs(2*(r[1]-cx)) <= lx * (1 + extended_eps()) &&
                abs(2*(r[2]-cy)) <= ly * (1 + extended_eps())
end

function _region_ellipsoid((rx, ry, rz), (cx, cy, cz) = (0, 0, 0))
    return r -> ((r[1]-cx)/rx)^2 + ((r[2]-cy)/ry)^2 + ((r[3]-cz)/rz)^2 <= 1 + eps()
end

function _region_cuboid((lx, ly, lz), (cx, cy, cz) = (0, 0, 0))
    return r -> abs(2*(r[1]-cx)) <= lx * (1 + extended_eps()) &&
                abs(2*(r[2]-cy)) <= ly * (1 + extended_eps()) &&
                abs(2*(r[3]-cy)) <= lz * (1 + extended_eps())
end

end # module