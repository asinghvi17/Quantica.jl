@testset "show methods" begin
    hs = HP.graphene(orbitals = 2), HP.graphene(orbitals = (2,1))
    for h in hs
        b = bands(h, subdiv(0,2pi,10), subdiv(0,2pi,10))
        g = greenfunction(supercell(h) |> attach(@onsite(ω -> im*I)) |> attach(nothing), GS.Spectrum())
        @test nothing === show(stdout, sublat((0,0)))
        @test nothing === show(stdout, LP.honeycomb())
        @test nothing === show(stdout, LP.honeycomb()[cells = (0,0)])
        @test nothing === show(stdout, siteselector(; cells = (0,0)))
        @test nothing === show(stdout, hopselector(; range = 1))
        @test nothing === show(stdout, onsite(1) + hopping(1))
        @test nothing === show(stdout, @onsite(()->1) + @hopping(()->1))
        @test nothing === show(stdout, @onsite!(o->1))
        @test nothing === show(stdout, @hopping!(t->1))
        @test nothing === show(stdout, h)
        @test nothing === show(stdout, h |> hamiltonian(@onsite!(o->2o)))
        @test nothing === show(stdout, h |> attach(nothing, cells = (0,0)))
        @test nothing === show(stdout, current(h))
        @test nothing === show(stdout, ES.LinearAlgebra())
        @test nothing === show(stdout, spectrum(h, (0,0)))
        @test nothing === show(stdout, b)
        @test nothing === show(stdout, b[(0,0)])
        @test nothing === show(stdout, Quantica.slice(b, (0,0)))
        @test nothing === show(stdout, Quantica.slice(b, (0,0)))
        @test nothing === show(stdout, g)
        @test nothing === show(stdout, g[cells = ()])
        @test nothing === show(stdout, MIME("text/plain"), g[diagonal(cells = ())](0.1))
        @test nothing === show(stdout, g(0.1))
        @test nothing === show(stdout, ldos(g[1]))
        @test nothing === show(stdout, ldos(g(0.1)))
        @test nothing === show(stdout, current(g[1]))
        @test nothing === show(stdout, current(g(0.1)))
        @test nothing === show(stdout, conductance(g[1]))
        @test nothing === show(stdout, transmission(g[1,2]))
        @test nothing === show(stdout, densitymatrix(g[1]))
        @test nothing === show(stdout, MIME("text/plain"), densitymatrix(g[1])())
        @test nothing === show(stdout, sites(1))
        @test nothing === show(stdout, sites(SA[1], 2:4))
        @test nothing === show(stdout, sites(:))
        @test nothing === show(stdout, sites(SA[0], :))
    end
    h = first(hs)
    g = greenfunction(supercell(h) |> attach(@onsite(ω -> im*I)) |> attach(nothing))
    @test nothing === show(stdout, josephson(g[1], 2))
    @test nothing === show(stdout, densitymatrix(g[1], 2))
    h = supercell(h, 3) |> supercell
    g = greenfunction(supercell(h) |> attach(nothing), GS.KPM())
    @test nothing === show(stdout, densitymatrix(g[1]))
    g = greenfunction(supercell(h) |> attach(@onsite(ω -> im*I)) |> attach(nothing), GS.Spectrum())
    @test nothing === show(stdout, densitymatrix(g[1]))
    b = LP.honeycomb() |> Quantica.builder(orbitals = 2)
    @test nothing === show(stdout, b)
    w = EP.wannier90("wannier_test_tb.dat");
    @test nothing === show(stdout, w)
    @test nothing === show(stdout, position(w))
end
