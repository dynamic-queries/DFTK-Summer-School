using DFTK
using ForwardDiff
using LinearAlgebra
using Random

kgrid = [1, 1, 1]
Ecut = 5
H = ElementPsp(:H, psp=load_psp("hgh/lda/h-q1"))
atoms = [H,H]
lattice = 16 * Mat3(Diagonal(ones(3)))

function sample_positions(seed)
    randoms = + randn(MersenneTwister(seed),3)
    positions = [
        [0.45312500031210007, 1/2, 1/2] + randoms,
        [0.5468749996028622, 1/2, 1/2],
    ]
    positions
end 

function scfres_from_positions(positions,xc)
    terms = [Kinetic(), AtomicLocal(), xc]
    model = Model(lattice, atoms, positions; terms)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    self_consistent_field(basis, is_converged=DFTK.ScfConvergenceDensity(1e-4))
end 


# how to compute errors between (complex-valued) orbitals
function compute_wavefunction_error(ϕ, ψ)
    map(zip(ϕ, ψ)) do (ϕk, ψk)
        S = ψk'ϕk
        U = S*sqrt(inv(S'S))
        ϕk - ψk*U
    end
end

# Loss
function loss(scfres1,scfres2)
    compute_wavefunction_error(scfres1.ψ,scfres2.ψ)
end 

# Seeds -> Positions
Ndatapoints = 100
seeds = 1:Ndatapoints
positions = [sample_positions(seed) for seed in seeds]

ground_truth = sin
xc = LocalNonlinearity(ground_truth)
scfres_data = [scfres_from_positions(position,xc) for position in positions]

xrange = extrema(scfres.ρ)
