using DFTK
using ForwardDiff
using LinearAlgebra
using Random
using Optim

kgrid = [1, 1, 1]
Ecut = 5
H = ElementPsp(:H, psp=load_psp("hgh/lda/h-q1"))
atoms = [H,H]
lattice = 16 * Mat3(Diagonal(ones(3)))

function sample_positions(seed)
    randoms = 0.01*randn(MersenneTwister(seed),3)
    positions = [
        [0.45312500031210007, 1/2, 1/2] + randoms,
        [0.5468749996028622, 1/2, 1/2],
    ]
    positions
end 

function scfres_from_positions(positions,xc; kwargs...)
    T = typeof(xc.f(0.0))
    terms = [Kinetic(), AtomicLocal(), xc]
    model = Model(T.(lattice), atoms, positions; terms)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    self_consistent_field(basis;is_converged=DFTK.ScfConvergenceDensity(1e-4), kwargs...)
end 


# how to compute errors between (complex-valued) orbitals
function compute_wavefunction_error(ϕ, ψ)
    err = map(zip(ϕ, ψ)) do (ϕk, ψk)
        S = ψk'ϕk
        U = S*sqrt(inv(S'S))
        reshape(ϕk - ψk*U,:)
    end
    sum(abs2,reduce(vcat,err)) 
end

# Loss
function scfres_loss(scfres1,scfres2)
    compute_wavefunction_error(scfres1.ψ,scfres2.ψ)
end 

# Seeds -> Positions
Ndatapoints = 1
seeds = 1:Ndatapoints
positions = [sample_positions(seed) for seed in seeds]

k = 5
A = 0.05
xc_ground_truth = LocalNonlinearity(x-> (-3/4 * cbrt(3/π * x) *x ))
scfres_data = [scfres_from_positions(position,xc_ground_truth) for position in positions];

xrange = extrema(scfres_data[1].ρ)

# XC surrogate model
nparams = 10
xsbasis = collect(range(0, 0.3, length=nparams))
params = (1/nparams) .* ones(nparams)
lengtscale = xsbasis[2] - xsbasis[1]
rbf(x1, x2) = exp(-0.5*sum(abs2, x1 - x2) / lengtscale^2)
xc_model(x, params) = sum(params[i] * rbf(xsbasis[i], x) for i in 1:nparams)

function loss_params(params)
    xc_term = LocalNonlinearity(x -> xc_model(x, params))
    errors = map(zip(positions, scfres_data)) do (pos, scfres)
        scfres_pred = scfres_from_positions(pos,xc_term;maxiter=10)
        scfres_loss(scfres,scfres_pred)
    end
    sum(errors)
end 

# Optimizer

ForwardDiff.derivative(h->loss_params(params*h),0.0)

# opt = optimize(loss_params,params,LBFGS(),autodiff= :forward)
pred = loss_params(params)