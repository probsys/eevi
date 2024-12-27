import Gen

using Gen: ChoiceMap, GenerativeFunction, Selection, Trace
using Gen: categorical, choicemap, generate, propose, simulate
using Gen: @gen

import LinearAlgebra
import Distributions

# Trace for Multivariate Normal GenerativeFunction
struct MvnTrace{U <: Real} <: Gen.Trace
    gen_fn::GenerativeFunction
    mu::AbstractVector{U}
    cov::AbstractMatrix{U}
    constrained::Set{Int}
    choices::ChoiceMap
    retval::AbstractVector{U}
    score::Float64
end

# Multivariate Normal GenerativeFunction
abstract type MvnGF{U <: Real} <: GenerativeFunction{AbstractVector{U}, MvnTrace} end
struct MvnGFLW{U} <: MvnGF{U} end         # Likelihood weighting
struct MvnGFExact{U} <: MvnGF{U} end      # Exact inference

# Singleton types.
const mvnorm_lw = MvnGFLW{Real}()
const mvnorm_exact = MvnGFExact{Real}()

# Trace API.
Gen.get_gen_fn(trace::MvnTrace) = trace.gen_fn
Gen.get_args(trace::MvnTrace) = (trace.mu, trace.cov)
Gen.get_score(trace::MvnTrace) = trace.score
Gen.get_retval(trace::MvnTrace) = trace.retval
Base.getindex(trace::MvnTrace, addr::Integer) = trace.choices[addr]
Gen.get_choices(trace::MvnTrace) = trace.choices

# SIMULATE
function Gen.simulate(gen_fn::MvnGF{U}, args::Tuple) where {U}
    (mu, cov) = args
    dist = Distributions.MvNormal(mu, cov)
    constrained = Set{Int}()
    retval = rand(dist)
    choices = choicemap(((i, x) for (i, x) in enumerate(retval))...)
    score = Distributions.logpdf(dist, retval)
    trace = MvnTrace(gen_fn, mu, cov, constrained, choices, retval, score)
    return trace
end

# PROJECT
function Gen.project(trace::MvnTrace, selection::Selection)
    if isempty(selection)
        return 0.
    end
    # No type annotations causes StackOverflowError?
    (mu::Vector{Float64}, cov::Matrix{Float64}) = Gen.get_args(trace)
    ndim = length(mu)
    constrained = Set{Int}(keys(Gen.get_subselections(selection)))
    @assert all((1 <= i <= ndim for i in constrained))
    retval = Gen.get_retval(trace)
    # All constrained.
    if length(constrained) == ndim
        dist = Distributions.MvNormal(mu, cov)
        score = Distributions.logpdf(dist, retval)
        return score
    end
    # Partial constraints.
    gen_fn::MvnGF = Gen.get_gen_fn(trace)
    return project_mvn_partial(gen_fn, mu, cov, retval, constrained)
end

function project_mvn_partial(::MvnGFLW,
        mu::AbstractVector,
        cov::AbstractMatrix,
        retval::AbstractVector,
        constrained::Set{Int})
    # p(x,y) / p(x), where x is the unconstrained part.
    ndim = length(mu)
    unconstrained = [i for i=1:ndim if !(i in constrained)]
    dist_sim = Distributions.MvNormal(mu[unconstrained], cov[unconstrained,unconstrained])
    logp_sim = Distributions.logpdf(dist_sim, retval[unconstrained])
    dist = Distributions.MvNormal(mu, cov)
    logp = Distributions.logpdf(dist, retval)
    score = logp - logp_sim
    return score
end

function project_mvn_partial(::MvnGFExact,
        mu::AbstractVector,
        cov::AbstractMatrix,
        retval::AbstractVector,
        constrained::Set{Int})
    # p(x,y) / p(x|y) = p(y), where x is the unconstrained part.
    ndim = length(mu)
    idx = [i for i=1:ndim if (i in constrained)]
    dist = Distributions.MvNormal(mu[idx], cov[idx,idx])
    return Distributions.logpdf(dist, retval[idx])
end

# GENERATE
function Gen.generate(gen_fn::MvnGF{U}, args::Tuple, constraints::ChoiceMap) where {U}
    # No constraints.
    if isempty(constraints)
        trace = Gen.simulate(gen_fn, args)
        score = 0.
        return (trace, score)
    end
    # Extract arguments.
    (mu, cov) = args
    ndim = length(mu)
    constrained = Set{Int}(keys(Gen.get_values_shallow(constraints)))
    # All constrained.
    if length(constrained) == ndim
        dist = Distributions.MvNormal(mu, cov)
        retval = [constraints[i] for i=1:ndim]
        choices = choicemap(((i, retval[i]) for i=1:ndim)...)
        score = Distributions.logpdf(dist, retval)
        trace = MvnTrace{U}(gen_fn, mu, cov, constrained, choices, retval, score)
        return (trace, score)
    end
    # Partial constraints.
    return generate_mvn_partial(gen_fn, mu, cov, constrained, constraints)
end

function generate_mvn_partial(
        gen_fn::MvnGFLW{U},
        mu::AbstractVector,
        cov::AbstractMatrix,
        constrained::Set{Int},
        constraints::ChoiceMap) where {U}
    # Partial constraints (likelihood weighting).
    ndim = length(mu)
    unconstrained = [i for i=1:ndim if !(i in constrained)]
    dist_sim = Distributions.MvNormal(mu[unconstrained], cov[unconstrained, unconstrained])
    retval_sim = rand(dist_sim)
    logp_sim = Distributions.logpdf(dist_sim, retval_sim)
    retval = Vector{U}(undef, ndim)
    for (i, j) in enumerate(unconstrained)
        retval[j] = retval_sim[i]
    end
    for i in constrained
        retval[i] = constraints[i]
    end
    choices = choicemap(((i, retval[i]) for i=1:ndim)...)
    dist = Distributions.MvNormal(mu, cov)
    logp = Distributions.logpdf(dist, retval)
    score = logp - logp_sim
    trace = MvnTrace{U}(gen_fn, mu, cov, constrained, choices, retval, score)
    return trace, score
end

function generate_mvn_partial(
        gen_fn::MvnGFExact{U},
        mu::AbstractVector,
        cov::AbstractMatrix,
        constrained::Set{Int},
        constraints::ChoiceMap) where {U}
    # Partial constraints (exact inference).
    # p(x|y) with weight p(x,y) / p(x|y) = p(y)
    ndim = length(mu)
    unconstrained = [i for i=1:ndim if !(i in constrained)]
    mu_sim, cov_sim = mvn_condition(mu, cov, constraints)
    dist_sim = Distributions.MvNormal(mu_sim, cov_sim)
    retval_sim = rand(dist_sim)
    retval = Vector{U}(undef, ndim)
    for (i, j) in enumerate(unconstrained)
        retval[j] = retval_sim[i]
    end
    for i in constrained
        retval[i] = constraints[i]
    end
    choices = choicemap(((i, retval[i]) for i=1:ndim)...)
    score = project_mvn_partial(gen_fn, mu, cov, retval, constrained)
    trace = MvnTrace{U}(gen_fn, mu, cov, constrained, choices, retval, score)
    return trace, score
end

function mvn_condition(mu::AbstractVector, cov::AbstractMatrix, constraints::ChoiceMap)
    # Partition indexes.
    ndim = length(mu)
    idx_obs_set = keys(Gen.get_values_shallow(constraints))
    idx_obs = [i for i=1:(ndim) if i in idx_obs_set]
    idx_new = [i for i=1:(ndim) if !(i in idx_obs_set)]
    ys = [constraints[i] for i in idx_obs]
    # Extract mean and covariance submatrices.
    mu1 = mu[idx_obs]
    mu2 = mu[idx_new]
    cov_11 = cov[idx_obs, idx_obs]
    cov_22 = cov[idx_new, idx_new]
    cov_12 = cov[idx_obs, idx_new]
    cov_21 = cov[idx_new, idx_obs]
    @assert cov_12 == cov_21'
    # Make condition
    conditional_mu = mu2 + cov_21 * (cov_11 \ (ys - mu1))
    conditional_cov = cov_22 - cov_21 * (cov_11 \ cov_12)
    conditional_cov = .5 * conditional_cov + .5 * conditional_cov'
    return (conditional_mu, conditional_cov)
end

# JULIA

# Allows calling MvnGF as a regular function.
function (gen_fn::MvnGF)(args...)
    trace = simulate(gen_fn, args)
    return Gen.get_retval(trace)
end
