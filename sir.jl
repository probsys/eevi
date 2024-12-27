import Gen
using Gen: GenerativeFunction, Trace, ChoiceMap
using Gen: simulate, generate, categorical

#########################
# resampling combinator #
#########################

# compile-time arguments to the combinator (generative function constructor)
# 1. the model generative function
# 2. the proposal generative function
# 3. the arguments to the model
# 4. the number of particles (K)

# the resulting generative function has the same support over choice maps as
# the proposal generative function

# run-time arguments to the generative function:
# 1. the arguments to the proposal, whose last entry is the
#    observations on the model

# requirement: for any choice map in the support of the  proposal generative
# function on the given arguments, the merger of that choice map with the
# observations must result in a choice map that is in the support of the model
# generative function's distribution on choice maps

# NOTE: project, update, regenerate, choice_gradients, and
# accumulate_param_gradients!  are not yet implemented for the resulting
# functions

struct SIRGF{T,U} <: GenerativeFunction{T,Trace}
    model::GenerativeFunction
    proposal::GenerativeFunction{T,U}
    model_args::Tuple
    num_particles::Int
end

# construct using default constructor

struct SIRGFTrace{T,U} <: Gen.Trace
    gen_fn::SIRGF{T,U}
    # arguments to the model generative function
    model_args::Tuple
    # arguments to the proposal generative function
    proposal_args::Tuple
    # when combined with choices made by proposal,
    # uniquely determine a model trace
    observations::ChoiceMap
    # number of particles
    num_particles::Int
    # the chosen trace of proposal generative function
    chosen_particle::U
    # score
    score::Float64
end

Gen.get_gen_fn(trace::SIRGFTrace) = trace.gen_fn

function Gen.get_args(trace::SIRGFTrace)
    return trace.proposal_args
end

Gen.get_score(trace::SIRGFTrace) = trace.score
Gen.get_retval(trace::SIRGFTrace) = get_retval(trace.chosen_particle)
Base.getindex(trace::SIRGFTrace, addr) = trace.chosen_particle[addr]
Gen.get_choices(trace::SIRGFTrace) = get_choices(trace.chosen_particle)

function Gen.simulate(gen_fn::SIRGF{T,U}, args::Tuple) where {T,U}
    model = gen_fn.model
    model_args = gen_fn.model_args
    num_particles = gen_fn.num_particles
    proposal_args = args
    observations = proposal_args[end]
    proposal_traces = Vector{U}(undef, num_particles)
    log_weights = Vector{Float64}(undef, num_particles)
    model_scores = Vector{Float64}(undef, num_particles)
    for i in 1:num_particles
        # sample from proposal
        proposal_trace = simulate(gen_fn.proposal, proposal_args)
        proposal_traces[i] = proposal_trace
        proposed_choices = get_choices(proposal_trace)
        # combine latents with observations to form a model trace
        constraints = merge(observations, proposed_choices)
        (model_trace, model_score) = generate(model, model_args, constraints)
        # for now, make sure all of the choices in the model are constrained
        @assert isapprox(model_score, get_score(model_trace))
        # record the model joint density (model_score) and importance weight
        model_scores[i] = model_score
        log_weights[i] = model_score - get_score(proposal_trace)
    end
    # sample particle in proposal to weights
    log_total_weight = Gen.logsumexp(log_weights)
    normalized_weights = exp.(log_weights .- log_total_weight)
    chosen_idx = categorical(normalized_weights)
    chosen_particle = proposal_traces[chosen_idx]
    # compute score (our estimate of the marginal density of our choices)
    log_ml_estimate = log_total_weight - log(num_particles)
    score = model_scores[chosen_idx] - log_ml_estimate
    # NOTE: we do not retain the other particles in our trace
    return SIRGFTrace(
        gen_fn,
        model_args,
        proposal_args,
        observations,
        num_particles,
        chosen_particle,
        score)
end

function Gen.generate(gen_fn::SIRGF, args::Tuple, constraints::ChoiceMap)
    model = gen_fn.model
    model_args = gen_fn.model_args
    num_particles = gen_fn.num_particles
    proposal_args = args
    observations = proposal_args[end]
    # combine observations and constraints to form model trace
    constraints_model = merge(observations, constraints)
    (model_trace, model_score) = generate(model, model_args, constraints_model)
    # for now, make sure all of the choices in the model are constrained
    @assert isapprox(model_score, get_score(model_trace))
    # form the chosen particle (the trace of the proposal)
    # and check that all of the proposal choices are constrained
    (chosen_particle, proposal_score) = generate(gen_fn.proposal, proposal_args, constraints)
    @assert isapprox(get_score(chosen_particle), proposal_score)
    # sample the other particles, just to compute the score
    # (we do not retain the other particles in our trace)
    log_weights = Vector{Float64}(undef, num_particles)
    log_weights[1] = model_score - proposal_score
    for i in 2:num_particles
        # sample from proposal
        proposal_trace = simulate(gen_fn.proposal, proposal_args)
        proposed_choices = get_choices(proposal_trace)
        # combine latents with observations to form a model trace
        constraints_i = merge(observations, proposed_choices)
        (model_trace_i, model_score_i) = generate(model, model_args, constraints_i)
        # for now, make sure all of the choices in the model are constrained
        @assert isapprox(model_score_i, get_score(model_trace_i))
        # record the importance weight
        log_weights[i] = model_score_i - get_score(proposal_trace)
    end
    # compute the score
    log_total_weight = Gen.logsumexp(log_weights)
    log_ml_estimate = log_total_weight - log(num_particles)
    score = model_score - log_ml_estimate
    # NOTE: we do not retain the other particles in our trace
    trace = SIRGFTrace(
        gen_fn,
        model_args,
        proposal_args,
        observations,
        num_particles,
        chosen_particle,
        score)
    return (trace, score)
end
