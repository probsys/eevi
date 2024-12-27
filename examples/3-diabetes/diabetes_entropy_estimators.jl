import Gen

using Statistics: mean

using Gen:
    assess,
    complement,
    project,
    generate,
    GenerativeFunction,
    get_choices,
    get_score,
    get_selected,
    Selection,
    simulate

function logmeanexp(arr::AbstractArray{T}) where {T <: Real}
    return Gen.logsumexp(arr) - log(length(arr))
end

# Implementation of entropy estimators for including
# background constraints that are always included in the
# forward simulation and model. It is essential that these
# constraints are root choices, such that their weights are
# the same for any execution of generate.
#
# It is especially needed for upper bounds, since we have to
# generate an exact sample of (latents, targets) from the
# joint model distribution. Consider
#   latent -> constraint -> target
# In this case, sampling latent from the prior and constraining
# constraint and target no longer maintains the property that
# the joint sample gives us a posterior sample from distribution
#   latent | (constraint, target)

"""Entropy lower bound using default proposal."""
function entropy_lower_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        targets::Selection,
        constraints::Gen.ChoiceMap,
        N::Integer,
        M::Integer,
        K::Integer,
        model_traces::Vector{Gen.Trace}=Gen.Trace[];
        return_weights::Bool=false)
    # Verify offset is constant.
    log_w_offset = generate(model, model_args, constraints)[2]
    @assert log_w_offset == generate(model, model_args, constraints)[2]
    # Run estimator.
    wi_list = []
    for i=1:N
        # Sample observations from model.
        tr_p = model_traces[i]
        observations = get_selected(get_choices(tr_p), targets)
        observations_and_constraints = merge(observations, constraints)
        wj_list = []
        for j=1:M
            wk_list::Vector{Float64} = []
            for k=1:K
                # Sample latents (particle k) from default proposal
                # and retrieve weight w_k,
                # which is p(observations | latents)
                tr_q, log_w = generate(model, model_args, observations_and_constraints)
                push!(wk_list, log_w - log_w_offset)
            end
            # Compute overall log importance weight,
            # which is log [1/K sum w_k]
            wk_avg = logmeanexp(wk_list)
            !isinf(wk_avg) || throw("Invalid proposal (infinite log weight)")
            push!(wj_list, wk_avg)
        end
        wj_avg = mean(wj_list)
        push!(wi_list, wj_avg)
    end
    return return_weights ? wi_list : mean(wi_list)
end

"""Entropy upper bound using default proposal."""
function entropy_upper_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        targets::Selection,
        constraints::Gen.ChoiceMap,
        N::Integer,
        M::Integer,
        K::Integer,
        model_traces::Vector{Gen.Trace}=Gen.Trace[];
        return_weights::Bool=false)
    @assert(M == 1)
    # Verify offset is constant
    log_w_offset = generate(model, model_args, constraints)[2]
    @assert log_w_offset == generate(model, model_args, constraints)[2]
    # Run estimator
    wi_list = []
    for i=1:N
        # Sample latents + observations from model.
        tr_p = model_traces[i]
        observations = get_selected(get_choices(tr_p), targets)
        observations_and_constraints = merge(observations, constraints)
        wj_list = []
        for j=1:M
            tr_pk = tr_p
            wk_list::Vector{Float64} = []
            # Retrieve weight of exact posterior sample (particle 1)
            # using default proposal, which is
            # p(observations | latents)
            log_w = Gen.project(tr_pk, targets)
            push!(wk_list, log_w)
            for k=2:K
                # Sample latents (particle k) from default proposal
                # and retrieve weight w_k,
                # which is p(observations | latents)
                tr_pk, log_w = generate(model, model_args, observations_and_constraints)
                push!(wk_list, log_w - log_w_offset)
            end
            # Compute overall log importance weight, which is
            # log [1 / (1/K sum w_k)]
            # = - log [1/K sum w_k]
            wk_avg = -logmeanexp(wk_list)
            @assert !isinf(wk_avg)
            push!(wj_list, wk_avg)
            # TODO: MCMC step iterating the latents in tr_p
            # if j < M
            #   tr_p = metropolis_hastings(tr_p, complement(targets);
            #       check=true, observations=observations)
            # end
        end
        wj_avg = -mean(wj_list)
        push!(wi_list, wj_avg)
    end
    return return_weights ? wi_list : mean(wi_list)
end
