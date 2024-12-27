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

"""Entropy lower bound using custom proposal."""
function entropy_lower_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        proposal::GenerativeFunction,
        proposal_args::Tuple,
        targets::Selection,
        N::Integer,
        M::Integer;
        model_traces::Vector{Gen.Trace}=Gen.Trace[])
    wi_list = []
    for i=1:N
        # Sample observations from model.
        tr_p = isempty(model_traces) ? simulate(model, model_args) : model_traces[i]
        observations = get_selected(get_choices(tr_p), targets)
        wj_list = []
        for j=1:M
            # Sample latents from proposal and compute score.
            tr_q = simulate(proposal, (proposal_args..., observations,))
            log_q = get_score(tr_q)
            # Compute score of latents + observations under model.
            choices = merge(observations, get_choices(tr_q))
            _, log_p = generate(model, model_args, choices)
            # Compute importance weight.
            log_w = log_p - log_q
            push!(wj_list, log_w)
        end
        wj_avg = mean(wj_list)
        push!(wi_list, wj_avg)
    end
    return mean(wi_list)
end

"""Entropy upper bound using custom proposal."""
function entropy_upper_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        proposal::GenerativeFunction,
        proposal_args::Tuple,
        targets::Selection,
        N::Integer,
        M::Integer;
        model_traces::Vector{Gen.Trace}=Gen.Trace[])
    @assert(M == 1)
    wi_list = []
    for i=1:N
        # Sample latents + observations from model and compute score.
        tr_p = isempty(model_traces) ? simulate(model, model_args) : model_traces[i]
        wj_list = []
        for j=1:M
            # Compute score of latents + observations under model.
            log_p = get_score(tr_p)
            # Compute score of latents under proposal.
            observations = get_selected(get_choices(tr_p), targets)
            latents = get_selected(get_choices(tr_p), complement(targets))
            _, log_q = generate(proposal, (proposal_args..., observations,), latents)
            # Compute importance weight.
            log_w = log_q - log_p
            push!(wj_list, log_w)
            # TODO: MCMC step iterating the latents in tr_p
            # if j < M
            #   tr_p = metropolis_hastings(tr_p, complement(targets);
            #       check=true, observations=observations)
            # end
        end
        wj_avg = -mean(wj_list)
        push!(wi_list, wj_avg)
    end
    return mean(wi_list)
end

"""Entropy lower bound using default proposal."""
function entropy_lower_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        targets::Selection,
        N::Integer,
        M::Integer,
        K::Integer;
        model_traces::Vector{Gen.Trace}=Gen.Trace[],
        return_weights::Bool=false)
    wi_list = []
    for i=1:N
        # Sample observations from model.
        tr_p = isempty(model_traces) ? simulate(model, model_args) : model_traces[i]
        observations = get_selected(get_choices(tr_p), targets)
        wj_list = []
        for j=1:M
            wk_list::Vector{Float64} = []
            for k=1:K
                # Sample latents (particle k) from default proposal
                # and retrieve weight w_k,
                # which is p(observations | latents)
                tr_q, log_w = generate(model, model_args, observations)
                push!(wk_list, log_w)
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
        N::Integer,
        M::Integer,
        K::Integer;
        model_traces::Vector{Gen.Trace}=Gen.Trace[],
        return_weights::Bool=false)
    @assert(M == 1)
    wi_list = []
    for i=1:N
        # Sample latents + observations from model.
        tr_p = isempty(model_traces) ? simulate(model, model_args) : model_traces[i]
        observations = get_selected(get_choices(tr_p), targets)
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
                tr_pk, log_w = generate(model, model_args, observations)
                push!(wk_list, log_w)
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
