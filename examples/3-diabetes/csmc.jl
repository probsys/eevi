import Gen
import Distributions

function log_ml_estimate(
        state::Gen.ParticleFilterState,
        star_log_weight::Float64)
    num_particles =  length(state.traces)
    log_weights = Vector{Float64}(undef, num_particles + 1)
    log_weights[1:num_particles] = state.log_weights
    log_weights[end] = star_log_weight
    return state.log_ml_est + Gen.logsumexp(log_weights) - log(1 + num_particles)
end

function maybe_resample!(
        state::Gen.ParticleFilterState{U},
        star_trace::Gen.Trace,
        star_log_weight::Float64;
        ess_threshold::Real=(1+length(state.traces))/2,
        verbose=false) where {U}

    num_particles = length(state.traces)

    # Extend the log weights.
    log_weights_sample = Vector{Float64}(undef, num_particles + 1)
    log_weights_sample[1:num_particles] = state.log_weights
    log_weights_sample[end] = star_log_weight

    # Check resampling criteria.
    (log_total_weight, log_normalized_weights) = Gen.normalize_weights(log_weights_sample)
    ess = Gen.effective_sample_size(log_normalized_weights)
    do_resample = ess < ess_threshold

    verbose && println("effective sample size: $ess, doing resample: $do_resample")
    if do_resample

        weights = exp.(log_normalized_weights)
        dist = Distributions.Categorical(weights / sum(weights))
        Distributions.rand!(dist, state.parents)
        state.log_ml_est += log_total_weight - log(1 + num_particles)

        for i=1:num_particles
            if state.parents[i] <= num_particles
                state.new_traces[i] = state.traces[state.parents[i]]
            else
                @assert state.parents[i] == num_particles + 1
                state.new_traces[i] = star_trace
            end
            state.log_weights[i] = 0.
        end

        # swap references
        tmp = state.traces
        state.traces = state.new_traces
        state.new_traces = tmp
    end
    return do_resample
end
