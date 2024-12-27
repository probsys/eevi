using Combinatorics: combinations

import Gen
import Random

include("diabetes_ts.jl")
include("diabetes_entropy_estimators.jl")
include("csmc.jl")

Random.seed!(1)

# ==============================================================================
# PREAMBLE
# ==============================================================================

N_model = 25
model_args = (N_model,)
trace = Gen.simulate(model, model_args)

lookup_meal = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
lookup_ins_abs = [100_0, 70_7, 50_0, 35_5, 25_0, 17_7, 12_5, 8_9, 6_4, 3_2, 1_6]

evidence_scenarios_meal = Dict(
    0 => [0, 30, 0, 30, 0, 0, 50, 0, 20, 0, 0, 0, 50, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    1 => [0, 0, 0, 100, 0, 0, 0, 0, 20, 0, 50, 0, 50, 0, 30, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0],
)

evidence_scenarios_ins_abs = Dict(
    0 => [1_6, 1_6, 3_2, 3_2, 6_4, 6_4, 6_4, 6_4, 6_4, 3_2, 3_2, 3_2, 3_2, 3_2, 3_2, 6_4, 6_4, 6_4, 6_4, 3_2, 3_2, 3_2, 3_2, 3_2],
    1 => [6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 3_2, 3_2, 3_2, 3_2, 3_2, 3_2, 6_4, 6_4, 6_4, 6_4, 3_2, 3_2, 3_2, 3_2, 3_2],
    2 => [70_7, 50_0, 50_0, 50_0, 35_5, 35_5, 35_5, 6_4, 6_4, 3_2, 25_0, 25_0, 25_0, 25_0, 25_0, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4, 6_4],
)

function make_evidence(addr, values, lookup)
    return [((addr => i), findall(x -> x==v, lookup)[1]) for (i, v) in enumerate(values)]
end

# Selected evidence scenario.
idx_meal = 0
idx_ins_abs = 0

scenario_meal = evidence_scenarios_meal[idx_meal]
scenario_ins_abs = evidence_scenarios_ins_abs[idx_ins_abs]

# Build evidence.
k_meal = min(N_model, length(scenario_meal))
k_ins_abs = min(N_model, length(scenario_ins_abs))
evidence_meal_all = make_evidence(:meal, scenario_meal[1:k_meal], lookup_meal)
evidence_ins_abs_all = make_evidence(:ins_abs, scenario_ins_abs[1:k_ins_abs], lookup_ins_abs)
evidence_all = Gen.choicemap(vcat(evidence_meal_all, evidence_ins_abs_all)...)

# Weight offsets from intervention variables for times 1:N_model
log_weight_intervention = generate(model, model_args, evidence_all)[2]
@assert log_weight_intervention == generate(model, model_args, evidence_all)[2]

log_weights_intervention = Vector{Float64}(undef, N_model)
for i=1:N_model
    x = generate(model, (i,), evidence_all)[2]
    @assert x == generate(model, (i,), evidence_all)[2]
    log_weights_intervention[i] = x
end
@assert log_weights_intervention[end] == log_weight_intervention

# Query traces.
N_samples_Y = 1000
model_traces = Gen.Trace[generate(model, model_args, evidence_all)[1] for i=1:N_samples_Y]

# Quick assertion on offsets.
for j=1:N_model
    j_meal = min(j, length(evidence_meal_all))
    j_ins_abs = min(j, length(evidence_ins_abs_all))
    addr_meal_j = [t[1] for t in evidence_meal_all[1:j_meal]]
    addr_ins_abs_j = [t[1] for t in evidence_ins_abs_all[1:j_ins_abs]]
    selection = Gen.select(addr_meal_j..., addr_ins_abs_j...)
    for i=1:5
        x = Gen.project(model_traces[i], selection)
        @assert isapprox(x, log_weights_intervention[j])
    end
end

# ==============================================================================
# (CONDITIONAL) SEQUENTIAL MONTE CARLO
# ==============================================================================

function make_evidence_cell(addr, i, value, lookup)
    value_i = findall(x -> x==value, lookup)[1]
    return (addr => i, value_i)
end

# Initial constraints.
function get_constraints_initial_smc(
        observation::Gen.ChoiceMap,
        ins_sens::Bool,
        bg_addrs::Vector{Pair{Symbol,Int64}})
    # Prepare the constraints.
    constraints = []
    if ins_sens
        c_ins_sens = (:ins_sens, observation[:ins_sens])
        push!(constraints, c_ins_sens)
    end
    if bg_addrs[1][end] == 1
        c_bg = (bg_addrs[1], observation[bg_addrs[1]])
        push!(constraints, c_bg)
    end
    c_meal_value = scenario_meal[1]
    c_meal = make_evidence_cell(:meal, 1, c_meal_value, lookup_meal)
    push!(constraints, c_meal)
    c_ins_abs_value = scenario_ins_abs[1]
    c_ins_abs = make_evidence_cell(:ins_abs, 1, c_ins_abs_value, lookup_ins_abs)
    push!(constraints, c_ins_abs)
    # Return constraints as Gen.ChoiceMap
    return Gen.choicemap(constraints...)
end

function get_constraints_t_smc(
        t::Integer,
        observation::Gen.ChoiceMap,
        bg_addrs::Vector{Pair{Symbol,Int64}})
    @assert 1 < t
    # Obtain meal and insulin interventions at t.
    constraints = []
    if t == bg_addrs[1][end]
        c_bg_1 = (bg_addrs[1], observation[bg_addrs[1]])
        push!(constraints, c_bg_1)
    end
    if t == bg_addrs[2][end]
        c_bg_2 = (bg_addrs[2], observation[bg_addrs[2]])
        push!(constraints, c_bg_2)
    end
    if t <= length(scenario_meal)
        c_meal_value = scenario_meal[t]
        c_meal = make_evidence_cell(:meal, t, c_meal_value, lookup_meal)
        push!(constraints, c_meal)
    end
    if t <= length(scenario_ins_abs)
        c_ins_abs_value = scenario_ins_abs[t]
        c_ins_abs = make_evidence_cell(:ins_abs, t, c_ins_abs_value, lookup_ins_abs)
        push!(constraints, c_ins_abs)
    end
    # Return constraints as Gen.ChoiceMap
    return Gen.choicemap(constraints...)
end

function run_smc(
        N_model::Integer,
        observation::Gen.ChoiceMap,
        ins_sens::Bool,
        bg_addrs::Vector{Pair{Symbol,Int64}},
        N_particles::Integer,
        seed::Integer;
        star_traces::Vector{Gen.Trace}=Gen.Trace[],
        star_log_weights::Vector{Float64}=Float64[])
    # Which times is BG being queried?
    t1 = bg_addrs[1][end]
    t2 = bg_addrs[2][end]
    @assert 1 <= t1 < t2 <= N_model
    # Verify star traces.
    @assert length(star_traces) == length(star_log_weights)
    @assert length(star_traces) in [0, t2, N_model]
    csmc = length(star_traces) > 1
    # Fix the seed.
    Random.seed!(seed)
    # Initialize the PF.
    evidence_init = get_constraints_initial_smc(observation, ins_sens, bg_addrs)
    state = Gen.initialize_particle_filter(model, (1,), evidence_init, N_particles)
    # Run through 2:t2
    # No need to run up to N_model
    for t=2:t2
        new_args = (t,)
        if !csmc
            Gen.maybe_resample!(state)
        else
            maybe_resample!(
                state,
                star_traces[t-1],
                star_log_weights[t-1])
        end
        evidence_t = get_constraints_t_smc(t, observation, bg_addrs)
        # Run the PF step.
        argdiffs = (Gen.UnknownChange(),)
        log_inc_w, = Gen.particle_filter_step!(state, new_args, argdiffs, evidence_t)
        @assert length(log_inc_w) == N_particles
    end
    # Compute ML estimate.
    if !csmc
        log_ml_est = Gen.log_ml_estimate(state)
    else
        log_ml_est = log_ml_estimate(state, star_log_weights[t2])
    end
    # Offset the ML estimate by the intervention weight.
    log_ml_est_offset = log_ml_est - log_weights_intervention[t2]
    return (state, log_ml_est_offset)
end

# ==============================================================================
# ENTROPY LOWER BOUND
# ==============================================================================

function compute_entropies_lower_bound(
            N_model::Integer,
            ins_sens::Bool,
            bg_addrs::Vector{Pair{Symbol,Int64}},
            N_particles::Integer,
            N_samples_Y::Integer,
            M_samples_Y::Integer)
    wi_list::Vector{Float64} = Vector{Float64}(undef, N_samples_Y)
    Threads.@threads for i=1:N_samples_Y
        # Extract the observation.
        sel_xy = Gen.select(:ins_sens, bg_addrs...)
        observation = Gen.get_selected(get_choices(model_traces[i]), sel_xy)
        wj_list::Vector{Float64} = []
        for j=1:M_samples_Y
            seed = max(i, j)^2 + max(j, 2*j-i)
            state, log_ml_est = run_smc(
                        N_model,
                        observation,
                        ins_sens,
                        bg_addrs,
                        N_particles,
                        seed)
            push!(wj_list, log_ml_est)
        end
        wj_avg = mean(wj_list)
        wi_list[i] = wj_avg
    end
    return -mean(wi_list)
end

# ==============================================================================
# ENTROPY UPPER BOUND
# ==============================================================================

function get_star_trajectory(
            N_model::Integer,
            ins_sens::Bool,
            bg_addrs::Vector{Pair{Symbol,Int64}},
            star_trace::Gen.Trace
        )
    # Which times is BG being queried?
    t1 = bg_addrs[1][end]
    t2 = bg_addrs[2][end]
    @assert 1 <= t1 < t2 <= N_model
    # Extract star trajectories for star trace.
    star_choices = Gen.get_choices(star_trace)
    star_traces::Vector{Gen.Trace} = []
    star_log_weights::Vector{Float64} = []
    # Compute selections in star trace from 1:t2
    for j=1:t2
        star_selections::Vector{Any} = []
        # Global insulin sensitivity.
        if ins_sens
            push!(star_selections, :ins_sens)
        end
        # Meal interventions from 1:j
        j_meal = min(j, length(evidence_meal_all))
        addr_meal_j = [t[1] for t in evidence_meal_all[1:j_meal]]
        append!(star_selections, addr_meal_j)
        # Insulin interventions from 1:j
        j_ins_abs = min(j, length(evidence_ins_abs_all))
        addr_ins_abs_j = [t[1] for t in evidence_ins_abs_all[1:j_ins_abs]]
        append!(star_selections, addr_ins_abs_j)
        # Query variables.
        if bg_addrs[1][end] <= j
            push!(star_selections, bg_addrs[1])
        end
        if bg_addrs[2][end] <= j
            push!(star_selections, bg_addrs[2])
        end
        # Compute star trace observation weight from 1:j.
        log_weight_j = Gen.project(star_trace, Gen.select(star_selections...))
        push!(star_log_weights, log_weight_j)
        # Create a prefix star trace from 1:j
        trace_j, = Gen.generate(model, (j,), star_choices)
        push!(star_traces, trace_j)
        # Confirm weights from project and generate agree.
        log_weight_check = Gen.project(trace_j, Gen.select(star_selections...))
        @assert isapprox(log_weight_j, log_weight_check)
    end
    # Return the star trajectory and log weights.
    return (star_traces, star_log_weights)
end

function compute_entropies_upper_bound(
            N_model::Integer,
            ins_sens::Bool,
            bg_addrs::Vector{Pair{Symbol,Int64}},
            N_particles::Integer,
            N_samples_Y::Integer,
            M_samples_Y::Integer)
    wi_list::Vector{Float64} = Vector{Float64}(undef, N_samples_Y)
    Threads.@threads for i=1:N_samples_Y
        # Extract the observation and star trajectory.
        star_trace = model_traces[i]
        sel_xy = Gen.select(:ins_sens, bg_addrs...)
        observation = Gen.get_selected(get_choices(star_trace), sel_xy)
        star_traces, star_log_weights =
            get_star_trajectory(N_model, ins_sens, bg_addrs, star_trace)
        # Run CSMC.
        wj_list::Vector{Float64} = []
        for j=1:M_samples_Y
            seed = max(i, j)^2 + max(j, 2*j-i)
            state, log_ml_est =
                run_smc(
                    N_model,
                    observation,
                    ins_sens,
                    bg_addrs,
                    N_particles,
                    seed;
                    star_traces=star_traces,
                    star_log_weights=star_log_weights)
            push!(wj_list, log_ml_est)
        end
        wj_avg = mean(wj_list)
        wi_list[i] = wj_avg
    end
    return -mean(wi_list)
end

# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

# Query address pairs.
bg_all_pairs = [collect(:bg => i for i in x) for x in combinations(1:N_model, 2)]

bg_addrs = bg_all_pairs[end]
ins_sens = true
N_particles = 100
N_samples_Y = 100
M_samples_Y = 1

# The naming is reversed, as the lower bound is computed
# on log[p(Y)] not -log[p(Y)], so entropy_lower_bound
# actually gives the upper bound, and entropy_upper_bound
# gives the lower bound.
for bg_addrs in bg_all_pairs
    H_hi = compute_entropies_lower_bound(
                N_model,
                ins_sens,
                bg_addrs,
                N_particles,
                N_samples_Y,
                M_samples_Y)
    H_lo = compute_entropies_upper_bound(
                N_model,
                ins_sens,
                bg_addrs,
                N_particles,
                N_samples_Y,
                M_samples_Y)
    line = "bg_$(bg_addrs[1][end]),bg_$(bg_addrs[2][end]),$(H_lo),$(H_hi)"
    println(line)
end
