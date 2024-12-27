using Gen: @gen, normal, simulate, generate, get_choices, choicemap
using Gen: has_value, get_score

include("sir.jl")

@gen function model(a)
    x ~ normal(a, 1.0)
    y ~ normal(x, 1.0)
end

@gen function proposal(a, observations::ChoiceMap)
    y = observations[:y]
    x ~ normal((a + y) / 2.0, 3.0)
end


a = 0.
y = -2
observations = choicemap((:y, y))
model_args = (a,)
num_particles = 100

sir = SIRGF(model, proposal, model_args, num_particles)

proposal_args = (a, observations)

# smoke test for simulate
sir_trace = simulate(sir, proposal_args)
sir_choices = get_choices(sir_trace)
display(sir_choices)
println(get_score(sir_trace))
@assert has_value(sir_choices, :x)
@assert !has_value(sir_choices, :y)

# smoke test for generate
choices = choicemap((:x, -1.5))
(sir_trace, sir_score) = generate(sir, proposal_args, choices)
sir_choices = get_choices(sir_trace)
display(sir_choices)
println(get_score(sir_trace))
@assert has_value(sir_choices, :x)
@assert !has_value(sir_choices, :y)
@assert sir_choices[:x] == -1.5

# TODO: next steps
#
# 1. work out what the posterior density of x given y is using Gaussian math
#
# 2. as num_particles increases, the score returned by calling get_score(trace)
# where trace is obtained by calling simulate on the SIRGF, should approach the
# log posterior density on x given y for the sampled value of x; verify this is
# indeed the case
#
# 3. do the same for the log_weight that is returned by calling generate on the
# SIRGF, which should approach the log posterior density for the given value of
# x that is passed in the constraints to generate; verify that this is indeed the case
