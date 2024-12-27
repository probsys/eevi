include("diabetes_ts.jl")
include("diabetes_entropy_estimators.jl")

import Gen
import Random

using Combinatorics: combinations

Random.seed!(1)

n = 24
model_args = (n,)
trace = Gen.simulate(model, model_args)

meal_lookup = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
meal_values = [0, 30, 0, 30, 0, 0, 50, 0, 20, 0, 0, 0, 50, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
meal_evidence = [((:meal => i), findall(x -> x==v, meal_lookup)[1]) for (i, v) in enumerate(meal_values)]

ins_abs_lookup = [100_0, 70_7, 50_0, 35_5, 25_0, 17_7, 12_5, 8_9, 6_4, 3_2, 1_6]
ins_abs_values = [1_6, 1_6, 3_2, 3_2, 6_4, 6_4, 6_4, 6_4, 6_4, 3_2, 3_2, 3_2, 3_2, 3_2, 3_2, 6_4, 6_4, 6_4, 6_4, 3_2, 3_2, 3_2, 3_2, 3_2]
ins_abs_evidence = [((:ins_abs => i), findall(x -> x==v, ins_abs_lookup)[1]) for (i, v) in enumerate(ins_abs_values)]

evidence = Gen.choicemap(vcat(meal_evidence, ins_abs_evidence)...)

N = 1000
K = 100

# Model traces must be generated with background evidence.
model_traces = Gen.Trace[generate(model, model_args, evidence)[1] for i=1:N]
log_w_baseline = generate(model, model_args, evidence)[2]

x = [:ins_sens]

indexes = collect(combinations(1:n, 2))
selections = [collect(:bg => i for i in x) for x in indexes]
y = selections[1]

Threads.@threads for y in selections[1:5]
    println("Analyzing $(y)")
    sel_xy = Gen.select(vcat(x, y)...)
    sel_y = Gen.select(y...)
    # H(X,Y) bounds
    H_xy_lo = entropy_lower_bound(model, model_args, sel_xy, evidence, N, 1, K, model_traces)
    H_xy_hi = entropy_upper_bound(model, model_args, sel_xy, evidence, N, 1, K, model_traces)
    H_xy_approx = mean([H_xy_lo, H_xy_hi])
    # H(Y) bounds
    H_y_lo = entropy_lower_bound(model, model_args, sel_y, evidence, N, 1, K, model_traces)
    H_y_hi = entropy_upper_bound(model, model_args, sel_y, evidence, N, 1, K, model_traces)
    H_y_approx = mean([H_y_lo, H_y_hi])
    # H(X|Y) bounds
    CH_lo = H_y_lo - H_xy_hi
    CH_hi = H_y_hi - H_xy_lo
    # Report
    println((y, CH_lo, CH_hi))
end
