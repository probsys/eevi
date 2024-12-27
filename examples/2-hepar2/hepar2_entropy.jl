include("hepar2.jl")
include("../../entropy.jl")

import Gen
import Random

using DelimitedFiles: writedlm

Random.seed!(1)

OUTDIR = joinpath("results")

# ==============================================================================
# RUNTIME EXPERIMENT

# Crash test model.
tr = Gen.simulate(model, ())
model_args = ()

# Define nodes.
diseases = [:PBC, :Steatosis, :Cirrhosis, :fibrosis, :ChHepatitis,]
properties = [:surgery, :sex, :age, :joints, :hepatomegaly, :bilirubin, :proteins, :platelet, :inr, :encephalopathy]
leaves_order = [:upper_pain, :fat, :flatulence, :amylase, :anorexia, :nausea, :ama, :le_cells, :pain, :triglycerides, :pain_ruq, :fatigue, :pressure_ruq, :ESR, :ggtp, :cholesterol, :hbc_anti, :hcv_anti, :hbeag, :hepatalgia, :hbsag_anti, :phosphatase, :edema, :alcohol, :alt, :ast, :spleen, :spiders, :albumin, :edge, :irregular_liver, :palms, :carcinoma, :itching, :skin, :jaundice, :ascites, :bleeding, :urea, :density, :consciousness]

# ==============================================================================
# RUNTIME EXPERIMENT

function experiment_entropy_runtime(observations; N=5000, M=1)
    stats = []
    for K in [1, 2, 3, 4, 5, 6, 7, 8, 16]
        runtime = @elapsed begin
            H_lo = entropy_lower_bound(model, model_args, Gen.select(observations...), N, M, K)
            H_hi = entropy_upper_bound(model, model_args, Gen.select(observations...), N, 1, K)
        end
        push!(stats, (runtime, H_lo, H_hi))
        println(stats[end])
    end
    return stats
end

function run_experiment_entropy_runtime()
    println("Runtime (Precompilation)")
    experiment_entropy_runtime(diseases)
    for n in [10, 20, 40]
        println("Runtime Experiment $(n)")
        stats = experiment_entropy_runtime(leaves_order[1:n])
        fname = joinpath(OUTDIR, "hepar2.cmi.$(n).runtime")
        open(fname, "w") do io
            for stat in stats
                data = join(stat, ",")
                write(io, data)
                write(io, "\n")
            end
        end
        println(fname)
    end
end

# ==============================================================================
# CONDITIONAL ENTROPY RANKING EXPERIMENT

function experiment_conditional_entropy_ranking(diseases, observations, tests; N=100, M=1, K=100)
    d = diseases
    O = observations
    stats = []
    model_traces::Vector{Gen.Trace} = [simulate(model, model_args) for i=1:N]
    Threads.@threads for t in tests
        println("Running $(d) $(t)")
        sel_tdO = Gen.select(vcat([t], d, O)...)
        sel_td = Gen.select(vcat([t], O)...)
        # H(X,Y,Z)
        H_tdO_lower = entropy_lower_bound(model, model_args, sel_tdO, N, M, K; model_traces=model_traces)
        H_tdO_upper = entropy_upper_bound(model, model_args, sel_tdO, N, 1, K; model_traces=model_traces)
        H_tdO_approx = mean([H_tdO_lower, H_tdO_upper])
        # H (Y,Z)
        H_td_lower = entropy_lower_bound(model, model_args, sel_td, N, M, K; model_traces=model_traces)
        H_td_upper = entropy_upper_bound(model, model_args, sel_td, N, 1, K; model_traces=model_traces)
        H_td_approx = mean([H_td_lower, H_td_upper])
        # MI(X:Y|Z) without constant terms
        CH_lo = H_tdO_lower - H_td_upper
        CH_hi = H_tdO_upper - H_td_lower
        CH_approx = H_tdO_approx - H_td_approx
        stat = (
            diseases, t,
            CH_approx, CH_lo, CH_hi,
            H_tdO_approx, H_tdO_lower, H_tdO_upper,
            H_td_approx, H_td_lower, H_td_upper,
        )
        push!(stats, stat)
    end
    return stats
end

# CONDITIONAL ENTROPY EXPERIMENT
function run_experiment_conditional_entropy_ranking()
    observations = vcat(leaves_order[end-5:end], properties)
    tests = leaves_order[1:end-5]
    for d in diseases
        stats = experiment_conditional_entropy_ranking([d], observations, tests)
        stats = sort(stats, by=t->t[3], rev=true)
        for stat in stats
            println(stat)
        end
        fname = joinpath(OUTDIR, "hepar2.$(d).rankings")
        writedlm(fname, [stat[2:end] for stat in stats], ",")
        println(fname)
    end
end

# ==============================================================================
# ENTROPY VARIANCE EXPERIMENT

function experiment_entropy_variance(iid, diseases, observations, test, Nrep;
        N=1000, M=1, K=100)
    # The sign here is the opposite of experiment_conditional_entropy, i.e.,
    # the correct signs are used.
    d = diseases
    O = observations
    entropies = []
    weights = []
    Threads.@threads for i=1:Nrep
        println("Running variance $(d) $(test) $(i)")
        sel_dtO = Gen.select(vcat(d, test, O)...)
        sel_tO = Gen.select(vcat(test, O)...)
        model_traces::Vector{Gen.Trace} = !iid ? [simulate(model, model_args) for i=1:N] : []
        # Lower bound.
        H_tO_lower_weights = entropy_lower_bound(model, model_args, sel_tO, N, M, K;
            model_traces=model_traces, return_weights=true)
        H_dtO_upper_weights = entropy_upper_bound(model, model_args, sel_dtO, N, 1, K;
            model_traces=model_traces, return_weights=true)
        H_dtO_lower = -mean(H_dtO_upper_weights)
        H_tO_upper = -mean(H_tO_lower_weights)
        CH_lo = H_dtO_lower - H_tO_upper
        # Upper bound.
        H_tO_upper_weights = entropy_upper_bound(model, model_args, sel_tO, N, 1, K;
            model_traces=model_traces, return_weights=true)
        H_dtO_lower_weights = entropy_lower_bound(model, model_args, sel_dtO, N, M, K;
            model_traces=model_traces, return_weights=true)
        H_dtO_upper = -mean(H_dtO_lower_weights)
        H_tO_lower = -mean(H_tO_upper_weights)
        CH_hi = H_dtO_upper - H_tO_lower
        push!(entropies, (H_dtO_lower, H_dtO_upper, H_tO_lower, H_tO_upper))
        push!(weights, (
            H_dtO_lower_weights,
            H_dtO_upper_weights,
            H_tO_lower_weights,
            H_tO_upper_weights
        ))
    end
    return entropies, weights
end

function run_experiment_entropy_variance()
    disease, test = :PBC, :ama
    observations = vcat(leaves_order[end-5:end], properties)
    N = 10000
    K = 200
    for iid in [false, true]
        entropies, weights =
            experiment_entropy_variance(iid, [disease], observations, test, 18;
                N=N, K=K)
        # Write the Nrep entropies.
        fname = joinpath(OUTDIR, "hepar2.entropies.$(disease).$(test).$(N).$(K).$(iid)")
        rows = [vcat(row...) for row in entropies]
        lines = Matrix(hcat(rows...)')
        writedlm(fname, lines, ",")
        println(fname)
        # Write the N weights from Nrep=1.
        fname = joinpath(OUTDIR, "hepar2.weights.$(disease).$(test).$(N).$(K).$(iid)")
        lines = hcat(weights[1]...)
        writedlm(fname, lines, ",")
        println(fname)
    end
end

run_experiment_entropy_runtime()
run_experiment_conditional_entropy_ranking()
run_experiment_entropy_variance()
