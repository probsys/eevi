include("mvn.jl")
include("../../entropy.jl")
include("../../sir.jl")

import LinearAlgebra
import Random
import ArgParse
import DelimitedFiles

using Gen: @gen, @trace

function entropy_theoretical(cov::AbstractMatrix{U}, ix::Vector{Int}) where {U<:Real}
    @assert LinearAlgebra.isposdef(cov)
    @assert 0 < length(ix)
    n = length(ix)
    subcov = view(cov, ix, ix)
    det = LinearAlgebra.det(subcov)
    return (n/2.) * (1 + log(2*pi)) + .5 * log(det)
end

function neg_entropy_theoretical(cov::AbstractMatrix{U}, ix::Vector{Int}) where {U <: Real}
    return -entropy_theoretical(cov, ix)
end

@gen function mvn_proposal_indep(mu::AbstractVector, cov::AbstractMatrix,
        baseaddr, constraints::ChoiceMap)
    ndim = length(mu)
    idx_obs_set = Gen.get_values_shallow(constraints)
    idx_obs = [i for i=1:(ndim) if i in keys(idx_obs_set)]
    idx_new = [i for i=1:(ndim) if !(i in keys(idx_obs_set))]
    mu_sim, cov_sim = mvn_condition(mu, cov, constraints)
    make_addr = (i) -> isnothing(baseaddr) ? i : baseaddr => i
    retval = Vector{Float64}(undef, length(idx_new))
    for (i, j) in enumerate(idx_new)
        mu_i = mu_sim[i]
        std_i = sqrt(cov_sim[i, i])
        retval[i] = @trace(Gen.normal(mu_i, std_i), make_addr(j))
    end
    return retval
end

# Small case.
function test_crash()
    mu = [0., 0., 0., 0.]
    cov = hcat(
        [3.,  2.,  1.,  1.],
        [2.,  3.,  1.,  2.],
        [1.,  1.,  3.,  1.],
        [1.,  2.,  1.,  3.])

    N = 1000
    M = 1
    K = 10
    targets = [3, 4]
    H_ex = neg_entropy_theoretical(cov, targets)
    H_lo = entropy_lower_bound(mvnorm_lw, (mu, cov), Gen.select(targets...), N, M, K)
    H_hi = entropy_upper_bound(mvnorm_lw, (mu, cov), Gen.select(targets...), N, 1, K)
end
test_crash()

function make_random_cov(ndim::Int, seed::Int64)
    @assert ndim % 2 == 0
    cov = rand(Distributions.Wishart(ndim, Matrix(LinearAlgebra.I(ndim))))
    return cov
end

# Exact inference.
function experiment_exact_inference(
        mu::AbstractVector, cov::AbstractMatrix, targets::AbstractVector,
        N::Integer, M::Integer, K::Integer)
    H_lo = entropy_lower_bound(mvnorm_exact, (mu, cov), Gen.select(targets...), N, 1, 1)
    H_hi = entropy_upper_bound(mvnorm_exact, (mu, cov), Gen.select(targets...), N, 1, 1)
    return (H_lo, H_hi)
end

# Independent proposal.
function experiment_independent_proposal(
        mu::AbstractVector, cov::AbstractMatrix, targets::AbstractVector,
        N::Integer, M::Integer, K::Integer)
    proposal = K == 1 ? mvn_proposal_indep :
        SIRGF(mvnorm_exact, mvn_proposal_indep, (mu, cov), K)
    H_lo = entropy_lower_bound(
        mvnorm_exact, (mu, cov),
        proposal, (mu, cov, nothing),
        Gen.select(targets...),
        N, M)
    H_hi = entropy_upper_bound(
        mvnorm_exact, (mu, cov),
        proposal, (mu, cov, nothing),
        Gen.select(targets...),
        N, 1)
    return (H_lo, H_hi)
end

function experiment_likelihood_weighting(
        mu::AbstractVector, cov::AbstractMatrix, targets::AbstractVector,
        N::Integer, M::Integer, K::Integer)
    H_lo = entropy_lower_bound(mvnorm_lw, (mu, cov), Gen.select(targets...), N, M, K)
    H_hi = entropy_upper_bound(mvnorm_lw, (mu, cov), Gen.select(targets...), N, 1, K)
    return (H_lo, H_hi)
end

# experiment_independent_proposal(200, 10, 1)
# experiment_independent_proposal(200, 10, 10)
# experiment_likelihood_weighting(200, 10, 1)
# experiment_likelihood_weighting(200, 10, 10)

settings = ArgParse.ArgParseSettings()
ArgParse.@add_arg_table settings begin
    "--outdir"
        arg_type = String
        help = "output directory"
        required = false
        default = "."
    "cov_path"
        arg_type = String
        help = "path to target covariance matrix"
        required = true
    "experiment"
        help = "name of experiment to run"
        required = true
        arg_type = String
    "Nrep"
        arg_type = Int64
        help = "number of repetitions"
        required = true
        default = 2
    "N"
        help = "number of outer samples"
        required = true
        arg_type = Int64
    "M"
        help = "number of inner samples"
        required = true
        arg_type = Int64
    "K"
        help = "number of SIR samples"
        required = true
        arg_type = Int64
end
args = ArgParse.parse_args(ARGS, settings)

experiments = Dict(
    "exact_inference"       => experiment_exact_inference,
    "independent_proposal"  => experiment_independent_proposal,
    "likelihood_weighting"  => experiment_likelihood_weighting,
)

# Prepare experiment.
cov = DelimitedFiles.readdlm(args["cov_path"], ',')
ndim = size(cov)[1]
mu = zeros(ndim)
targets = Vector{Int}(1:ndim/2)
func = experiments[args["experiment"]]

# Prepare output file.
fkeys = ["experiment", "N", "M", "K"]
fname = join([join((k, args[k]), "@") for k in fkeys], ".")
pathname = joinpath(args["outdir"], fname)
rm(pathname, force=true)

# Run warm-up.
H_ex = neg_entropy_theoretical(cov, targets)
println(H_ex)
func(mu, cov, targets, 1, 1, 1)

for i=1:args["Nrep"]
    seed = i
    Random.seed!(seed)
    # Run timed code.
    runtime = @elapsed begin
        (H_lo, H_hi) = func(mu, cov, targets, args["N"], args["M"], args["K"])
    end
    # Save to disk.
    data = join([seed, runtime, H_lo, H_hi, ], ",")
    open(pathname, "a") do io
        write(io, data)
        write(io, "\n")
    end
    println(seed, " ", H_lo, " ", H_hi)
end
println(pathname)
