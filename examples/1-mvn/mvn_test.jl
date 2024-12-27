include("mvn.jl")

import Distributions
import Gen
import Random

using Gen: @gen, @trace

Random.seed!(1)

mu = [0., 0.]
cov = [[1., .5] [.5, 1.]]
dist = Distributions.MvNormal(mu, cov)

@gen function model()
    xs = @trace(mvnorm_lw(mu, cov), :ys)
    return xs
end

trace = Gen.simulate(mvnorm_lw, (mu, cov))
trace, score = Gen.generate(mvnorm_lw, (mu, cov))
trace, score = Gen.generate(mvnorm_lw, (mu, cov), Gen.choicemap((1, 0.)))

weights = Vector{Float64}()
for i=1:1000
    tr, sc = Gen.generate(mvnorm_lw, (mu, cov), choicemap((1, 0)))
    push!(weights, sc)
    sc2 = Gen.project(tr, Gen.select(1))
    @assert isapprox(sc, sc2)
end
logp_est = Gen.logsumexp(weights) - Gen.log(length(weights))
dist_exact = Distributions.MvNormal(mu[1:1], cov[1:1, 1:1])
@assert abs(logp_est - Distributions.logpdf(dist_exact, [0.])) < 0.1

ys = model()
@assert length(ys) == length(mu)

trace = Gen.simulate(mvnorm_lw, (mu, cov))
@assert isapprox(
    Distributions.logpdf(dist, Gen.get_retval(trace)),
    Gen.get_score(trace))
@assert isapprox(0, Gen.project(trace, Gen.select()))
@assert isapprox(Gen.get_score(trace), Gen.project(trace, Gen.select(1, 2)))

tr, w = Gen.generate(model, (), choicemap((:ys => 1, 0)))
@assert Gen.get_choices(tr)[:ys=>1] == 0
@assert Gen.get_choices(tr)[:ys=>2] != 0

tr, w = Gen.generate(model, (), choicemap((:ys => 2, 0)))
@assert Gen.get_choices(tr)[:ys=>1] != 0
@assert Gen.get_choices(tr)[:ys=>2] == 0

tr, w = Gen.generate(model, (), choicemap((:ys => 1, 0), (:ys => 2, 0)))
@assert Gen.get_choices(tr)[:ys=>1] == 0
@assert Gen.get_choices(tr)[:ys=>2] == 0
