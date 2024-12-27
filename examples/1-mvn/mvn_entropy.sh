#!/bin/sh

set -Ceux

experiments="likelihood_weighting independent_proposal"

outdir=${1}
cov_file=${2}

test -d ${outdir} || (echo no such ${outdir} && exit 1)

K_max=4
K_list=$(python -c "print(' '.join([str(2**i) for i in range(${K_max})]))")
N=100
M=1
N_rep=10

for experiment in ${experiments}; do
    echo ${K_list} | xargs -I% -d' ' -n1 -P4 \
        julia mvn_entropy.jl \
        --outdir=${outdir} \
        ${cov_file} \
        ${experiment} \
        ${N_rep} ${N} ${M} %
done
