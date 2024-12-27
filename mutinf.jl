include("entropy.jl")

using Gen:
    GenerativeFunction,
    Selection

struct OrSelection <: Selection
    selections::Vector{Selection}
end
get_address_schema(::Type{OrSelection}) = @assert false # ???
Base.isempty(selection::OrSelection) = all(isempty(s) for s in selection.selections)
Base.in(addr, selection::OrSelection) = any(addr in s for s in selection.selections)
function Base.getindex(selection::OrSelection, addr)
    for s in selection.selections
        sel = s[addr]
        if !isempty(sel)
            return sel
        end
    end
    return EmptySelection()
end

"""Mutual information lower bound using default proposal."""
function mutinf_lower_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        X::Selection,
        Y::Selection,
        Z::Selection,
        N::Integer,
        M::Integer,
        K::Integer)
    H_XYZ_lower = entropy_lower_bound(model, model_args, OrSelection([X, Y, Z]), N, M, K)
    H_XZ_upper = entropy_upper_bound(model, model_args, OrSelection([X, Z]), N, 1, K)
    H_YZ_upper = entropy_upper_bound(model, model_args, OrSelection([Y, Z]), N, 1, K)
    H_Z_lower = isempty(Z) ? 0 : entropy_lower_bound(model, model_args, Z, N, M, K)
    return H_XYZ_lower - H_XZ_upper - H_YZ_upper + H_Z_lower
end

"""Mutual information upper bound using default proposal."""
function mutinf_upper_bound(
        model::GenerativeFunction,
        model_args::Tuple,
        X::Selection,
        Y::Selection,
        Z::Selection,
        N::Integer,
        M::Integer,
        K::Integer)
    H_XYZ_upper = entropy_upper_bound(model, model_args, OrSelection([X, Y, Z]), N, 1, K)
    H_XZ_lower = entropy_lower_bound(model, model_args, OrSelection([X, Z]), N, M, K)
    H_YZ_lower = entropy_lower_bound(model, model_args, OrSelection([Y, Z]), N, M, K)
    H_Z_upper = isempty(Z) ? 0 : entropy_upper_bound(model, model_args, Z, N, 1, K)
    return H_XYZ_upper - H_XZ_lower - H_YZ_lower + H_Z_upper
end
