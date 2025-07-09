"""
    all_violations!(bm::BatchModel, X::AbstractMatrix)

Compute all constraint and variable violations for a batch of solutions.
"""
function all_violations!(bm::BatchModel, X::AbstractMatrix)
    Vc = constraint_violations!(bm, X)
    Vb = bound_violations!(bm, X)

    return Vc, Vb
end

"""
    all_violations!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Compute all constraint and variable violations for a batch of solutions and parameters.
"""
function all_violations!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    Vc = constraint_violations!(bm, X, Θ)
    Vb = bound_violations!(bm, X)

    return Vc, Vb
end

"""
    constraint_violations!(bm::BatchModel, X::AbstractMatrix)

Compute constraint violations for a batch of constraint primal values.
"""
function constraint_violations!(bm::BatchModel, X::AbstractMatrix)
    V = constraints!(bm, X)
    return _constraint_violations!(bm, V)
end
"""
    constraint_violations!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Compute constraint violations for a batch of constraint primal values.
"""
function constraint_violations!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    V = constraints!(bm, X, Θ)
    return _constraint_violations!(bm, V)
end

function _constraint_violations!(bm::BatchModel, V::AbstractMatrix)
    viols_cons_out = _maybe_view(bm, :viols_cons_out, V)
    
    _violation!.(eachcol(viols_cons_out), eachcol(V), bm.viols_cons)

    return viols_cons_out
end

"""
    bound_violations!(bm::BatchModel, X::AbstractMatrix)

Compute variable violations for a batch of variable primal values.
"""
function bound_violations!(bm::BatchModel, X::AbstractMatrix)
    viols_vars_out = _maybe_view(bm, :viols_vars_out, X)
    
    _violation!.(eachcol(viols_vars_out), eachcol(X), bm.viols_vars)

    return viols_vars_out
end

@inline _violation!(d, v, s::S) where {S} = begin
    d .= _violation(v, s)
end
