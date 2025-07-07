"""
    all_violations!(bm::BatchModel, X::AbstractMatrix)

Compute all constraint and variable violations for a batch of solutions.
"""
function all_violations!(bm::BatchModel, X::AbstractMatrix)
    V = cons_nln_batch!(bm, X)

    Vc = constraint_violations!(bm, V)
    Vb = bound_violations!(bm, X)

    return Vc, Vb
end

"""
    all_violations!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Compute all constraint and variable violations for a batch of solutions and parameters.
"""
function all_violations!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    V = cons_nln_batch!(bm, X, Θ)

    Vc = constraint_violations!(bm, V)
    Vb = bound_violations!(bm, X)

    return Vc, Vb
end

"""
    constraint_violations!(bm::BatchModel, V::AbstractMatrix)

Compute constraint violations for a batch of constraint primal values.
"""
function constraint_violations!(bm::BatchModel, V::AbstractMatrix)
    viols_cons_out = _maybe_view(bm, :viols_cons_out, V)
    
    violation!.(eachcol(viols_cons_out), eachcol(V), bm.viols_cons)

    return viols_cons_out
end

"""
    bound_violations!(bm::BatchModel, V::AbstractMatrix)

Compute variable violations for a batch of variable primal values.
"""
function bound_violations!(bm::BatchModel, V::AbstractMatrix)
    viols_vars_out = _maybe_view(bm, :viols_vars_out, V)
    
    violation!.(eachcol(viols_vars_out), eachcol(V), bm.viols_vars)

    return viols_vars_out
end

"""
    violation!(d, v, s::S) where {S}

Store the distance between the point `v` and the set `s` in `d`.
"""
@inline violation!(d, v, s::S) where {S} = begin
    d .= violation(v, s)
end

"""
    violation(v, s::S) where {S}

Compute the distance between the point `v` and the set `s`.
"""
function violation end
