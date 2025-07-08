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
    
    _violation!.(eachcol(viols_cons_out), eachcol(V), bm.viols_cons)

    return viols_cons_out
end

"""
    bound_violations!(bm::BatchModel, V::AbstractMatrix)

Compute variable violations for a batch of variable primal values.
"""
function bound_violations!(bm::BatchModel, V::AbstractMatrix)
    viols_vars_out = _maybe_view(bm, :viols_vars_out, V)
    
    _violation!.(eachcol(viols_vars_out), eachcol(V), bm.viols_vars)

    return viols_vars_out
end

@inline _violation!(d, v, s::S) where {S} = begin
    d .= _violation(v, s)
end


"""
    Interval{VT}

Represents the RHS of M constraints g(xᵢ) ∈ [lᵢ, uᵢ]  ∀i ∈ 1:M.
"""
struct Interval{VT}
    l::VT
    u::VT
end
@inline _violation(v, s::Interval{VT}) where {VT} = begin
    @. max(s.l - v, v - s.u, zero(v))
end

Base.broadcastable(s::Interval) = Ref(s)
Base.isempty(s::Interval{VT}) where {VT} = isempty(s.l) || isempty(s.u)

# empty support (unconstrained)
Interval(::Nothing) = Interval()
Interval() = Interval(nothing, nothing)
Base.isempty(::Interval{Nothing}) = true
_violation(v, ::Interval{Nothing}) = zero(v)
