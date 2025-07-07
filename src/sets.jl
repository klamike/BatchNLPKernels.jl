abstract type AbstractSet end
abstract type AbstractDistance end
struct DefaultDistance <: AbstractDistance end
struct EpigraphViolationDistance <: AbstractDistance end
struct NormedEpigraphDistance{p} <: AbstractDistance end

struct LessThan{VT} <: AbstractSet
    u::VT
end
struct GreaterThan{VT} <: AbstractSet
    l::VT
end
struct EqualTo{VT} <: AbstractSet
    v::VT
end
struct Interval{VT} <: AbstractSet
    l::VT
    u::VT
end

@inline violation(v, s) = violation(DefaultDistance(), v, s)
@inline violation(::DefaultDistance, v, s::S) where {S} = violation(EpigraphViolationDistance(), v, s)
@inline violation(::NormedEpigraphDistance{p}, s::S) where {p,S} = LinearAlgebra.norm(violation(EpigraphViolationDistance(), v, s), p)

violation!(d, v, s) = begin
    d .= violation(DefaultDistance(), v, s)
end
violation!(::DefaultDistance, d, v, s::S) where {S} = begin
    d .= violation(EpigraphViolationDistance(), v, s)
end
violation!(::NormedEpigraphDistance{p}, d, v, s::S) where {p,S} = begin
    d .= LinearAlgebra.norm(violation(EpigraphViolationDistance(), v, s), p)
end


@inline violation(::EpigraphViolationDistance, s::LessThan) = begin
    @. max(v - s.u, zero(v))
end
@inline violation(::EpigraphViolationDistance, s::GreaterThan) = begin
    @. max(s.l - v, zero(v))
end
@inline violation(::EpigraphViolationDistance, s::EqualTo) = begin
    @. abs(v - s.v)
end
@inline violation(::EpigraphViolationDistance, s::Interval) = begin
    @. max(s.l - v, v - s.u, zero(v))
end

# FIXME is interval slow?
struct BatchViolation{MT,E}
    model::E
    batch_size::Int

    # constraints
    in_cons_out::MT
    in_cons::Interval
    
    # variable bounds
    in_vars_out::MT
    in_vars::Interval
end

function BatchViolation(model::E, batch_size::Int) where {E}
    lcon = model.meta.lcon
    ucon = model.meta.ucon

    in_cons_out = similar(lcon, length(lcon), batch_size)
    in_cons = Interval(lcon, ucon)

    lvar = model.meta.lvar
    uvar = model.meta.uvar

    in_vars_out = similar(lvar, length(lvar), batch_size)
    in_vars = Interval(lvar, uvar)

    return BatchViolation(
        model, batch_size,
        in_cons_out, in_cons,
        in_vars_out, in_vars
    )
end


function _constraint_violations!(b::BatchViolation, V::AbstractMatrix)
    violation!.(eachcol(b.in_cons_out), eachcol(V), Ref(b.in_cons))
end

function all_violations!(bm::BatchModel, b::BatchViolation, X::AbstractMatrix)
    V = cons_nln_batch!(bm, X, Θ)
    _constraint_violations!(b, V)
    violation!.(eachcol(b.in_vars_out), eachcol(X), Ref(b.in_vars))
end
function all_violations!(bm::BatchModel, b::BatchViolation, X::AbstractMatrix, Θ::AbstractMatrix)
    V = cons_nln_batch!(bm, X, Θ)
    _constraint_violations!(b, V)
    violation!.(eachcol(b.in_vars_out), eachcol(X), Ref(b.in_vars))
end
