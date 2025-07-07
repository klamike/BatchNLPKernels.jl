
"""
    Interval{VT}

Represents the RHS of M constraints g(xᵢ) ∈ [lᵢ, uᵢ]  ∀i ∈ 1:M.
"""
struct Interval{VT}
    l::VT
    u::VT
end
@inline violation(v, s::Interval{VT}) where {VT} = begin
    @. max(s.l - v, v - s.u, zero(v))
end

Base.broadcastable(s::Interval) = Ref(s)
Base.isempty(s::Interval{VT}) where {VT} = isempty(s.l) || isempty(s.u)

# empty support (unconstrained)
Interval(::Nothing) = Interval()
Interval() = Interval(nothing, nothing)
Base.isempty(s::Interval{Nothing}) = true
violation(v, s::Interval{Nothing}) = zero(v)