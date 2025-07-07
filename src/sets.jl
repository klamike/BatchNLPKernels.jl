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

@inline distance_to_set(v, s) = distance_to_set(DefaultDistance(), v, s)
@inline distance_to_set(::DefaultDistance, v, s::S) where {S} = distance_to_set(EpigraphViolationDistance(), v, s)
@inline distance_to_set(::NormedEpigraphDistance{p}, s::S) where {p,S} = LinearAlgebra.norm(distance_to_set(EpigraphViolationDistance(), v, s), p)

distance_to_set!(d, v, s) = begin
    d .= distance_to_set(DefaultDistance(), v, s)
end
distance_to_set!(::DefaultDistance, d, v, s::S) where {S} = begin
    d .= distance_to_set(EpigraphViolationDistance(), v, s)
end
distance_to_set!(::NormedEpigraphDistance{p}, d, v, s::S) where {p,S} = begin
    d .= LinearAlgebra.norm(distance_to_set(EpigraphViolationDistance(), v, s), p)
end


@inline distance_to_set(::EpigraphViolationDistance, s::LessThan) = begin
    @. max(v - s.u, zero(v))
end
@inline distance_to_set(::EpigraphViolationDistance, s::GreaterThan) = begin
    @. max(s.l - v, zero(v))
end
@inline distance_to_set(::EpigraphViolationDistance, s::EqualTo) = begin
    @. abs(v - s.v)
end
@inline distance_to_set(::EpigraphViolationDistance, s::Interval) = begin
    @. max(s.l - v, v - s.u, zero(v))
end
