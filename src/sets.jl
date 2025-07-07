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

distance_to_set(v, s) = distance_to_set(DefaultDistance(), v, s)
distance_to_set(::DefaultDistance, v, s) = distance_to_set(EpigraphViolationDistance(), v, s)
distance_to_set(::NormedEpigraphDistance{p}, v, s) where {p} = LinearAlgebra.norm(distance_to_set(EpigraphViolationDistance(), v, s), p)

distance_to_set(::EpigraphViolationDistance, v, s::LessThan) = begin
    max(v - s.u, zero(v))
end
distance_to_set(::EpigraphViolationDistance, v, s::GreaterThan) = begin
    max(s.l - v, zero(v))
end
distance_to_set(::EpigraphViolationDistance, v, s::EqualTo) = begin
    abs(v - s.v)
end
distance_to_set(::EpigraphViolationDistance, v, s::Interval) = begin
    max(s.l - v, v - s.u, zero(v))
end
