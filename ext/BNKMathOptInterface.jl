module BNKMathOptInterface

using BatchNLPKernels
using KernelAbstractions
using MathOptInterface

const MOI = MathOptInterface
const KA = KernelAbstractions


function BNK.LessThan(sets::Vector{MOI.LessThan{T}}; backend=KA.CPU(), Tv=T) where T
    u = zeros(Tv, length(sets))
    for (i, set) in enumerate(sets)
        u[i] = set.upper
    end
    return BNK.LessThan(BNK._copyto_backend(backend, u))
end

function BNK.GreaterThan(sets::Vector{MOI.GreaterThan{T}}; backend=KA.CPU(), Tv=T) where T
    l = zeros(Tv, length(sets))
    for (i, set) in enumerate(sets)
        l[i] = set.lower
    end
    return BNK.GreaterThan(BNK._copyto_backend(backend, l))
end

function BNK.EqualTo(sets::Vector{MOI.EqualTo{T}}; backend=KA.CPU(), Tv=T) where T
    v = zeros(Tv, length(sets))
    for (i, set) in enumerate(sets)
        v[i] = set.value
    end
    return BNK.EqualTo(BNK._copyto_backend(backend, v))
end

function BNK.Interval(sets::Vector{MOI.Interval{T}}; backend=KA.CPU(), Tv=T) where T
    l = zeros(Tv, length(sets))
    u = zeros(Tv, length(sets))
    for (i, set) in enumerate(sets)
        l[i] = set.lower
        u[i] = set.upper
    end
    return BNK.Interval(
        BNK._copyto_backend(backend, l),
        BNK._copyto_backend(backend, u)
    )
end



end # module BNKMathOptInterface