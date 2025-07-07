module BNKMathOptInterface

using BatchNLPKernels

using KernelAbstractions
const KA = KernelAbstractions

using MathOptInterface
const MOI = MathOptInterface


# TODO: use AbstractToIntervalBridge
function BNK.Interval(sets::Vector{MOI.Interval{T}}; backend=KA.CPU(), Tv=T) where T
    l = zeros(Tv, length(sets))
    u = zeros(Tv, length(sets))
    for (i, set) in enumerate(sets)
        l[i] = set.lower
        u[i] = set.upper
    end
    return BNK.Interval(
        _copyto_backend(backend, l),
        _copyto_backend(backend, u)
    )
end

function _copyto_backend(backend, v)
    arr = KA.allocate(backend, eltype(v), length(v)) 
    copyto!(arr, v)
    return arr
end



end # module BNKMathOptInterface