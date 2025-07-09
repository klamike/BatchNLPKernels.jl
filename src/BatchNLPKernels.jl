module BatchNLPKernels

using ExaModels
using KernelAbstractions

const ExaKA = Base.get_extension(ExaModels, :ExaModelsKernelAbstractions)
const KAExtension = ExaKA.KAExtension

include("interval.jl")
include("batch_model.jl")

const BOI = BatchNLPKernels
export BOI, BatchModel, BatchModelConfig
export objective!, objective_gradient!, constraints!, constraints_jacobian!, lagrangian_hessian!
export constraints_jprod!, constraints_jtprod!, lagrangian_hprod!
export all_violations!, constraint_violations!, bound_violations!

include("utils.jl")
include("kernels.jl")
include("api/cons.jl")
include("api/grad.jl")
include("api/hess.jl")
include("api/jac.jl")
include("api/obj.jl")
include("api/jprod.jl")
include("api/hprod.jl")
include("api/viols.jl")

end # module BatchNLPKernels