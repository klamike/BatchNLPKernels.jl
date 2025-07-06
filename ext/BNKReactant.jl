module BNKReactant

using BatchNLPKernels
using Reactant, KernelAbstractions
RKA = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
using GPUArraysCore
using ExaModels

function to_reactant(bm::BNK.BatchModel; MT=Reactant.ConcreteRArray)
    return BNK.BatchModel(
        bm.model,
        bm.batch_size,
        MT(bm.obj_work),
        MT(bm.cons_work),
        MT(bm.cons_out),
        MT(bm.grad_work),
        MT(bm.grad_out),
        MT(bm.jprod_work),
        MT(bm.hprod_work),
        MT(bm.jprod_out),
        MT(bm.jtprod_out),
        MT(bm.hprod_out),
        RKA.ReactantBackend(),
    )
end

end # module BNKReactant