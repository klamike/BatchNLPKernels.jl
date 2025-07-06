module BNKReactant

using BatchNLPKernels
using Reactant, KernelAbstractions
using ExaModels


function to_reactant_KA(bm::BNK.BatchModel)
    RKA = Base.get_extension(Reactant, :ReactantKernelAbstractionsExt)
    if !occursin("CUDA", string(bm.model.ext.backend))
        error("ExaModel must be built with CUDABackend")
    end
    return BNK.BatchModel(
        bm.model,
        bm.batch_size,
        Reactant.to_rarray(bm.obj_work),
        Reactant.to_rarray(bm.cons_work),
        Reactant.to_rarray(bm.cons_out),
        Reactant.to_rarray(bm.grad_work),
        Reactant.to_rarray(bm.grad_out),
        Reactant.to_rarray(bm.jprod_work),
        Reactant.to_rarray(bm.hprod_work),
        Reactant.to_rarray(bm.jprod_out),
        Reactant.to_rarray(bm.jtprod_out),
        Reactant.to_rarray(bm.hprod_out),
        RKA.ReactantBackend(),
    )
end

function to_reactant(bm::BNK.BatchModel)
    return BNK.BatchModel(
        bm.model,
        bm.batch_size,
        Reactant.to_rarray(bm.obj_work),
        Reactant.to_rarray(bm.cons_work),
        Reactant.to_rarray(bm.cons_out),
        Reactant.to_rarray(bm.grad_work),
        Reactant.to_rarray(bm.grad_out),
        Reactant.to_rarray(bm.jprod_work),
        Reactant.to_rarray(bm.hprod_work),
        Reactant.to_rarray(bm.jprod_out),
        Reactant.to_rarray(bm.jtprod_out),
        Reactant.to_rarray(bm.hprod_out),
        nothing,
    )
end

end # module BNKReactant