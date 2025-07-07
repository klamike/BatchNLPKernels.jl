"""
    BatchModelConfig

Configuration struct for controlling which buffers are allocated in a BatchModel.

## Fields
- `obj::Bool`: Allocate objective buffer (default: true)
- `cons::Bool`: Allocate constraint buffers (default: true)
- `grad::Bool`: Allocate gradient buffers (default: false)
- `jac::Bool`: Allocate jacobian buffer (default: false)
- `hess::Bool`: Allocate hessian buffer (default: false)
- `jprod::Bool`: Allocate jacobian-vector product buffer (default: false)
- `jtprod::Bool`: Allocate jacobian transpose-vector product buffer (default: false)
- `hprod::Bool`: Allocate hessian-vector product buffer (default: false)
- `viols::Bool`: Allocate constraint and variable violation buffers (default: false)
"""
struct BatchModelConfig
    obj::Bool
    cons::Bool
    grad::Bool
    jac::Bool
    hess::Bool
    jprod::Bool
    jtprod::Bool
    hprod::Bool
    viols::Bool
end

"""
    BatchModelConfig(; obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false, viols=false)

Create a BatchModelConfig with specified buffer allocations.
"""
function BatchModelConfig(; obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false, viols=false)
    return BatchModelConfig(obj, cons, grad, jac, hess, jprod, jtprod, hprod, viols)
end

"""
    BatchModelConfig(:minimal)

Minimal configuration with only objective and constraint buffers.
"""
BatchModelConfig(::Val{:minimal}) = BatchModelConfig(obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false, viols=false)

"""
    BatchModelConfig(:gradients)

Configuration to support obj, cons, and their gradients (grad, jtprod).
"""
BatchModelConfig(::Val{:gradients}) = BatchModelConfig(obj=true, cons=true, grad=true, jac=true, hess=false, jprod=false, jtprod=true, hprod=false, viols=false)

"""
    BatchModelConfig(:violations)

Configuration to support obj, cons, and constraint/variable violations.
"""
BatchModelConfig(::Val{:violations}) = BatchModelConfig(obj=true, cons=true, grad=false, jac=false, hess=false, jprod=false, jtprod=false, hprod=false, viols=true)

"""
    BatchModelConfig(:viol_grad)

Configuration to support obj, cons, constraint/variable violations, and their gradients.
"""
BatchModelConfig(::Val{:viol_grad}) = BatchModelConfig(obj=true, cons=true, grad=true, jac=true, hess=false, jprod=false, jtprod=true, hprod=false, viols=true)

"""
    BatchModelConfig(:full)

Full configuration with all buffers allocated.
"""
BatchModelConfig(::Val{:full}) = BatchModelConfig(obj=true, cons=true, grad=true, jac=true, hess=true, jprod=true, jtprod=true, hprod=true, viols=true)

BatchModelConfig(s::Symbol) = BatchModelConfig(Val(s))

"""
    BatchModel{MT,E}

A wrapper around ExaModel that pre-initializes buffers for batch operations.
Allows efficient evaluation of multiple points simultaneously.

## Fields
- `model::ExaModel`: The underlying ExaModel
- `batch_size::Int`: Number of points to evaluate simultaneously
- `obj_work::MT`: Batch objective values (nobj × batch_size)
- `cons_work::MT`: Batch constraint values (nconaug × batch_size)
- `cons_out::MT`: Dense constraint output buffer (ncon × batch_size)
- `grad_work::MT`: Batch gradient values (nnzg × batch_size)
- `grad_out::MT`: Dense gradient output buffer (nvar × batch_size)
- `jprod_work::MT`: Batch jacobian values (nnzj × batch_size)
- `hprod_work::MT`: Batch hessian values (nnzh × batch_size)
- `jprod_out::MT`: Batch jacobian-vector product buffer (ncon × batch_size)
- `jtprod_out::MT`: Batch jacobian transpose-vector product buffer (nvar × batch_size)
- `hprod_out::MT`: Batch hessian-vector product buffer (nvar × batch_size)
- `viols_cons_out::MT`: Constraint violation output buffer (ncon × batch_size)
- `viols_vars_out::MT`: Variable violation output buffer (nvar × batch_size)
- `viols_cons::Interval`: Constraint bounds as interval set
- `viols_vars::Interval`: Variable bounds as interval set
"""
struct BatchModel{MT,E,C,V} # FIXME: C,V needed?
    model::E
    batch_size::Int

    obj_work::MT
    cons_work::MT
    cons_out::MT
    grad_work::MT
    grad_out::MT
    jprod_work::MT
    hprod_work::MT
    jprod_out::MT
    jtprod_out::MT
    hprod_out::MT

    viols_cons_out::MT
    viols_vars_out::MT
    viols_cons::Interval{C}
    viols_vars::Interval{V}
end

"""
    BatchModel(model::ExaModel, batch_size::Int; config=BatchModelConfig())

Create a BatchModel wrapper around an ExaModel with pre-allocated buffers
for batch operations. Use `config` to specify which buffers to allocate.
"""
function BatchModel(model::ExaModels.ExaModel{T,VT,E,O,C}, batch_size::Int; config=BatchModelConfig(:minimal)) where {T,VT,E,O,C}
    @assert batch_size > 0 "Batch size must be positive"
    @assert E <: KAExtension "ExaModel must be created with a KernelAbstractions backend"
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nnzj = model.meta.nnzj
    nnzh = model.meta.nnzh

    nobj = length(model.ext.objbuffer)
    nnzg = length(model.ext.gradbuffer) 
    nconaug = length(model.ext.conbuffer)

    has_prodhelper = !isnothing(_get_prodhelper(model))
    if (config.jprod || config.jtprod || config.hprod) && !has_prodhelper
        error("Matrix-vector operations (jprod, jtprod, hprod) require ExaModel to be initialized with prod = true")
    end
    if (config.jprod || config.jtprod) && !config.jac
        error("jprod and jtprod operations require jac = true in BatchModelConfig")
    end
    if config.hprod && !config.hess
        error("hprod operation requires hess = true in BatchModelConfig")
    end

    o = model.ext.objbuffer
    obj_work = config.obj ? similar(o, nobj, batch_size) : similar(o, 0, batch_size)
    cons_work = config.cons ? similar(o, nconaug, batch_size) : similar(o, 0, batch_size)
    cons_out = config.cons ? similar(o, ncon, batch_size) : similar(o, 0, batch_size)
    grad_work = config.grad ? similar(o, nnzg, batch_size) : similar(o, 0, batch_size)
    grad_out = config.grad ? similar(o, nvar, batch_size) : similar(o, 0, batch_size)
    jprod_work = config.jac ? similar(o, nnzj, batch_size) : similar(o, 0, batch_size)
    hprod_work = config.hess ? similar(o, nnzh, batch_size) : similar(o, 0, batch_size)
    
    jprod_out = config.jprod ? similar(o, ncon, batch_size) : similar(o, 0, batch_size)
    jtprod_out = config.jtprod ? similar(o, nvar, batch_size) : similar(o, 0, batch_size)
    hprod_out = config.hprod ? similar(o, nvar, batch_size) : similar(o, 0, batch_size)

    # FIXME: consider don't allocate vars if there are no bound constraints
    viols_cons_out = config.viols ? similar(o, ncon, batch_size) : similar(o, 0, batch_size)
    viols_vars_out = config.viols ? similar(o, nvar, batch_size) : similar(o, 0, batch_size)
    viols_cons = config.viols ? Interval(model.meta.lcon, model.meta.ucon) : Interval()
    viols_vars = config.viols ? Interval(model.meta.lvar, model.meta.uvar) : Interval()
    
    return BatchModel(
        model,
        batch_size,
        obj_work,
        cons_work,
        cons_out,
        grad_work,
        grad_out,
        jprod_work,
        hprod_work,
        jprod_out,
        jtprod_out,
        hprod_out,
        viols_cons_out,
        viols_vars_out,
        viols_cons,
        viols_vars,
    )
end

function Base.show(io::IO, bm::BatchModel{T,VT}) where {T,VT}
    println(io, "BatchModel{$T, $VT, ...} with max batch size = $(bm.batch_size)")
    println(io, "Wrapping:")
    Base.show(io, bm.model)
end