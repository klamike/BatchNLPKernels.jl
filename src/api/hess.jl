"""
    lagrangian_hessian!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian coordinates for a batch of points.
"""
function lagrangian_hessian!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)
    H_view = _maybe_view(bm, :hprod_work, X)
    lagrangian_hessian!(bm, X, Θ, Y, H_view; obj_weight=obj_weight)
    return H_view
end

"""
    lagrangian_hessian!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian coordinates for a batch of points.
"""
function lagrangian_hessian!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix; obj_weight=1.0)
    Θ = _repeat_params(bm, X)
    lagrangian_hessian!(bm, X, Θ, Y; obj_weight=obj_weight)
end

function lagrangian_hessian!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    Y::AbstractMatrix,
    H::AbstractMatrix;
    obj_weight=1.0,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(Y) eachcol(H)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Y)
    @lencheck bm.model.meta.nnzh eachrow(H)
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)
    
    fill!(H, zero(eltype(H)))
    _obj_lagrangian_hessian!(backend, H, bm.model.objs, X, Θ, obj_weight)
    _con_lagrangian_hessian!(backend, H, bm.model.cons, X, Θ, Y)
    return H
end

function _obj_lagrangian_hessian!(backend, H, objs, X, Θ, obj_weight)
    shessian_batch!(backend, H, nothing, objs, X, Θ, obj_weight, zero(eltype(H)))
    _obj_lagrangian_hessian!(backend, H, objs.inner, X, Θ, obj_weight)
    synchronize(backend)
end
function _obj_lagrangian_hessian!(backend, H, objs::ExaModels.ObjectiveNull, X, Θ, obj_weight) end

function _con_lagrangian_hessian!(backend, H, cons, X, Θ, Y)
    shessian_batch!(backend, H, nothing, cons, X, Θ, Y, zero(eltype(H)))
    _con_lagrangian_hessian!(backend, H, cons.inner, X, Θ, Y)
    synchronize(backend)
end
function _con_lagrangian_hessian!(backend, H, cons::ExaModels.ConstraintNull, X, Θ, Y) end

function shessian_batch!(
    backend::B,
    y1,
    y2,
    f,
    X,
    Θ,
    adj,
    adj2,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        batch_size = size(X, 2)
        kerh_batch(backend)(y1, y2, f.f, f.itr, X, Θ, adj, adj2; ndrange = (length(f.itr), batch_size))
    end
end

function shessian_batch!(
    backend::B,
    y1,
    y2,
    f,
    X,
    Θ,
    adj::AbstractMatrix,
    adj2,
) where {B<:KernelAbstractions.Backend}
    if !isempty(f.itr)
        batch_size = size(X, 2)
        kerh2_batch(backend)(y1, y2, f.f, f.itr, X, Θ, adj, adj2; ndrange = (length(f.itr), batch_size))
    end
end
