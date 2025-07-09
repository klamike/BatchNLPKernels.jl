"""
    objective!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)

Evaluate objective function for a batch of points.
"""
function objective!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix)
    obj_work = _maybe_view(bm, :obj_work, X)
    return objective!(bm, obj_work, X, Θ)
end


"""
    objective!(bm::BatchModel, X::AbstractMatrix)

Evaluate objective function for a batch of points.
"""
function objective!(bm::BatchModel, X::AbstractMatrix)
    Θ = _repeat_params(bm, X)
    return objective!(bm, X, Θ)
end

function objective!(
    bm::BatchModel,
    obj_work,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ)
    @lencheck bm.model.meta.nvar eachrow(X)
    @lencheck length(bm.model.θ) eachrow(Θ)
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)

    _obj_batch(backend, obj_work, bm.model.objs, X, Θ)
    return vec(sum(obj_work, dims=1))  # FIXME
end

function _obj_batch(backend, obj_work, obj, X, Θ)
    if !isempty(obj.itr)
        batch_size = size(X, 2)
        kerf_batch(backend)(obj_work, obj.f, obj.itr, X, Θ; ndrange = (length(obj.itr), batch_size))
    end
    _obj_batch(backend, obj_work, obj.inner, X, Θ)
    synchronize(backend)
end
function _obj_batch(backend, obj_work, f::ExaModels.ObjectiveNull, X, Θ) end
