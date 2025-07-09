"""
    lagrangian_hprod!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian-vector products for a batch of points.
"""
function lagrangian_hprod!(bm::BatchModel, X::AbstractMatrix, Θ::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)
    Hv = _maybe_view(bm, :hprod_out, X)
    lagrangian_hprod!(bm, X, Θ, Y, V, Hv; obj_weight=obj_weight)
    return Hv
end 

"""
    lagrangian_hprod!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)

Evaluate Hessian-vector products for a batch of points.
"""
function lagrangian_hprod!(bm::BatchModel, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix; obj_weight=1.0)
    Θ = _repeat_params(bm, X)
    lagrangian_hprod!(bm, X, Θ, Y, V; obj_weight=obj_weight)
    return Hv
end

function lagrangian_hprod!(
    bm::BatchModel,
    X::AbstractMatrix,
    Θ::AbstractMatrix,
    Y::AbstractMatrix,
    V::AbstractMatrix,
    Hv::AbstractMatrix;
    obj_weight=1.0,
)
    batch_size = size(X, 2)
    @lencheck batch_size eachcol(X) eachcol(Θ) eachcol(Y) eachcol(V) eachcol(Hv)
    @lencheck bm.model.meta.nvar eachrow(X) eachrow(V) eachrow(Hv)
    @lencheck length(bm.model.θ) eachrow(Θ)
    @lencheck bm.model.meta.ncon eachrow(Y)
    _assert_batch_size(batch_size, bm.batch_size)
    backend = _get_backend(bm.model)
    ph = _get_prodhelper(bm.model)
    
    H_batch = _maybe_view(bm, :hprod_work, X)

    lagrangian_hessian!(bm, X, Θ, Y, H_batch; obj_weight=obj_weight)
    
    fill!(Hv, zero(eltype(Hv)))
    kersyspmv_batch(backend)(
        Hv,
        V,
        ph.hesssparsityi,
        H_batch,
        ph.hessptri;
        ndrange = (length(ph.hessptri) - 1, batch_size),
    )
    synchronize(backend)
    
    kersyspmv2_batch(backend)(
        Hv,
        V,
        ph.hesssparsityj,
        H_batch,
        ph.hessptrj;
        ndrange = (length(ph.hessptrj) - 1, batch_size),
    )
    synchronize(backend)
    
    return Hv
end
