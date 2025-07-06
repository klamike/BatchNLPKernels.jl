@kernel function kerf_batch(Y, @Const(f), @Const(itr), @Const(X), @Const(Θ))
    I, batch_idx = @index(Global, NTuple)
    @inbounds Y[ExaModels.offset0(f, itr, I), batch_idx] = f.f(itr[I], view(X, :, batch_idx), view(Θ, :, batch_idx))
end
function kerf_batch_cpu!(Y::AbstractMatrix, f, itr, X::AbstractMatrix, Θ::AbstractMatrix)
    @assert size(X, 2) == size(Θ, 2) == size(Y, 2) "Batch dimension mismatch"
    @inbounds for batch_idx in axes(X, 2)
        x_batch = view(X, :, batch_idx)
        θ_batch = view(Θ, :, batch_idx)
        @simd for I in eachindex(itr)
            row = ExaModels.offset0(f, itr, I)
            Y[row, batch_idx] = f.f(itr[I], x_batch, θ_batch)
        end
    end
    return Y
end


@kernel function kerf2_batch(Y, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(oa))
    I, batch_idx = @index(Global, NTuple)
    @inbounds Y[oa+I, batch_idx] = f.f(itr[I], view(X, :, batch_idx), view(Θ, :, batch_idx))
end
function kerf2_batch_cpu!(Y::AbstractMatrix, f, itr, X::AbstractMatrix, Θ::AbstractMatrix, oa::Integer)
    @assert size(X, 2) == size(Θ, 2) == size(Y, 2) "Batch dimension mismatch"
    @inbounds for batch_idx in axes(X, 2)
        x_batch = view(X, :, batch_idx)
        θ_batch = view(Θ, :, batch_idx)
        @simd for I in eachindex(itr)
            Y[oa + I, batch_idx] = f.f(itr[I], x_batch, θ_batch)
        end
    end
    return Y
end


@kernel function kerg_batch(Y, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adj))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.grpass(
        f.f(itr[I], ExaModels.AdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp1,
        view(Y, :, batch_idx),
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end
function kerg_batch_cpu!(Y::AbstractMatrix, f, itr, X::AbstractMatrix, Θ::AbstractMatrix, adj)
    @assert size(X, 2) == size(Θ, 2) == size(Y, 2) "Batch dimension mismatch"
    @inbounds for batch_idx in axes(X, 2)
        x_batch = view(X, :, batch_idx)
        θ_batch = view(Θ, :, batch_idx)
        y_batch = view(Y, :, batch_idx)
        @simd for I in eachindex(itr)
            ExaModels.grpass(
                f.f(itr[I], ExaModels.AdjointNodeSource(x_batch), θ_batch),
                f.comp1,
                y_batch,
                ExaModels.offset1(f, I),
                0,
                adj,
            )
        end
    end
    return Y
end


@kernel function kerj_batch(Y1, Y2, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adj))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.jrpass(
        f.f(itr[I], ExaModels.AdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp1,
        ExaModels.offset0(f, itr, I),
        isnothing(Y1) ? nothing : view(Y1, :, batch_idx),
        isnothing(Y2) ? nothing : view(Y2, :, batch_idx),
        ExaModels.offset1(f, I),
        0,
        adj,
    )
end
function kerj_batch_cpu!(Y1, Y2, f, itr, X::AbstractMatrix, Θ::AbstractMatrix, adj)
    @assert size(X, 2) == size(Θ, 2) "Batch dimension mismatch"
    nbatch = size(X, 2)
    @inbounds for batch_idx in 1:nbatch
        x_batch = view(X, :, batch_idx)
        θ_batch = view(Θ, :, batch_idx)
        y1_view = isnothing(Y1) ? nothing : view(Y1, :, batch_idx)
        y2_view = isnothing(Y2) ? nothing : view(Y2, :, batch_idx)
        @simd for I in eachindex(itr)
            ExaModels.jrpass(
                f.f(itr[I], ExaModels.AdjointNodeSource(x_batch), θ_batch),
                f.comp1,
                ExaModels.offset0(f, itr, I),
                y1_view,
                y2_view,
                ExaModels.offset1(f, I),
                0,
                adj,
            )
        end
    end
    return nothing
end



@kernel function kerh_batch(Y1, Y2, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adj1), @Const(adj2))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.hrpass0(
        f.f(itr[I], ExaModels.SecondAdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp2,
        isnothing(Y1) ? nothing : view(Y1, :, batch_idx),
        isnothing(Y2) ? nothing : view(Y2, :, batch_idx),
        ExaModels.offset2(f, I),
        0,
        adj1,
        adj2,
    )
end
function kerh_batch_cpu!(Y1, Y2, f, itr, X::AbstractMatrix, Θ::AbstractMatrix, adj1, adj2)
    @assert size(X, 2) == size(Θ, 2) "Batch dimension mismatch"
    nbatch = size(X, 2)
    @inbounds for batch_idx in 1:nbatch
        x_batch = view(X, :, batch_idx)
        θ_batch = view(Θ, :, batch_idx)
        y1_view = isnothing(Y1) ? nothing : view(Y1, :, batch_idx)
        y2_view = isnothing(Y2) ? nothing : view(Y2, :, batch_idx)
        @simd for I in eachindex(itr)
            ExaModels.hrpass0(
                f.f(itr[I], ExaModels.SecondAdjointNodeSource(x_batch), θ_batch),
                f.comp2,
                y1_view,
                y2_view,
                ExaModels.offset2(f, I),
                0,
                adj1,
                adj2,
            )
        end
    end
    return nothing
end


@kernel function kerh2_batch(Y1, Y2, @Const(f), @Const(itr), @Const(X), @Const(Θ), @Const(adjs1), @Const(adj2))
    I, batch_idx = @index(Global, NTuple)
    @inbounds ExaModels.hrpass0(
        f.f(itr[I], ExaModels.SecondAdjointNodeSource(view(X, :, batch_idx)), view(Θ, :, batch_idx)),
        f.comp2,
        isnothing(Y1) ? nothing : view(Y1, :, batch_idx),
        isnothing(Y2) ? nothing : view(Y2, :, batch_idx),
        ExaModels.offset2(f, I),
        0,
        adjs1[ExaModels.offset0(f, itr, I), batch_idx],
        adj2,
    )
end
function kerh2_batch_cpu!(Y1, Y2, f, itr, X::AbstractMatrix, Θ::AbstractMatrix, adjs1, adj2)
    @assert size(X, 2) == size(Θ, 2) == size(adjs1, 2) "Batch dimension mismatch"
    nbatch = size(X, 2)
    @inbounds for batch_idx in 1:nbatch
        x_batch = view(X, :, batch_idx)
        θ_batch = view(Θ, :, batch_idx)
        y1_view = isnothing(Y1) ? nothing : view(Y1, :, batch_idx)
        y2_view = isnothing(Y2) ? nothing : view(Y2, :, batch_idx)
        @simd for I in eachindex(itr)
            ExaModels.hrpass0(
                f.f(itr[I], ExaModels.SecondAdjointNodeSource(x_batch), θ_batch),
                f.comp2,
                y1_view,
                y2_view,
                ExaModels.offset2(f, I),
                0,
                adjs1[ExaModels.offset0(f, itr, I), batch_idx],
                adj2,
            )
        end
    end
    return nothing
end


@kernel function compress_to_dense_batch(Y, @Const(Y0), @Const(ptr), @Const(sparsity))
    I, batch_idx = @index(Global, NTuple)
    @inbounds for j = ptr[I]:(ptr[I+1]-1)
        (k, l) = sparsity[j]
        Y[k, batch_idx] += Y0[l, batch_idx]
    end
end
function compress_to_dense_batch_cpu!(Y::AbstractMatrix, Y0::AbstractMatrix, ptr, sparsity)
    @assert size(Y, 2) == size(Y0, 2) "Batch dimension mismatch"
    nbatch = size(Y, 2)
    @inbounds for batch_idx in 1:nbatch
        @simd for I in 1:(length(ptr) - 1)
            @simd for j in ptr[I]:(ptr[I + 1] - 1)
                k, l = sparsity[j]
                Y[k, batch_idx] += Y0[l, batch_idx]
            end
        end
    end
    return Y
end
@inline function _run_compress_to_dense_batch!(backend, Y, Y0, ptr, sparsity, batch_size)
    compress_to_dense_batch(backend)(Y, Y0, ptr, sparsity; ndrange = (length(ptr) - 1, batch_size))
    synchronize(backend)
    return Y
end
@inline function _run_compress_to_dense_batch!(::Nothing, Y, Y0, ptr, sparsity, batch_size)
    compress_to_dense_batch_cpu!(Y, Y0, ptr, sparsity)
    return Y
end


@kernel function kerspmv_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        Y[i, batch_idx] += V[ind, batch_idx] * X[j, batch_idx]
    end
end
function kerspmv_batch_cpu!(Y::AbstractMatrix, X::AbstractMatrix, coord, V::AbstractMatrix, ptr)
    @assert size(Y, 2) == size(X, 2) == size(V, 2) "Batch dimension mismatch"
    nbatch = size(Y, 2)
    nidx = length(ptr) - 1
    @inbounds for batch_idx in 1:nbatch
        for idx in 1:nidx
            @simd for l in ptr[idx]:(ptr[idx + 1] - 1)
                ((i, j), ind) = coord[l]
                Y[i, batch_idx] += V[ind, batch_idx] * X[j, batch_idx]
            end
        end
    end
    return Y
end
@inline function _run_kerspmv_batch!(backend, Y, X, coord, V, ptr, batch_size)
    kerspmv_batch(backend)(Y, X, coord, V, ptr; ndrange = (length(ptr) - 1, batch_size))
    synchronize(backend)
    return Y
end
@inline function _run_kerspmv_batch!(::Nothing, Y, X, coord, V, ptr, batch_size)
    kerspmv_batch_cpu!(Y, X, coord, V, ptr)
    return Y
end


@kernel function kerspmv2_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        Y[j, batch_idx] += V[ind, batch_idx] * X[i, batch_idx]
    end
end
function kerspmv2_batch_cpu!(Y::AbstractMatrix, X::AbstractMatrix, coord, V::AbstractMatrix, ptr)
    @assert size(Y, 2) == size(X, 2) == size(V, 2) "Batch dimension mismatch"
    nbatch = size(Y, 2)
    nidx = length(ptr) - 1
    @inbounds @simd for batch_idx in 1:nbatch
        for idx in 1:nidx
            @simd for l in ptr[idx]:(ptr[idx + 1] - 1)
                ((i, j), ind) = coord[l]
                Y[j, batch_idx] += V[ind, batch_idx] * X[i, batch_idx]
            end
        end
    end
    return Y
end
@inline function _run_kerspmv2_batch!(backend, Y, X, coord, V, ptr, batch_size)
    kerspmv2_batch(backend)(Y, X, coord, V, ptr; ndrange = (length(ptr) - 1, batch_size))
    synchronize(backend)
    return Y
end
@inline function _run_kerspmv2_batch!(::Nothing, Y, X, coord, V, ptr, batch_size)
    kerspmv2_batch_cpu!(Y, X, coord, V, ptr)
    return Y
end


@kernel function kersyspmv_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        Y[i, batch_idx] += V[ind, batch_idx] * X[j, batch_idx]
    end
end
function kersyspmv_batch_cpu!(Y::AbstractMatrix, X::AbstractMatrix, coord, V::AbstractMatrix, ptr)
    @assert size(Y, 2) == size(X, 2) == size(V, 2) "Batch dimension mismatch"
    nbatch = size(Y, 2)
    nidx = length(ptr) - 1
    @inbounds @simd for batch_idx in 1:nbatch
        for idx in 1:nidx
            @simd for l in ptr[idx]:(ptr[idx + 1] - 1)
                ((i, j), ind) = coord[l]
                Y[i, batch_idx] += V[ind, batch_idx] * X[j, batch_idx]
            end
        end
    end
    return Y
end
@inline function _run_kersyspmv_batch!(backend, Y, X, coord, V, ptr, batch_size)
    kersyspmv_batch(backend)(Y, X, coord, V, ptr; ndrange = (length(ptr) - 1, batch_size))
    synchronize(backend)
    return Y
end
@inline function _run_kersyspmv_batch!(::Nothing, Y, X, coord, V, ptr, batch_size)
    kersyspmv_batch_cpu!(Y, X, coord, V, ptr)
    return Y
end


@kernel function kersyspmv2_batch(Y, @Const(X), @Const(coord), @Const(V), @Const(ptr))
    idx, batch_idx = @index(Global, NTuple)
    @inbounds for l = ptr[idx]:(ptr[idx+1]-1)
        ((i, j), ind) = coord[l]
        if i != j
            Y[j, batch_idx] += V[ind, batch_idx] * X[i, batch_idx]
        end
    end
end
function kersyspmv2_batch_cpu!(Y::AbstractMatrix, X::AbstractMatrix, coord, V::AbstractMatrix, ptr)
    @assert size(Y, 2) == size(X, 2) == size(V, 2) "Batch dimension mismatch"
    nbatch = size(Y, 2)
    nidx = length(ptr) - 1
    @inbounds for batch_idx in 1:nbatch
        for idx in 1:nidx
            @simd for l in ptr[idx]:(ptr[idx + 1] - 1)
                ((i, j), ind) = coord[l]
                if i != j
                    Y[j, batch_idx] += V[ind, batch_idx] * X[i, batch_idx]
                end
            end
        end
    end
    return Y
end 
@inline function _run_kersyspmv2_batch!(backend, Y, X, coord, V, ptr, batch_size)
    kersyspmv2_batch(backend)(Y, X, coord, V, ptr; ndrange = (length(ptr) - 1, batch_size))
    synchronize(backend)
    return Y
end
@inline function _run_kersyspmv2_batch!(::Nothing, Y, X, coord, V, ptr, batch_size)
    kersyspmv2_batch_cpu!(Y, X, coord, V, ptr)
    return Y
end