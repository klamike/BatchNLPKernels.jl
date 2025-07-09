module BNKChainRulesCore

using BatchNLPKernels
using ChainRulesCore

function ChainRulesCore.rrule(::typeof(BatchNLPKernels.objective!), bm::BatchModel, X, Θ)
    y = BatchNLPKernels.objective!(bm, X, Θ)
    
    function obj_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        gradients = BatchNLPKernels.objective_gradient!(bm, X, Θ)
        
        X̄ = gradients .* Ȳ'
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄, ChainRulesCore.NoTangent()
    end
    
    return y, obj_batch_pullback
end
function ChainRulesCore.rrule(::typeof(BatchNLPKernels.objective!), bm::BatchModel, X)
    y = BatchNLPKernels.objective!(bm, X)
    
    function obj_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        gradients = BatchNLPKernels.objective_gradient!(bm, X)

        X̄ = gradients .* Ȳ'
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄
    end
    
    return y, obj_batch_pullback
end


function ChainRulesCore.rrule(::typeof(BatchNLPKernels.constraints!), bm::BatchModel, X, Θ)
    y = BatchNLPKernels.constraints!(bm, X, Θ)
    
    function cons_nln_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        X̄ = BatchNLPKernels.constraints_jtprod!(bm, X, Θ, Ȳ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄, ChainRulesCore.NoTangent()
    end
    
    return y, cons_nln_batch_pullback
end
function ChainRulesCore.rrule(::typeof(BatchNLPKernels.constraints!), bm::BatchModel, X)
    y = BatchNLPKernels.constraints!(bm, X)
    
    function cons_nln_batch_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        X̄ = BatchNLPKernels.constraints_jtprod!(bm, X, Ȳ)
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄
    end
    
    return y, cons_nln_batch_pullback
end


function ChainRulesCore.rrule(::typeof(BatchNLPKernels.constraint_violations!), bm::BatchModel, V)
    Vc = BatchNLPKernels.constraint_violations!(bm, V)
    
    function constraint_violations_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        
        # violation(v, s) = max(s.l - v, v - s.u, 0)
        # ∂violation/∂v = -1 if v < s.l, +1 if v > s.u, 0 otherwise
        
        V̄ = if isempty(bm.viols_cons)
            zeros(eltype(V), size(V))
        else
            lower_viols = V .< bm.viols_cons.l
            upper_viols = V .> bm.viols_cons.u
            lower_viols .* (-Ȳ) .+ upper_viols .* Ȳ
        end
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), V̄
    end
    
    return Vc, constraint_violations_pullback
end
function ChainRulesCore.rrule(::typeof(BatchNLPKernels.bound_violations!), bm::BatchModel, X)
    Vb = BatchNLPKernels.bound_violations!(bm, X)
    
    function bound_violations_pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        
        # violation(x, s) = max(s.l - x, x - s.u, 0)
        # ∂violation/∂x = -1 if x < s.l, +1 if x > s.u, 0 otherwise
        
        X̄ = if isempty(bm.viols_vars)
            zeros(eltype(X), size(X))
        else
            lower_viols = X .< bm.viols_vars.l
            upper_viols = X .> bm.viols_vars.u
            lower_viols .* (-Ȳ) .+ upper_viols .* Ȳ
        end
        
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), X̄
    end
    
    return Vb, bound_violations_pullback
end

end # module BNKChainRulesCore 