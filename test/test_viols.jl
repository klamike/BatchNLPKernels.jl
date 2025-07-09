function test_violations_correctness(model::ExaModel, batch_size::Int; 
                                   atol::Float64=1e-10, rtol::Float64=1e-10, MOD=OpenCL)
    bm = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:violations))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)

    X = MOD.randn(nvar, batch_size)
    Θ = MOD.randn(nθ, batch_size)

    @allowscalar if !isempty(model.meta.lvar) && !isempty(model.meta.uvar)
        if isfinite(model.meta.lvar[1])
            X[1, :] .= model.meta.lvar[1] - 0.1
        end
        if isfinite(model.meta.uvar[end])
            X[end, :] .= model.meta.uvar[end] + 0.1
        end
    end
    
    @testset "Violations Correctness: $(nvar) vars, $(ncon) cons, $(nθ) params" begin
        
        @testset "All Violations" begin
            if nθ > 0
                Vc, Vb = BOI.all_violations!(bm, X, Θ)
                @test size(Vc) == (ncon, batch_size)
                @test size(Vb) == (nvar, batch_size)
                @test all(>=(0), Vc)
                @test all(>=(0), Vb)
                @test all(isfinite, Vc)
                @test all(isfinite, Vb)
                @allowscalar begin
                    if !isempty(model.meta.lvar) && !isempty(model.meta.uvar)
                        if isfinite(model.meta.lvar[1])
                            @test Vb[1, :] .≈ 0.1 atol=atol rtol=rtol
                        end
                        if isfinite(model.meta.uvar[end])
                            @test Vb[end, :] .≈ 0.1 atol=atol rtol=rtol
                        end
                    end
                end
            end

            Vc, Vb = BOI.all_violations!(bm, X)
            @test size(Vc) == (ncon, batch_size)
            @test size(Vb) == (nvar, batch_size)
            @test all(>=(0), Vc)
            @test all(>=(0), Vb)
            @test all(isfinite, Vc)
            @test all(isfinite, Vb)
        end
        
        @testset "Constraint Violations" begin
            if ncon > 0
                V_cons = BOI.constraints!(bm, X, Θ)
                Vc = BOI.constraint_violations!(bm, V_cons)
                
                @test size(Vc) == (ncon, batch_size)
                @test all(>=(0), Vc)
                @test all(isfinite, Vc)
                
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    cons_vals = MOD.zeros(ncon)
                    @allowscalar ExaModels.cons_nln!(model, X[:, i], cons_vals)
                    
                    @allowscalar for j in 1:ncon
                        if isempty(model.meta.lcon) && isempty(model.meta.ucon)
                            expected_viol = 0.0
                        else
                            lcon = model.meta.lcon[j]
                            ucon = model.meta.ucon[j]
                            
                            lower_viol = isfinite(lcon) ? max(lcon - cons_vals[j], 0.0) : 0.0
                            upper_viol = isfinite(ucon) ? max(cons_vals[j] - ucon, 0.0) : 0.0
                            expected_viol = lower_viol + upper_viol
                        end
                        @test Vc[j, i] ≈ expected_viol atol=atol rtol=rtol
                    end
                end
            end
        end
        
        @testset "Bound Violations" begin
            Vb = BOI.bound_violations!(bm, X)
            @test size(Vb) == (nvar, batch_size)
            @test all(>=(0), Vb)
            @test all(isfinite, Vb)
            
            for i in 1:batch_size
                @allowscalar for j in 1:nvar
                    lvar = model.meta.lvar[j]
                    uvar = model.meta.uvar[j]
                    
                    lower_viol = isfinite(lvar) ? max(lvar - X[j, i], 0.0) : 0.0
                    upper_viol = isfinite(uvar) ? max(X[j, i] - uvar, 0.0) : 0.0
                    expected_viol = lower_viol + upper_viol
                    
                    @test Vb[j, i] ≈ expected_viol atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Feasible Points" begin
            X_feasible = MOD.zeros(nvar, batch_size)
            if !isempty(model.meta.lvar) && !isempty(model.meta.uvar)
                @allowscalar begin
                    X_feasible .= (model.meta.lvar .+ model.meta.uvar) ./ 2
                    unconstr = findall(isinf.(model.meta.lvar) .&& isinf.(model.meta.uvar))
                    X_feasible[unconstr, :] .= zero(model.meta.lvar)[unconstr]
                    lunconstr = findall(isinf.(model.meta.lvar) .&& isfinite.(model.meta.uvar))
                    X_feasible[lunconstr, :] .= model.meta.uvar[lunconstr] .- 0.1
                    uunconstr = findall(isfinite.(model.meta.lvar) .&& isinf.(model.meta.uvar))
                    X_feasible[uunconstr, :] .= model.meta.lvar[uunconstr] .+ 0.1
                    @assert all(isfinite, X_feasible)
                end
                Vb_feasible = BOI.bound_violations!(bm, X_feasible)
                @test all(==(0), Vb_feasible)
            end
        end
        
        @testset "Dimension Validation" begin
            X_wrong = MOD.randn(nvar + 1, batch_size)
            @test_throws DimensionMismatch BOI.all_violations!(bm, X_wrong)
            @test_throws DimensionMismatch BOI.bound_violations!(bm, X_wrong)
            
            if nθ > 0
                Θ_wrong = MOD.randn(nθ + 1, batch_size)
                @test_throws DimensionMismatch BOI.all_violations!(bm, X, Θ_wrong)
            end
            
            if ncon > 0
                V_wrong = MOD.randn(ncon + 1, batch_size)
                @test_throws DimensionMismatch BOI.constraint_violations!(bm, V_wrong)
            end
        end
        
        @testset "Batch Size Validation" begin
            X_large = MOD.randn(nvar, batch_size + 1)
            @test_throws AssertionError BOI.all_violations!(bm, X_large)
            @test_throws AssertionError BOI.bound_violations!(bm, X_large)
            
            if ncon > 0
                V_large = MOD.randn(ncon, batch_size + 1)
                @test_throws AssertionError BOI.constraint_violations!(bm, V_large)
            end
        end
    end
end

function test_violations_differentiability_gpu(model::ExaModel, batch_size::Int; MOD=OpenCL)
    bm = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:viol_grad))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X_gpu = MOD.randn(nvar, batch_size)
    Θ_gpu = MOD.randn(nθ, batch_size)
    
    @testset "Violations Differentiability GPU" begin
        @testset "All Violations Sum" begin
            function f_all_viols(params)
                X = params[1:nvar, :]
                Θ = params[nvar+1:end, :]
                Vc, Vb = BOI.all_violations!(bm, X, Θ)
                return sum(Vc) + sum(Vb)
            end
            
            params = vcat(X_gpu, Θ_gpu)
            grad = DI.gradient(f_all_viols, AutoZygote(), params)
            @test size(grad) == size(params)
            @test all(isfinite, grad)
        end
        
        @testset "Constraint Violations Sum" begin
            if ncon > 0
                function f_cons_viols(params)
                    X = params[1:nvar, :]
                    Θ = params[nvar+1:end, :]
                    V_cons = BOI.constraints!(bm, X, Θ)
                    Vc = BOI.constraint_violations!(bm, V_cons)
                    return sum(Vc)
                end
                
                params = vcat(X_gpu, Θ_gpu)
                grad = DI.gradient(f_cons_viols, AutoZygote(), params)
                @test size(grad) == size(params)
                @test all(isfinite, grad)
            end
        end
        
        @testset "Bound Violations Sum" begin
            function f_bound_viols(X)
                Vb = BOI.bound_violations!(bm, X)
                return sum(Vb)
            end
            
            grad = DI.gradient(f_bound_viols, AutoZygote(), X_gpu)
            @test size(grad) == size(X_gpu)
            @test all(isfinite, grad)
        end
    end
end

function test_violations_differentiability_cpu(model::ExaModel, batch_size::Int)
    bm = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:viol_grad))
    
    nvar = model.meta.nvar
    nθ = length(model.θ)
    
    X_cpu = randn(nvar, batch_size)
    Θ_cpu = randn(nθ, batch_size)
    
    @testset "Violations Differentiability CPU" begin
        @testset "All Violations Sum" begin
            function f_all_viols(params)
                X = params[1:nvar, :]
                Θ = params[nvar+1:end, :]
                Vc, Vb = BOI.all_violations!(bm, X, Θ)
                return sum(Vc) + sum(Vb)
            end
            
            params = vcat(X_cpu, Θ_cpu)
            grad = DI.gradient(f_all_viols, AutoZygote(), params)
            @test grad isa AbstractMatrix
            @test size(grad) == size(params)
            @test all(isfinite, grad)
            @testset "FiniteDifferences" begin
                gradfd = DI.gradient(f_all_viols, AutoFiniteDifferences(fdm=FiniteDifferences.central_fdm(3,1)), params)
                @test gradfd[1:nvar,:] ≈ grad[1:nvar,:] atol=1e-4 rtol=1e-4
            end
        end
        
        @testset "Bound Violations Sum" begin
            function f_bound_viols(X)
                Vb = BOI.bound_violations!(bm, X)
                return sum(Vb)
            end
            
            grad = DI.gradient(f_bound_viols, AutoZygote(), X_cpu)
            @test grad isa AbstractMatrix
            @test size(grad) == size(X_cpu)
            @test all(isfinite, grad)
            @testset "FiniteDifferences" begin
                gradfd = DI.gradient(f_bound_viols, AutoFiniteDifferences(fdm=FiniteDifferences.central_fdm(3,1)), X_cpu)
                @test gradfd ≈ grad atol=1e-4 rtol=1e-4
            end
        end
    end
end

function test_violations_config_errors(MOD=OpenCL)
    model = create_luksan_vlcek_model(5; M = 1)
    batch_size = 2
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X = MOD.randn(nvar, batch_size)
    Θ = MOD.randn(nθ, batch_size)
    
    @testset "Config Errors" begin
        bm_no_viols = BOI.BatchModel(model, batch_size, config=BOI.BatchModelConfig(:minimal))
        @test_throws ArgumentError BOI.all_violations!(bm_no_viols, X, Θ)
        @test_throws ArgumentError BOI.bound_violations!(bm_no_viols, X)
        
        if ncon > 0
            V = MOD.randn(ncon, batch_size)
            @test_throws ArgumentError BOI.constraint_violations!(bm_no_viols, V)
        end
    end
end

@testset "Violations API - Luksan" begin
    @testset "Config Errors" begin
        test_violations_config_errors(OpenCL)
    end
    
    cpu_models, names = create_luksan_models(CPU())
    gpu_models, _ = create_luksan_models(OpenCLBackend())
    
    for (name, (cpu_model, gpu_model)) in zip(names, zip(cpu_models, gpu_models))
        @testset "$name Model" begin
            for batch_size in [1, 4]
                @testset "Batch Size $batch_size" begin
                    @testset "Correctness" begin
                        test_violations_correctness(gpu_model, batch_size, atol=1e-5, rtol=1e-5)
                    end
                    @testset "CPU Differentiability" begin
                        test_violations_differentiability_cpu(cpu_model, batch_size)
                    end
                    @testset "GPU Differentiability" begin
                        test_violations_differentiability_gpu(gpu_model, batch_size)
                    end
                end
            end
        end
    end
end

@testset "Violations API - Power" begin
    cpu_models_p, names_p = create_power_models(CPU())
    gpu_models_p, _       = create_power_models(OpenCLBackend())

    for (name, (cpu_model, gpu_model)) in zip(names_p, zip(cpu_models_p, gpu_models_p))
        @testset "$(name) Model" begin
            for batch_size in [1, 4]
                @testset "Batch Size $(batch_size)" begin
                    @testset "Correctness" begin
                        test_violations_correctness(gpu_model, batch_size; atol=1e-5, rtol=1e-5)
                    end
                    @testset "CPU Differentiability" begin
                        test_violations_differentiability_cpu(cpu_model, batch_size)
                    end
                    @testset "GPU Differentiability" begin
                        test_violations_differentiability_gpu(gpu_model, batch_size)
                    end
                end
            end
        end
    end
end 

if haskey(ENV, "BNK_TEST_CUDA")
    @testset "Violations API - CUDA" begin
        gpu_models_p, names_p = create_power_models(CUDABackend())
        for (name, gpu_model) in zip(names_p, gpu_models_p)
            @testset "$name Model" begin
                for batch_size in [1, 4]
                    @testset "Batch Size $batch_size" begin
                        test_violations_correctness(gpu_model, batch_size, atol=1e-5, rtol=1e-5, MOD=CUDA)
                        test_violations_differentiability_gpu(gpu_model, batch_size, MOD=CUDA)
                    end
                end
            end
        end
    end
end