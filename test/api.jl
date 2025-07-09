function test_batch_model(model::ExaModel, batch_size::Int; 
                                   atol::Float64=1e-10, rtol::Float64=1e-10, MOD=OpenCL)
    
    bm = BNK.BatchModel(model, batch_size, config=BNK.BatchModelConfig(:full))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nnzh = model.meta.nnzh
    nnzj = model.meta.nnzj
    nθ = length(model.θ)
    
    X = MOD.randn(nvar, batch_size)
    Θ = MOD.randn(nθ, batch_size)
    
    @testset "Model Info: $(nvar) vars, $(ncon) cons, $(nθ) params" begin
        @testset "Objective" begin
            obj_vals = BNK.objective!(bm, X, Θ)
            @test length(obj_vals) == batch_size
            @test all(isfinite, obj_vals)
            for i in 1:batch_size
                @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                @allowscalar @test obj_vals[i] ≈ ExaModels.obj(model, X[:, i]) atol=atol rtol=rtol
            end
        end
        
        @testset "Constraint" begin
            if ncon > 0
                cons_vals = BNK.constraints!(bm, X, Θ)
                @test size(cons_vals) == (ncon, batch_size)
                @test all(isfinite, cons_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    cons_single = similar(cons_vals, ncon)
                    @allowscalar ExaModels.cons_nln!(model, X[:, i], cons_single)
                    @allowscalar @test cons_vals[:, i] ≈ cons_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Gradient" begin
            grad_vals = BNK.objective_gradient!(bm, X, Θ)
            @test size(grad_vals) == (nvar, batch_size)
            @test all(isfinite, grad_vals)
            for i in 1:batch_size
                @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                grad_single = similar(grad_vals, nvar)
                @allowscalar ExaModels.grad!(model, X[:, i], grad_single)
                @allowscalar @test grad_vals[:, i] ≈ grad_single atol=atol rtol=rtol
            end
        end

        @testset "Jacobian" begin
            if ncon > 0
                jac_vals = BNK.constraints_jacobian!(bm, X, Θ)
                @test size(jac_vals) == (nnzj, batch_size)
                @test all(isfinite, jac_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    jac_single = similar(jac_vals, nnzj)
                    @allowscalar ExaModels.jac_coord!(model, X[:, i], jac_single)
                    @allowscalar @test jac_vals[:, i] ≈ jac_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Jacobian-Vector Product" begin
            if ncon > 0
                V = MOD.randn(nvar, batch_size)
                jprod_vals = BNK.constraints_jprod!(bm, X, Θ, V)
                @test size(jprod_vals) == (ncon, batch_size)
                @test all(isfinite, jprod_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    jprod_single = similar(jprod_vals, ncon)
                    @allowscalar ExaModels.jprod_nln!(model, X[:, i], V[:, i], jprod_single)
                    @allowscalar @test jprod_vals[:, i] ≈ jprod_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Jacobian-Transpose-Vector Product" begin
            if ncon > 0
                V = MOD.randn(ncon, batch_size)
                jtprod_vals = BNK.constraints_jtprod!(bm, X, Θ, V)
                @test size(jtprod_vals) == (nvar, batch_size)
                @test all(isfinite, jtprod_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    jtprod_single = similar(jtprod_vals, nvar)
                    @allowscalar ExaModels.jtprod_nln!(model, X[:, i], V[:, i], jtprod_single)
                    @allowscalar @test jtprod_vals[:, i] ≈ jtprod_single atol=atol rtol=rtol
                end
            end
        end

        @testset "Hessian" begin
            if ncon > 0
                Y = MOD.randn(ncon, batch_size)
                hess_vals = BNK.lagrangian_hessian!(bm, X, Θ, Y)
                @test size(hess_vals) == (nnzh, batch_size)
                @test all(isfinite, hess_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    hess_single = similar(hess_vals, nnzh)
                    @allowscalar ExaModels.hess_coord!(model, X[:, i], Y[:, i], hess_single)
                    @allowscalar @test hess_vals[:, i] ≈ hess_single atol=atol rtol=rtol
                end
            end
        end
        @testset "Hessian-Vector Product" begin
            V = MOD.randn(nvar, batch_size)
            if ncon > 0
                Y = MOD.randn(ncon, batch_size)
                hprod_vals = BNK.lagrangian_hprod!(bm, X, Θ, Y, V)
                @test size(hprod_vals) == (nvar, batch_size)
                @test all(isfinite, hprod_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    hprod_single = similar(hprod_vals, nvar)
                    @allowscalar ExaModels.hprod!(model, X[:, i], Y[:, i], V[:, i], hprod_single)
                    @allowscalar @test hprod_vals[:, i] ≈ hprod_single atol=atol rtol=rtol
                end
            else
                Y = MOD.zeros(ncon, batch_size)
                hprod_vals = BNK.lagrangian_hprod!(bm, X, Θ, Y, V)
                @test size(hprod_vals) == (nvar, batch_size)
                @test all(isfinite, hprod_vals)
                for i in 1:batch_size
                    @allowscalar nθ > 0 && (model.θ .= Θ[:, i])
                    hprod_single = similar(hprod_vals, nvar)
                    @allowscalar ExaModels.hprod!(model, X[:, i], Y[:, i], V[:, i], hprod_single)
                    @allowscalar @test hprod_vals[:, i] ≈ hprod_single atol=atol rtol=rtol
                end
            end
        end
        
        @testset "Batch Size Validation" begin
            X_large = MOD.randn(nvar, batch_size + 1)
            @test_throws AssertionError BNK.objective!(bm, X_large)
            
            if ncon > 0
                @test_throws AssertionError BNK.constraints!(bm, X_large)
            end
            
            @test_throws AssertionError BNK.objective_gradient!(bm, X_large)
            
            if ncon > 0
                V_jprod = MOD.randn(nvar, batch_size + 1)
                @test_throws AssertionError BNK.constraints_jprod!(bm, X_large, V_jprod)
                
                V_jtprod = MOD.randn(ncon, batch_size + 1)
                @test_throws AssertionError BNK.constraints_jtprod!(bm, X_large, V_jtprod)
            end
            
            V_hprod = MOD.randn(nvar, batch_size + 1)
            if ncon > 0
                Y_large = MOD.randn(ncon, batch_size + 1)
                @test_throws AssertionError BNK.lagrangian_hprod!(bm, X_large, Y_large, V_hprod)
            else
                Y_large = MOD.zeros(ncon, batch_size + 1)
                @test_throws AssertionError BNK.lagrangian_hprod!(bm, X_large, Y_large, V_hprod)
            end
        end
        
        @testset "Dimension Validation" begin
            X_wrong = MOD.randn(nvar + 1, batch_size)
            @test_throws DimensionMismatch BNK.objective!(bm, X_wrong)

            if nθ > 0
                Θ_wrong = MOD.randn(nθ + 1, batch_size)
                @test_throws DimensionMismatch BNK.objective!(bm, X, Θ_wrong)
            end
            
            if ncon > 0
                V_jprod_wrong = MOD.randn(nvar + 1, batch_size)
                @test_throws DimensionMismatch BNK.constraints_jprod!(bm, X, V_jprod_wrong)
                
                V_jtprod_wrong = MOD.randn(ncon + 1, batch_size)
                @test_throws DimensionMismatch BNK.constraints_jtprod!(bm, X, V_jtprod_wrong)
                
                Y_wrong = MOD.randn(ncon + 1, batch_size)
                V_hprod = MOD.randn(nvar, batch_size)
                @test_throws DimensionMismatch BNK.lagrangian_hprod!(bm, X, Y_wrong, V_hprod)
            end

            V_hprod_wrong = MOD.randn(nvar + 1, batch_size)
            if ncon > 0
                Y = MOD.randn(ncon, batch_size)
                @test_throws DimensionMismatch BNK.lagrangian_hprod!(bm, X, Y, V_hprod_wrong)
            else
                Y = MOD.zeros(ncon, batch_size)
                @test_throws DimensionMismatch BNK.lagrangian_hprod!(bm, X, Y, V_hprod_wrong)
            end
        end
    end
end

@testset "API - Luksan - OpenCL" begin
    models, names = create_luksan_models(OpenCLBackend())
    
    for (name, model) in zip(names, models)
        @testset "$name Model" begin
            for batch_size in [1, 2, 4]
                @testset "Batch Size $batch_size" begin
                    test_batch_model(model, batch_size, atol=1e-5, rtol=1e-5, MOD=OpenCL)
                end
            end
        end
    end
end

@testset "API - Power - OpenCL" begin
    models_p, names_p = create_power_models(OpenCLBackend())

    for (name, model) in zip(names_p, models_p)
        @testset "$(name) Model" begin
            for batch_size in [1, 2, 4]
                @testset "Batch Size $(batch_size)" begin
                    test_batch_model(model, batch_size; atol=1e-5, rtol=1e-5, MOD=OpenCL)
                end
            end
        end
    end
end


if haskey(ENV, "BNK_TEST_CUDA")
    @testset "API - Luksan - CUDA" begin
        models_c, names = create_luksan_models(CUDABackend())
        
        for (name, model) in zip(names, models_c)
            @testset "$name Model" begin
                for batch_size in [1, 2, 4]
                    @testset "Batch Size $batch_size" begin
                        test_batch_model(model, batch_size, atol=1e-5, rtol=1e-5, MOD=CUDA)
                    end
                end
            end
        end
    end

    @testset "API - Power - CUDA" begin
        models_pc, names_p = create_power_models(CUDABackend())

        for (name, model) in zip(names_p, models_pc)
            @testset "$(name) Model" begin
                for batch_size in [1, 2, 4]
                    @testset "Batch Size $(batch_size)" begin
                        test_batch_model(model, batch_size; atol=1e-5, rtol=1e-5, MOD=CUDA)
                    end
                end
            end
        end
    end
end