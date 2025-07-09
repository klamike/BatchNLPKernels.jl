function test_diff_gpu(model::ExaModel, batch_size::Int; MOD=OpenCL)
    bm = BNK.BatchModel(model, batch_size, config=BNK.BatchModelConfig(:full))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X_gpu = MOD.randn(nvar, batch_size)
    Θ_gpu = MOD.randn(nθ, batch_size)

    @testset "objective!" begin
        y = BNK.objective!(bm, X_gpu, Θ_gpu)
        @test size(y) == (batch_size,)

        function f_gpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BNK.objective!(bm, X, Θ))
        end
        
        params = vcat(X_gpu, Θ_gpu)
        grad = DI.gradient(f_gpu, AutoZygote(), params)
        @test grad isa AbstractMatrix
        @test size(grad) == size(params)
    end
    
    ncon == 0 && return

    @testset "constraints!" begin
        y = BNK.constraints!(bm, X_gpu, Θ_gpu)
        @test size(y) == (ncon, batch_size)

        function f_gpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BNK.constraints!(bm, X, Θ))
        end
        
        params = vcat(X_gpu, Θ_gpu)
        grad = DI.gradient(f_gpu, AutoZygote(), params)
        @test grad isa AbstractMatrix
        @test size(grad) == size(params)
    end
end

function test_diff_cpu(model::ExaModel, batch_size::Int)
    bm = BNK.BatchModel(model, batch_size, config=BNK.BatchModelConfig(:full))
    
    nvar = model.meta.nvar
    ncon = model.meta.ncon
    nθ = length(model.θ)
    
    X_cpu = randn(nvar, batch_size)
    Θ_cpu = randn(nθ, batch_size)
    
    @testset "objective! CPU" begin
        y = BNK.objective!(bm, X_cpu, Θ_cpu)
        @test size(y) == (batch_size,)

        function f_cpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BNK.objective!(bm, X, Θ))
        end
        
        params = vcat(X_cpu, Θ_cpu)
        grad = DI.gradient(f_cpu, AutoZygote(), params)
        @test grad isa AbstractMatrix
        @test size(grad) == size(params)

        @testset "FiniteDifferences objective!" begin
            gradfd = DI.gradient(f_cpu, AutoFiniteDifferences(fdm=FiniteDifferences.central_fdm(3,1)), params)
            @test gradfd[1:nvar,:] ≈ grad[1:nvar,:] atol=1e-4 rtol=1e-4
        end
    end

    ncon == 0 && return
    
    @testset "constraints! CPU" begin
        y = BNK.constraints!(bm, X_cpu, Θ_cpu)
        @test size(y) == (ncon, batch_size)

        function f_cpu(params)
            X = params[1:nvar, :]
            Θ = params[nvar+1:end, :]
            return sum(BNK.constraints!(bm, X, Θ))
        end
        
        params = vcat(X_cpu, Θ_cpu)
        grad = DI.gradient(f_cpu, AutoZygote(), params)
        @test grad isa AbstractMatrix
        @test size(grad) == size(params)

        @testset "FiniteDifferences constraints!" begin
            gradfd = DI.gradient(f_cpu, AutoFiniteDifferences(fdm=FiniteDifferences.central_fdm(3,1)), params)
            @test gradfd[1:nvar,:] ≈ grad[1:nvar,:] atol=1e-4 rtol=1e-4
        end
    end
end


@testset "AD rules - Luksan" begin
    cpu_models, names = create_luksan_models(CPU())
    gpu_models, _ = create_luksan_models(OpenCLBackend())
    
    for (name, (cpu_model, gpu_model)) in zip(names, zip(cpu_models, gpu_models))
        @testset "$name Model" begin
            for batch_size in [1, 4]
                @testset "Batch Size $batch_size" begin
                    @testset "CPU Diff" begin
                        test_diff_cpu(cpu_model, batch_size)
                    end
                    @testset "GPU Diff" begin
                        test_diff_gpu(gpu_model, batch_size)
                    end
                end
            end
        end
    end
end

@testset "AD rules - Power" begin
    cpu_models_p, names_p = create_power_models(CPU())
    gpu_models_p, _       = create_power_models(OpenCLBackend())

    for (name, (cpu_model, gpu_model)) in zip(names_p, zip(cpu_models_p, gpu_models_p))
        @testset "$(name) Model" begin
            for batch_size in [1, 4]
                @testset "Batch Size $(batch_size)" begin
                    @testset "CPU Diff" begin
                        test_diff_cpu(cpu_model, batch_size)
                    end
                    @testset "OpenCL Diff" begin
                        test_diff_gpu(gpu_model, batch_size)
                    end
                end
            end
        end
    end
end


if haskey(ENV, "BNK_TEST_CUDA")
    @testset "AD rules - Luksan - CUDA" begin
        gpu_models, names = create_luksan_models(CUDABackend())
        
        for (name, gpu_model) in zip(names, gpu_models)
            @testset "$name Model" begin
                for batch_size in [1, 4]
                    @testset "Batch Size $batch_size" begin
                        @testset "GPU Diff" begin
                            test_diff_gpu(gpu_model, batch_size, MOD=CUDA)
                        end
                    end
                end
            end
        end
    end
    
    @testset "AD rules - Power - CUDA" begin
        gpu_models_p, names_p = create_power_models(CUDABackend())
    
        for (name, gpu_model) in zip(names_p, gpu_models_p)
            @testset "$(name) Model" begin
                for batch_size in [1, 4]
                    @testset "Batch Size $(batch_size)" begin
                        @testset "OpenCL Diff" begin
                            test_diff_gpu(gpu_model, batch_size, MOD=CUDA)
                        end
                    end
                end
            end
        end
    end
end