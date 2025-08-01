using BatchNLPKernels
using Test
using ExaModels
using KernelAbstractions
using DifferentiationInterface
const DI = DifferentiationInterface

import Zygote
import FiniteDifferences

using PowerModels
PowerModels.silence()
using PGLib
using LinearAlgebra

using OpenCL, pocl_jll, AcceleratedKernels

using Lux
using LuxCUDA
using Lux.Training
using MLUtils
using Optimisers
using CUDA
using Random
import GPUArraysCore: @allowscalar


if haskey(ENV, "BNK_TEST_CUDA")
    @info "CUDA detected"
end

@testset "Empty Intervals" begin
    empt = BNK.Interval()
    @test isempty(empt)
    @test isnothing(empt.l)
    @test isnothing(empt.u)
    @test BNK._violation([1.0, 2.0, 3.0], empt) == [0.0, 0.0, 0.0]
end

include("luksan.jl")
include("power.jl")
include("test_viols.jl")
include("test_diff.jl")
include("api.jl")
include("config.jl")
include("test_penalty.jl")