using Lux
using LuxCUDA
using Lux.Training
using MLUtils
using Optimisers
using CUDA
using Random

dev_gpu = gpu_device()

function feed_forward_builder(
    num_p::Integer,
    num_y::Integer,
    hidden_layers::AbstractVector{<:Integer};
    activation = relu,
)
    """
    Builds a Chain of Dense layers with Lux
    """
    # Combine all layers: input size, hidden sizes, output size
    layer_sizes = [num_p; hidden_layers; num_y]
    
    # Build up a list of Dense layers
    dense_layers = Any[]
    for i in 1:(length(layer_sizes)-1)
        if i < length(layer_sizes) - 1
            # Hidden layers with activation
            push!(dense_layers, Dense(layer_sizes[i], layer_sizes[i+1], activation))
        else
            # Final layer with no activation
            push!(dense_layers, Dense(layer_sizes[i], layer_sizes[i+1]))
        end
    end
    
    return Chain(dense_layers...)
end
# backend = CUDABackend()
batch_size = 32
model = create_parametric_ac_power_model("pglib_opf_case14_ieee.m"; backend = CPU())
bm = BNK.BatchModel(model, batch_size, config=BNK.BatchModelConfig(:full))

function PenaltyLoss(model, ps, st, Θ)
    X̂ , st_new = model(Θ, ps, st)

    obj = BNK.objective!(bm, X̂, Θ)
    Vc, Vb = BNK.all_violations!(bm, X̂, Θ)

    return sum(obj) + 1000 * sum(Vc) + 1000 * sum(Vb), st_new, (;obj=sum(obj), Vc=sum(Vc), Vb=sum(Vb))
end

MOD = CUDA
dataset_size = 3200
rng = Random.default_rng()

nvar = model.meta.nvar
ncon = model.meta.ncon
nθ = length(model.θ)

# X̂ = MOD.randn(nvar, dataset_size)
# Θ_train = randn(nθ, dataset_size)
Θ_train = MOD.randn(nθ, dataset_size)

lux_model = feed_forward_builder(nθ, nvar, [320, 320])

ps_model, st_model = Lux.setup(rng, lux_model)
X̂ , _ = lux_model(Θ_train, ps_model, st_model)

y = BNK.objective!(bm, X̂[:,1:32], Θ_train[:,1:32])
Vc, Vb = BNK.all_violations!(bm, X̂[:,1:32], Θ_train[:,1:32])

train_state = Training.TrainState(lux_model, ps_model, st_model, Optimisers.Adam(1e-5))

data = DataLoader((Θ_train); batchsize=batch_size, shuffle=true) .|> dev_gpu
for (Θ) in data
    _, loss_val, stats, train_state = Training.single_train_step!(
        AutoZygote(),          # AD backend
        PenaltyLoss,
        (Θ),  # data
        train_state
    )
end
