# start_training.jl
include("HanabiLogic.jl")
include("Training.jl") # This will now also include ActionSpace.jl
Training.run_training_loop()
