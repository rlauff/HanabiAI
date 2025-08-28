# Training.jl (With Discard Pile and History)

module Training

# Import game logic and action space mapping
using ..HanabiLogic
include("ActionSpace.jl")
using .ActionSpace

# Import required packages
using Flux
using CUDA
using MLUtils
using BSON: @save, @load
using DataStructures
using Random
using Statistics
using Base.Threads
using Crayons
using Printf
using Distributions: Categorical
using Dates

# Configuration constants
const NUM_ACTIONS = 20
const INPUT_CHANNELS = 29 # 14 (base) + 5 (discard) + 10 (history)
const BATCH_SIZE = 128
const TRAINING_ITERATIONS = 10000
const GAMES_PER_ITERATION = 200
const STEPS_PER_ITERATION = 500
const MCTS_SIMULATIONS = 200
const SNAPSHOT_DIR = "snapshots"
const MIN_BUFFER_FILL = BATCH_SIZE * 10
const NUM_SELF_PLAY_WORKERS = 2 # Number of games to play in parallel

#SECTION: Core Network and Data Structures
struct ResBlock
    conv1
    bn1
    conv2
    bn2
end

function ResBlock(channels::Int)
    ResBlock(Conv((3, 3), channels => channels, pad=1), BatchNorm(channels, relu), Conv((3, 3), channels => channels, pad=1), BatchNorm(channels))
end

(m::ResBlock)(x) = m.bn2(m.conv2(m.bn1(m.conv1(x)))) + x |> relu

struct HanabiNet
    base
    policy_head
    value_head
end
function create_model(input_channels::Int, num_actions::Int, res_blocks::Int=5, channels::Int=64)
    base = Chain(Conv((3, 3), input_channels => channels, pad=1), BatchNorm(channels, relu), [ResBlock(channels) for _ in 1:res_blocks]...)
    policy_head = Chain(Conv((1, 1), channels => 16), BatchNorm(16, relu), Flux.flatten, Dense(16 * 5 * 5, num_actions))
    value_head = Chain(Conv((1, 1), channels => 4), BatchNorm(4, relu), Flux.flatten, Dense(4 * 5 * 5, 32, relu), Dense(32, 1))
    return HanabiNet(base, policy_head, value_head)
end
(m::HanabiNet)(x) = (m.policy_head(m.base(x)), m.value_head(m.base(x)))
struct AIPlayer <: AbstractPlayer
    model::HanabiNet
end
export AIPlayer
struct TrainingSample
    state::Array{Float32, 3}
    policy_target::Vector{Float32}
    value_target::Float32
end
const REPLAY_BUFFER = CircularBuffer{TrainingSample}(100_000)
const BUFFER_LOCK = ReentrantLock()

function encode_state(player_view::PlayerView)::Array{Float32, 3}
    board = zeros(Float32, 5, 5, INPUT_CHANNELS)

    # My hand knowledge (Channels 1-10)
    for i in 1:player_view.my_hand_size
        if player_view.my_knowledge[i].is_color_known
            color = Int(log2(player_view.my_knowledge[i].possible_colors)) + 1
            board[i, 1, color] = 1.0f0
        end
        if player_view.my_knowledge[i].is_rank_known
            rank = Int(log2(player_view.my_knowledge[i].possible_ranks)) + 1
            board[i, 2, 5 + rank] = 1.0f0
        end
    end
    # Partner's hand (Channels 11-12)
    partner_hand = player_view.other_player_hands[1]
    for (i, card_id) in enumerate(partner_hand)
        card = CANONICAL_DECK[card_id + 1]
        board[i, 3, 11] = card.color / 5.0f0
        board[i, 4, 12] = card.rank / 5.0f0
    end
    # Fireworks (Channel 13)
    for c in 1:NUM_COLORS
        board[c, 5, 13] = player_view.fireworks[c] / 5.0f0
    end
    # Game state scalars (Channel 14)
    board[1, 5, 14] = player_view.clue_tokens / CLUE_TOKENS_MAX
    board[2, 5, 14] = player_view.error_tokens / ERROR_TOKENS_MAX
    board[3, 5, 14] = player_view.deck_size / 50.0f0

    # Discard Pile (Channels 15-19)
    discard_counts = zeros(Int, 5, 5) # rank, color
    for card_id in player_view.discard_pile
        card = CANONICAL_DECK[card_id + 1]
        discard_counts[card.rank, card.color] += 1
    end
    for color in 1:NUM_COLORS
        for rank in 1:5
            board[rank, 1, 14 + color] = discard_counts[rank, color] / RANKS[rank]
        end
    end

    # Last 10 Moves (Channels 20-29)
    last_10_actions = last(player_view.history, 10)
    for (i, action) in enumerate(last_10_actions)
        channel_idx = 19 + i
        action_idx = action_to_index(action)
        if action_idx > 0
            row = (action_idx - 1) % 5 + 1
            col = (action_idx - 1) ÷ 5 + 1
            board[row, col, channel_idx] = 1.0f0
        end
    end

    return board
end

#SECTION: MCTS Implementation
mutable struct MCTSNode
    parent::Union{MCTSNode, Nothing}
    children::Dict{Action, MCTSNode}
    action::Union{Action, Nothing}
    visit_count::Int
    total_action_value::Float64
    prior_prob::Float32
    MCTSNode(parent, action, prior) = new(parent, Dict(), action, 0, 0.0, prior)
end
function puct_value(node::MCTSNode, cpuct::Float64)
    parent_visits = node.parent === nothing ? node.visit_count : node.parent.visit_count
    q_value = node.visit_count == 0 ? 0.0 : node.total_action_value / node.visit_count
    u_value = cpuct * node.prior_prob * sqrt(parent_visits) / (1 + node.visit_count)
    return q_value + u_value
end
function select_child(node::MCTSNode, cpuct::Float64)
    best_child = nothing
    max_val = -Inf
    for child_node in values(node.children)
        val = puct_value(child_node, cpuct)
        if val > max_val
            max_val = val
            best_child = child_node
        end
    end
    return best_child
end
function expand!(node::MCTSNode, gs::GameState, model::HanabiNet)
    p_idx = gs.current_player
    player_view = get_player_view(gs, p_idx)
    legal_actions = get_legal_actions(player_view)
    if isempty(legal_actions) return end
    device = Flux.get_device(model)
    state_tensor = encode_state(player_view) |> device |> x -> unsqueeze(x, 4)
    policy_logits, _ = model(state_tensor)
    probabilities = softmax(policy_logits) |> cpu
    for action in legal_actions
        if !haskey(node.children, action)
            idx = action_to_index(action)
            prior = probabilities[idx]
            node.children[action] = MCTSNode(node, action, prior)
        end
    end
end
function backpropagate!(node::MCTSNode, value::Float64)
    while node !== nothing
        node.visit_count += 1
        node.total_action_value += value
        node = node.parent
    end
end
function run_mcts(gs::GameState, model::HanabiNet; simulations::Int=100, cpuct::Float64=1.0)
    root = MCTSNode(nothing, nothing, 0.0f0)
    expand!(root, gs, model)

    for _ in 1:simulations
        current_node = root
        sim_gs = deepcopy(gs)

        while !isempty(current_node.children)
            current_node = select_child(current_node, cpuct)
            apply_action!(sim_gs, current_node.action)
        end

        value = 0.0

        if !is_game_over(sim_gs)
            expand!(current_node, sim_gs, model)
            device = Flux.get_device(model)
            player_view = get_player_view(sim_gs, sim_gs.current_player)
            state_tensor = encode_state(player_view) |> device |> x -> unsqueeze(x, 4)
            _ , value_tensor = model(state_tensor)
            value = Float64((value_tensor |> cpu)[1])
        else
            value = Float64(get_score(sim_gs)) / 25.0
        end

        backpropagate!(current_node, value)
    end
    
    policy = zeros(Float32, NUM_ACTIONS)
    if root.visit_count > 0
        for (action, child) in root.children
            idx = action_to_index(action)
            policy[idx] = child.visit_count / root.visit_count
        end
    end
    return policy
end

#SECTION: Core RL Loop functions
function get_action(player::AIPlayer, player_view::PlayerView; policy::Vector{Float32}, deterministic=false)::Action
    if sum(policy) < 1e-6
        legal_actions = get_legal_actions(player_view)
        return isempty(legal_actions) ? Action(:discard, 1, 0, :_, 0) : rand(legal_actions)
    end

    if deterministic
        action_idx = argmax(policy)
    else
        action_idx = rand(Categorical(policy))
    end

    action = index_to_action(action_idx)
    if action.type == :clue
        target_player = (player_view.player_idx % NUM_PLAYERS) + 1
        return Action(action.type, action.card_index, target_player, action.clue_type, action.value)
    end
    return action
end

function self_play_worker(model::HanabiNet, games_to_play::Int)
    players = [AIPlayer(model), AIPlayer(model)]
    total_score = 0.0
    for _ in 1:games_to_play
        game_history = []
        gs = setup_game()
        while !is_game_over(gs)
            p_idx = gs.current_player
            player_view = get_player_view(gs, p_idx)
            policy = run_mcts(gs, model, simulations=MCTS_SIMULATIONS)
            
            state_tensor = encode_state(player_view)
            push!(game_history, (state_tensor, policy))
            action = get_action(players[p_idx], player_view, policy=policy, deterministic=false)
            apply_action!(gs, action)
        end
        
        final_score = Float32(get_score(gs))
        normalized_score = final_score / 25.0f0 
        total_score += final_score

        lock(BUFFER_LOCK) do
            for (state, policy) in game_history
                push!(REPLAY_BUFFER, TrainingSample(state, policy, normalized_score))
            end
        end
    end
    return total_score / games_to_play
end
function loss(model, state, policy_target, value_target)
    policy_logits, value_pred = model(state)
    value_loss = Flux.mse(value_pred, value_target)
    policy_loss = Flux.logitcrossentropy(policy_logits, policy_target)
    return value_loss + policy_loss
end

function training_worker(model::HanabiNet, opt, training_steps::Int, batch_size::Int)
    if length(REPLAY_BUFFER) < MIN_BUFFER_FILL return model, -1.0f0 end
    final_loss = 0.0f0
    for step in 1:training_steps
         batch = lock(BUFFER_LOCK) do
            rand(REPLAY_BUFFER, batch_size)
        end
        states = cat([s.state for s in batch]...; dims=4) |> gpu
        
        # <<< FIXED LINE >>>
        # Removed the unsupported 'dims=2' keyword from the hcat call.
        policy_targets = hcat([s.policy_target for s in batch]...) |> gpu

        value_targets = reshape([s.value_target for s in batch], 1, :) |> gpu
        grads = gradient(m -> loss(m, states, policy_targets, value_targets), model)
        Flux.update!(opt, model, grads[1])
        if step == training_steps
            final_loss = loss(model, states, policy_targets, value_targets) |> cpu
        end
    end
    return model, final_loss
end

function evaluate_model(model::HanabiNet, num_games::Int=100)
    model_cpu = model |> cpu
    players = [AIPlayer(model_cpu), AIPlayer(model_cpu)]
    total_score = 0.0
    for _ in 1:num_games
        gs = setup_game()
        while !is_game_over(gs)
            p_idx = gs.current_player
            player_view = get_player_view(gs, p_idx)
            policy = run_mcts(gs, model_cpu, simulations=20)
            action = get_action(players[p_idx], player_view, policy=policy, deterministic=true)
            apply_action!(gs, action)
        end
        total_score += get_score(gs)
    end
    return total_score / num_games
end

function run_training_loop()
    mkpath(SNAPSHOT_DIR)

    latest_iter = 0
    latest_snapshot_path = ""
    model = nothing

    # Find the latest model snapshot
    try
        files = readdir(SNAPSHOT_DIR)
        model_files = filter(f -> occursin(r"model_iter_\d+\.bson", f), files)
        if !isempty(model_files)
            latest_iter = maximum(f -> parse(Int, match(r"model_iter_(\d+)\.bson", f).captures[1]), model_files)
            latest_snapshot_path = joinpath(SNAPSHOT_DIR, "model_iter_$(latest_iter).bson")
        end
    catch e
        println("Could not read snapshot directory: $e")
    end

    start_iter = 1
    if latest_iter > 0
        println(crayon"cyan", "Resuming training from snapshot: $(latest_snapshot_path)", crayon"reset")
        @load latest_snapshot_path model_cpu
        model = model_cpu |> gpu
        start_iter = latest_iter + 1

        # Load the corresponding replay buffer
        latest_buffer_path = joinpath(SNAPSHOT_DIR, "buffer_iter_$(latest_iter).bson")
        if isfile(latest_buffer_path)
            println(crayon"cyan", "Loading replay buffer from: $(latest_buffer_path)", crayon"reset")
            @load latest_buffer_path buffer_copy
            lock(BUFFER_LOCK) do
                empty!(REPLAY_BUFFER)
                for sample in buffer_copy
                    push!(REPLAY_BUFFER, sample)
                end
            end
            println(crayon"green", "Buffer successfully loaded with $(length(REPLAY_BUFFER)) samples.", crayon"reset")
        else
            println(crayon"yellow", "No buffer snapshot found. Starting with an empty buffer.", crayon"reset")
        end
    else
        println(crayon"cyan", "No snapshots found. Starting a new training session.", crayon"reset")
        model = create_model(INPUT_CHANNELS, NUM_ACTIONS) |> gpu
    end
    
    opt = Flux.setup(Adam(1e-4), model)

    @printf("%-5s | %-15s | %-12s | %-15s | %-15s\n", "Iter", "Avg SP Score", "Final Loss", "Avg Eval Score", "Buffer Size")
    println("-"^70)
    
    for iter in start_iter:TRAINING_ITERATIONS
        # <<< MODIFIED SECTION START >>>
        # Create a CPU copy of the latest model for the self-play workers.
        model_cpu_for_sp = model |> cpu
        
        # Distribute the games across multiple parallel workers
        games_per_worker = GAMES_PER_ITERATION ÷ NUM_SELF_PLAY_WORKERS
        
        # Spawn workers, giving each one the CPU-based model.
        tasks = [Threads.@spawn self_play_worker(model_cpu_for_sp, games_per_worker) for _ in 1:NUM_SELF_PLAY_WORKERS]
        
        # Wait for all tasks to finish and collect their average scores
        worker_scores = fetch.(tasks)
        avg_sp_score = mean(worker_scores)
        # <<< MODIFIED SECTION END >>>

        model, final_loss = training_worker(model, opt, STEPS_PER_ITERATION, BATCH_SIZE)
        
        # ... (the rest of the loop is the same) ...
    end
    println("\nTraining complete.")
end

end
