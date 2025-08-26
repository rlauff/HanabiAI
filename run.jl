# run.jl (Modified to run a game with the latest AI model)

# Include the necessary logic and training modules
include("HanabiLogic.jl")
include("Training.jl")

# Bring the modules and BSON for loading into scope
using .HanabiLogic
using .Training
using BSON: @load
using Crayons

# This function finds the latest model, loads it, and runs a visualized game
function run_ai_game()
    println("Looking for the latest AI model snapshot...")

    # --- 1. Find the latest model snapshot ---
    SNAPSHOT_DIR = "snapshots"
    if !isdir(SNAPSHOT_DIR)
        println("Error: Snapshot directory '$(SNAPSHOT_DIR)' not found. Please train a model first.")
        return
    end

    latest_iter = -1
    latest_snapshot_path = ""
    model_files = readdir(SNAPSHOT_DIR)

    for file in model_files
        m = match(r"model_iter_(\d+).bson", file)
        if m !== nothing
            iter_num = parse(Int, m.captures[1])
            if iter_num > latest_iter
                latest_iter = iter_num
                latest_snapshot_path = joinpath(SNAPSHOT_DIR, file)
            end
        end
    end

    if latest_iter == -1
        println("Error: No valid model snapshots found in '$(SNAPSHOT_DIR)'.")
        return
    end

    # --- 2. Load the model ---
    println("Loading model from: $(latest_snapshot_path)")
    @load latest_snapshot_path model_cpu

    # --- 3. Set up and run the game ---
    players = [AIPlayer(model_cpu), AIPlayer(model_cpu)]
    gs = setup_game()

    println("\nStarting AI vs. AI game. Press Enter to advance each turn.")

    while !is_game_over(gs)
        # Render the current game state
        render(gs)

        p_idx = gs.current_player
        player = players[p_idx]
        player_view = get_player_view(gs, p_idx)

        # The AI runs MCTS to decide on the best move
        # We use a healthy number of simulations for a good quality move.
        println("Player $p_idx is thinking (running MCTS)...")
        policy = run_mcts(gs, model_cpu, simulations=100)
        action = get_action(player, player_view, policy=policy, deterministic=true)

        # Describe the chosen action and wait for user input
        println(crayon"bold yellow", "\nPlayer $p_idx will now ", describe_action(action), "...", crayon"reset")
        println("Press Enter to apply.")
        readline()

        apply_action!(gs, action)
    end

    # --- 4. Show the final game state ---
    println(crayon"bold cyan", "\nGAME OVER!", crayon"reset")
    render(gs)
    println(crayon"bold cyan", "Final Score: $(get_score(gs))", crayon"reset")
end

# Run the main function
run_ai_game()
