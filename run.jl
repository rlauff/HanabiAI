# run.jl (FIXED)

include("HanabiLogic.jl")
using .HanabiLogic

# --- Run a single game with visualization ---
println("Starting a single visualized game with Random Players...")
players = [RandomPlayer(), RandomPlayer()]
simulate_game(players, render_game=true)


# --- Run multiple games to test performance and get average score ---
println("\nRunning 10,000 simulations to check performance...")
num_simulations = 10000
total_score = 0
@time begin
    for _ in 1:num_simulations
        # (FIXED) Use 'global' to modify the variable outside the loop's scope
        global total_score += simulate_game(players)
    end
end
println("Average score for RandomPlayer over $num_simulations games: $(total_score / num_simulations)")
