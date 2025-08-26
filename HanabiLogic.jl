# HanabiLogic.jl (v6 - Observer View)

module HanabiLogic

using Crayons
using Random

export GameState, PlayerView, AbstractPlayer, RandomPlayer, simulate_game, render, Action, setup_game, is_game_over, get_player_view, get_legal_actions, CANONICAL_DECK, NUM_COLORS, CLUE_TOKENS_MAX, ERROR_TOKENS_MAX, apply_action!, get_score, NUM_PLAYERS, describe_action, RANKS

#SECTION: Core Types and Constants

const NUM_PLAYERS = 2
const HAND_SIZE = 5
const NUM_COLORS = 5
const COLORS = ["Red", "Green", "Blue", "White", "Yellow"]
const COLOR_CRAYONS = [crayon"red", crayon"green", crayon"blue", crayon"white", crayon"yellow"]
const RANKS = [3, 2, 2, 2, 1]
const CARD_IDS = 0:49
const CLUE_TOKENS_MAX = 8
const ERROR_TOKENS_MAX = 3

struct Card
    id::Int
    color::Int
    rank::Int
end

function create_deck()
    cards = Vector{Card}(undef, 50)
    id = 0
    for color in 1:NUM_COLORS, rank in 1:5, _ in 1:RANKS[rank]
        cards[id + 1] = Card(id, color, rank)
        id += 1
    end
    return cards
end

const CANONICAL_DECK = create_deck()

mutable struct CardKnowledge
    possible_colors::UInt8
    possible_ranks::UInt8
    is_color_known::Bool
    is_rank_known::Bool
    CardKnowledge() = new(0b11111, 0b11111, false, false)
end

abstract type AbstractPlayer end

struct Action
    type::Symbol
    card_index::Int
    target_player::Int
    clue_type::Symbol
    value::Int
end

mutable struct GameState
    deck::Vector{Int}
    hands::Vector{Vector{Int}}
    player_knowledge::Vector{Vector{CardKnowledge}}
    fireworks::Vector{Int}
    discard_pile::Vector{Int}
    clue_tokens::Int
    error_tokens::Int
    current_player::Int
    turns_remaining::Int
    history::Vector{Action}

    GameState() = new(
        collect(CARD_IDS),
        [Int[] for _ in 1:NUM_PLAYERS],
        [[CardKnowledge() for _ in 1:HAND_SIZE] for _ in 1:NUM_PLAYERS],
        zeros(Int, NUM_COLORS),
        Int[],
        CLUE_TOKENS_MAX, 0, 1, -1,
        Action[]
    )
end

struct PlayerView
    player_idx::Int
    clue_tokens::Int
    error_tokens::Int
    deck_size::Int
    fireworks::Vector{Int}
    discard_pile::Vector{Int}
    my_hand_size::Int
    my_knowledge::Vector{CardKnowledge}
    other_player_hands::Vector{Vector{Int}}
    history::Vector{Action}
end

function get_player_view(gs::GameState, player_idx::Int)::PlayerView
    other_player_indices = filter(i -> i != player_idx, 1:NUM_PLAYERS)
    
    return PlayerView(
        player_idx,
        gs.clue_tokens,
        gs.error_tokens,
        length(gs.deck),
        gs.fireworks,
        gs.discard_pile,
        length(gs.hands[player_idx]),
        gs.player_knowledge[player_idx],
        [gs.hands[i] for i in other_player_indices],
        gs.history
    )
end

function setup_game()::GameState
    gs = GameState()
    shuffle!(gs.deck)
    for p in 1:NUM_PLAYERS, _ in 1:HAND_SIZE
        push!(gs.hands[p], pop!(gs.deck))
    end
    return gs
end

function is_game_over(gs::GameState)::Bool
    return gs.error_tokens >= ERROR_TOKENS_MAX ||
           all(gs.fireworks .== 5) ||
           gs.turns_remaining == 0
end

function get_score(gs::GameState)::Int
    return sum(gs.fireworks)
end

function draw_card!(gs::GameState, player_idx::Int, hand_idx::Int)
    if !isempty(gs.deck)
        new_card_id = pop!(gs.deck)
        gs.hands[player_idx][hand_idx] = new_card_id
        gs.player_knowledge[player_idx][hand_idx] = CardKnowledge()
        if isempty(gs.deck)
            gs.turns_remaining = NUM_PLAYERS
        end
    else
        deleteat!(gs.hands[player_idx], hand_idx)
        deleteat!(gs.player_knowledge[player_idx], hand_idx)
    end
end

function apply_action!(gs::GameState, action::Action)
    push!(gs.history, action)
    p_idx = gs.current_player
    if action.type == :play
        card_id = gs.hands[p_idx][action.card_index]
        card = CANONICAL_DECK[card_id + 1]
        if gs.fireworks[card.color] == card.rank - 1
            gs.fireworks[card.color] += 1
            if card.rank == 5 && gs.clue_tokens < CLUE_TOKENS_MAX
                gs.clue_tokens += 1
            end
        else
            gs.error_tokens += 1
            push!(gs.discard_pile, card_id)
        end
        draw_card!(gs, p_idx, action.card_index)
    elseif action.type == :discard
        if gs.clue_tokens < CLUE_TOKENS_MAX
            gs.clue_tokens += 1
        end
        card_id = gs.hands[p_idx][action.card_index]
        push!(gs.discard_pile, card_id)
        draw_card!(gs, p_idx, action.card_index)
    elseif action.type == :clue
        gs.clue_tokens -= 1
        target_hand = gs.hands[action.target_player]
        target_knowledge = gs.player_knowledge[action.target_player]
        for (i, card_id) in enumerate(target_hand)
            card = CANONICAL_DECK[card_id + 1]
            if action.clue_type == :color && card.color == action.value
                target_knowledge[i].is_color_known = true
                target_knowledge[i].possible_colors = 1 << (card.color - 1)
            elseif action.clue_type == :rank && card.rank == action.value
                target_knowledge[i].is_rank_known = true
                target_knowledge[i].possible_ranks = 1 << (card.rank - 1)
            end
        end
    end
    
    gs.current_player = (gs.current_player % NUM_PLAYERS) + 1
    if gs.turns_remaining > 0
        gs.turns_remaining -= 1
    end
end

function get_legal_actions(view::PlayerView)
    actions = Action[]
    if view.my_hand_size > 0
        for i in 1:view.my_hand_size
            push!(actions, Action(:play, i, 0, :_, 0))
            if view.clue_tokens < CLUE_TOKENS_MAX
                push!(actions, Action(:discard, i, 0, :_, 0))
            end
        end
    end
    if view.clue_tokens > 0
        target_player_real_idx = (view.player_idx % NUM_PLAYERS) + 1
        target_hand = view.other_player_hands[1]
        unique_colors = Set(CANONICAL_DECK[cid+1].color for cid in target_hand)
        unique_ranks = Set(CANONICAL_DECK[cid+1].rank for cid in target_hand)
        for color in unique_colors
            push!(actions, Action(:clue, 0, target_player_real_idx, :color, color))
        end
        for rank in unique_ranks
            push!(actions, Action(:clue, 0, target_player_real_idx, :rank, rank))
        end
    end
    return actions
end

struct RandomPlayer <: AbstractPlayer end

function get_action(player::RandomPlayer, view::PlayerView)::Action
    return rand(get_legal_actions(view))
end

function describe_action(action::Action)::String
    if action.type == :play
        return "PLAY card #$(action.card_index)"
    elseif action.type == :discard
        return "DISCARD card #$(action.card_index)"
    elseif action.type == :clue
        target = action.target_player
        if action.clue_type == :color
            color_name = COLORS[action.value]
            return "give a CLUE to Player $target about '$color_name' cards"
        else # :rank
            return "give a CLUE to Player $target about '$(action.value)'s"
        end
    end
    return "take an unknown action"
end

function simulate_game(players::Vector{<:AbstractPlayer}; render_game=false)
    gs = setup_game()

    if render_game
        render(gs)
    end

    while !is_game_over(gs)
        p_idx = gs.current_player
        player = players[p_idx]
        view = get_player_view(gs, p_idx)
        action = get_action(player, view)

        if render_game
            println(crayon"bold yellow", "\nPlayer $p_idx will now ", describe_action(action), "...", crayon"reset")
            println("Press Enter to apply.")
            readline()
        end

        apply_action!(gs, action)

        if render_game
            render(gs)
        end
    end

    if render_game
        println(crayon"bold cyan", "\nGAME OVER! Final Score: $(get_score(gs))", crayon"reset")
    end

    return get_score(gs)
end

function render_card(card::Card)
    c = COLOR_CRAYONS[card.color]
    print(c, " ", card.rank, " ")
end

function render_knowledge(k::CardKnowledge)
    color_str = "?"
    if k.is_color_known
        color_idx = Int(log2(k.possible_colors)) + 1
        color_str = COLORS[color_idx][1:1]
    end
    rank_str = "?"
    if k.is_rank_known
        rank_str = string(Int(log2(k.possible_ranks)) + 1)
    end
    
    crayon_to_use = k.is_color_known ? COLOR_CRAYONS[Int(log2(k.possible_colors)) + 1] : crayon"default"
    print(crayon_to_use, "[", color_str, rank_str, "]")
end

# (MODIFIED) This function is updated to show all hands.
function render(gs::GameState)
    print("\x1b[2J\x1b[H") # Clear screen
    println(crayon"bold underline", "Hanabi Game State", crayon"reset")
    
    info_str = "Clues: $(gs.clue_tokens) | Errors: $(gs.error_tokens)/$(ERROR_TOKENS_MAX)"
    deck_str = "Deck: $(length(gs.deck))"
    println(crayon"yellow", info_str, " "^(40-length(info_str)), deck_str, crayon"reset")
    println("-"^50)
    
    print(crayon"bold", "Fireworks: ", crayon"reset")
    for color_idx in 1:NUM_COLORS
        c = COLOR_CRAYONS[color_idx]
        val = gs.fireworks[color_idx]
        print(c, val > 0 ? " $(val) " : " 0 ", crayon"reset")
    end
    println("\n" * "-"^50)

    for p_idx in 1:NUM_PLAYERS
        is_current = p_idx == gs.current_player
        print(crayon"bold", is_current ? "-> Player $(p_idx)'s Hand:" : "   Player $(p_idx)'s Hand:", crayon"reset", "  ")
        
        # Always show the actual cards for every player
        for card_id in gs.hands[p_idx]
            render_card(CANONICAL_DECK[card_id+1])
        end
        println()

        # If it's the current player, add a second line showing their knowledge
        if is_current
            print("   (Player's View):  ")
            for k in gs.player_knowledge[p_idx]
                render_knowledge(k)
                print(" ")
            end
            println("\n") # Add extra space for clarity
        end
    end
    println("-"^50)
    
    print(crayon"bold", "Discard ($(length(gs.discard_pile))): ", crayon"reset")
    sorted_discard = sort(gs.discard_pile, by=id->(CANONICAL_DECK[id+1].color, CANONICAL_DECK[id+1].rank))
    for card_id in sorted_discard
        render_card(CANONICAL_DECK[card_id+1])
    end
    println("\n" * "="^50)
end

end # module
