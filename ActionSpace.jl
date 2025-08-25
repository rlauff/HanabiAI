# ActionSpace.jl

module ActionSpace

using ..HanabiLogic

export action_to_index, index_to_action, a_idcs

# Mapping:
# 1-5  : Play card 1-5
# 6-10 : Discard card 1-5
# 11-15: Clue Color 1-5 (R,G,B,W,Y)
# 16-20: Clue Rank 1-5

const a_idcs = (
    play = 1:5,
    discard = 6:10,
    clue_color = 11:15,
    clue_rank = 16:20
)

function action_to_index(action::Action)::Int
    if action.type == :play
        return a_idcs.play[action.card_index]
    elseif action.type == :discard
        return a_idcs.discard[action.card_index]
    elseif action.type == :clue
        if action.clue_type == :color
            return a_idcs.clue_color[action.value]
        else # rank
            return a_idcs.clue_rank[action.value]
        end
    end
    return -1 # Should not happen
end

function index_to_action(index::Int)::Action
    if index in a_idcs.play
        return Action(:play, index - a_idcs.play[1] + 1, 0, :_, 0)
    elseif index in a_idcs.discard
        return Action(:discard, index - a_idcs.discard[1] + 1, 0, :_, 0)
    elseif index in a_idcs.clue_color
        # For clues, target_player must be filled in later
        return Action(:clue, 0, -1, :color, index - a_idcs.clue_color[1] + 1)
    else # clue_rank
        return Action(:clue, 0, -1, :rank, index - a_idcs.clue_rank[1] + 1)
    end
end

end
