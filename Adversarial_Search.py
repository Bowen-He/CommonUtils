from adversarialsearchproblem import AdversarialSearchProblem
from queue import Queue
import time


def minimax(asp):
    """
    Implement the minimax algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    def MaxValue(State, Player):
        if asp.is_terminal_state(State):
            return asp.evaluate_state(State)[Player]
        Value = -9999
        for j in asp.get_available_actions(State):
            CounterPlayerValue = MinValue(asp.transition(State, j), Player)
            Value = CounterPlayerValue if Value < CounterPlayerValue else Value
        return Value

    def MinValue(State, Player):
        if asp.is_terminal_state(State):
            return asp.evaluate_state(State)[Player]
        Value = 9999
        for k in asp.get_available_actions(State):
            CounterPlayerValue = MaxValue(asp.transition(State, k), Player)
            Value = CounterPlayerValue if Value > CounterPlayerValue else Value
        return Value
    StartTime = time.time()
    StartState = asp.get_start_state()
    Actions = asp.get_available_actions(StartState)
    ActionValues = {}
    for i in Actions:
        ActionValues[i] = MinValue(asp.transition(StartState, i), StartState.player_to_move())
    print(time.time()-StartTime)
    return max(ActionValues, key=ActionValues.get)


def alpha_beta(asp):
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input: asp - an AdversarialSearchProblem
    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """

    def MaxValue(State, Player, alpha, beta):
        if asp.is_terminal_state(State):
            return [asp.evaluate_state(State)[Player], None]
        Value = [-9999, None]
        for j in asp.get_available_actions(State):
            CounterPlayerValue = MinValue(asp.transition(State, j), Player, alpha, beta)
            Value = [CounterPlayerValue[0], j] if Value[0] < CounterPlayerValue[0] else Value
            if Value[0] >= beta:
                return Value
            alpha = max([alpha, Value[0]])
        return Value

    def MinValue(State, Player, alpha, beta):
        if asp.is_terminal_state(State):
            return [asp.evaluate_state(State)[Player], None]
        Value = [9999, None]
        for k in asp.get_available_actions(State):
            CounterPlayerValue = MaxValue(asp.transition(State, k), Player, alpha, beta)
            Value = [CounterPlayerValue[0], k] if Value[0] > CounterPlayerValue[0] else Value
            if Value[0] <= alpha:
                return Value
            beta = min([beta, Value[0]])
        return Value

    StartTime = time.time()
    StartState = asp.get_start_state()
    ValueAndAction = MaxValue(StartState, StartState.player_to_move(), -9999, 9999)
    print(time.time() - StartTime)
    return ValueAndAction[1]


def alpha_beta_cutoff(asp, cutoff_ply, eval_func):
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Inputs:
            asp - an AdversarialSearchProblem
            cutoff_ply- an Integer that determines when to cutoff the search
                    and use eval_func.
                    For example, when cutoff_ply = 1, use eval_func to evaluate
                    states that result from your first move. When cutoff_ply = 2, use
                    eval_func to evaluate states that result from your opponent's
                    first move. When cutoff_ply = 3 use eval_func to evaluate the
                    states that result from your second move.
                    You may assume that cutoff_ply > 0.
            eval_func - a function that takes in a GameState and outputs
                    a real number indicating how good that state is for the
                    player who is using alpha_beta_cutoff to choose their action.
                    You do not need to implement this function, as it should be provided by
                    whomever is calling alpha_beta_cutoff, however you are welcome to write
                    evaluation functions to test your implemention. The eval_func we provide
        does not handle terminal states, so evaluate terminal states the
        same way you evaluated them in the previous algorithms.

    Output: an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    def MaxValue(State, Player, Count, alpha, beta):
        Value = [-9999, None]
        if asp.is_terminal_state(State):
            return [asp.evaluate_state(State)[Player], None]
        # Check if Count==0
        if Count == 0:
            return [eval_func(State), None]
        # Search further
        for j in asp.get_available_actions(State):
            CounterPlayerValue = MinValue(asp.transition(State, j), Player, Count-1, alpha, beta)
            Value = [CounterPlayerValue[0], j] if Value[0] < CounterPlayerValue[0] else Value
            if Value[0] >= beta:
                return Value
            alpha = max([alpha, Value[0]])
        return Value

    def MinValue(State, Player, Count, alpha, beta):
        Value = [9999, None]
        if asp.is_terminal_state(State):
            return [asp.evaluate_state(State)[Player], None]
        # Check if Count==0
        if Count == 0:
            return [eval_func(State), None]
        # Search further
        for k in asp.get_available_actions(State):
            CounterPlayerValue = MaxValue(asp.transition(State, k), Player, Count-1, alpha, beta)
            Value = [CounterPlayerValue[0], k] if Value[0] > CounterPlayerValue[0] else Value
            if Value[0] <= alpha:
                return Value
            beta = min([beta, Value[0]])
        return Value

    StartTime = time.time()
    StartState = asp.get_start_state()
    ValueAndAction = MaxValue(StartState, StartState.player_to_move(), cutoff_ply, -9999, 9999)
    print(time.time() - StartTime)
    return ValueAndAction[1]

