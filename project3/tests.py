from soccer import SoccerEnv


def test_action_encode():
    env = SoccerEnv()
    action1, action2 = 1, 2
    x = env.encode_action(1,2)
    assert (action1, action2) == env.decode_action(x)


def test_state_encode():
    env = SoccerEnv
    player1_row, player1_column, player2_row, player2_column, player1_possession = 0, 1, 0, 2, 1
    x = env.encode_state(player1_row, player1_column, player2_row, player2_column, player1_possession)
    assert (player1_row, player1_column, player2_row, player2_column, player1_possession) == env.decode_state(x)


def test_done():
    assert SoccerEnv.done((0, 2, 0, 1, 1)) is False
    assert SoccerEnv.done((0, 3, 0, 1, 1))
    assert SoccerEnv.done((0, 0, 0, 1, 1))
    assert SoccerEnv.done((0, 0, 0, 1, 0)) is False
    assert SoccerEnv.done((0, 0, 0, 1, 0)) is False
    assert SoccerEnv.done((0, 0, 0, 0, 0))
    assert SoccerEnv.done((0, 0, 0, 3, 0))


def test_reward():
    assert SoccerEnv.reward((0, 2, 0, 1, 1)) == 0
    assert SoccerEnv.reward((0, 3, 0, 1, 1)) == -100
    assert SoccerEnv.reward((0, 0, 0, 1, 1)) == 100
    assert SoccerEnv.reward((0, 0, 0, 1, 0)) == 0
    assert SoccerEnv.reward((0, 0, 0, 1, 0)) == 0
    assert SoccerEnv.reward((0, 0, 0, 0, 0)) == 100
    assert SoccerEnv.reward((0, 0, 0, 3, 0)) == -100


def test_transitions():
    transitions = SoccerEnv.transitions(0, 2, 0, 1, 0, SoccerEnv.Action.W, SoccerEnv.Action.Stick)
    expected_states = set([SoccerEnv.encode_state(0, 2, 0, 1, 0), SoccerEnv.encode_state(0, 2, 0, 1, 1)])
    assert len(transitions) == 2
    for next_state, reward, done in transitions:
        assert next_state in expected_states
        assert reward == 0
        assert done == 0

    state = SoccerEnv.encode_state(0, 2, 0, 1, 0)
    action = SoccerEnv.encode_action(SoccerEnv.Action.W, SoccerEnv.Action.Stick)
    env = SoccerEnv()
    transitions = env.P[state][action]
    assert len(transitions) == 2
    for prob, next_state, reward, done in transitions:
        assert abs(prob - 0.5) < 0.001
        assert next_state in expected_states
        assert reward == 0
        assert done == 0


def test_transition2():
    initial_state = SoccerEnv.encode_state(0,3,0,1,False)
    action = SoccerEnv.encode_action(SoccerEnv.Action.S, SoccerEnv.Action.N)
    env = SoccerEnv()
    transitions = env.P[initial_state][action]
    expected_next_state = SoccerEnv.encode_state(1,3,0,1,False)
    for prob, next_state, reward, done in transitions:
        assert next_state == expected_next_state
        assert reward == 0
        assert done == 0


def test_render():
    env = SoccerEnv()
    env.render()
    action = env.encode_action(SoccerEnv.Action.Stick, SoccerEnv.Action.Stick)
    env.step(action)
    env.render()
    return


test_action_encode()
test_state_encode()
test_done()
test_reward()
test_transitions()
test_transition2()
test_render()
