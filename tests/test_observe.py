"""Information-set tests on the Observation boundary."""

from catan_llm.observe import OpponentView, observe


def test_no_hidden_fields():
    fields = set(OpponentView.model_fields)
    assert "hand" not in fields
    assert "devs_in_hand" not in fields
    assert "vps_actual" not in fields


def test_observe(trajectory):
    game_rec, decisions = trajectory
    for rec in decisions:
        obs = observe(game_rec, rec)
        assert obs.you == rec.players[rec.actor]
        assert len(obs.opponents) == len(rec.players) - 1
        for opp in obs.opponents:
            truth = rec.players[opp.color]
            assert opp.color != rec.actor
            assert opp.card_count == sum(truth.hand.values())
            assert opp.dev_card_count == sum(truth.devs_in_hand.values())
            assert opp.vps_public == truth.vps_public
        assert obs.legal_actions == rec.legal_actions
        assert obs.turn_order == list(rec.players)
        assert obs.actor in obs.turn_order
