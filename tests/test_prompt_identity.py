"""Train/eval prompt identity: the dataset renderer must match the live one.

SFT samples are re-rendered from logs (decision_to_prompt) while the eval
renders from a running game (observe_live). Captures the live render at
every decision of a played game and asserts the logged render reproduces it.
"""

from catanatron import Game
from catanatron.game import GameAccumulator
from conftest import SEED, _players

from catan_llm.extract import TrajectoryAccumulator, observe_live
from catan_llm.schema import DecisionRecord, GameRecord
from catan_llm.serialize import decision_to_prompt, observation_to_prompt


class PromptCapture(GameAccumulator):
    def __init__(self):
        self.prompts = []

    def step(self, game, action):
        self.prompts.append(observation_to_prompt(observe_live(game)))


def test_logged_render_matches_live(tmp_path):
    trajectory = TrajectoryAccumulator(tmp_path)
    capture = PromptCapture()
    game = Game(_players(), seed=SEED)
    game.play(accumulators=[trajectory, capture])

    lines = trajectory.path.read_text().splitlines()
    game_rec = GameRecord.model_validate_json(lines[0])
    decisions = [DecisionRecord.model_validate_json(line) for line in lines[1:]]
    assert len(decisions) == len(capture.prompts)
    for rec, live in zip(decisions, capture.prompts):
        assert decision_to_prompt(game_rec, rec) == live, (
            f"prompt drift at decision {rec.i}"
        )
