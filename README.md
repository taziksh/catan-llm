- [x] install catanatron engine
- [x] trajectory design
- [x] game loop + trajectory logger
- [ ] devtool: replay 
- [x] 1v1 benchmark: find best policy
- [x] serializer: engine -> LLM prompt
- [x] parser: LLM output -> engine
- [x] verifiers: environment wrapper
- [x] verifiers: define rubric
- [ ] benchmark non-reasoning LLMs
- [ ] benchmark reasoning LLMs
- [ ] experiment: 4-player mirror play
- [ ] sft: generate traces
- [ ] sft: build dataset from logs
- [ ] sft: training
- [ ] rl: dpo, grpo etc

### scripted policies
Try policies in [leaderboard](https://docs.catanatron.com/advanced/making-catanatron-stronger)

### trajectory 
Schema is defined in `catan_llm/schema.py`.

### serializer
Serializer translates engine to a prompt for the LLM.

This is what the model sees:
<EXAMPLE_PROMPT>

- We start with a fresh context window per decision
- The model is presented with legal moves in a numbered list
- We map the board graph into a serialized text format:
tiles: [`tile_id:resource-dice_number`, ...]

### reward 
How does the model know it's doing well? Win/lose is too sparse and delayed of a reward signal by itself. We thus introduce VP as an additional reward term...

### rubric design
`reward_win: 1.0 | 0.0` # weight 1.0
`reward_vp: min(vps, 10) / 10` # weight 0.1

We normalize the `reward_vp` term, and downweight it, so that the model doesn't learn to score games with lots of victory points, but overall loss too highly.
Not all games are equally difficult. To compare fairly, we plan to normalize relative to the performance of an *expert* Catan agent, replaying the same seat (+ deterministic seed). This is left for future work.


We also log diagnostic data with weight `0`:
`invalid_rate, truncated, game_length, decisions, rank, vp_margin`

`vp_margin` is the relative difference in VPs of our model against the best opponent vp, operationalized as `(agent_vps - max(opponent_vps)) / 10`

`truncated` games keep their reward_vp, only reward_win is 0.

We err towards logging more metrics at this point, to gain a more granular understanding of failure modes. Not all of these will make their way into the final reward.




### open questions:
- what is the impact of seat ordering?

### references:
- https://docs.primeintellect.ai/verifiers/v1/overview
- https://djdumpling.github.io/2025/11/24/rl_envs.html
