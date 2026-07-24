### scratchpad

- [x] install catanatron engine
- [x] trajectory design
- [x] game loop + trajectory logger
- [ ] devtool: replay 
- [x] 1v1 benchmark: find best policy
- [x] serializer: engine -> LLM prompt
- [x] parser: LLM output -> engine
- [x] verifiers: environment wrapper
- [x] verifiers: define rubric
- [x] research: cost vs intelligence
- [x] methods: estimate statistical power required
- [x] eng: fix shared seed bug in catanatron
- [ ] benchmark non-reasoning + reasoning LLMs
    - [ ] run 1 game / model
    - [ ] run 5 games / model
    - [ ] run 10 games / model
    - [ ] run 25 games / model
- [ ] sft: generate traces
- [ ] sft: build dataset from logs
- [ ] sft: training
- [ ] rl: dpo, grpo etc
- [ ] experiment: 4-player mirror play

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

### how well do frontier models do?
I wanted to see how good frontier models are at playing Catan. We chose to evaluate these models: 

- Qwen3.5-9B
- DeepSeek V4 Flash
- MiniMax M3
- Kimi K3
- GPT 5.6 Sol
- Claude Fable 5
- GPT 5.6 Luna
- MiMo V2.5 Pro
- Claude Sonnet 5

Most of these sit on the cost-efficient frontier from [costs.md](costs.md). Luna, MiMo, and Sonnet 5 are near-frontier models that we decided to add to the fray: OpenAI models historically punch above their index on game-playing leaderboards, MiMo is ~tied with MiniMax M3 on cost while doing well on agentic evals, and Sonnet 5 gives us a lower-cost Anthropic comparison with strong agentic performance. Qwen3.5-9B is evaluated as we'll be using it for finetuning in later experiments.

IF GOOD: they're v good, so we use them as teacher, and obtain SFT traces from their trajectories.

IF BAD: let's see if we can do better by training a small model with SFT traces from our best performing policy.

### non-reasoning models

### cost vs intelligence 

`num_games * num_tokens * token_cost`
where
`num_tokens * token_cost = input_tokens * input_token_cost + output_tokens * output_token_cost`

For `input_tokens = 500K` and `output_tokens = 50K`, we present the estimated costs in costs.md.

We use this estimate to find the Pareto optimal models for our evaluation.

![model intelligence vs cost per game](assets/costs_scatter.png)

### experimental setup
We have two primary aims here:
- Is there enough statistical power to be sure our results are not just from chance?
- Do we have tight guardrails to prevent contamination/cheating?

**Statistical power**

How many games do we need to run to make sure our results are meaningful, and not just due to chance?

Let's first consider a single game of Catan.
In a game of Catan, there are two outcomes: you win ($p$) or lose ($1-p$). This can be modelled as a binomial distribution.

The mean, $E[X] = p(1) + (1-p)(0) = p$
The variance, $Var(X) = E[X^2] - (E[X])^2$ = $p-p^2$ = $p(1-p)$

Now what happens for $n$ games? Since the games are independent, we can add the variances!

If the total win count is W, then $W = X_1 + ... + X_n$.
The measured win rate is $\hat{p} = W/n$, and this is what we care about.

By the Central Limit Theorem (CLT), the sum of many independent random variables is approximately normally distributed. So, $\hat{p}$ is normal.

$Var(W) = n \cdot p(1-p)$

$Var(\hat{p}) = p(1-p)/n$

The variance is *squared*, so to end up in the same scale, we normalize by taking the square root

This is known as the "standard error",
$SE(\hat{p}) = \sqrt{p(1-p)/n}$

Now, we're comparing across models.

Let's do the pairwise case, of model $A$ and $B$. Each plays $n$ games, giving measured win rates $\hat{p}_A$ and $\hat{p}_B$.



$\hat{\Delta} = \hat{p}_A - \hat{p}_B $

The models may be playing the same game, so their variance is not independent. Hence, we have an additional covariance term

$Var(\hat{\Delta}) = \frac{p_A(1-p_A)}{n} + \frac{p_B(1-p_B)}{n} - 2 \cdot Cov(\hat{p}_A, \hat{p}_B)$

Because both models play the same initial boards and turn positions, we expect their outcomes to be positively correlated. Since covariance reduces the variance of the gap, ignoring it gives a conservative estimate of the standard error. We err on the side of running more games.

Let's also rewrite the two win rates around their average $\bar{p} = (p_A + p_B)/2$ and true gap $\Delta = p_A - p_B$:

$p_A = \bar{p} + \Delta/2, \quad p_B = \bar{p} - \Delta/2$

Substituting into the variance terms:

$p_A(1-p_A) + p_B(1-p_B) = 2\bar{p}(1-\bar{p}) - \frac{\Delta^2}{2}$

The $\Delta^2/2$ term is tiny, and dropping it again overestimates the variance. And thus, the standard error of the model gap is:

$SE(\hat{\Delta}) \approx \sqrt{\frac{2\bar{p}(1-\bar{p})}{n}}$

Let's say D_i represents the difference for the same seed $i$, but different models
$ D_i = X_{A,i}-X_{B,i} $

where
$$
D_i =
\begin{cases}
+1 & \text{only A wins,} \\
-1 & \text{only B wins,} \\
0  & \text{they get the same result.}
\end{cases}
$$

Since the difference between seeds is independent, we can say that 

$\hat{\Delta} = \frac{1}{n}\sum_{i=1}^{n}D_i$ is approximately normal

For a normal distribution, 95% of outcomes fall within 1.96 standard deviations.

So we only call a measured difference real if 

$|\hat{\Delta}| > 1.96 \cdot SE(\hat{\Delta})$

If the models are actually equal, for a single pairwise comparison, chance will fool us 5% of the time.

If the true gap between the models is exactly $1.96 \cdot SE(\hat{\Delta})$, we'll catch it 50% of the time, as our measurement is centered exactly on the threshold! Clearly, we'd want to notice this more than half the time. The standard choice of statistical power is 80%. For a normal distribution, 80% of outcomes lie above a point 0.84 SDs below the mean (check a Z-table). Therefore, if $\delta$ is the smallest gap we want to detect,

$\delta \geq (1.96 + 0.84) \cdot SE(\hat{\Delta}) = 2.80 \cdot SE(\hat{\Delta})$

Each individual game outcome, $X_{M,i}$ is Bernoulli i.e. $X_{A,i} \sim \text{Bernoulli}(p_A)$

Assuming the paired outcomes are non-negatively correlated, the variance of their difference is at most 1/4 + 1/4 = 1/2*

*($p(1-p)$ peaks at $1/4$, and the covariance term is negative)

Then, $SE(\hat{\Delta}) \leq \sqrt{\frac{1}{2n}}$

Plugging back into the earlier bound and rearranging for $n$, we get $n \geq \frac{3.92}{\delta^2}$

How sensitive do we want our measure of cross-model difference? We don't really need it to be fine. What point-difference between models do we need to establish a clear winner? Once we know that, we can decide how many games to run.

We expect strong models to do much better than weaker models. We set $\delta = 0.20$, meaning an absolute difference of 20 percentage points in win rate. We then get n=98, and round up to 100.

This is all based on relatively conservative estimates! I hypothesize that we won't need nearly this many games to get signal, at least for our initial pool of nine models. These models have non-trivial performance differences, so we expect the win rate differences to be coarse. Once we get to comparing similar tier models, it makes more sense to have high n  

TODO: note that our analysis above considered pairwise model differences, but not across the full matrix

Each model should play $n=TBD$ seeds (i.e. distinct games). Rather than replaying the same board many times, we choose to simply increase the seeded experiments. The model is "seat 0", turn order shuffles per seed. We fix the model's position across all games. We are interested in studying how models do in other turn positions in future work. The other three players are value_function bots. We report win counts. Scores remain comparable across models as they all play the same seeds. 

Note: alpha_beta is a stronger bot (36% vs 25% in our anchor runs) but ~8x slower. We thus use value_function as the opponent bots, and alpha_beta for our SFT teacher.

We start with n=1 for our initial experiments. Then, we plan on scaling up to 5, 10, 25, 50, 100. We may only do 25 experiments for the best (most expensive) frontier models -- Fable 5 and 5.6 Sol

**Scaffolding/guardrails**
Upstream catanatron seeds *global* randomness, so concurrent games in the same process corrupt each other's dice and shuffles! We fixed this in our fork ([taziksh/catanatron](https://github.com/taziksh/catanatron), branch `per-game-rng`): each game owns its own `random.Random(seed)`.

Eval and training seeds never overlap: the eval taskset only plays seeds below 10,000, and `log_games.py` starts at 10,000. So post-training traces can never share a board with an eval game.

The model gets a fresh context window per decision. It can't carry information across decisions, or across games.

Games are only seed-deterministic under `PYTHONHASHSEED=0`, so the env refuses to run without it.

### model benchmarks

### sft
we choose alpha_beta, as it was the strongest policy

![scripted-bot anchor win rates](assets/anchors_winrate.png)

**step 1: generate trajectories**

From some early experiments, each games has 50-70 decisions. We aim for ~100K decisions and so create traces for 3k games.

```
for n in 0...num_games:
    seed = 10_000 + n
    players = [alpha_beta(RED), value_function(...)...]
    game = new Game(seed)
    while !winner and turn < max_turns:
        actor = ...
        snapshot = {
            board, hands (all 4 players), bank, legal_actions, chosen_index
        }
        records.append(snapshot)
        game.execute(action)

    write file: data/games/<lineup>_s<seed>.jsonl
        line 1: game summary (seats, layout, winner, final VPs, etc)
        lines 2..N: decision snapshots
```


### references:
- https://docs.primeintellect.ai/verifiers/v1/overview
- https://djdumpling.github.io/2025/11/24/rl_envs.html
