# Reacher: first control and RL experiment

A learning experiment using Gymnasium's simulated two-joint **Reacher-v5** arm. The objective is to bring the fingertip near a target. This repository compares two hand-written controllers with early PPO-trained policies; it is **not** a production robotics system or proof of reliable reaching.

## What I tested

- **Fixed action:** send `[0.05, -0.10]` at every step.
- **Hand-written observation-based rule:** inspect the fingertip-to-target x-offset; send `[0.05, -0.10]` if negative, otherwise `[-0.05, -0.10]`. This rule is for learning about policies, not a validated controller.
- **PPO:** train an `MlpPolicy` on Reacher observations with Stable-Baselines3, then evaluate the saved policies using deterministic actions.

The original comparison uses reset seeds **42–51**. Distance is computed from both x and y offsets as `np.linalg.norm(observation[8:10])`, at reset and after every step. An episode counts as a *provisional success* if its **smallest** distance is `≤ 0.05` at any time, including the starting point. Reacher does not define this success threshold or stop when it is reached. Total reward is the sum of Reacher's step rewards; unlike the success check, it reflects distance and action cost across the entire episode.

| Controller / saved checkpoint | Successes on seeds 42–51 |
| --- | ---: |
| Fixed action | 0/10 |
| Hand-written x-offset rule | 3/10 |
| First PPO model (10,000 requested training steps) | 2/10 |
| Reward-selected checkpoint from longer PPO run | 0/10 |
| Final PPO model from longer run (50,000 requested steps) | 7/10 |

**These measures disagree:** the final PPO model passed close to the target more often but had much worse total episode rewards than the reward-selected checkpoint. Passing close once is not the same as staying close. These ten repeatedly used seeds are an exploratory comparison set, **not an untouched test set**.

## Additional evaluation on seeds 100–129

I evaluated the two saved PPO models again on 30 different simulator reset seeds, without retraining them:

| Saved model | Reached `≤ 0.05` at least once |
| --- | ---: |
| Reward-selected checkpoint | 6/30 |
| Final PPO model | 24/30 |

For each of these episodes, I also counted the post-action steps within `0.05` and the **longest uninterrupted stretch** of close steps. The final model reached the cutoff more often, but its longest stretch was **at most 4 steps** in any of these 30 episodes; the checkpoint reached it less often, but each of its six successful episodes had a stretch of **at least 6 steps**. The final model also had much larger action penalties. These are exploratory results in simulation, not proof of reliable sustained reaching. The new seeds have now been inspected and should not be treated as an untouched test set for future model selection. See [reacher-report.md](reacher-report.md) for per-seed results, reward breakdowns, charts, methodology, limitations, and a short presentation summary.

## Files

- [notebook.ipynb](notebook.ipynb) — step-by-step experiments, training, evaluations, and plots. This is a learning notebook with exploratory and repeated cells; execute the specific experiment you want rather than assuming every cell must be run in order.
- [reacher-report.md](reacher-report.md) — task definition, results, limitations, and next steps.
- `ppo_reacher_first_run.zip` — first PPO model, trained for 10,000 requested timesteps (10,240 collected).
- `ppo_reacher_checkpoints/best_model.zip` — checkpoint selected by `EvalCallback` for highest **mean evaluation reward** during a longer run; *not* selected using the `0.05` distance cutoff.
- `ppo_reacher_final_50k.zip` — final model saved after the longer run (51,200 collected timesteps).
- `ppo_reacher_first_run_20261011_062532.zip` — additional timestamped PPO training run; keep it distinct from the callback-selected and final models.

## Run locally

The project specifies **Python 3.13 or newer** and dependencies in [pyproject.toml](pyproject.toml). From the project directory, install the project into a virtual environment and open the notebook, for example:

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -e .
python -m notebook
```

If the environment is activated, use its Python for the install and notebook commands. The notebook can open a separate simulation window when it creates Reacher with `render_mode="human"`; a local desktop is needed for that view. Training cells use a non-rendered environment. The notebook uses relative paths for model files, so launch it from this directory or adjust paths to match your working directory. **Training cells create new models and may replace saved files with the same names**; load an existing model instead if you only want to evaluate it.

To use a saved model with your own evaluation loop:

```python
from stable_baselines3 import PPO

model = PPO.load("ppo_reacher_final_50k.zip")
# Inside an episode's step loop, after env.reset(...):
action, _ = model.predict(observation, deterministic=True)
```

The full reset/step loop, metrics, and plots are in the notebook. For a fair comparison, use the same seeds, episode limit, distance definition, and success rule for each controller.

## Next steps

For a future experiment, decide whether the main goal is reaching the cutoff, staying near the target, improving total reward, or balancing these measures. Compare multiple training runs with different training seeds and evaluate on conditions not used to develop or select the models. Add a `Monitor` wrapper to the callback evaluation environment for clearer episode accounting. The current exploratory experiment is complete; further training would be a new experiment.

## References

- [Gymnasium Reacher-v5](https://gymnasium.farama.org/environments/mujoco/reacher/)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Stable-Baselines3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
