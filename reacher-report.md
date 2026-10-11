# Reacher Task Definition and Evaluation Report

## Purpose

Reacher is a simulated two-joint robot arm. Its task is to move its fingertip toward a target. I use it as a small learning benchmark to understand how a controller interacts with a robot simulation and how to measure its performance. It is not a business product or a demonstration of a trained AI controller.

I first tested two hand-written controllers, then trained and evaluated a PPO policy. This gives me an initial comparison of action-selection rules under the same evaluation conditions. The experiment is exploratory, not evidence of reliable target reaching.

## Task and measurements

- **Observation:** Reacher returns 10 numbers describing joint angles, target position, joint velocities, and the fingertip's x/y offset from the target. The last two values, `observation[8:10]`, are fingertip position minus target position in x and y.
- **Action:** Two numbers command torques at the arm's two joints. They do not directly command movement in x or y.
- **Episode:** `env.reset(seed=...)` starts a new attempt. Each `env.step(action)` moves the arm from its current state and returns a new observation and a step reward. The default episode ends at the 50-step time limit.
- **Distance:** `np.linalg.norm(observation[8:10])` combines the x/y offsets into one fingertip-to-target distance. Smaller means closer. I measure it at reset (step 0) and after every action.
- **Reward:** Reacher scores each step using a distance penalty and an action-size penalty. Total reward below is the sum of step rewards, not the same thing as the closest distance.
- **Provisional success rule:** An episode succeeds if its **closest recorded distance, including step 0, is at most `0.05`**. This is my chosen evaluation cutoff, not a built-in Reacher success condition. It does not require the arm to remain close or finish close.

The numerical distances are in the simulator's position units. I have not established a physical precision requirement that would make `0.05` an official or optimal cutoff.

## Experiment setup

The hand-written controllers and evaluated PPO checkpoints used 10 comparison episodes with reset seeds **42–51**, the same distance calculation, the same 50-step limit, and the same provisional success rule. The PPO policies used `deterministic=True` when choosing evaluation actions. This shared setup allows a small, seed-by-seed comparison, but the ten episodes are not established as a held-out test set.

1. **Fixed-action baseline:** Send `[0.05, -0.10]` at every step, regardless of the observation.
2. **Hand-written observation-based policy:** Read the current x-offset (`observation[8]`) before every step. If it is negative, send `[0.05, -0.10]`; otherwise, send `[-0.05, -0.10]`. This is a teaching rule, not a validated method for controlling joint torques.

Neither hand-written controller changes its rule based on rewards. PPO, described below, **does** update a policy during training.

## Results

| Seed | Fixed closest | Fixed final | Fixed total reward | Fixed success | Observation-based closest | Observation-based final | Observation-based total reward | Observation-based success |
| ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |
| 42 | 0.117885 | 0.143159 | -7.600587 | No | 0.148185 | 0.172645 | -10.640841 | No |
| 43 | 0.100117 | 0.132823 | -8.017413 | No | 0.046760 | 0.115604 | -6.735297 | Yes |
| 44 | 0.184949 | 0.184949 | -14.779082 | No | 0.174253 | 0.203011 | -14.285330 | No |
| 45 | 0.094004 | 0.094014 | -8.725283 | No | 0.107455 | 0.117026 | -10.044383 | No |
| 46 | 0.103920 | 0.106769 | -7.687580 | No | 0.125959 | 0.137500 | -9.913775 | No |
| 47 | 0.172633 | 0.172633 | -13.909960 | No | 0.168530 | 0.189927 | -13.855076 | No |
| 48 | 0.139695 | 0.141191 | -9.292453 | No | 0.160263 | 0.171917 | -11.682013 | No |
| 49 | 0.145768 | 0.173213 | -10.512340 | No | 0.008943 | 0.154279 | -8.195874 | Yes |
| 50 | 0.111821 | 0.111821 | -11.441349 | No | 0.092137 | 0.129984 | -10.710031 | No |
| 51 | 0.112552 | 0.175248 | -8.011542 | No | 0.030199 | 0.158383 | -7.315249 | Yes |

The two controllers started at the same distance for each seed. Under the provisional “ever within `0.05`” rule, the fixed-action baseline succeeded **0/10** times and the observation-based rule succeeded **3/10** times (seeds 43, 49, and 51). This is a result on these ten seeded episodes, not evidence of general reliability. The observation-based rule was not better on every seed: for seed 42 it never improved on its starting distance, whereas the fixed action got closer.

A success does **not** mean the fingertip stayed at the target. For example, on seed 49 the observation-based rule got as close as `0.008943` but finished at `0.154279`. The closer-ever metric and final-distance metric tell different parts of the story.

## First PPO training and evaluation

I trained a neural-network policy with Stable-Baselines3 PPO (`MlpPolicy`, training seed 42). This is different from the two hand-written rules: PPO gathers observations, actions, and rewards and uses that experience to update its policy. My `total_reward` measurement is the **undiscounted sum of step rewards over an episode**. It is an evaluation measure; PPO may use discounting internally during learning.

The first run requested **10,000 training timesteps** and collected **10,240** to finish PPO rollouts. I saved it as `ppo_reacher_first_run.zip`. When evaluated with deterministic actions on comparison seeds 42–51, it met the `≤ 0.05` rule **2/10** times (seeds 43 and 51). Its observed closest distances were `0.019839` and `0.025155` on those successful episodes, but its final distances were `0.106533` and `0.154559`. This checkpoint is not evidence of consistent reaching or staying at the target.

## Longer PPO run and reward-selected checkpoint

I then ran a separate PPO training experiment requesting **50,000 timesteps**. It reached **51,200** training steps because PPO completes rollouts. An `EvalCallback` evaluated the evolving policy every **5,000 training steps**, using **10 episodes per evaluation** and deterministic actions, and saved a checkpoint when its **mean evaluation reward** improved. This selection rule is **not** my `0.05` distance threshold.

The callback's highest reported mean evaluation reward was **−9.57 at 5,000 steps**. Later logged mean evaluation rewards were lower (for example, **−58.12 at 50,000 steps**), so the callback retained an earlier reward-selected checkpoint. The training log also warned that the callback evaluation environment was not wrapped with `Monitor`; evaluation still ran, but I should wrap that environment in `Monitor` for clearer episode accounting on the next run. The final 51,200-step model and the callback's saved checkpoint are **different models**; both were evaluated separately below.

I loaded `ppo_reacher_checkpoints/best_model.zip` and tested it on the same comparison seeds 42–51 with `deterministic=True`. Results:

| Seed | Closest distance | Final distance | Total reward | Success (`≤ 0.05`) |
| ---: | ---: | ---: | ---: | :---: |
| 42 | 0.130230 | 0.144016 | -7.992446 | No |
| 43 | 0.100942 | 0.105929 | -7.741380 | No |
| 44 | 0.203840 | 0.217512 | -16.691490 | No |
| 45 | 0.105067 | 0.119694 | -9.982390 | No |
| 46 | 0.111088 | 0.118162 | -8.963556 | No |
| 47 | 0.187292 | 0.211350 | -15.070437 | No |
| 48 | 0.147119 | 0.156883 | -10.205513 | No |
| 49 | 0.151385 | 0.151385 | -10.323670 | No |
| 50 | 0.125672 | 0.148676 | -12.559329 | No |
| 51 | 0.093127 | 0.142781 | -6.612066 | No |

**Result: 0/10 successes on these comparison seeds.** The closest value, `0.093127`, was still above `0.05`. “Best model” means best **mean reward on the callback's evaluation episodes**, not best on every seed or by my success rule. This outcome does not establish that PPO is generally worse than either hand-written controller.

## Final model from the longer PPO run

I also loaded `ppo_reacher_final_50k.zip`, saved at the end of the longer run, and evaluated it separately with deterministic actions on comparison seeds 42–51. This time the action came from the **final model**, not the callback-selected checkpoint.

| Seed | Closest distance | Final distance | Total reward | Success (`≤ 0.05`) |
| ---: | ---: | ---: | ---: | :---: |
| 42 | 0.023784 | 0.137107 | -57.686005 | Yes |
| 43 | 0.053437 | 0.121021 | -55.842600 | No |
| 44 | 0.044222 | 0.208880 | -60.682184 | Yes |
| 45 | 0.046821 | 0.104872 | -56.584557 | Yes |
| 46 | 0.034551 | 0.106176 | -56.762329 | Yes |
| 47 | 0.007657 | 0.199266 | -60.300776 | Yes |
| 48 | 0.046304 | 0.141467 | -58.112797 | Yes |
| 49 | 0.054430 | 0.162238 | -57.458269 | No |
| 50 | 0.058216 | 0.136880 | -57.383019 | No |
| 51 | 0.016414 | 0.162512 | -57.111784 | Yes |

**Result: 7/10 successes on these comparison seeds.** Every successful episode ended farther from the target than its closest recorded point. For example, on seed 47 the fingertip came within `0.007657`, which passes my `0.05` cutoff, but finished at `0.199266`; its total reward was `−60.300776`. This means it passed close at least once, **not** that it stayed close. For this seed 47 episode of the final PPO model, I summed the reward components over 50 steps: the distance component was `−11.041945`, and the joint-action component was `−49.258839`. Their sum (`−60.300784`) differs from the accumulated total reward (`−60.300776`) by about `0.000008` due to numerical precision. The action penalty contributed most of the negative reward **in this episode**; I have not measured the component totals for the other nine seeds or analyzed when within this episode the penalties occurred.

The final model's `7/10` success count and the callback-selected model's `0/10` do not contradict each other. My threshold measures *ever close enough*, while Reacher's reward accounts for distance and action size at **every step**. `EvalCallback` selects by **mean evaluation reward on its evaluation episodes**, not by the count of episodes crossing my threshold. On these ten comparison seeds the final model has much more negative total reward despite more threshold successes. Ten episodes are too few for a general reliability claim.

| Controller / checkpoint | Successes on seeds 42–51 |
| --- | ---: |
| Fixed action | 0/10 |
| Hand-written x-offset rule | 3/10 |
| First PPO run, ~10,000 steps | 2/10 |
| Reward-selected checkpoint from longer PPO run | 0/10 |
| Final model from longer PPO run | 7/10 |

## Additional evaluation on seeds 100–129

I evaluated the saved reward-selected checkpoint and final PPO model on the same 30 new reset seeds, 100–129, with deterministic actions and the same provisional success rule: the smallest distance at reset or after any action must be `≤ 0.05`. This evaluation loads the saved models; it does not retrain them. The results in this section come from the saved notebook evaluation of seeds 100–129, not the earlier evaluation of seeds 42–51. These 30 seeds were not part of the earlier ten-seed comparison, but they are now part of my exploratory evaluation; they are not a guarantee of performance on other seeds.

| Saved model | Episodes reaching `≤ 0.05` at least once | Example of reward components per episode |
| --- | ---: | --- |
| Reward-selected checkpoint | 6/30 | Seed 100: distance `−4.458281`, action `−0.170643`, total `−4.628924` |
| Final PPO model | 24/30 | Seed 100: distance `−9.928332`, action `−48.423901`, total `−58.352230` |

The final model crossed my cutoff more often on these seeds, but its action penalties were much larger. The two seed 100 examples illustrate this difference; they do not stand in for average values across all 30 episodes. The final model was within the cutoff at just **one post-action observation** on many successful seeds (for example, seed 101), but at **20 of 50** on seed 116 and **17 of 50** on seed 120. The reward-selected checkpoint reached the cutoff less often, but recorded **11 of 50** close observations on seed 100 and **10 of 50** on seed 102.

I also checked the longest *consecutive* stretch of post-action observations within `0.05`, excluding the starting observation. The following results show both the total number of close steps and the longest uninterrupted stretch for each seed; `0` means no post-action observation was within the cutoff. Each episode has 50 post-action observations.

| Seed | Checkpoint: close steps | Checkpoint: longest stretch | Final: close steps | Final: longest stretch |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 11 | 11 | 3 | 2 |
| 101 | 0 | 0 | 1 | 1 |
| 102 | 10 | 9 | 1 | 1 |
| 103 | 0 | 0 | 1 | 1 |
| 104 | 0 | 0 | 1 | 1 |
| 105 | 0 | 0 | 1 | 1 |
| 106 | 0 | 0 | 1 | 1 |
| 107 | 0 | 0 | 1 | 1 |
| 108 | 0 | 0 | 1 | 1 |
| 109 | 0 | 0 | 1 | 1 |
| 110 | 0 | 0 | 1 | 1 |
| 111 | 0 | 0 | 2 | 1 |
| 112 | 0 | 0 | 0 | 0 |
| 113 | 0 | 0 | 0 | 0 |
| 114 | 0 | 0 | 1 | 1 |
| 115 | 10 | 6 | 8 | 2 |
| 116 | 9 | 9 | 20 | 4 |
| 117 | 0 | 0 | 1 | 1 |
| 118 | 0 | 0 | 1 | 1 |
| 119 | 0 | 0 | 1 | 1 |
| 120 | 10 | 10 | 17 | 2 |
| 121 | 0 | 0 | 1 | 1 |
| 122 | 7 | 7 | 1 | 1 |
| 123 | 0 | 0 | 1 | 1 |
| 124 | 0 | 0 | 0 | 0 |
| 125 | 0 | 0 | 0 | 0 |
| 126 | 0 | 0 | 0 | 0 |
| 127 | 0 | 0 | 0 | 0 |
| 128 | 0 | 0 | 1 | 1 |
| 129 | 0 | 0 | 1 | 1 |

On these seeds, the final model's longest stretch in any episode was **4 steps** (seed 116), even though it had 20 close steps in total on that seed. Its 17 close steps on seed 120 had a longest stretch of only **2**. For the reward-selected checkpoint, each of the six episodes that reached the cutoff had an uninterrupted close stretch of **at least 6 steps**; the longest was **11 steps** on seed 100. Thus, the final model reached the cutoff more often, while the checkpoint maintained longer uninterrupted close stretches *when it reached the cutoff*. These are counts at discrete simulation steps, not continuous real-world time. None of these measures alone proves reliable sustained reaching.

![Median fingertip-to-target distance over steps for both saved models on seeds 100–129](test%20with%20different%20seed.png)

*Each line is the median distance at a given step across 30 episodes; shading covers the middle 50%. A line above `0.05` does not mean no individual episode crossed the cutoff.*

![Final PPO model distance over steps for seed 116, with the 0.05 cutoff](final%20model%20seed%20116.png)

*Seed 116 illustrates the difference between 20 close steps in total and a longest uninterrupted stretch of 4: the distance repeatedly moves above and below the cutoff.*

## What I learned and next steps

A policy is a rule for choosing actions. The fixed policy ignores observations; the x-offset rule uses one observation value to select an action pair. Neither hand-written rule learns. PPO updates a policy from experience, but training longer did **not** improve both measures here: the final model crossed my distance threshold more often on the comparison seeds while its episode reward was much worse. The callback's reward-based selection and my distance-based success measure answer different questions.

These results do **not** establish overfitting, underfitting, or a “best fit.” The original ten seeds were used repeatedly for comparison, and the additional 30 seeds have now been inspected too; neither set should be treated as an untouched test set for future model selection. To assess generalization, I need separately chosen conditions, multiple training seeds, and evaluation of future models without tuning them on these same episodes. I should compare sustained proximity across more training runs and use `Monitor` on the callback's evaluation environment for clearer episode accounting. This remains an exploratory learning benchmark, not proof of robust target reaching.

## Short presentation version

> “Reacher is a simulated two-joint arm whose task is to move its fingertip toward a target. I tested two hand-written controllers and trained a PPO controller. On ten repeatable comparison episodes, my provisional ‘within 0.05 at any time’ rule was met 0 times by a fixed command, 3 times by an x-offset rule, 2 times by my first PPO model, and 7 times by the final model from a longer PPO run. A checkpoint selected during that longer run for high average reward met the same distance rule 0 times. The final model often passed near the target but did not stay near it; its episode rewards were worse than the reward-selected checkpoint's. These results show why how we define success matters. They are an exploratory learning exercise, not proof of reliable reaching or a diagnosis of overfitting.”




**References:** [Gymnasium Reacher-v5 task description](https://gymnasium.farama.org/environments/mujoco/reacher/); [Gymnasium basic environment usage](https://gymnasium.farama.org/introduction/basic_usage/); [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html); [Stable-Baselines3 EvalCallback](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html); [Stable-Baselines3 evaluation tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html). All numerical results above come from my local notebook outputs and training logs, not from the linked documentation.
