# Reacher Task Definition and Evaluation Plan

What is Reacher?

- Reacher is a small simulated robot-arm task.

in this case the arm has two joints, and its goal is move the fingertip close to the target. I use is as a safe, low cost test environment to learn how a controller's actions affect movement and how to measure whether it reaches a goal

why i doing it:
before training ai model to control a robot, i need to understand: what the arm observes, what action can it takes, and how i will measure it success. the reacher can help me to test a simple hand-chosen controller first. Later, i can train an RL controller and compare both across multiple episode to see whether the learning improves performance


1. Goal: Move the fingertip close to the red target
2. Observation: 
The environment returns 10 observation values, its tell me what the arm look like now, where the target is, how fast hinge rotate, and where is the fingertip is compared with the target

 3. Actions: there are 2 hinge, hinge 0: 0.05 and hinge 1: -0.10
3. Episode: One episode begins with env.reset(). My constant-actions is [0.05,-0.10] once per step. Each step the environment return a new observation, and reward after each action. the reacher truncate the episode after 50 steps.
4. Evaluation: after measure how closed the fingertip to the target with 10 values using hinge 0: 0.05 and hinge 1: -0.10 the results is the fingertip y is closed to the target in 0.00644788 m but the x is far 0.11789663 with 50 steps for the duration agent do the work

## Evaluation Plan

I will measure fingertip-to-target distance at the start and end of several episode and compare the result


so the Reinforcement Learning Workflow is:

1. Define the goal task: move the fingertip toward the target
2. set up reacher and inspect its observations, actions, rewards and episode limit
3. run a few episode; check how actions affect movement and whether your distance measurement make sense

4. test with fixed-action controller, like [0.05, -0.10]

5. choose an RL Algorithm and train a policy through many simulated episodes

6. run the trained policy on separate test episodes and compare it with the baseline


Report today:

-Ran reacher with fixed action [0.05, -0.10] across seed from range 42 - 51. record starting, closest, and final distance plus total reward. Next: document the baseline and choose a provisional success rule before training RL.


## Fixed-Action Baseline

- Environment: Gymnasium `Reacher-v5`.
- Controller: send `[0.05, -0.10]` to the two joints at every step. The action does not change based on observations.
- Episodes: 10, using reset seeds 42 through 51.
- Episode length: up to 50 steps; stop when `terminated` or `truncated`.
- Distance: calculate `np.linalg.norm(observation[8:10])` at reset (step 0) and after each step.
- Success rule: the **smallest distance recorded during an episode** must be `0.05` or less. This is my provisional evaluation rule, not a built-in Reacher success signal.
- Other measurements: starting distance, final distance, and sum of step rewards.

**Result:** 0 of 10 episodes met the success rule. All 10 episodes got closer at some point than they were at the start. These results describe this fixed-action controller only; no RL policy was trained.

Seed | Start distance | Closest distance | Final distance | Total reward | Success
-------------------------------------------------------------------------------------
  42 |       0.148185 |         0.117885 |       0.143159 |    -7.600587 | False
  43 |       0.204758 |         0.100117 |       0.132823 |    -8.017413 | False
  44 |       0.381065 |         0.184949 |       0.184949 |   -14.779082 | False
  45 |       0.232501 |         0.094004 |       0.094014 |    -8.725283 | False
  46 |       0.188319 |         0.103920 |       0.106769 |    -7.687580 | False
  47 |       0.352997 |         0.172633 |       0.172633 |   -13.909960 | False
  48 |       0.209893 |         0.139695 |       0.141191 |    -9.292453 | False
  49 |       0.255981 |         0.145768 |       0.173213 |   -10.512340 | False
  50 |       0.316714 |         0.111821 |       0.111821 |   -11.441349 | False
  51 |       0.159109 |         0.112552 |       0.175248 |    -8.011542 | False


Using Observation-based Policy:

I compared a fixed-action baseline with a hand-written policy that switches actions based on the fingertip’s x-offset. On 10 repeatable Reacher episodes, the fixed action met my provisional ‘ever within 0.05’ rule 0 times, and the observation-based rule met it 3 times. Neither controller was trained with RL.”

Seed | Start distance | Closest distance | Final distance | Total reward | Success
-------------------------------------------------------------------------------------
  42 |       0.148185 |         0.148185 |       0.172645 |   -10.640841 | False
  43 |       0.204758 |         0.046760 |       0.115604 |    -6.735297 | True
  44 |       0.381065 |         0.174253 |       0.203011 |   -14.285330 | False
  45 |       0.232501 |         0.107455 |       0.117026 |   -10.044383 | False
  46 |       0.188319 |         0.125959 |       0.137500 |    -9.913775 | False
  47 |       0.352997 |         0.168530 |       0.189927 |   -13.855076 | False
  48 |       0.209893 |         0.160263 |       0.171917 |   -11.682013 | False
  49 |       0.255981 |         0.008943 |       0.154279 |    -8.195874 | True
  50 |       0.316714 |         0.092137 |       0.129984 |   -10.710031 | False
  51 |       0.159109 |         0.030199 |       0.158383 |    -7.315249 | True


  I tested two hand-written controllers over the same 10 Reacher episodes (seeds 42–51). The fixed-action controller sent [0.05, -0.10] at every step and met my provisional ‘ever within 0.05’ rule in 0/10 episodes. The observation-based controller chose between [0.05, -0.10] and [-0.05, -0.10] based on the sign of the fingertip-to-target x-offset; it met the rule in 3/10 episodes. Success only means the fingertip came close at some point—it might move away later. Neither controller was trained with RL