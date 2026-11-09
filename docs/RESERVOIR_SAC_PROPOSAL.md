# Reservoir–SAC for Multi-System Boundary Tracking

## Motivation

The current reservoir computer uses a fixed recurrent reservoir and a linear
action readout. The readout is trained with the cross-entropy method (CEM): a
population of candidate weight matrices is sampled, each candidate receives an
episode-level reward, and the distribution is updated toward the candidates
with the highest rewards.

This approach works when candidate fitness can be estimated reliably. In the
multi-walk experiment, however, each policy must work across several systems,
random initial conditions, a changing parameter walk, basin-boundary
diagnostics, and Langevin noise. A small number of complete episodes therefore
produces a very noisy estimate of policy quality.

SAC does not depend on ranking a small population using a few complete episode
returns. It learns from individual transitions stored in a replay buffer and
can reuse each transition in many gradient updates. A promising alternative is
therefore to retain the reservoir as the policy's nonlinear memory while using
SAC to train its action readout.

## Evidence from the short diagnostic

The reservoir diagnostic produced the following results:

| System | Initial return | Trained return | Change | Boundary proximity | Action saturation |
|---|---:|---:|---:|---:|---:|
| Duffing | 8.482 | 24.654 | +16.172 | 0.244 | 0.0% |
| Langevin | 20.458 | 5.242 | -15.217 | 0.017 | 0.0% |
| Sextic | 14.928 | 37.320 | +22.392 | 0.400 | 0.0% |
| Smooth biquadratic | 24.210 | 12.521 | -11.689 | 0.122 | 0.0% |

The candidate rank correlation between the training seed set and a fresh seed
set was only **0.024**. A value near zero means that a candidate that appears
strong on the training rollouts is almost randomly placed when evaluated on
fresh rollouts. CEM is consequently likely to select candidates that were
lucky rather than candidates whose behavior is generally better.

The generation log supports the same conclusion. For example, a best training
return of 31.195 corresponded to a validation return of only 6.908. The search
also improved Duffing and sextic while substantially harming Langevin and the
smooth biquadratic system. Optimizing a single average return allows gains on
some systems to hide regressions on others.

Action saturation was zero for every system, and the CEM search standard
deviation stayed near 0.04. The immediate failure was therefore not action
clipping or an exploding search distribution. It was unreliable candidate
selection and uneven multi-system generalization.

## Proposed architecture

The proposed hybrid has three main components:

```text
observation
    |
    v
fixed recurrent reservoir
    |
    v
reservoir feature vector z_t
    |                         |
    v                         v
SAC actor readout        SAC Q critics
    |                    Q_1(z_t, a_t)
    v                    Q_2(z_t, a_t)
continuous action
```

The reservoir remains fixed. Its recurrent state provides a nonlinear summary
of the observation history. SAC replaces CEM as the method used to learn how
that summary should be converted into actions.

### Fixed reservoir

At control step `t`, the reservoir receives observation `o_t` and updates its
state:

```text
h_t = reservoir_step(h_(t-1), o_t)
```

The policy feature vector can include a bias, the normalized current
observation, and the reservoir state:

```text
z_t = [1, normalized(o_t), h_t]
```

For the physical Duffing reservoir, both oscillator position and velocity can
be included:

```text
z_t = [1, normalized(o_t), q_t, p_t]
```

The reservoir state is cleared at an episode boundary but retained between
steps within an episode.

### SAC actor

SAC uses a stochastic continuous-action policy. Two trainable readouts operate
on the reservoir features:

```text
mu_t      = W_mu z_t + b_mu
log_std_t = clip(W_std z_t + b_std, lower, upper)
```

An action is sampled with the reparameterization trick and passed through a
hyperbolic tangent so it remains within the environment's action limits:

```text
epsilon ~ Normal(0, I)
u_t = mu_t + exp(log_std_t) * epsilon
a_t = action_limit * tanh(u_t)
```

The reservoir supplies the nonlinear recurrent computation. Only the small
actor readouts need to be trained. During evaluation, the deterministic action
is obtained from `tanh(mu_t)`.

### SAC critics

SAC requires two critics to estimate the long-term value of a feature-action
pair:

```text
Q_1(z_t, a_t)
Q_2(z_t, a_t)
```

The first implementation should use small multilayer critics rather than a
purely linear readout. The value function can be highly nonlinear in both the
reservoir state and the continuous action. Using two small critics preserves
the reservoir-based actor while giving SAC enough capacity to learn a useful
training signal. The smaller of the two Q estimates is used in the target,
reducing positive value bias.

A later ablation can replace the critics with reservoir or linear models to
test how much conventional function approximation is actually necessary.

## Replay buffer and recurrent state

Standard SAC samples transitions in random order. A recurrent policy normally
creates a difficulty because the hidden state for a transition depends on all
earlier observations in its episode. This hybrid avoids reconstructing that
history by storing the fixed reservoir features when experience is collected:

```text
(z_t, a_t, r_t, z_(t+1), terminated)
```

Because the reservoir weights never change, stored feature vectors do not
become stale during training. Random replay batches can therefore be used
without replaying entire observation sequences. At collection time, the
reservoir still evolves in chronological order and is reset only when the
environment resets.

It is important to distinguish a time-limit truncation from a true terminal
state. SAC should normally bootstrap across a time-limit truncation but not
across a genuine terminal failure.

## SAC training update

For a replay transition, the target actor samples a next action from the next
reservoir feature vector. The target value is

```text
y_t = r_t + gamma * (1 - terminal_t) *
      [min(Q1_target(z_(t+1), a_(t+1)),
           Q2_target(z_(t+1), a_(t+1)))
       - alpha * log_pi(a_(t+1) | z_(t+1))]
```

Each critic minimizes squared temporal-difference error against this target.
The actor minimizes

```text
alpha * log_pi(a_t | z_t) - min(Q1(z_t, a_t), Q2(z_t, a_t)).
```

The entropy temperature `alpha` can be learned automatically. Entropy is
useful here because it discourages the actor from committing too early to a
strategy that happened to work for one system or one boundary location.

Target critics are updated slowly with Polyak averaging.

## Why this should improve multi-walk learning

The hybrid addresses the specific weaknesses observed in the diagnostic:

1. **More efficient use of experience.** CEM reduces an entire rollout to one
   scalar and then discards most of the trajectory information. SAC learns
   from every transition and reuses transitions in multiple batches.
2. **Less dependence on lucky episodes.** Critic targets are averaged across a
   large replay buffer rather than determined by the rank of one or a few
   episode returns.
3. **Better handling of stochastic dynamics.** Experiences from Langevin noise,
   randomized starts, and parameter walks accumulate in the same buffer.
4. **Smoother policy improvement.** Gradient updates to a small actor readout
   make incremental changes instead of resampling a full readout matrix each
   generation.
5. **Preserved reservoir memory.** The reservoir can still represent recent
   dynamical history and infer information that is not explicit in a single
   observation.

This approach does not guarantee that one policy will solve every system. It
does, however, give the optimizer far more evidence with which to distinguish
general behavior from seed-specific luck.

## Environment and reward

The environment dynamics and reward should remain unchanged for the initial
comparison. The reservoir–SAC agent should receive the same observation and
action spaces and should be trained with the same:

- double-well systems;
- randomized initial states;
- parameter-walk behavior;
- Langevin noise;
- boundary metric;
- control penalty;
- episode horizon; and
- evaluation seeds.

Storing reservoir features in the replay buffer changes the agent's internal
representation, not the physical environment or reward.

## Fair experimental comparison

Four agents should be evaluated:

1. **Existing SAC:** the current multilayer actor and critics.
2. **Reservoir–SAC actor:** a fixed reservoir, linear Gaussian actor readouts,
   and small SAC critics.
3. **Reservoir–SAC actor and reservoir critics:** a more aggressive reservoir
   variant used as an ablation after the hybrid is working.
4. **Reservoir–CEM:** the current fixed reservoir and CEM-trained linear
   readout.

Fairness requires more than giving every method the same number of optimizer
updates. SAC and CEM use data differently. Results should therefore be plotted
against both:

- total environment transitions; and
- wall-clock training time.

Every method should be evaluated on a fixed held-out suite that contains the
same number of episodes from each system. At minimum, report:

- mean return across all systems;
- return and standard deviation for each system;
- worst-system return;
- mean boundary proximity;
- action RMS and saturation rate;
- termination rate;
- performance across several independent training seeds; and
- wall-clock time and number of environment transitions.

The primary checkpoint criterion should not be the raw mean alone. A robust
criterion such as

```text
mean(per_system_return) - lambda * std(per_system_return)
```

or the minimum per-system return will discourage a policy from sacrificing
Langevin and smooth biquadratic performance to improve Duffing and sextic.
This robust score is for checkpoint selection and reporting; it does not
require changing the environment reward.

## Recommended implementation stages

### Stage 1: fixed feature correctness

- Add an interface that returns reservoir features separately from actions.
- Confirm that the reservoir state persists across steps.
- Confirm that an episode reset clears the state.
- Confirm that identical seeded observation histories produce identical
  features.
- Store `z_t` and `z_(t+1)` in a small test replay buffer and verify their
  alignment with actions and rewards.

### Stage 2: reservoir–SAC training

- Implement the linear Gaussian actor readouts.
- Implement twin critics and target critics.
- Add automatic entropy-temperature tuning.
- Add replay warm-up before gradient updates.
- Save checkpoints using a balanced held-out validation suite.
- Preserve the most recent good checkpoint on interruption.

### Stage 3: controlled validation

- First test one deterministic Duffing system.
- Then test all systems without parameter walking.
- Enable parameter walking.
- Finally enable the complete Langevin and basin-mixing configuration.

This progression is diagnostic rather than a change to the final task. It
identifies the point at which learning becomes unstable.

### Stage 4: ablations

- Conventional reservoir versus physical Duffing reservoir.
- Raw observation skip connection versus reservoir features only.
- Small neural critics versus reservoir critics.
- With and without system identification in the observation.
- Different reservoir sizes, leaking rates, and spectral radii.
- Learned versus fixed entropy temperature.

## Success criteria

The hybrid should be considered promising if it satisfies all of the
following on held-out episodes:

- positive improvement over the untrained reservoir on every system;
- substantially better Langevin and smooth biquadratic performance than CEM;
- no large worst-system collapse hidden by the overall mean;
- consistent results across independent seeds;
- low action saturation; and
- equal or better sample efficiency than conventional SAC.

If the hybrid improves only Duffing and sextic, the bottleneck is likely the
shared representation or partial observability rather than CEM alone. If its
critics learn but the linear actor does not improve, the actor readout may need
one small hidden layer. If both actor and critics fail even on deterministic
Duffing, reservoir feature scaling and state-update alignment should be checked
before increasing training time.

## Expected outcome

The diagnostic already shows that the reservoir contains useful features:
short CEM training produced large improvements on two systems. The problem is
that CEM could not reliably identify readouts that generalize across seeds and
systems. Reservoir–SAC keeps the useful recurrent representation and replaces
the weakest part of the current method—the noisy population-level optimizer—
with transition-level off-policy learning.

The most practical first target is therefore a fixed reservoir, a SAC-trained
linear Gaussian actor, and small twin SAC critics. It retains the central
reservoir-computing idea while using SAC where SAC is strongest: extracting a
stable learning signal from a large, diverse replay buffer.

## Implemented files

The first practical version described above is implemented in:

- `src/reservoir_sac.py`: replay buffer, linear Gaussian actor, twin critics,
  entropy tuning, updates, and checkpoint persistence;
- `src/train_reservoir_sac_multi.py`: chronological reservoir interaction,
  off-policy training, balanced validation, and interruption-safe checkpoints;
- `src/evaluate_reservoir_sac_multi.py`: balanced held-out per-system evaluation;
  and
- `tests/test_reservoir_sac.py`: replay, recurrence, optimization, action-bound, and
  persistence tests.

The environment dynamics and reward implementation were not changed.
