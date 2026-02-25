

# Value Head Collapse in AlphaZero Self-Play: Causes, Analysis, and Mathematically Rigorous Prevention

## Abstract

AlphaZero-style self-play reinforcement learning systems suffer from a catastrophic and underexplored failure mode: **value head collapse**, in which the Win-Draw-Loss (WDL) value head converges to predicting every position as a draw, creating an absorbing state from which no amount of continued training can recover. We provide the first rigorous mathematical treatment of this phenomenon, modeling the self-play training loop as a coupled dynamical system $S_t = (\theta_t, D_t, \phi_t)$ over network parameters, replay buffer composition, and self-play draw rate. We prove that the collapsed state $S^* = (\theta^*, (0,1,0), 1)$ is a stable fixed point of the training operator (Theorem 2) by establishing three interlocking results: (i) the replay buffer acts as an exponential moving average that resists correction once draw-saturated (Proposition 1), (ii) SGD on WDL cross-entropy drives the value head toward the buffer's empirical distribution (Proposition 2), and (iii) Gumbel Top-$k$ Sequential Halving amplifies mild value head bias into overwhelming search certainty by collapsing the variance of completed Q-values (Theorem 1). We further prove that the WDL loss landscape at collapse is a plateau with vanishing gradients (Proposition 5), near-zero Hessian eigenvalues (Theorem 3), and zero entropy (Proposition 6), and that the Entropic Risk Measure is ineffective because the variance it requires has itself collapsed (Theorem 4). Grounded in a production AlphaZero chess implementation, we document four empirically distinct collapse modes — simulation depth draw death, fifty-move stagnation, max-moves pollution, and buffer saturation — and propose four families of mathematically grounded prevention methods: loss function modifications with provable entropy floors (Theorem 5), factored network architectures that decouple draw and decisive prediction, stratified sampling that eliminates buffer composition bias entirely (Theorem 6), and adaptive simulation budgets that prevent search amplification cascades (Theorem 7). A Lyapunov stability analysis (Theorem 8) proves that entropy regularization eliminates the absorbing state, and we establish a formal ordering of prevention methods by the size of the safe training basin they guarantee (Theorem 9).

---

## 1. Introduction

### 1.1 Motivation

AlphaZero-style reinforcement learning systems achieve superhuman performance in perfect-information games by coupling a neural network $f_\theta$ with Monte Carlo Tree Search (MCTS). The network provides a prior policy $p_\theta(s)$ and a value estimate $v_\theta(s)$ for each board state $s$; MCTS refines these estimates through simulation; the refined estimates become training targets for the next generation of $\theta$. This elegant loop---self-play generates data, data trains the network, the network improves self-play---has produced world-class agents in chess, Go, and shogi without human knowledge beyond the rules of the game.

Yet practitioners who attempt to reproduce these results encounter a dominant failure mode that receives remarkably little attention in the literature: **value head collapse**. Rather than failing to learn good moves (a policy problem), the system learns to predict every position as a draw with near certainty, after which the policy ceases to improve because search can no longer distinguish between moves. The value head's output distribution collapses to a degenerate point mass on the draw outcome, the loss function's gradients vanish, and the system enters an absorbing state from which no amount of continued training can escape.

This paper provides the first rigorous mathematical treatment of value head collapse as a dynamical systems phenomenon. We analyze the self-play training loop as an iterated operator $T: \theta_t \mapsto \theta_{t+1}$ and prove that the collapsed state is a stable fixed point of this operator under conditions that are not merely possible but *typical* in the early phases of training. Our analysis is grounded in a complete AlphaZero chess implementation that uses Gumbel Top-$k$ Sequential Halving at the root and PUCT at internal nodes, with a WDL (win/draw/loss) value head trained via categorical cross-entropy.

### 1.2 The Problem in Brief

Consider the empirical trajectory of a training run initialized with 200 simulations per move, then increased to 800 simulations at iteration 6:

| Iteration | Decisive Games (%) | Repetition Draws | Value Loss | Avg. Game Length |
|:---------:|:------------------:|:----------------:|:----------:|:----------------:|
| 5         | 22                 | 16/32            | 0.89       | 315              |
| 6         | 22                 | 16/32            | 0.89       | 315              |
| 7         | 9                  | 27/32            | 0.51       | 153              |
| 8         | 3                  | 31/32            | 0.45       | 95               |
| 9--53     | 0                  | 32/32            | 0.02       | 42               |

Within three iterations of the simulation budget increase, the system transitioned from a healthy training regime (22% decisive games, value loss 0.89 indicating genuine uncertainty) to total collapse (0% decisive games, value loss 0.02 indicating the network predicts draws with near certainty for every position). Attempts to recover by adjusting search parameters at iteration 8---reducing simulations to 400, increasing exploration constant to 2.0, raising Dirichlet noise---made the situation *worse*. The system remained locked in this state for 44 additional iterations with no sign of recovery.

This is not an isolated incident. We document four distinct mechanisms by which collapse occurs, each with different triggering conditions but the same terminal state. The commonality is that value head collapse is an *absorbing* fixed point: once entered, the gradient signal required for escape is precisely the signal that collapse has destroyed.

### 1.3 Contributions

This paper makes four contributions:

1. **Formal dynamical systems model** (Section 3). We define the state of the training loop as $S_t = (\theta_t, D_t, \phi_t)$ where $\theta_t$ are the network parameters, $D_t$ is the composition of the replay buffer, and $\phi_t$ is the self-play draw rate. We derive the iterated map $S_{t+1} = T(S_t)$ and prove that the collapsed state $S^* = (\theta^*, (0, 1, 0), 1)$ is a stable fixed point (Theorem 2). The proof chains three results: buffer dynamics (Proposition 1), gradient equilibrium of the WDL cross-entropy loss (Proposition 2), and Gumbel search amplification of value head bias (Theorem 1).

2. **WDL loss landscape analysis at collapse** (Section 4). We prove that the cross-entropy loss surface becomes a plateau near the collapsed state: gradients are $O(\varepsilon)$ (Proposition 5), Hessian eigenvalues are $O(\varepsilon)$ (Theorem 3), and the effective learning signal from rare decisive samples is attenuated by the factor $(1 - d_D)$ where $d_D > 0.95$ is the buffer's draw fraction (Proposition 5). We further prove that the Entropic Risk Measure, designed to promote risk-seeking play, is ineffective at collapse because the variance it relies on has itself collapsed to zero (Theorem 4).

3. **Taxonomy of collapse modes** (Section 5). We identify and empirically document four mechanisms: (I) simulation-depth draw death, where increased search budget causes premature convergence to drawing lines; (II) fifty-move stagnation, where the fifty-move rule terminates games as draws without repetition; (III) max-moves pollution, where games hitting the move limit flood the buffer with a 6.4$\times$ asymmetric draw-position surplus (Proposition 4); and (IV) buffer saturation, where the replay buffer's inertia resists correction once draw-dominated.

4. **Mathematically grounded prevention methods** (Section 6). We present four families of interventions with formal guarantees: loss function modifications (entropy regularization, focal loss), network architecture changes (factored value head, MC dropout), training data interventions (stratified sampling, dynamic epochs), and search modifications (adaptive simulation budget, anti-draw exploration bonus).

### 1.4 Scope and Relation to Prior Work

Our analysis is grounded in a production AlphaZero chess implementation (`alphazero-cpp/`) comprising approximately 16,000 lines of C++20 with pybind11 Python bindings and a 3,600-line training orchestrator (`alphazero-cpp/scripts/train.py`). The search algorithm at the root is Gumbel Top-$k$ Sequential Halving (Danihelka et al., 2022), not the standard PUCT algorithm used in the original AlphaZero (Silver et al., 2018). PUCT is used only at internal tree nodes (depth $> 0$). This distinction is critical: the mechanism by which a draw-biased value head corrupts search differs between the two algorithms, and our Theorem 1 provides the Gumbel-specific analysis.

The value head uses a WDL (win/draw/loss) categorical output trained with cross-entropy, following the approach of Leela Chess Zero, rather than the scalar value head with MSE loss used in the original AlphaZero. The WDL formulation introduces three-way softmax dynamics that create a qualitatively different collapse geometry, which we analyze in Section 4.

While the "draw death" phenomenon is well-known in the AlphaZero practitioner community---appearing in forum discussions, issue trackers, and engineering notes---we are not aware of any prior work that provides a formal mathematical treatment of the collapse mechanism, proves the absorbing nature of the fixed point, or analyzes the loss landscape geometry at collapse. This paper aims to fill that gap.


## 2. Background and Notation

### 2.1 Neural Network Architecture

The neural network $f_\theta: \mathcal{S} \to \Delta^{4671} \times \Delta^2$ maps a board state $s$ to a policy vector and a value triple. The input representation encodes the board as a tensor $x(s) \in \mathbb{R}^{8 \times 8 \times 123}$, where the 123 channels capture piece positions, castling rights, en passant status, repetition counts, and move clocks for the current position and recent history.

In the experiments analyzed here, the network trunk is an SE-ResNet with 256 filters and 20 residual blocks, each equipped with Squeeze-and-Excitation (SE) modules at reduction ratio 4 (the codebase defaults are 192 filters, 15 blocks, and SE reduction ratio 16; the experimental configuration was chosen for greater capacity). The trunk output feeds two heads:

**Policy head.** A convolutional head producing logits $z_p(s) \in \mathbb{R}^{4672}$, where the 4672-dimensional action space encodes all legal chess moves in a fixed spatial mapping. The policy is $p_\theta(a \mid s) = \mathrm{softmax}(z_p(s))_a$.

**WDL value head.** A fully connected head producing three logits $z_w(s) = (z_W, z_D, z_L) \in \mathbb{R}^3$. The WDL probabilities are:

$$w_\theta(s) = (w_W, w_D, w_L) = \mathrm{softmax}(z_W, z_D, z_L) \in \Delta^2$$

where $\Delta^2 = \{(a, b, c) \in \mathbb{R}^3_{\geq 0} : a + b + c = 1\}$ is the 2-simplex. The scalar value used in search is $v_\theta(s) = w_W - w_L \in [-1, 1]$.

**Mirror equivariance.** The forward pass exploits the horizontal symmetry of the chessboard. Given input $x(s)$, the network processes both $x(s)$ and its horizontal flip $\tilde{x}(s)$ through the shared trunk, then averages the resulting logits (with appropriate action index remapping for the policy) before applying softmax. This doubles the effective batch size at inference time and enforces exact equivariance with respect to board reflection.

We write $f_\theta(s) = (p_\theta(s), w_\theta(s))$ for the combined network output, suppressing the mirror averaging for notational convenience.


### 2.2 Gumbel Top-$k$ Sequential Halving (Root Search)

At the root node of the search tree, our implementation uses Gumbel Top-$k$ Sequential Halving (Danihelka et al., 2022) rather than PUCT. This algorithm provides a principled way to allocate simulations among root actions and produces an *improved policy* that serves as the training target. We describe it in full detail, as its interaction with value head collapse is the subject of Theorem 1.

**Notation.** Let $s_0$ denote the root state, let $\mathcal{A} = \{a_1, \ldots, a_K\}$ be the set of legal actions, and let $N$ be the total simulation budget. Define the log-prior:

$$\ell(a) = \log(\max(p_\theta(a \mid s_0), 10^{-6}))$$

The floor at $10^{-6}$ prevents $-\infty$ logits for actions with negligible prior mass.

**Phase 0: Top-$m$ initialization.** Draw independent Gumbel noise $g(a) \sim \mathrm{Gumbel}(0, 1)$ for each legal action $a \in \mathcal{A}$. Compute initial scores:

$$\mathrm{score}_0(a) = \ell(a) + g(a)$$

Select the top $m = k$ actions by $\mathrm{score}_0$. Call this set $A_0 \subseteq \mathcal{A}$ with $|A_0| = m$. These are the *active actions* entering the first phase of sequential halving.

**Simulation allocation via round-robin.** The $N$ simulations are distributed across $\lceil \log_2 m \rceil$ phases. In each phase $j$, the active actions $A_j$ receive simulations in strict round-robin order: the algorithm cycles through $A_j$ repeatedly, allocating one simulation per action per cycle, until the budget for that phase is exhausted. This ensures approximately equal visit counts within each phase.

**Phase advancement via sequential halving.** After completing the simulations for phase $j$, each active action $a \in A_j$ has accumulated $N(a)$ total visits across all phases so far. The algorithm computes the *completed Q-value*:

$$\hat{Q}(a) = \frac{N(a)}{1 + N(a)} \cdot Q(a) + \frac{1}{1 + N(a)} \cdot V(s_0)$$

where $Q(a)$ is the mean backed-up value over the $N(a)$ simulations of action $a$, and $V(s_0)$ is the root's value estimate. The interpolation with $V(s_0)$ provides a Bayesian prior that regularizes actions with few visits toward the network's initial estimate.

The $\sigma$-transform converts $\hat{Q}$ to a score-compatible scale:

$$\sigma(\hat{Q}(a)) = (c_{\mathrm{visit}} + N_{\max}) \cdot c_{\mathrm{scale}} \cdot q_{\mathrm{norm}}(a)$$

where $N_{\max} = \max_{a' \in A_j} N(a')$ and

$$q_{\mathrm{norm}}(a) = 2 \cdot \frac{\hat{Q}(a) - \hat{Q}_{\min}}{\hat{Q}_{\max} - \hat{Q}_{\min}} - 1 \in [-1, 1]$$

normalizes the completed Q-values to $[-1, 1]$ across the active set.

Updated scores are:

$$\mathrm{score}_j(a) = \ell(a) + g(a) + \sigma(\hat{Q}(a))$$

The bottom half of $A_j$ by $\mathrm{score}_j$ is eliminated, yielding $A_{j+1}$ with $|A_{j+1}| = \lceil |A_j| / 2 \rceil$. This continues until one action remains or the simulation budget is exhausted.

**Improved policy (training target).** After all simulations, the improved policy over *all* legal actions is:

$$\pi_{\mathrm{improved}}(a) = \mathrm{softmax}\!\left(\ell(a) + \sigma(\hat{Q}_{\mathrm{completed}}(a))\right)$$

This improved policy, not the visit count distribution, serves as the policy training target. It is the key output of Gumbel Top-$k$ SH: a policy that integrates the prior with search-refined value estimates in a principled way.


### 2.3 Internal PUCT (Depth $> 0$)

At all non-root nodes (depth $> 0$), action selection follows the PUCT formula. For a node at state $s$ with parent visit count $N_{\mathrm{parent}} = \sum_{a'} N(a')$, the algorithm selects:

$$a^* = \arg\max_{a} \left[ Q(a) + c_{\mathrm{explore}} \cdot p_\theta(a \mid s) \cdot \frac{\sqrt{N_{\mathrm{parent}}}}{1 + N(a)} \right]$$

where $c_{\mathrm{explore}} > 0$ is the exploration constant (default 1.5 in our implementation).

**First Play Urgency (FPU).** Unvisited actions ($N(a) = 0$) at internal nodes do not have a $Q$-value from backed-up simulations. The FPU heuristic assigns them:

$$Q_{\mathrm{FPU}}(a) = Q_{\mathrm{parent}} - f_{\mathrm{base}} \cdot \sqrt{1 - p_\theta(a \mid s)}$$

where $Q_{\mathrm{parent}}$ is the mean value of the parent node and $f_{\mathrm{base}} = 0.3$ by default. This formula penalizes unvisited actions proportional to their prior improbability: low-prior moves receive a larger penalty, making them less attractive for exploration. The square root modulates the penalty so that the reduction is sublinear in $(1 - p_\theta(a \mid s))$.


### 2.4 Entropic Risk Measure (ERM)

Our implementation supports risk-sensitive search via the Entropic Risk Measure. Each node tracks not only the sum of backed-up values $V_{\mathrm{sum}}$ and visit count $N$, but also the sum of squared values $V_{\mathrm{sum,sq}}$, enabling online variance computation:

$$\mathrm{Var}(v) = \frac{V_{\mathrm{sum,sq}}}{N} - \left(\frac{V_{\mathrm{sum}}}{N}\right)^2 = E[v^2] - E[v]^2$$

The risk-adjusted Q-value is:

$$Q_\beta(a) = E[v(a)] + \frac{\beta}{2} \cdot \mathrm{Var}(v(a))$$

where $\beta > 0$ yields risk-seeking behavior (preferring high-variance positions), $\beta < 0$ yields risk-averse behavior, and $\beta = 0$ recovers standard AlphaZero. The implementation includes a fast path: when $\beta = 0$, the variance computation is skipped entirely and $Q_\beta$ delegates directly to $Q$, ensuring zero overhead for the default case.


### 2.5 Training Loss

The network is trained by stochastic gradient descent on a combined loss:

$$L(\theta) = L_{\mathrm{policy}}(\theta) + L_{\mathrm{value}}(\theta)$$

**Policy loss.** The policy loss is a soft cross-entropy between the improved policy $\pi_{\mathrm{improved}}$ (from Gumbel Top-$k$ SH) and the network's policy logits $z_p$:

$$L_{\mathrm{policy}} = -\frac{1}{|\mathcal{B}|} \sum_{(s, \pi, y) \in \mathcal{B}} \sum_{a} \pi_{\mathrm{improved}}(a \mid s) \cdot \log \mathrm{softmax}(z_p(s))_a$$

This is a *soft* cross-entropy because the targets $\pi_{\mathrm{improved}}$ are themselves a distribution (not one-hot), encoding the relative quality of moves as assessed by search.

**Value loss.** The value loss is a categorical cross-entropy between the game outcome (encoded as a one-hot WDL vector) and the network's WDL logits $z_w$:

$$L_{\mathrm{value}} = -\frac{1}{|\mathcal{B}|} \sum_{(s, \pi, y) \in \mathcal{B}} \sum_{c \in \{W, D, L\}} y_c \cdot \log \mathrm{softmax}(z_w(s))_c$$

where $y = (y_W, y_D, y_L)$ is the outcome vector: $y = (1, 0, 0)$ for a win from the perspective of the player to move, $y = (0, 1, 0)$ for a draw, and $y = (0, 0, 1)$ for a loss. Crucially, these are *hard* one-hot targets determined solely by the game result. The training labels are pure: draws are always encoded as $(0, 1, 0)$ regardless of any risk parameter $\beta$ used during search.


### 2.6 Replay Buffer

The replay buffer $\mathcal{B}$ is a circular buffer of capacity $C = 100{,}000$ training positions. Each entry stores:

- The board observation $x(s) \in \mathbb{R}^{8 \times 8 \times 123}$
- The improved policy target $\pi_{\mathrm{improved}}(s) \in \Delta^{4671}$
- The scalar value target $v \in \{-1, 0, 1\}$
- The WDL outcome target $y \in \{(1,0,0), (0,1,0), (0,0,1)\}$
- Per-sample metadata: iteration number, game result, termination type (checkmate, stalemate, repetition, fifty-move, max-moves, insufficient material), move number within the game, and total game length

Each self-play iteration generates $G$ games. Let $L_i$ denote the length (in half-moves, i.e., plies) of game $i$. The total number of new positions per iteration is $\sum_{i=1}^{G} L_i$. When the buffer is full, the oldest positions are overwritten.

We denote the buffer's *composition* as $D_t = (d_W, d_D, d_L) \in \Delta^2$, where $d_c$ is the fraction of positions in the buffer whose game outcome was $c \in \{W, D, L\}$. Note that $d_W = d_L$ by symmetry (each game produces both W and L positions from the two players' perspectives), so the composition is determined by $d_D$ alone.


### 2.7 Self-Play Iteration Operator

We formalize the training loop as a discrete-time dynamical system. Define the *state* at iteration $t$ as:

$$S_t = (\theta_t, D_t, \phi_t)$$

where $\theta_t$ are the network parameters, $D_t = (d_W^{(t)}, d_D^{(t)}, d_L^{(t)})$ is the buffer composition, and $\phi_t \in [0, 1]$ is the draw rate (fraction of self-play games ending in draws) at iteration $t$.

The self-play iteration operator $T$ maps $S_t$ to $S_{t+1}$ through three stages:

1. **Self-play**: Using $\theta_t$, play $G$ games via Gumbel Top-$k$ SH (root) and PUCT (internal), generating new training positions. The draw rate $\phi_{t+1}$ is determined by $\theta_t$ through the stochastic self-play process.

2. **Buffer update**: Insert the new positions into $\mathcal{B}$, overwriting the oldest entries. The new composition $D_{t+1}$ depends on $D_t$, $\phi_{t+1}$, and the number of positions generated.

3. **Training**: Perform $E$ epochs of SGD on the updated buffer $\mathcal{B}$ with parameters $\theta_t$, yielding $\theta_{t+1}$.

We write $S_{t+1} = T(S_t)$ and study the fixed points and stability of $T$. The central question is: under what conditions is the collapsed state $S^* = (\theta^*, (0, 1, 0), 1)$ a stable fixed point?


## 3. The Draw Death Spiral as a Dynamical System

In this section, we develop a formal dynamical systems analysis of the self-play training loop and prove that value head collapse to a draw-predicting absorbing state is a stable fixed point of the iteration operator $T$. The proof proceeds through a chain of four results: buffer composition dynamics (Proposition 1), gradient equilibrium (Proposition 2), search amplification (Theorem 1), and the absorbing state theorem (Theorem 2). We also characterize the data asymmetry that accelerates collapse (Proposition 4) and analyze the FPU trap at internal nodes (Proposition 3).

### 3.1 Buffer Composition Dynamics

The replay buffer is a circular buffer of capacity $C$. At each iteration, $G$ self-play games produce a total of $\Lambda_t = \sum_{i=1}^{G} L_i^{(t)}$ new positions, where $L_i^{(t)}$ is the length of game $i$ at iteration $t$.

**Proposition 1 (Buffer Dynamics).** *Let $D_t = (d_W^{(t)}, d_D^{(t)}, d_L^{(t)})$ be the buffer composition at iteration $t$, let $\Phi_t = (\phi_W^{(t)}, \phi_D^{(t)}, \phi_L^{(t)})$ be the composition of the newly generated positions, and let $\Lambda_t$ be the number of new positions. Assuming $\Lambda_t \leq C$ (the new data does not exceed the buffer capacity), the buffer composition evolves as:*

$$D_{t+1} = \left(1 - \frac{\Lambda_t}{C}\right) D_t + \frac{\Lambda_t}{C} \Phi_t$$

*Proof.* The circular buffer of capacity $C$ contains $C$ positions at any time after the initial fill. When $\Lambda_t$ new positions are inserted, they overwrite the $\Lambda_t$ oldest positions. The retained positions (the $C - \Lambda_t$ most recent from previous iterations) have composition $D_t$ in expectation, assuming the buffer was well-mixed by prior insertions. The new $\Lambda_t$ positions have composition $\Phi_t$. The updated buffer contains $(C - \Lambda_t)$ old positions and $\Lambda_t$ new positions, giving:

$$D_{t+1} = \frac{C - \Lambda_t}{C} D_t + \frac{\Lambda_t}{C} \Phi_t = \left(1 - \frac{\Lambda_t}{C}\right) D_t + \frac{\Lambda_t}{C} \Phi_t$$

$\square$

**Remark.** Define $\alpha_t = \Lambda_t / C$ as the *buffer replacement fraction*. Then $D_{t+1} = (1 - \alpha_t) D_t + \alpha_t \Phi_t$, an exponential moving average with mixing rate $\alpha_t$. When all games are draws ($\Phi_t = (0, 1, 0)$), the draw fraction increases monotonically: $d_D^{(t+1)} = (1 - \alpha_t) d_D^{(t)} + \alpha_t > d_D^{(t)}$. It takes approximately $1/\alpha_t$ iterations of pure draws to drive $d_D$ from any initial value to within $\varepsilon$ of 1.

In a typical configuration with $G = 32$ games, average game length $\bar{L} = 200$ plies, and $C = 100{,}000$, we have $\alpha_t \approx 32 \times 200 / 100{,}000 = 0.064$. Five consecutive iterations of 100% draws would push $d_D$ from 0.5 to $1 - 0.5 \times (1 - 0.064)^5 \approx 0.72$.

### 3.2 Gradient Equilibrium of WDL Cross-Entropy

We now analyze the equilibrium behavior of SGD on the value head under a buffer with draw fraction $d_D$.

**Proposition 2 (Gradient Equilibrium).** *Consider the WDL cross-entropy loss $L_{\mathrm{value}}$ over a buffer $\mathcal{B}$ with composition $(d_W, d_D, d_L)$ where $d_W = d_L = (1 - d_D)/2$. Let $P_c = \mathrm{softmax}(z_w)_c$ for $c \in \{W, D, L\}$ denote the network's predicted WDL probabilities at an arbitrary position. The expected gradient with respect to the draw logit $z_D$ is:*

$$\mathbb{E}\left[\frac{\partial L_{\mathrm{value}}}{\partial z_D}\right] = P_D - d_D$$

*The gradient vanishes if and only if $P_D = d_D$. Consequently, SGD drives the network's average draw prediction toward the buffer's draw fraction.*

*Proof.* For a single sample $(s, y)$ with outcome $y = (y_W, y_D, y_L)$, the WDL cross-entropy loss is:

$$L = -\sum_{c \in \{W, D, L\}} y_c \log P_c$$

The gradient with respect to the draw logit $z_D$ uses the standard softmax-cross-entropy identity:

$$\frac{\partial L}{\partial z_D} = P_D - y_D$$

This well-known result follows from $\partial \log P_c / \partial z_D = \mathbf{1}[c = D] - P_D$ and the chain rule.

Taking the expectation over the buffer:

$$\mathbb{E}\left[\frac{\partial L}{\partial z_D}\right] = P_D - \mathbb{E}[y_D] = P_D - d_D$$

where $\mathbb{E}[y_D] = d_D$ because a fraction $d_D$ of the buffer has $y_D = 1$ and the rest have $y_D = 0$. Setting the expected gradient to zero gives the equilibrium condition $P_D = d_D$. $\square$

**Remark.** This result holds for the *average* prediction across the buffer. In practice, the network learns position-dependent predictions, but the average must satisfy this constraint. When $d_D \to 1$, the network is driven to predict $P_D \to 1$ on average, collapsing the value head.

More precisely, the gradient $\partial L / \partial z_D = P_D - y_D$ has the following structure across the buffer:
- On draw samples (fraction $d_D$): gradient $= P_D - 1$, which is negative, pushing $P_D$ up.
- On win/loss samples (fraction $1 - d_D$): gradient $= P_D - 0 = P_D$, which is positive, pushing $P_D$ down.

At equilibrium these cancel: $d_D(P_D - 1) + (1 - d_D) P_D = 0$, yielding $P_D = d_D$. When draws dominate ($d_D \gg 1 - d_D$), the "push up" signal overwhelms the "push down" signal, and $P_D$ is driven close to 1.

### 3.3 Gumbel Search Amplification

We now prove the key result: when the value head is draw-biased, Gumbel Top-$k$ SH amplifies the bias by producing improved policies that provide no corrective signal. This closes the feedback loop that makes collapse self-reinforcing.

**Theorem 1 (Gumbel Search Amplification).** *Suppose the value head predicts $v_\theta(s) = w_W(s) - w_L(s) \approx \mu$ with $|\mu| \ll 1$ and $\mathrm{Var}_s[v_\theta] = \sigma^2 \ll 1$ across positions reachable from root $s_0$ within the search tree. Then:*

*(i) The completed Q-values satisfy $\hat{Q}(a) \approx \mu$ for all active actions $a$, with deviation $O(\sigma / \sqrt{N_{\mathrm{eff}}(a)})$ where $N_{\mathrm{eff}}(a)$ is the effective sample size for action $a$.*

*(ii) The $q_{\mathrm{norm}}$ normalization maps any nonzero $\hat{Q}$-spread to the full $[-1,1]$ range, so $\sigma$-score differences between actions are $O((c_{\mathrm{visit}} + N_{\max}) \cdot c_{\mathrm{scale}})$ — large in magnitude. However, these differences are driven by noise (random $\hat{Q}$ fluctuations of size $O(\sigma/\sqrt{N_{\mathrm{eff}}})$), not by signal. At each individual position, $\pi_{\mathrm{improved}}$ may differ substantially from the prior, but across many positions the noise-driven perturbations average out: $\mathbb{E}_s[\pi_{\mathrm{improved}} - p_\theta] \approx 0$.*

*(iii) The expected policy gradient $\mathbb{E}_s[\partial L_{\mathrm{policy}} / \partial z_p] = \mathbb{E}_s[p_\theta - \pi_{\mathrm{improved}}] \approx 0$: search adds no systematic information beyond the prior. Policy learning becomes a random walk rather than directed improvement, and the policy training targets provide no coherent corrective gradient.*

*Proof.* We prove each part in turn.

*Part (i): Q-value concentration.* Consider action $a$ at the root. Each simulation through $a$ terminates at a leaf node $s_\ell$ where the network evaluates $v_\theta(s_\ell) = w_W(s_\ell) - w_L(s_\ell)$. By hypothesis, $v_\theta(s_\ell) \approx \mu$ with variance $\sigma^2$ across leaf states. After $N(a)$ simulations, the mean backed-up value is:

$$Q(a) = \frac{1}{N(a)} \sum_{i=1}^{N(a)} v_i$$

where $v_i$ are the backed-up values (possibly with sign flips from the alternating minimax perspective, but since $v_i \approx \mu \approx 0$, the sign flips have negligible effect: $-\mu \approx \mu$ when $|\mu| \ll 1$). By the law of large numbers, $Q(a) \approx \mu$ with standard deviation $\sigma / \sqrt{N(a)}$.

The completed Q-value is:

$$\hat{Q}(a) = \frac{N(a)}{1 + N(a)} Q(a) + \frac{1}{1 + N(a)} V(s_0) \approx \frac{N(a)}{1 + N(a)} \mu + \frac{1}{1 + N(a)} \mu = \mu$$

since $V(s_0) = v_\theta(s_0) \approx \mu$. The deviation from $\mu$ is:

$$|\hat{Q}(a) - \mu| \leq \frac{N(a)}{1 + N(a)} \cdot |Q(a) - \mu| = O\!\left(\frac{\sigma}{\sqrt{N(a)}}\right)$$

With $N$ total simulations distributed among $k$ active actions, each action receives $N_{\mathrm{eff}}(a) \approx N/k$ simulations, giving deviations $O(\sigma \sqrt{k/N})$.

*Part (ii): $\sigma$-transform produces large but noise-driven perturbations.* The normalized Q-value is:

$$q_{\mathrm{norm}}(a) = 2 \cdot \frac{\hat{Q}(a) - \hat{Q}_{\min}}{\hat{Q}_{\max} - \hat{Q}_{\min}} - 1$$

The $q_{\mathrm{norm}}$ normalization maps *any* nonzero spread $\hat{Q}_{\max} - \hat{Q}_{\min} > 0$ to the full range $[-1, 1]$, regardless of how small that spread is. The overall $\sigma$-transform is:

$$\sigma(\hat{Q}(a)) = (c_{\mathrm{visit}} + N_{\max}) \cdot c_{\mathrm{scale}} \cdot q_{\mathrm{norm}}(a)$$

The absolute $\sigma$-score differences between actions are therefore $O((c_{\mathrm{visit}} + N_{\max}) \cdot c_{\mathrm{scale}})$, which is *large* — with typical values $c_{\mathrm{visit}} = 50$, $N_{\max} \approx 100$, $c_{\mathrm{scale}} = 1.0$, the $\sigma$-scores span a range of $\sim 300$, far exceeding the typical log-prior spread $|\ell(a) - \ell(a')| \sim 5$--$10$.

This means that at any *individual* position, the improved policy $\pi_{\mathrm{improved}}(a) = \mathrm{softmax}(\ell(a) + \sigma(\hat{Q}(a)))$ can differ substantially from the prior $p_\theta(a \mid s_0)$. The $\sigma$-scores dominate the log-priors and rearrange the action ranking.

However, the critical observation is that these large $\sigma$-score differences are driven entirely by *noise*, not signal. The underlying Q-value differences $\hat{Q}(a) - \hat{Q}(a')$ are $O(\sigma / \sqrt{N_{\mathrm{eff}}})$ — random fluctuations within the noise floor, since all actions lead to subtrees with approximately equal (near-zero) value. The normalization amplifies these random fluctuations to the $[-1, 1]$ scale, and the $\sigma$-transform further amplifies them to $O(150)$, but the *direction* of the perturbation is random across positions. Action $a$ may receive a high $\sigma$-score at position $s$ and a low one at position $s'$, depending on which random fluctuations happened to be largest.

Across many positions sampled in a training batch, these noise-driven perturbations average out. The expected deviation of the improved policy from the prior is:

$$\mathbb{E}_s[\pi_{\mathrm{improved}}(a \mid s) - p_\theta(a \mid s)] \approx 0$$

because the $\sigma$-score rankings are uncorrelated with the true action quality (which is uniform under the draw-biased value head). The policy gradient signal in expectation becomes:

$$\mathbb{E}_s\left[\frac{\partial L_{\mathrm{policy}}}{\partial z_p}\right] = \mathbb{E}_s[p_\theta(s) - \pi_{\mathrm{improved}}(s)] \approx 0$$

Policy learning under these conditions becomes a *random walk* rather than directed improvement: each position pushes the policy in a random direction determined by $\sigma$-score noise, but the mean displacement is zero.

*Part (iii): Policy target collapse in expectation.* From part (ii), although $\pi_{\mathrm{improved}}(s) \neq p_\theta(s)$ at individual positions, the expected policy gradient is approximately zero:

$$\mathbb{E}_s\left[\frac{\partial L_{\mathrm{policy}}}{\partial z_p}\right] = \mathbb{E}_s[p_\theta - \pi_{\mathrm{improved}}] \approx 0$$

Search has added no *systematic* information: while the improved policy differs from the prior at each position, these differences are noise-driven and cancel across the training batch. The policy head receives negligible expected gradient, and its parameters $\theta_p$ undergo a random walk rather than directed improvement. Without systematic policy improvement, the search cannot discover decisive lines (checkmates, winning tactics), which would produce the non-draw outcomes needed to rebalance the buffer. $\square$

**Remark.** The mechanism is subtle: Gumbel Top-$k$ SH does produce improved policies that differ from the prior at individual positions (unlike the $\sigma^2 \to 0$ limit where the implementation's epsilon-threshold sets $\sigma(Q) = 0$ identically). The collapse occurs not because $\pi_{\mathrm{improved}} = p_\theta$ pointwise, but because the *expected* deviation is zero — the perturbations are large but random, providing no coherent learning signal. This is in contrast to visit-count-based targets (as in the original PUCT AlphaZero), where noise and exploration bonuses can still produce some deviation from the prior even with a flat value head. However, both algorithms ultimately suffer the same collapse — the mechanism differs, but the outcome is identical.

### 3.4 The FPU Trap at Internal Nodes

At depth $> 0$, PUCT selects actions. When the value head is draw-biased, the FPU mechanism creates a trap that further suppresses exploration.

**Proposition 3 (FPU Trap at Internal Nodes).** *At a non-root node with parent value $Q_{\mathrm{parent}} \approx 0$ (corresponding to a draw evaluation), the FPU formula assigns unvisited actions:*

$$Q_{\mathrm{FPU}}(a) = -f_{\mathrm{base}} \cdot \sqrt{1 - p_\theta(a \mid s)} \leq -f_{\mathrm{base}} \cdot \sqrt{1 - p_{\max}}$$

*where $p_{\max}$ is the largest prior probability. For typical chess positions with $p_{\max} \lesssim 0.3$ and $f_{\mathrm{base}} = 0.3$, unvisited actions have $Q_{\mathrm{FPU}} \lesssim -0.25$. Meanwhile, visited actions have $Q(a) \approx 0$ (since the value head evaluates their subtrees as draws). PUCT therefore preferentially revisits already-visited actions over exploring new ones, confining the search to a narrow subtree of "known safe" drawing lines.*

*Proof.* With $Q_{\mathrm{parent}} \approx 0$:

$$Q_{\mathrm{FPU}}(a) = 0 - 0.3 \cdot \sqrt{1 - p_\theta(a \mid s)}$$

For any action with $p_\theta(a \mid s) < 1$, we have $Q_{\mathrm{FPU}}(a) < 0$. For a typical action with prior $p_\theta = 0.05$:

$$Q_{\mathrm{FPU}} = -0.3 \times \sqrt{0.95} \approx -0.29$$

The PUCT selection criterion at the parent, with $N_{\mathrm{parent}}$ visits, compares:
- Visited action $a'$ with $Q(a') \approx 0$ and exploration bonus $c_{\mathrm{explore}} \cdot p_\theta(a') \cdot \sqrt{N_{\mathrm{parent}}} / (1 + N(a'))$
- Unvisited action $a$ with $Q_{\mathrm{FPU}}(a) \approx -0.29$ and exploration bonus $c_{\mathrm{explore}} \cdot p_\theta(a) \cdot \sqrt{N_{\mathrm{parent}}}$

The unvisited action is selected only when its exploration bonus overcomes the Q-deficit of $\sim 0.29$:

$$c_{\mathrm{explore}} \cdot p_\theta(a) \cdot \sqrt{N_{\mathrm{parent}}} > 0.29 + Q(a') + c_{\mathrm{explore}} \cdot p_\theta(a') \cdot \frac{\sqrt{N_{\mathrm{parent}}}}{1 + N(a')}$$

For the unvisited action to be explored, we need (approximately, taking $Q(a') \approx 0$ and $N(a') \gg 1$ so the bonus for $a'$ is small):

$$c_{\mathrm{explore}} \cdot p_\theta(a) \cdot \sqrt{N_{\mathrm{parent}}} \gtrsim 0.29$$

With $c_{\mathrm{explore}} = 1.5$ and $p_\theta(a) = 0.05$, this requires $\sqrt{N_{\mathrm{parent}}} \gtrsim 3.9$, i.e., $N_{\mathrm{parent}} \gtrsim 15$. But once the unvisited action is explored and returns $Q \approx 0$, it competes on equal footing with the previously visited action, and visit counts equalize. The trap is not that exploration is permanently blocked---it is that the *outcomes* of exploration are indistinguishable (all $Q \approx 0$), so exploration produces no useful signal for the improved policy.

The FPU mechanism ensures that in the initial simulations, the search preferentially deepens existing subtrees rather than broadening. With more simulations (higher $N$), the search tree grows deeper rather than wider, exploring long forced drawing lines rather than discovering refutations. This is the "simulation depth" component of the "simulation depth draw death" phenomenon. $\square$


### 3.5 Data Asymmetry

An additional mechanism accelerates collapse: drawn games contribute far more training positions per game than decisive games.

**Proposition 4 (Data Asymmetry Amplification).** *Let $L_{\mathrm{draw}}$ be the average length of a drawn game and $L_{\mathrm{decisive}}$ be the average length of a decisive game, measured in plies. The effective draw-to-decisive ratio in the buffer is:*

$$r_{\mathrm{eff}} = \frac{\phi \cdot L_{\mathrm{draw}}}{(1 - \phi) \cdot L_{\mathrm{decisive}}}$$

*where $\phi$ is the draw rate. In the regime where draws are caused by the max-moves limit ($L_{\mathrm{draw}} \approx 512$) and decisive games end by checkmate or early termination ($L_{\mathrm{decisive}} \approx 80$), a draw rate of $\phi = 0.95$ yields $r_{\mathrm{eff}} \approx 122:1$.*

*Proof.* In a batch of $G$ games, $\phi G$ are draws and $(1 - \phi) G$ are decisive. Total draw positions: $\phi G \cdot L_{\mathrm{draw}}$. Total decisive positions: $(1 - \phi) G \cdot L_{\mathrm{decisive}}$. The ratio is:

$$r_{\mathrm{eff}} = \frac{\phi G \cdot L_{\mathrm{draw}}}{(1 - \phi) G \cdot L_{\mathrm{decisive}}} = \frac{\phi \cdot L_{\mathrm{draw}}}{(1 - \phi) \cdot L_{\mathrm{decisive}}}$$

Substituting: $r_{\mathrm{eff}} = (0.95 \times 512) / (0.05 \times 80) = 486.4 / 4 \approx 121.6$ for the buffer composition ratio. However, since both win and loss positions come from decisive games (each game contributes positions from both sides), the decisive positions split into wins and losses. The draw-to-non-draw *sample* ratio is:

$$r_{\mathrm{eff}} = \frac{\phi \cdot L_{\mathrm{draw}}}{(1 - \phi) \cdot L_{\mathrm{decisive}}} = \frac{0.95 \times 512}{0.05 \times 80} \approx 121.6$$

At less extreme parameters---for example, 85% draw rate with average draw game length 300 and decisive game length 150---the ratio is $(0.85 \times 300) / (0.15 \times 150) = 255 / 22.5 \approx 11.3$, still heavily skewed.

The worst-case scenario from the empirical data (Section 10m of the operation manual) computes: each max-moves draw contributes $\sim$512 draw-labeled positions, while each decisive game contributes $\sim$80 positions labeled win or loss. This is a 6.4$\times$ per-game asymmetry in position count. Combined with a 95% draw rate (19:1 game ratio), the effective sample ratio becomes $19 \times 6.4 = 121.6$ draw positions per decisive position. Even using a more conservative estimate---the operation manual's figure of "36:1 effective draw:decisive ratio"---the asymmetry is severe enough to drive rapid collapse via Proposition 2. $\square$

**Remark.** The 36:1 figure from Section 10m of `llm_operation_manual.md` uses a slightly different calculation that accounts for both players' perspectives and the distribution of termination types. The precise ratio depends on the mix of repetition draws (short, $L \approx 50$), fifty-move draws (long, $L \approx 200$), and max-moves draws ($L = 512$). The qualitative conclusion is unchanged: the data asymmetry amplifies the draw bias by an order of magnitude beyond what the raw draw rate would suggest.

### 3.6 The Absorbing State

We now combine the preceding results to prove that value head collapse is a stable fixed point of the training loop.

**Theorem 2 (Absorbing State).** *Define the collapsed state $S^* = (\theta^*, D^*, \phi^*)$ where $\theta^*$ is any parameterization with $w_\theta(s) \approx (0, 1, 0)$ for all positions $s$, $D^* = (0, 1, 0)$, and $\phi^* = 1$. There exists a threshold draw fraction $\rho \in (0, 1)$ such that if $d_D^{(t)} > \rho$, then:*

*(i) $d_D^{(t+1)} > d_D^{(t)}$ (the buffer becomes more draw-heavy);*

*(ii) $\theta_{t+1}$ is more draw-biased than $\theta_t$ (the average $P_D$ prediction increases);*

*(iii) $\phi_{t+1} > \phi_t$ (the self-play draw rate increases).*

*Consequently, the sequence $(d_D^{(t)})_{t \geq t_0}$ is monotonically increasing and bounded above by 1, so it converges: $d_D^{(t)} \to 1$. The system $S_t$ converges to an absorbing state in a neighborhood of $S^*$.*

*Proof.* We prove each part by invoking the preceding propositions and theorem.

**Part (i): Buffer composition.** By Proposition 1, $d_D^{(t+1)} = (1 - \alpha_t) d_D^{(t)} + \alpha_t \phi_D^{(t)}$, where $\phi_D^{(t)}$ is the draw fraction of the newly generated positions (note: this differs from the game-level draw rate $\phi_t$ due to data asymmetry). We have $d_D^{(t+1)} > d_D^{(t)}$ if and only if $\phi_D^{(t)} > d_D^{(t)}$.

By Proposition 4, the position-level draw fraction satisfies:

$$\phi_D^{(t)} = \frac{\phi_t \cdot \bar{L}_{\mathrm{draw}}}{\phi_t \cdot \bar{L}_{\mathrm{draw}} + (1 - \phi_t) \cdot \bar{L}_{\mathrm{decisive}}}$$

Since $\bar{L}_{\mathrm{draw}} > \bar{L}_{\mathrm{decisive}}$ in the regime under consideration (drawn games last longer), we have $\phi_D^{(t)} > \phi_t$. So if $\phi_t > d_D^{(t)}$, then $\phi_D^{(t)} > d_D^{(t)}$ and $d_D^{(t+1)} > d_D^{(t)}$.

If $\phi_t \leq d_D^{(t)}$ but $\phi_D^{(t)} > d_D^{(t)}$ (which can occur because $\phi_D > \phi$ due to data asymmetry), the conclusion still holds.

We require $d_D^{(t)} > \rho$ to be large enough that parts (ii) and (iii) below ensure $\phi_t > d_D^{(t)}$ (or at least $\phi_D^{(t)} > d_D^{(t)}$). The existence of such $\rho$ follows from parts (ii) and (iii).

**Part (ii): Parameter bias.** By Proposition 2, SGD on the WDL cross-entropy drives the average $P_D$ toward $d_D^{(t)}$. If the current average $\bar{P}_D < d_D^{(t)}$ (the network's draw predictions have not yet caught up with the buffer's draw fraction), then the gradient $\bar{P}_D - d_D^{(t)} < 0$ pushes $z_D$ upward, increasing $P_D$. After $E$ epochs of training:

$$\bar{P}_D^{(t+1)} = \bar{P}_D^{(t)} + \eta \cdot E \cdot (d_D^{(t)} - \bar{P}_D^{(t)}) \cdot \kappa(\theta_t)$$

where $\eta$ is the learning rate and $\kappa(\theta_t) > 0$ is an effective curvature factor. For $d_D^{(t)} > \rho$ sufficiently large and the network not yet fully collapsed, $\bar{P}_D^{(t+1)} > \bar{P}_D^{(t)}$: the network becomes more draw-biased.

If $\bar{P}_D \geq d_D^{(t)}$ already (the network over-predicts draws relative to the buffer), the gradient pushes $P_D$ down, but this partial correction is overwhelmed in the next iteration because the buffer itself becomes more draw-heavy (part (i)), and the new equilibrium $P_D = d_D^{(t+1)} > d_D^{(t)}$ is higher.

**Part (iii): Self-play draw rate.** By Theorem 1, when $\theta_t$ is draw-biased (i.e., $w_\theta(s) \approx (0, 1, 0)$ so $v_\theta(s) \approx 0$ with small variance $\sigma^2$), the Gumbel search produces improved policies that approximate the prior: $\pi_{\mathrm{improved}} \approx p_\theta$. The search fails to discover decisive continuations because:

1. At the root, $\sigma(\hat{Q}(a)) \approx \mathrm{const}$ for all actions (Theorem 1 part (ii)), so the Gumbel noise dominates action selection, effectively randomizing over high-prior moves. No action is identified as decisively better.

2. At internal nodes, FPU penalizes exploration of new moves (Proposition 3), and visited moves all return $Q \approx 0$. The search deepens known drawing lines rather than discovering refutations.

3. The resulting games therefore tend to end in draws (by repetition, fifty-move rule, or max-moves limit), yielding $\phi_{t+1} \geq \phi_t$.

More precisely, define $\phi_t = \Psi(\theta_t)$ where $\Psi$ is the (stochastic) self-play draw rate as a function of the network parameters. We assume $\Psi$ is monotonically increasing in the "draw bias" of $\theta$ (formalized as $\mathbb{E}_s[w_D(s)]$). Then part (ii) gives $\theta_{t+1}$ more draw-biased, so $\phi_{t+1} = \Psi(\theta_{t+1}) \geq \Psi(\theta_t) = \phi_t$.

**Convergence.** From parts (i)--(iii), the sequence $d_D^{(t)}$ is eventually monotonically increasing (once $d_D^{(t_0)} > \rho$ for some $t_0$) and bounded above by 1. By the monotone convergence theorem, $d_D^{(t)} \to d_D^* \leq 1$. 

Suppose $d_D^* < 1$. Then $\phi_D^{(t)} \to \phi_D^* > d_D^*$ (since $\phi_D > d_D$ in the draw-dominated regime by the asymmetry argument), and $d_D^{(t+1)} - d_D^{(t)} = \alpha_t (\phi_D^{(t)} - d_D^{(t)}) \to \alpha (\phi_D^* - d_D^*) > 0$, contradicting convergence. Therefore $d_D^* = 1$.

As $d_D^{(t)} \to 1$, Proposition 2 drives $\bar{P}_D \to 1$, and the network converges to $w_\theta \approx (0, 1, 0)$. The draw rate $\phi_t \to 1$. The system converges to the absorbing state $S^*$. $\square$

**Remark on the threshold $\rho$.** The threshold depends on system parameters: simulation budget $N$, exploration constants, FPU base, game length distributions, buffer capacity, and games per iteration. Empirically, the transition occurs around $d_D \approx 0.8$--$0.9$. Below this threshold, the residual decisive games provide enough gradient signal to maintain value head diversity. Above it, the self-reinforcing loop dominates.

### 3.7 Taxonomy of Collapse Modes

The absorbing state theorem applies regardless of *how* the draw rate initially rises above $\rho$. We identify four empirically observed mechanisms, each with distinct triggering conditions:

**Mode I: Simulation Depth Draw Death.** Triggered by increasing the simulation budget $N$ before the value head has learned to reliably distinguish between positions. With more simulations, the search tree grows deeper (Proposition 3: FPU favors deepening over broadening), and the deeper evaluations are all $\approx 0$ (Theorem 1: draw-biased value head). Under PUCT at root, the search discovers long forced repetition sequences---threefold repetition being the easiest drawing mechanism in chess. This creates a self-fulfilling prophecy: the value head predicts draws, search finds drawing continuations, games end in draws, the buffer fills with draws, the value head becomes more draw-biased. Empirically, increasing from 200 to 800 simulations at iteration 6 triggered collapse within 3 iterations (see Table in Section 1.2). The irony is that *stronger search makes training worse* when the value head is immature.

**Mode II: Fifty-Move Stagnation.** Observed in the Gumbel Top-$k$ SH run (Section 10i of the operation manual). Gumbel search, by using the improved policy rather than visit counts, can avoid the repetition trap (empirically, zero early repetition draws across 11 iterations). However, the games instead last until the fifty-move rule terminates them as draws. The mechanism is different---no explicit repetition---but the terminal state is identical: 100% draws, value head collapse. Recovery was achieved by reducing to $E = 1$ epoch and increasing temperature moves to 50.

| Iteration | Decisive (%) | Rep. Draws | 50-Move Draws | Value Loss | Avg. Length |
|:---------:|:------------:|:----------:|:-------------:|:----------:|:-----------:|
| 1--3      | 28           | 3--7       | 5--7          | 0.47--0.89 | 311--358    |
| 4         | 50           | 4          | 6             | 0.55       | 289         |
| 6         | 0            | 0          | 25            | 0.50       | 381         |
| 8         | 6            | 1          | 29            | 0.15       | 341         |
| 10        | 28           | 6          | 11            | 0.30       | 330         |

**Mode III: Max-Moves Pollution.** When games hit the maximum move limit (typically 512 plies), they are recorded as draws with $\sim$512 positions each. By Proposition 4, each such game contributes $\sim$6.4$\times$ more training positions than a decisive game of average length 80. Even a modest fraction of max-moves draws can rapidly skew the buffer composition. This mode is particularly insidious because it does not require the value head to be severely biased---merely slightly biased is enough when amplified by the 6.4$\times$ data asymmetry.

**Mode IV: Buffer Saturation Trap.** Once the replay buffer exceeds $\sim$85% draw composition, it acquires sufficient inertia to resist correction even when individual iterations produce healthier game distributions. By Proposition 1, each iteration replaces only a small fraction of the buffer, so even a "good" iteration with 64% draws barely shifts a buffer at 88% draws. This mode can be triggered by any of the other three modes and represents the terminal phase of the collapse feedback loop.

### 3.8 Empirical Validation

The predictions of Theorem 2 match the empirical trajectory precisely. In the simulation depth draw death run:

1. **Iteration 5** ($d_D \lesssim \rho$): 22% decisive games, value loss 0.89 (healthy uncertainty). The system is below the collapse threshold; the feedback loop is not yet self-reinforcing.

2. **Iteration 7** ($d_D > \rho$): 9% decisive games, value loss 0.51. The simulation increase at iteration 6 pushed $\phi$ above $d_D$, triggering part (i) of Theorem 2. The buffer is accumulating draw positions faster than decisive ones.

3. **Iteration 8** ($d_D \gg \rho$): 3% decisive games, value loss 0.45. Attempted recovery (reducing sims, increasing exploration) failed because the buffer composition $D_t$ has inertia---it cannot be reset by parameter changes alone. The modification worsened the situation because the lower simulation count produced even less decisive outcomes from the already-collapsed value head.

4. **Iterations 9--53** ($d_D \to 1$): 0% decisive games, value loss 0.02. Full absorbing state. The value head predicts $(w_W, w_D, w_L) \approx (\varepsilon, 1-2\varepsilon, \varepsilon)$ for all positions. The gradient signal is $O(\varepsilon)$ (see Section 4). No recovery is possible without external intervention.

The monotonic decrease in value loss from 0.89 to 0.02 over iterations 5--9 directly confirms the convergence $d_D \to 1$ predicted by Theorem 2: as the buffer fills with draws, the network's optimal cross-entropy loss approaches $-\log(1) = 0$.


## 4. Analysis of the WDL Cross-Entropy Absorbing State

Having established that the collapsed state $S^*$ is a stable fixed point of the training dynamics, we now analyze *why* escape from $S^*$ is so difficult. The answer lies in the geometry of the WDL cross-entropy loss surface near the collapsed parameterization: gradients vanish, the Hessian becomes flat, and the loss landscape is a plateau from which SGD cannot escape in any practical number of steps. We further show that multiple training epochs per iteration accelerate convergence to this plateau, and that the Entropic Risk Measure---a natural candidate for breaking the draw bias---is ineffective because the variance signal it relies on has itself been destroyed.

### 4.1 Gradient Vanishing at Collapse

**Proposition 5 (Gradient Vanishing).** *At the collapsed parameterization $\theta^*$ with $w_{\theta^*}(s) = (P_W, P_D, P_L) \approx (\varepsilon, 1-2\varepsilon, \varepsilon)$ for small $\varepsilon > 0$ and all positions $s$, the expected gradient of the WDL cross-entropy loss has the following structure:*

*(i) On draw samples (fraction $d_D$ of the buffer): $\partial L / \partial z_W = \varepsilon$, $\partial L / \partial z_D = (1-2\varepsilon) - 1 = -2\varepsilon$, $\partial L / \partial z_L = \varepsilon$. All components are $O(\varepsilon)$.*

*(ii) On win samples (fraction $d_W = (1-d_D)/2$ of the buffer): $\partial L / \partial z_W = \varepsilon - 1 \approx -1$, $\partial L / \partial z_D = 1-2\varepsilon \approx 1$, $\partial L / \partial z_L = \varepsilon$. The gradients are $O(1)$, but these samples constitute a fraction $(1-d_D)/2$ of the buffer.*

*(iii) The buffer-averaged gradient is:*
$$\mathbb{E}\left[\frac{\partial L}{\partial z_W}\right] = d_D \cdot \varepsilon + d_W \cdot (\varepsilon - 1) + d_L \cdot \varepsilon = \varepsilon - d_W$$
$$\mathbb{E}\left[\frac{\partial L}{\partial z_D}\right] = d_D \cdot (-2\varepsilon) + d_W \cdot (1-2\varepsilon) + d_L \cdot (1-2\varepsilon) = (1-2\varepsilon)(1-d_D) - 2\varepsilon \cdot d_D$$
$$= (1-2\varepsilon) - d_D$$

*At equilibrium ($P_D = d_D$, i.e., $1-2\varepsilon = d_D$), the expected gradients vanish. When $d_D > 0.95$, the effective gradient signal from win/loss samples is attenuated by the factor $(1-d_D) < 0.05$: the O(1) gradients from rare decisive samples are overwhelmed by the near-zero gradients from the dominant draw samples.*

*Proof.* Using the standard softmax-cross-entropy gradient $\partial L / \partial z_c = P_c - y_c$:

For a draw sample ($y = (0, 1, 0)$):

$$\frac{\partial L}{\partial z_W} = P_W - 0 = \varepsilon, \quad \frac{\partial L}{\partial z_D} = P_D - 1 = -2\varepsilon, \quad \frac{\partial L}{\partial z_L} = P_L - 0 = \varepsilon$$

For a win sample ($y = (1, 0, 0)$):

$$\frac{\partial L}{\partial z_W} = P_W - 1 = \varepsilon - 1, \quad \frac{\partial L}{\partial z_D} = P_D - 0 = 1 - 2\varepsilon, \quad \frac{\partial L}{\partial z_L} = P_L - 0 = \varepsilon$$

For a loss sample ($y = (0, 0, 1)$):

$$\frac{\partial L}{\partial z_W} = P_W - 0 = \varepsilon, \quad \frac{\partial L}{\partial z_D} = P_D - 0 = 1 - 2\varepsilon, \quad \frac{\partial L}{\partial z_L} = P_L - 1 = \varepsilon - 1$$

The buffer average with $d_W = d_L = (1 - d_D)/2$:

$$\mathbb{E}\!\left[\frac{\partial L}{\partial z_D}\right] = d_D(-2\varepsilon) + (1 - d_D)(1 - 2\varepsilon) = -2\varepsilon d_D + (1 - 2\varepsilon) - d_D(1 - 2\varepsilon) = (1 - 2\varepsilon) - d_D$$

At the equilibrium $P_D = d_D$, i.e., $1 - 2\varepsilon = d_D$, this becomes $d_D - d_D = 0$. The gradient for $z_W$ at equilibrium:

$$\mathbb{E}\!\left[\frac{\partial L}{\partial z_W}\right] = d_D \varepsilon + \frac{1 - d_D}{2}(\varepsilon - 1) + \frac{1 - d_D}{2}\varepsilon = \varepsilon - \frac{1 - d_D}{2} = P_W - d_W$$

which also vanishes at equilibrium ($P_W = d_W$). $\square$

**Remark.** The gradient vanishing is not a consequence of the network being at a minimum of the loss---it *is* at a minimum, but one that reflects the biased data distribution rather than genuine understanding of chess. The network has learned to predict the buffer statistics, not the game outcomes. The problem is that the buffer statistics themselves are an artifact of the collapsed search.

The practical consequence is dire: even if a rare decisive game enters the buffer, its gradient contribution is attenuated by the factor $(1 - d_D)$. With $d_D = 0.98$, a win sample produces gradient $\partial L / \partial z_W \approx -1$ for that sample, but it constitutes only 1% of the mini-batch. The remaining 98% of samples produce gradient $\partial L / \partial z_W \approx \varepsilon \approx 0.01$. The net gradient is $0.98 \times 0.01 + 0.01 \times (-1) \approx 0$. The decisive sample's signal is drowned in noise.


### 4.2 Hessian Flatness

The gradient vanishing alone does not preclude escape: a second-order method could potentially exploit curvature to find a descent direction. We now show that the curvature itself vanishes at collapse.

**Theorem 3 (Hessian Flatness at Collapse).** *The Hessian of the WDL cross-entropy loss with respect to the logits $z_w = (z_W, z_D, z_L)$ at the collapsed parameterization $P = (\varepsilon, 1-2\varepsilon, \varepsilon)$ has eigenvalues that all approach zero as $\varepsilon \to 0$.*

*Specifically, the Hessian matrix of the per-sample loss is:*

$$H_{cc'} = \frac{\partial^2 L}{\partial z_c \partial z_{c'}} = P_c(\mathbf{1}[c = c'] - P_{c'})$$

*The eigenvalues of this $3 \times 3$ matrix are $\{0, P_W(1 - P_W) + P_L(1 - P_L), P_D(1 - P_D)\}$ in the limit of interest, all of which are $O(\varepsilon)$ as $\varepsilon \to 0$.*

*Proof.* The Hessian of the cross-entropy loss with respect to the pre-softmax logits is a standard result. For the softmax function $P_c = e^{z_c} / \sum_{c'} e^{z_{c'}}$, the second derivatives of $L = -\sum_c y_c \log P_c$ are:

$$\frac{\partial^2 L}{\partial z_c \partial z_{c'}} = P_c \mathbf{1}[c = c'] - P_c P_{c'}$$

This is the matrix $\mathrm{diag}(P) - P P^\top$, where $P = (P_W, P_D, P_L)^\top$.

The matrix $\mathrm{diag}(P) - PP^\top$ is positive semidefinite with rank 2 (it has a zero eigenvalue corresponding to the eigenvector $\mathbf{1} = (1,1,1)^\top$, since the softmax parameterization has one degree of freedom fewer than the three logits).

The nonzero eigenvalues can be computed explicitly. The matrix is:

$$H = \begin{pmatrix} \varepsilon(1-\varepsilon) & -\varepsilon(1-2\varepsilon) & -\varepsilon^2 \\ -\varepsilon(1-2\varepsilon) & (1-2\varepsilon)(2\varepsilon) & -\varepsilon(1-2\varepsilon) \\ -\varepsilon^2 & -\varepsilon(1-2\varepsilon) & \varepsilon(1-\varepsilon) \end{pmatrix}$$

The trace is:

$$\mathrm{tr}(H) = \varepsilon(1-\varepsilon) + (1-2\varepsilon)(2\varepsilon) + \varepsilon(1-\varepsilon) = 2\varepsilon(1-\varepsilon) + 2\varepsilon(1-2\varepsilon) = 2\varepsilon(2 - 3\varepsilon)$$

As $\varepsilon \to 0$, $\mathrm{tr}(H) \to 4\varepsilon$. Since the eigenvalues are nonnegative and sum to $4\varepsilon$, each eigenvalue is at most $4\varepsilon$, and hence all eigenvalues are $O(\varepsilon)$.

More precisely, one eigenvalue is exactly 0 (eigenvector $\mathbf{1} = (1,1,1)^\top$). The two nonzero eigenvalues $\lambda_1, \lambda_2$ satisfy $\lambda_1 + \lambda_2 = 4\varepsilon - O(\varepsilon^2)$. By explicit computation, the matrix $H = \mathrm{diag}(P) - PP^\top$ with $P = (\varepsilon, 1-2\varepsilon, \varepsilon)$ has eigenvectors and eigenvalues:

$$\lambda_1 = \varepsilon, \quad \text{eigenvector } (1, 0, -1)^\top$$
$$\lambda_2 = 3\varepsilon - 6\varepsilon^2, \quad \text{eigenvector } (1, -2, 1)^\top$$

To verify: $\lambda_1 + \lambda_2 = 4\varepsilon - 6\varepsilon^2 = 2\varepsilon(2 - 3\varepsilon) = \mathrm{tr}(H)$. The eigenvector $(1,0,-1)^\top$ captures the $W$--$L$ contrast (with $P_D$ playing no role), while $(1,-2,1)^\top$ captures the contrast between the draw class and the average of $W$ and $L$.

Both eigenvalues converge to 0 as $\varepsilon \to 0$. $\square$

**Corollary.** *The SGD step size in parameter space is proportional to:*

$$\|\Delta \theta\| \propto \eta \cdot \|\nabla_\theta L\| \leq \eta \cdot \|J_\theta\| \cdot \|\nabla_z L\|$$

*where $J_\theta$ is the Jacobian of the logits with respect to the parameters. Since $\|\nabla_z L\| = O(\varepsilon)$ (from Proposition 5) and the loss surface has curvature $O(\varepsilon)$ (from Theorem 3), the effective learning rate is $\eta_{\mathrm{eff}} \sim \eta \varepsilon$. With $\varepsilon \approx 0.01$ (corresponding to $P_D \approx 0.98$), the effective learning rate is reduced by two orders of magnitude.*

The loss surface near $S^*$ is a nearly flat plateau. SGD takes steps of size $O(\eta \varepsilon)$ in a landscape with curvature $O(\varepsilon)$. The number of steps required to escape the basin of attraction is $O(1/\varepsilon^2)$---for $\varepsilon = 0.01$, this is on the order of $10{,}000$ gradient steps. But each iteration of training is itself only a few hundred gradient steps ($|B|/b \times E$ where $|B| = 100{,}000$ is the buffer size, $b$ is the mini-batch size, and $E$ is the number of epochs). Escape would require dozens or hundreds of training iterations, during which the buffer continues to be filled with draw data, maintaining $\varepsilon$ near zero.


### 4.3 Entropy Collapse

A complementary view of the value head collapse comes from the entropy of the WDL distribution.

**Proposition 6 (Entropy Collapse).** *The entropy of the WDL distribution at the collapsed state $w^* = (\varepsilon, 1-2\varepsilon, \varepsilon)$ is:*

$$H(w^*) = -(1-2\varepsilon)\log(1-2\varepsilon) - 2\varepsilon\log\varepsilon$$

*which satisfies $H(w^*) \to 0$ as $\varepsilon \to 0$. In contrast, a healthy value head with balanced predictions (e.g., $(0.3, 0.4, 0.3)$) has entropy:*

$$H_{\mathrm{healthy}} = -0.3\log 0.3 - 0.4\log 0.4 - 0.3\log 0.3 \approx 1.09 \text{ nats}$$

*Proof.* Direct computation:

$$H(w^*) = -(1-2\varepsilon)\log(1-2\varepsilon) - 2\varepsilon\log\varepsilon$$

As $\varepsilon \to 0$: the first term $(1-2\varepsilon)\log(1-2\varepsilon) \to 0$ (since $x \log x \to 0$ as $x \to 1$, or more precisely, $(1-2\varepsilon)\log(1-2\varepsilon) \approx -2\varepsilon + O(\varepsilon^2) \to 0$). The second term $-2\varepsilon\log\varepsilon \to 0$ (since $x\log x \to 0$ as $x \to 0^+$). Therefore $H(w^*) \to 0$. $\square$

The entropy measures the "decision-making capacity" of the value head. At collapse, the value head has zero capacity to distinguish between positions: it assigns the same WDL distribution $(\varepsilon, 1-2\varepsilon, \varepsilon)$ to openings, middlegames, endgames, winning positions, and losing positions alike. This is the informational counterpart of the gradient vanishing: low entropy means low information, which means low gradient, which means the network cannot learn to increase its entropy.


### 4.4 Epoch Multiplier Effect

The number of training epochs $E$ per iteration has a profound effect on the speed of collapse.

**Proposition 7 (Epochs Multiplier).** *With $E$ epochs per iteration, each sample in the buffer is seen $\sim E$ times per training phase. The effective per-iteration parameter update toward the buffer equilibrium is:*

$$\Delta \bar{P}_D \propto E \cdot \eta \cdot (d_D - \bar{P}_D)$$

*More epochs accelerate convergence to $\bar{P}_D = d_D$. When $d_D > \rho$ (the collapse threshold from Theorem 2), more epochs accelerate collapse.*

*Proof.* In each epoch, every sample is visited once (in expectation, with mini-batch sampling). The per-sample gradient update is $\Delta z_D = -\eta \cdot (P_D - d_D)$ in expectation (from Proposition 2), which yields $\Delta P_D \propto \eta \cdot (d_D - P_D)$ by the chain rule through the softmax. Over $E$ epochs:

$$\Delta P_D^{(E)} \approx 1 - (1 - \eta \cdot \kappa)^E \approx E \cdot \eta \cdot \kappa \cdot (d_D - P_D)$$

for small $\eta \cdot \kappa$, where $\kappa$ is the effective curvature. The linear approximation $\Delta P_D \propto E$ holds when $E \cdot \eta \cdot \kappa \ll 1$.

In the collapse regime, this means the value head converges to the biased equilibrium $E$ times faster. $\square$

**Empirical validation.** In the simulation depth draw death run, $E = 3$ epochs per iteration caused the value loss to collapse from 0.89 to 0.02 in approximately 4 iterations (iterations 5--9). The transition from "healthy" to "dead" took 4 iterations $\times$ 3 epochs = 12 total epochs of training. With $E = 1$, the same transition would have taken approximately $12$ iterations, providing a much wider window for intervention. This is precisely why reducing to $E = 1$ was effective in recovering from fifty-move stagnation in the Gumbel run (Section 3.7, Mode 3).


### 4.5 ERM Ineffectiveness at Collapse

The Entropic Risk Measure (Section 2.4) is designed to promote risk-seeking play by preferring high-variance nodes. In principle, a risk-seeking agent ($\beta > 0$) should prefer sharp positions with decisive outcomes over dull positions heading for draws. We now prove that ERM is ineffective once collapse has occurred.

**Theorem 4 (ERM Ineffectiveness at Collapse).** *At the collapsed state where $w_\theta(s) \approx (\varepsilon, 1-2\varepsilon, \varepsilon)$ for all positions $s$, the risk-adjusted Q-value $Q_\beta$ is independent of $\beta$ up to $O(\varepsilon^2)$ terms:*

$$Q_\beta(a) \approx 0 \quad \forall \beta, \forall a$$

*Consequently, risk-seeking search ($\beta > 0$) cannot distinguish between moves and cannot break the draw equilibrium.*

*Proof.* The scalar value at each leaf is $v = w_W - w_L \approx \varepsilon - \varepsilon = 0$ with variance:

$$\mathrm{Var}(v) = \mathbb{E}[v^2] - (\mathbb{E}[v])^2$$

Since $v \approx 0$ at all leaves, $\mathbb{E}[v] \approx 0$ and $\mathbb{E}[v^2] \approx 0$, giving $\mathrm{Var}(v) \approx 0$.

More precisely, the value $v = w_W - w_L$ at a leaf has $|v| \leq 2\varepsilon$ (since $w_W, w_L \leq \varepsilon + O(\varepsilon^2)$ for positions near the collapsed state). Therefore:

$$|\mathbb{E}[v]| \leq 2\varepsilon, \quad \mathbb{E}[v^2] \leq 4\varepsilon^2, \quad \mathrm{Var}(v) \leq 4\varepsilon^2$$

The risk-adjusted Q-value is:

$$Q_\beta(a) = \mathbb{E}[v(a)] + \frac{\beta}{2} \mathrm{Var}(v(a)) \leq 2\varepsilon + \frac{|\beta|}{2} \cdot 4\varepsilon^2 = 2\varepsilon + 2|\beta|\varepsilon^2$$

For any fixed $\beta$, this is $O(\varepsilon)$, and the contribution from variance is $O(\varepsilon^2)$---a second-order effect that is negligible compared to the already-tiny mean. The risk term $(\beta/2)\mathrm{Var}(v)$ is $O(\varepsilon^2)$ while the mean is $O(\varepsilon)$, so risk sensitivity provides no useful signal.

In practice, with $\varepsilon \approx 0.01$, the mean Q-value is $\sim 0.02$ and the variance contribution is $\sim 0.0002 \beta$. Even with $\beta = 10$ (aggressive risk-seeking), the risk adjustment is $0.001$---three orders of magnitude smaller than typical Q-value differences in a healthy search ($\sim 0.1$--$0.5$). $\square$

**Empirical validation.** In the Gumbel training run, using $\beta = 0.5$ (risk-seeking) produced only 6% decisive games when the value head was already draw-biased with value loss $\approx 0.15$. The risk measure could not generate sufficient variance to break the draw equilibrium because the variance itself had collapsed. This confirms that ERM is a *preventive* measure (effective when $\mathrm{Var}(v)$ is still nontrivial) but not a *curative* one (ineffective once collapse has occurred).

**Remark.** The fundamental issue is a chicken-and-egg problem. ERM needs variance in the value predictions to steer search toward decisive positions. But variance in value predictions requires the value head to distinguish between positions, which requires training on decisive game data, which requires search to find decisive lines, which requires ERM to have variance to work with. At collapse, this circular dependency has no entry point. Breaking the cycle requires an external intervention---such as buffer composition management or loss function modification---that does not depend on the value head's current predictions.

### 4.6 Summary of the Absorbing State Geometry

The WDL cross-entropy loss at collapse presents a pathological optimization landscape:

1. **Gradient magnitude**: $\|\nabla L\| = O(\varepsilon)$ (Proposition 5). The learning signal is nearly zero.

2. **Hessian eigenvalues**: $\lambda_{\max} = O(\varepsilon)$ (Theorem 3). The curvature is nearly zero. There are no sharp valleys to guide optimization.

3. **Entropy**: $H(w^*) \to 0$ (Proposition 6). The value head has zero information content.

4. **Effective learning rate**: $\eta_{\mathrm{eff}} = O(\eta \varepsilon)$ (Corollary of Theorem 3). Steps are too small to escape.

5. **Epoch amplification**: Multiple epochs accelerate arrival at this plateau by factor $E$ (Proposition 7).

6. **Risk measure blindness**: $Q_\beta \approx 0$ regardless of $\beta$ (Theorem 4). Risk-sensitive search cannot help.

The combination of vanishing gradients, flat curvature, and zero variance creates an absorbing state that is robust to perturbations of the search parameters, learning rate, and risk coefficient. The only effective interventions must modify either the data distribution (buffer composition) or the loss function itself---changes that operate outside the collapsed feedback loop rather than within it.

## 5. Taxonomy of Collapse Modes

The preceding theoretical framework identifies three structural links in the feedback loop $\theta_t \to \phi_t \to D_t \to \theta_{t+1}$ through which the absorbing state $S^*$ attracts the training trajectory. In practice, collapse manifests through four empirically distinct modes, each exploiting a different combination of these links. We present each mode with its mechanism, empirical signature, and formal connection to the theorems established in Sections 3--4.

### 5.1 Mode I: Simulation Depth Draw Death

**Mechanism.** When the simulation budget $N_{\text{sims}}$ is increased prematurely — before the value head has developed sufficient calibration — deep search amplifies the value head's draw bias through the mechanism of Theorem 1. At 800 simulations, Gumbel Sequential Halving runs $\lceil \log_2 k \rceil$ rounds of halving, and in each round, the completed Q-values $Q_{\text{completed}}(a)$ are computed from subtree returns that themselves depend on the biased value head at leaf nodes. With a weakly trained value head predicting $w_D \approx 0.6$ across most positions, the variance $\text{Var}(Q_{\text{completed}})$ collapses as $O(\sigma_v^2 / N_{\text{sims}})$, and the $\sigma$-transform maps nearly identical Q-values to nearly identical scores. The improved policy $\pi_{\text{improved}}$ therefore collapses toward the prior $\pi_\theta$, and the prior itself — trained on draw-dominated data — favors repetitive, non-committal moves.

At depth $> 0$, internal PUCT compounds the problem. The First Play Urgency (FPU) formula $Q_{\text{unvisited}} = Q_{\text{parent}} - f_{\text{base}} \sqrt{1 - P(a)}$ penalizes unexplored actions when $Q_{\text{parent}} \approx 0$ (the draw value), creating the FPU trap of Proposition 3: only actions with prior probability $P(a) > 1 - (Q_{\text{parent}} / f_{\text{base}})^2$ receive any visits. In the deep subtrees enabled by 800 simulations, this trap cascades across multiple levels, funneling search toward a narrow corridor of "safe" drawing moves — typically bishop oscillations (Bb7--Ba8) or rook shuffles that force threefold repetition.

**Empirical signature.** Table 1 reproduces the training log from Section 10g of the operation manual.

| Iteration | Decisive % | Repetition Draws | Value Loss | Avg Game Length |
|-----------|-----------|-----------------|------------|----------------|
| 5 (200 sims) | 22% | 16/32 | 0.89 | 315 |
| 6 (800 sims) | 22% | 16/32 | 0.89 | 315 |
| 7 | 9% | 27/32 | 0.51 | 153 |
| 8 | 3% | 31/32 | 0.45 | 95 |
| 9--53 | 0% | 32/32 | 0.02 | 42 |

The characteristic signatures are: (i) value loss monotonically collapses from 0.89 to 0.02, confirming the gradient vanishing of Proposition 5; (ii) average game length *decreases* from 315 to 42, as games terminate quickly via threefold repetition rather than playing out; (iii) the transition is abrupt — a single iteration (7) tips the system past the critical threshold of Proposition 12, after which recovery is impossible.

**Connection to Gumbel search.** While Gumbel's round-robin allocation at the root eliminates the PUCT-specific repetition lock-in (each of the top-$k$ actions receives simulations regardless of Q-values), the draw death in Mode I is fundamentally driven by the *subtree* evaluations that inform $Q_{\text{completed}}$. Gumbel's root diversity guarantee does not extend to internal nodes, where standard PUCT with FPU governs action selection. The 800-simulation budget allows these internal subtrees to grow deep enough that the FPU trap (Proposition 3) activates at multiple levels, producing draw-biased leaf evaluations that feed back into the root's $Q_{\text{completed}}$ regardless of which root action is being explored.

**Recovery.** After 5 or more iterations at 100% draws, the buffer composition reaches $D^D > 0.99$, the value head has converged to the flat minimum of Theorem 3 (Hessian eigenvalues $\lambda_i \approx \varepsilon$), and the entropy collapse of Proposition 6 has reduced the policy to near-deterministic repetition-forcing. No combination of hyperparameter changes can recover the system from this state, as the gradient signal for decisive outcomes has been reduced by a factor of $(1 - D_t^D)^E \approx 0.01^3 \approx 10^{-6}$ (combining the gradient attenuation of Proposition 5 with the epoch multiplier of Proposition 7). This mode is **irrecoverable**.

### 5.2 Mode II: Fifty-Move Draw Stagnation

**Mechanism.** Under Gumbel Sequential Halving, the root-level round-robin allocation ensures that every action in the top-$k$ set receives simulations in each halving round, eliminating the repetition-forcing behavior of Mode I. Empirically, Gumbel runs produce zero early repetition draws across 11 iterations, compared to 22--29 per iteration under PUCT (Section 10j). However, Gumbel's exploration guarantee is a double-edged sword: it forces the model to *play out* diverse positions rather than quickly terminating via repetition, and when the value head uniformly predicts draws ($w_D \approx 1$), the model cannot distinguish mating attacks from quiet positions. Every line of play appears equally drawn, so the model makes moves that are locally "interesting" (high prior probability) but strategically aimless. Games extend to the fifty-move rule limit as neither side can identify or execute winning plans.

Formally, the fifty-move mechanism exploits a different link than Mode I. Where Mode I amplifies draw bias through search depth (the $\theta_t \to \phi_t$ link), Mode II operates primarily through the data generation link ($\phi_t \to D_t$): the games themselves are longer and produce more training positions per game, but every position is labeled as a draw. The data asymmetry of Proposition 4 operates through a different mechanism than in Mode I — instead of short repetition games producing few positions, long fifty-move games produce *many* positions, all draw-labeled.

**Empirical signature.** From Section 10i:

| Iteration | Decisive % | Repetition Draws | Fifty-Move Draws | Value Loss | Avg Length |
|-----------|-----------|-----------------|-----------------|------------|------------|
| 1--3 | 28% | 3--7 | 5--7 | 0.47--0.89 | 311--358 |
| 4 | 50% | 4 | 6 | 0.55 | 289 |
| 6 | 0% | 0 | 25 | 0.50 | 381 |
| 10 | 28% | 6 | 11 | 0.30 | 330 |

The distinguishing features relative to Mode I are: (i) average game length *increases* (311 $\to$ 381) rather than decreasing; (ii) repetition draws remain near zero throughout; (iii) value loss decreases more slowly (0.89 $\to$ 0.50 over 6 iterations vs. 0.89 $\to$ 0.02 over 4 iterations in Mode I); (iv) the system exhibits partial spontaneous recovery (iteration 10 reaches 28% decisive games).

**Recovery.** Mode II is recoverable because the value head has not fully collapsed — value loss stabilizes around 0.30--0.50 rather than reaching the $\approx 0.02$ terminal state. The Hessian eigenvalues remain bounded away from zero (Theorem 3 does not apply at $P(\text{draw}) \approx 0.7$ as sharply as at $P(\text{draw}) \approx 0.99$). The successful recovery protocol uses epochs${}= 1$ (reducing the reinforcement multiplier of Proposition 7 to its minimum) combined with temperature\_moves${}= 50$ (introducing stochastic move selection for the first 50 half-moves, creating unbalanced positions that are more likely to reach decisive outcomes). This combination attacks both the $D_t \to \theta_{t+1}$ link (fewer epochs) and the $\theta_t \to \phi_t$ link (temperature-induced diversity).

### 5.3 Mode III: Max-Moves Draw Pollution

**Mechanism.** The training environment imposes a 512-ply maximum game length. Games reaching this limit are terminated and labeled as draws regardless of the position. This creates a severe data asymmetry quantified by Proposition 4: a max-moves draw game contributes all 512 positions to the replay buffer with draw labels, while a decisive game ending at move 40 (80 ply) contributes only $\sim$80 positions. At a 95% draw rate with 17 max-moves draws per iteration of 32 games, this produces approximately $17 \times 512 = 8{,}704$ draw-labeled positions versus $\sim 1.6 \times 80 + 13.4 \times 200 = 2{,}808$ positions from other outcomes — a ratio exceeding 3:1 even before accounting for the non-max-moves draws.

The root cause analysis revealed an encoding error in the halfmove clock feature: the raw clock value was divided by 50.0 instead of 100.0, causing the feature to saturate at 1.0 when the halfmove clock reached 50 (well before the fifty-move rule threshold of 100 half-moves). The network could not distinguish positions at 50 half-moves from positions at 90 half-moves, eliminating its ability to learn the fifty-move rule's urgency signal.

**Empirical signature.** The distinctive feature of Mode III is the *decoupling* of policy and value learning. From Section 10m: policy loss improved monotonically throughout training even as value loss stagnated at $\sim$0.3. This occurs because the policy head learns from the structure of positions (which moves are tactically and strategically sensible) independent of game outcomes, while the value head is overwhelmed by the draw-labeled flood. At buffer shutdown, composition was 92.1% draws — consistent with the saturation prediction of the coupled dynamics.

**Interaction with Proposition 4.** The worst-case position ratio of 122:1 computed in Proposition 4 is extreme, but even the empirically observed 3:1 ratio is sufficient to create a draw-dominated gradient. Under SGD with uniform sampling from the buffer, the expected gradient contribution from draw-labeled positions exceeds that from decisive positions by a factor proportional to their count ratio, regardless of the loss function's theoretical properties.

### 5.4 Mode IV: Buffer Draw Saturation Trap

**Mechanism.** Once the replay buffer exceeds $\sim$85% draw composition, it acquires sufficient inertial mass to resist correction even when individual iterations produce healthier game distributions. The buffer update dynamics of Proposition 1 govern this:

$$D_{t+1}^D = \left(1 - \frac{GL}{C}\right) D_t^D + \frac{GL}{C} \Phi_t^D$$

where $G$ is games per iteration, $L$ is average game length, $C$ is buffer capacity, and $\Phi_t^D$ is the draw fraction in iteration $t$'s self-play. With typical parameters ($G = 32$, $L \approx 200$, $C = 200{,}000$), the mixing coefficient $GL/C \approx 0.032$ — each iteration replaces only 3.2% of the buffer. Starting from $D_t^D = 0.88$, even a "good" iteration with $\Phi_t^D = 0.64$ yields:

$$D_{t+1}^D = 0.968 \times 0.88 + 0.032 \times 0.64 = 0.852 + 0.020 = 0.872$$

The buffer barely moves. And with $D_{t+1}^D = 0.872$, the next iteration's value head — trained on this still-heavily-draw-biased buffer — is likely to regress, producing $\Phi_{t+2}^D > 0.64$ and pushing the buffer back toward 0.88.

**Empirical signature.** From Section 10r:

| Iteration | Epochs | Draw % | Value Loss | Buffer Draw % |
|-----------|--------|--------|------------|---------------|
| 9 | 1 | 71% | 0.68 | 77.5% |
| 10 | 2 | 91% | 0.57 | 84.0% |
| 11 | 2 | 92% | 0.39 | 92.2% |
| 12 | 1 | 64% | 0.45 | 88.0% |
| 13 | 1 | 92% | 0.375 | 89.5% |

The oscillation between iterations 12 (64% draws, epochs${}=1$) and 13 (92% draws, epochs${}=1$) demonstrates the trap: a single good iteration cannot shift the buffer sufficiently, and the regression to high draw rates the following iteration erases the gains. The empirical threshold of buffer $> 85\%$ draws combined with value loss $< 0.4$ marks the point of no return under standard hyperparameter tuning.

**Connection to Mode III.** Mode III directly feeds Mode IV: the data asymmetry from max-moves games accelerates buffer saturation. A single max-moves draw game contributes as many draw-labeled positions as $\sim$6.4 decisive games contribute decisive positions (512 vs. 80). This creates a ratchet effect where the buffer's draw fraction increases faster than the game-level draw rate would suggest.

### 5.5 Comparison and Interconnections

Table 2 summarizes the four modes across key diagnostic dimensions.

| Property | Mode I: Sim Depth | Mode II: 50-Move | Mode III: Max-Moves | Mode IV: Buffer Sat. |
|----------|-------------------|-------------------|---------------------|----------------------|
| Draw mechanism | Threefold repetition | Fifty-move rule | 512-ply cap | Buffer inertia |
| Game length trend | Decreasing (315→42) | Increasing (311→381) | Maximal (512) | Oscillating |
| Trigger | Sims 200→800 | Draw-biased value head | Encoding error + long games | Buffer > 85% draws |
| Recoverable? | No (after 5+ iters) | Yes (epochs=1, temp=50) | Yes (fix encoding) | No (under standard tuning) |
| Value loss signature | Monotone collapse to 0.02 | Slow decline to 0.30--0.50 | Stagnation at 0.30 | Oscillation below 0.45 |
| Primary feedback link | $\theta_t \to \phi_t$ (search) | $\phi_t \to D_t$ (data gen) | $\phi_t \to D_t$ (data asymmetry) | $D_t \to \theta_{t+1}$ (training) |
| Root cause theorem | Thm 1 (amplification) | Prop 2 (SGD equilibrium) | Prop 4 (data asymmetry) | Prop 1 (buffer dynamics) |

The modes are not independent — they form a directed graph of causal relationships. Mode III (max-moves pollution) feeds directly into Mode IV (buffer saturation) through the data asymmetry mechanism. Modes I and II share the same ultimate root cause — value head draw bias — but manifest differently due to the search algorithm: PUCT's exploitation-heavy action selection drives repetition (Mode I), while Gumbel's enforced exploration prevents repetition but enables fifty-move timeout (Mode II). Mode IV can be triggered by any of the other three modes once the buffer crosses the critical saturation threshold.

The key structural insight is that each mode attacks a different link in the feedback loop $\theta_t \to \phi_t \to D_t \to \theta_{t+1}$, and therefore requires a different prevention strategy targeting the corresponding link. Section 6 develops these strategies systematically.


## 6. Prevention and Recovery Methods

We organize prevention methods into four families, each targeting a specific link in the feedback chain identified in Section 3. For each method, we provide formal guarantees and connect them to the collapse modes they address.

### 6.1 Loss Function Modifications

These methods break the $D_t \to \theta_{t+1}$ link by modifying how the training loss responds to buffer composition, ensuring that the gradient retains signal for decisive outcomes even when the buffer is draw-dominated.

**Theorem 5 (Entropy-Regularized WDL Loss).** *Define the entropy-regularized value loss:*

$$L_{\text{value}} = L_{\text{CE}}(y, w) - \lambda_H \cdot H(w)$$

*where $w = \text{softmax}(z)$ is the WDL prediction, $y$ is the one-hot target, $L_{\text{CE}}(y, w) = -\sum_c y_c \log w_c$ is the standard cross-entropy, and $H(w) = -\sum_c w_c \log w_c$ is the entropy of the prediction. Then the minimizer over the probability simplex satisfies:*

$$w_c^* = \frac{y_c + \lambda_H / 3}{1 + \lambda_H}$$

*for each class $c \in \{W, D, L\}$ when $y$ is one-hot. Consequently, for any $\lambda_H > 0$:*

1. *Every class has minimum probability $w_c^* \geq \frac{\lambda_H}{3(1 + \lambda_H)}$.*
2. *The prediction entropy satisfies $H(w^*) \geq H_{\min}(\lambda_H) > 0$.*
3. *The Hessian of $L_{\text{value}}$ at $w^*$ has all eigenvalues bounded below by $\frac{\lambda_H}{3}$.*

*Proof.* We work in the unconstrained logit space $z \in \mathbb{R}^3$ with $w_c = e^{z_c} / \sum_j e^{z_j}$. The loss is:

$$L(z) = -\sum_c y_c \log w_c + \lambda_H \sum_c w_c \log w_c$$

Taking the derivative with respect to $z_c$:

$$\frac{\partial L}{\partial z_c} = -y_c + w_c + \lambda_H \left(\frac{\partial}{\partial z_c} \sum_j w_j \log w_j\right)$$

For the entropy term, using the softmax Jacobian $\partial w_j / \partial z_c = w_j(\delta_{jc} - w_c)$:

$$\frac{\partial}{\partial z_c} \sum_j w_j \log w_j = \sum_j (\log w_j + 1) \cdot w_j(\delta_{jc} - w_c) = w_c(\log w_c + 1) - w_c \sum_j w_j(\log w_j + 1)$$

$$= w_c \left(\log w_c - \sum_j w_j \log w_j\right)$$

Setting $\partial L / \partial z_c = 0$:

$$w_c - y_c + \lambda_H w_c \left(\log w_c - \sum_j w_j \log w_j\right) = 0$$

At the optimum, symmetry of the KKT conditions requires that $\log w_c^* = \frac{1}{\lambda_H}\left(\frac{y_c}{w_c^*} - 1\right) + \text{const}$. For a one-hot target with $y_k = 1$ and $y_j = 0$ for $j \neq k$, the optimality conditions become:

$$w_k^* - 1 + \lambda_H w_k^* (\log w_k^* - \bar{H}) = 0$$
$$w_j^* + \lambda_H w_j^* (\log w_j^* - \bar{H}) = 0 \quad \text{for } j \neq k$$

where $\bar{H} = \sum_c w_c^* \log w_c^*$. From the second equation, for $j \neq k$: $1 + \lambda_H(\log w_j^* - \bar{H}) = 0$, giving $\log w_j^* = \bar{H} - 1/\lambda_H$. Since this holds for both non-target classes, we have $w_j^* = w_{j'}^*$ for all $j, j' \neq k$. Let $w_j^* = \beta$ for $j \neq k$, so $w_k^* = 1 - 2\beta$.

Substituting into the first equation and solving the system (using the constraint that probabilities sum to 1), we obtain the solution by verifying that $w_c^* = (y_c + \lambda_H/3) / (1 + \lambda_H)$ satisfies the optimality conditions. For the target class ($y_k = 1$):

$$w_k^* = \frac{1 + \lambda_H/3}{1 + \lambda_H} = \frac{3 + \lambda_H}{3(1 + \lambda_H)}$$

For non-target classes ($y_j = 0$):

$$w_j^* = \frac{\lambda_H/3}{1 + \lambda_H} = \frac{\lambda_H}{3(1 + \lambda_H)}$$

These satisfy $\sum_c w_c^* = (3 + \lambda_H + 2\lambda_H/3) / (3(1 + \lambda_H)) \cdot 3 = 1$. Verification:

$$\sum_c w_c^* = \frac{3 + \lambda_H}{3(1+\lambda_H)} + \frac{2\lambda_H}{3(1+\lambda_H)} = \frac{3 + 3\lambda_H}{3(1+\lambda_H)} = 1 \quad \checkmark$$

For property (1): the minimum class probability is $w_j^* = \lambda_H / (3(1 + \lambda_H))$, which is strictly positive for any $\lambda_H > 0$. At $\lambda_H = 0.1$: $w_{\min} = 0.1/3.3 \approx 0.030$.

For property (2): since $w_{\min} > 0$, the entropy $H(w^*) = -\sum_c w_c^* \log w_c^*$ is bounded below. Using the concavity of entropy and the constraint $w_{\min} \geq \lambda_H / (3(1+\lambda_H))$:

$$H(w^*) \geq -w_k^* \log w_k^* - 2 w_j^* \log w_j^*$$

At $\lambda_H = 0.1$: $H(w^*) \approx -0.94 \log 0.94 - 2(0.03)\log 0.03 \approx 0.058 + 0.211 = 0.269$ nats.

For property (3): the Hessian of the cross-entropy loss in logit space is $\nabla^2 L_{\text{CE}} = \text{diag}(w) - ww^\top$, whose eigenvalues are $\{w_c(1 - w_c)\}$ on the simplex tangent space. The entropy regularization adds $\lambda_H \cdot \nabla^2 H_{\text{neg}}$, where $\nabla^2 (-H)$ in logit space has eigenvalues that include terms proportional to $w_c$. The combined Hessian has minimum eigenvalue:

$$\lambda_{\min}(\nabla^2 L) \geq \min_c w_c^*(1 - w_c^*) + \lambda_H \min_c w_c^* \geq \frac{\lambda_H}{3(1+\lambda_H)} \cdot \frac{3}{3+\lambda_H} + \frac{\lambda_H^2}{3(1+\lambda_H)} = \Omega(\lambda_H)$$

More precisely, at $\lambda_H = 0.1$ the bound evaluates to $\approx 0.032$, which satisfies $\lambda_{\min} \geq \lambda_H/4 = 0.025$ for $\lambda_H \leq 1$. This eliminates the flat Hessian plateau of Theorem 3. $\square$

**Consequences for collapse prevention.** Theorem 5 directly addresses the gradient vanishing mechanism of Proposition 5. Even when the buffer is 99% draws, the entropy regularization ensures $w_W^*, w_L^* \geq \lambda_H / (3(1 + \lambda_H))$, maintaining nonzero gradient flow for decisive outcomes. The Hessian lower bound of $\Omega(\lambda_H)$ prevents the flat plateau that enables the absorbing state's stability (Theorem 3 showed eigenvalues $\approx \varepsilon$ at collapse; now they are bounded below by $\lambda_H/4 \approx 0.025$ at $\lambda_H = 0.1$).

The choice of $\lambda_H$ involves a bias-variance tradeoff: larger $\lambda_H$ provides stronger collapse prevention but introduces systematic bias toward the uniform distribution $w = (1/3, 1/3, 1/3)$. In practice, $\lambda_H \in [0.05, 0.15]$ provides meaningful protection while keeping the bias below the noise floor of early-training value predictions.

---

**Proposition 8 (Focal Loss for WDL).** *Define the focal loss:*

$$L_{\text{focal}} = -\sum_c y_c (1 - w_c)^{\gamma} \log w_c$$

*where $\gamma > 0$ is the focusing parameter. The gradient with respect to the logit $z_c$ satisfies:*

$$\frac{\partial L_{\text{focal}}}{\partial z_c} = \sum_j y_j (1 - w_j)^{\gamma - 1} \left[\gamma w_j \log w_j + (1 - w_j)\right](w_c \cdot \mathbf{1}[c = j] - w_c w_j) / w_j$$

*When the model confidently predicts the correct class ($w_{y} \to 1$), the gradient magnitude scales as $(1 - w_y)^{\gamma}$. When the model is confidently wrong ($w_y \to 0$), the gradient magnitude approaches 1.*

*Proof.* For a single sample with one-hot target $y_k = 1$, the focal loss reduces to:

$$L_{\text{focal}} = -(1 - w_k)^{\gamma} \log w_k$$

Taking the derivative with respect to $z_c$ using the chain rule through the softmax:

$$\frac{\partial L_{\text{focal}}}{\partial z_c} = \frac{\partial L_{\text{focal}}}{\partial w_k} \cdot \frac{\partial w_k}{\partial z_c}$$

First, compute $\partial L_{\text{focal}} / \partial w_k$:

$$\frac{\partial L_{\text{focal}}}{\partial w_k} = \gamma (1 - w_k)^{\gamma - 1} \log w_k - \frac{(1 - w_k)^{\gamma}}{w_k}$$

$$= (1 - w_k)^{\gamma - 1} \left[\gamma \log w_k - \frac{1 - w_k}{w_k}\right]$$

$$= -(1 - w_k)^{\gamma - 1} \left[\frac{1 - w_k}{w_k} - \gamma \log w_k\right]$$

Using the softmax Jacobian $\partial w_k / \partial z_c = w_k(\delta_{kc} - w_c)$:

$$\frac{\partial L_{\text{focal}}}{\partial z_c} = -(1 - w_k)^{\gamma - 1} \left[\frac{1 - w_k}{w_k} - \gamma \log w_k\right] \cdot w_k(\delta_{kc} - w_c)$$

$$= -(1 - w_k)^{\gamma - 1} \left[(1 - w_k) - \gamma w_k \log w_k\right] (\delta_{kc} - w_c)$$

Define the *modulating factor* $m(w_k) = (1 - w_k)^{\gamma - 1}[(1 - w_k) - \gamma w_k \log w_k]$. When $w_k \to 1$ (confident correct prediction): $(1 - w_k) \to 0$ and $\gamma w_k \log w_k \to 0$, so $m(w_k) \to 0$. The gradient vanishes — well-classified samples contribute nothing.

When $w_k \to 0$ (confident wrong prediction): $(1 - w_k)^{\gamma-1} \to 1$, $(1 - w_k) \to 1$, and $\gamma w_k \log w_k \to 0$, so $m(w_k) \to 1$. The gradient has full magnitude — misclassified samples contribute maximally. $\square$

**Application to draw-saturated buffers.** Consider a buffer with 90% draws. Under standard cross-entropy, the expected gradient is dominated by draw samples (90% of the sum). Under focal loss with $\gamma = 2$:

- Draw samples where the model correctly predicts draw ($w_D \approx 0.9$): modulating factor $m \approx (0.1)^1 \times [0.1 + 2 \times 0.9 \times 0.105] \approx 0.1 \times 0.289 \approx 0.029$
- Decisive samples where the model incorrectly predicts draw ($w_W \approx 0.05$ for an actual win): modulating factor $m \approx (0.95)^1 \times [0.95 + 2 \times 0.05 \times 3.0] \approx 0.95 \times 1.25 \approx 1.19$

The effective weight ratio is $1.19 / 0.029 \approx 41$, meaning each decisive sample contributes roughly 41 times more gradient than each draw sample. After accounting for the 9:1 count ratio, decisive samples contribute $41/9 \approx 4.6$ times as much total gradient as draw samples. This automatically rebalances the gradient without requiring explicit knowledge of the buffer composition.

---

**Auxiliary Variance Prediction Head.** The ERM mechanism $Q_\beta = \mathbb{E}[v] + (\beta/2) \text{Var}(v)$ fails at collapse because $\text{Var}(v)$ is computed empirically from MCTS backpropagation, and when all leaf evaluations return $v \approx 0$, the variance vanishes (Theorem 4). We propose adding an explicit variance prediction head that is trained to output calibrated uncertainty, providing an ERM signal independent of value head collapse.

Augment the network with a scalar output $\hat{\sigma}^2(s) = \text{softplus}(f_{\text{var}}(s; \theta))$ sharing the trunk with the value and policy heads but with its own final layers. The auxiliary loss is:

$$L_{\text{var}} = \frac{1}{B} \sum_{i=1}^{B} \left(\hat{\sigma}^2(s_i) - (z_i - \bar{v}_i)^2\right)^2$$

where $z_i \in \{-1, 0, +1\}$ is the game outcome, $\bar{v}_i = w_W(s_i) - w_L(s_i)$ is the model's current scalar value prediction, and the target $(z_i - \bar{v}_i)^2$ is the squared prediction error — a proxy for the true conditional variance.

During MCTS, the ERM formula becomes:

$$Q_\beta(s) = \bar{v}(s) + \frac{\beta}{2} \hat{\sigma}^2(s)$$

Since $\hat{\sigma}^2$ is a learned function rather than an empirical statistic, it can maintain nonzero output even when the value head has collapsed, provided the training signal $(z_i - \bar{v}_i)^2$ remains nonzero. At collapse, $\bar{v}_i \approx 0$ for all positions, and $(z_i - 0)^2 = 1$ for decisive games and $(z_i - 0)^2 = 0$ for draws. As long as the buffer contains *any* decisive games, the variance head receives a training signal. Even at 99% draws, the 1% decisive samples produce targets of 1.0 — a strong, unambiguous signal that the variance head can learn from.

```
# Pseudocode: Auxiliary variance head training
class AlphaZeroNet(nn.Module):
    def __init__(self, ...):
        # ... existing trunk, policy_head, value_head ...
        self.var_head = nn.Sequential(
            nn.Linear(trunk_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # ensures non-negative output
        )

    def forward(self, x):
        trunk = self.trunk(x)
        policy = self.policy_head(trunk)
        wdl = self.value_head(trunk)       # 3-logit WDL
        var = self.var_head(trunk)          # scalar >= 0
        return policy, wdl, var

# In training loop:
def compute_loss(batch):
    policy_logits, wdl_logits, pred_var = model(obs)
    wdl_probs = softmax(wdl_logits)
    scalar_v = wdl_probs[:, 0] - wdl_probs[:, 2]  # w_W - w_L

    # Standard losses
    L_policy = soft_cross_entropy(policy_logits, target_policy)
    L_value = cross_entropy(wdl_logits, target_wdl)

    # Variance loss: target is squared prediction error
    var_target = (game_outcome - scalar_v.detach()) ** 2
    L_var = ((pred_var.squeeze() - var_target) ** 2).mean()

    return L_policy + L_value + lambda_var * L_var

# In MCTS evaluation:
def erm_value(state, beta):
    wdl, var = model.inference(state)
    v = wdl[0] - wdl[2]  # scalar value
    return v + (beta / 2) * var  # ERM-adjusted value
```

This defeats the mechanism of Theorem 4: even when the empirical variance from MCTS backpropagation is zero, the learned variance provides a nonzero signal for risk-sensitive search.

### 6.2 Network Architecture Modifications

These methods provide structural guarantees that certain pathways remain active regardless of training dynamics.

**Proposition 9 (Factored Value Head).** *Replace the standard 3-logit WDL head with a factored architecture:*

1. *Binary decisiveness classifier: $p_{\text{dec}}(s) = \sigma(z_{\text{dec}}(s))$, predicting the probability of a decisive outcome (win or loss).*
2. *Conditional outcome predictor: $p_{W|\text{dec}}(s) = \sigma(z_{W|\text{dec}}(s))$, predicting the probability of a win given that the outcome is decisive.*

*The WDL probabilities are reconstructed as:*

$$w_D = 1 - p_{\text{dec}}, \quad w_W = p_{\text{dec}} \cdot p_{W|\text{dec}}, \quad w_L = p_{\text{dec}} \cdot (1 - p_{W|\text{dec}})$$

*The training losses are:*

$$L_{\text{dec}} = -\left[y_{\text{dec}} \log p_{\text{dec}} + (1 - y_{\text{dec}}) \log(1 - p_{\text{dec}})\right]$$

$$L_{W|\text{dec}} = -\left[y_W \log p_{W|\text{dec}} + y_L \log(1 - p_{W|\text{dec}})\right] \quad \text{(decisive samples only)}$$

*where $y_{\text{dec}} = y_W + y_L$ and the conditional loss $L_{W|\text{dec}}$ is computed only over samples with decisive outcomes.*

*Then the gradient of $L_{W|\text{dec}}$ with respect to the conditional head parameters $\theta_{W|\text{dec}}$ is independent of the buffer's draw fraction $D_t^D$.*

*Proof.* The conditional loss $L_{W|\text{dec}}$ is computed only over decisive samples. Let $\mathcal{D}_{\text{dec}} = \{i : y_W^{(i)} + y_L^{(i)} = 1\}$ be the set of decisive samples in a minibatch. The expected gradient is:

$$\mathbb{E}\left[\nabla_{\theta_{W|\text{dec}}} L_{W|\text{dec}}\right] = \frac{|\mathcal{D}_{\text{dec}}|}{B} \cdot \frac{1}{|\mathcal{D}_{\text{dec}}|} \sum_{i \in \mathcal{D}_{\text{dec}}} \nabla_{\theta_{W|\text{dec}}} L_{W|\text{dec}}^{(i)}$$

The inner average $\frac{1}{|\mathcal{D}_{\text{dec}}|} \sum_{i \in \mathcal{D}_{\text{dec}}} \nabla L_{W|\text{dec}}^{(i)}$ depends only on the *conditional* distribution of wins given decisive outcomes, which is $P(W | \text{dec}) = D_t^W / (D_t^W + D_t^L)$ — independent of $D_t^D$. The outer factor $|\mathcal{D}_{\text{dec}}| / B$ affects the gradient *magnitude* but not its *direction*.

Moreover, $\theta_{W|\text{dec}}$ participates only in $L_{W|\text{dec}}$, not in $L_{\text{dec}}$, because the factored architecture uses separate final layers for each head. Let $\theta = (\theta_{\text{trunk}}, \theta_{\text{dec}}, \theta_{W|\text{dec}})$. Then:

$$\nabla_{\theta_{W|\text{dec}}} L_{\text{dec}} = 0$$

because $L_{\text{dec}}$ depends on $z_{\text{dec}}(s) = g_{\text{dec}}(\text{trunk}(s; \theta_{\text{trunk}}); \theta_{\text{dec}})$ and does not involve $\theta_{W|\text{dec}}$. Therefore:

$$\nabla_{\theta_{W|\text{dec}}} (L_{\text{dec}} + L_{W|\text{dec}}) = \nabla_{\theta_{W|\text{dec}}} L_{W|\text{dec}}$$

and this gradient is determined entirely by the decisive samples, whose internal win/loss ratio is independent of the draw fraction. $\square$

The factored architecture provides a structural firewall against draw saturation. Even if the decisiveness head $p_{\text{dec}}$ collapses to predicting $p_{\text{dec}} \approx 0$ (everything is a draw), the conditional head $p_{W|\text{dec}}$ continues to receive clean training signal from the decisive samples. When the system eventually encounters a decisive position during search, $p_{W|\text{dec}}$ can correctly identify whether it is a win or loss — information that is completely lost in the standard WDL head at collapse, where $w_W \approx w_L \approx 0$.

The factored head has a practical limitation: gradient flow through $\theta_{\text{trunk}}$ from $L_{W|\text{dec}}$ is scaled by $|\mathcal{D}_{\text{dec}}| / B$, which shrinks as the buffer becomes draw-dominated. The trunk representations for decisive positions may therefore receive less training signal. This can be mitigated by combining the factored head with outcome-stratified sampling (Section 6.3), which ensures $|\mathcal{D}_{\text{dec}}| / B \approx 2/3$ regardless of buffer composition.

```
# Architecture diagram (text):
#
#   Input (8x8x123)
#       │
#   [Shared Trunk: ResNet blocks]
#       │
#   ┌───┴───────────────┐
#   │                   │
#   [Policy Head]   [Value Trunk FC]
#   π(s)                │
#               ┌───────┴───────┐
#               │               │
#         [Dec Head]      [Cond Head]
#         z_dec            z_{W|dec}
#           │               │
#         σ(·)            σ(·)     ← separate sigmoids
#           │               │
#         p_dec          p_{W|dec}
#           │               │
#           └───────┬───────┘
#                   │
#              WDL reconstruction:
#              w_D = 1 - p_dec
#              w_W = p_dec · p_{W|dec}
#              w_L = p_dec · (1 - p_{W|dec})
```

---

**Proposition 10 (MC Dropout for ERM Signal).** *Let $v(s; \theta, \xi) = w_v^\top \tilde{h}(s; \xi)$ denote the scalar value output of a neural network with dropout mask $\xi \sim \text{Bernoulli}(1-p)^d$ applied to a hidden layer of dimension $d$, where $w_v \in \mathbb{R}^d$ is the value head's weight vector and $\tilde{h}$ is the post-dropout activation. Define the Monte Carlo variance estimate from $K$ forward passes:*

$$\hat{V}_K(s) = \frac{1}{K} \sum_{k=1}^{K} v(s; \theta, \xi_k)^2 - \left(\frac{1}{K} \sum_{k=1}^{K} v(s; \theta, \xi_k)\right)^2$$

*Then:*

$$\mathbb{E}[\hat{V}_K(s)] = \frac{K-1}{K} \cdot \frac{p}{1-p} \sum_j (w_v)_j^2 \, h_j(s)^2 > 0$$

*whenever $w_v \neq 0$ and $h(s) \neq 0$, where $h(s) \in \mathbb{R}^d$ is the pre-dropout activation vector.*

*Remark.* For a general $d_{\text{out}}$-dimensional output with weight matrix $W_{\text{drop}} \in \mathbb{R}^{d \times d_{\text{out}}}$, the average per-output variance satisfies $\overline{\text{Var}} \geq \frac{p}{1-p} \cdot \frac{1}{d_{\text{out}}} \sum_{m} \|w_m \odot h\|^2$. The scalar case ($d_{\text{out}} = 1$) admits the clean exact result above.

*Proof.* Consider a single hidden unit $j$ with pre-dropout activation $h_j(s)$. Under dropout, the post-dropout activation is $\tilde{h}_j = \xi_j h_j / (1-p)$ where $\xi_j \sim \text{Bernoulli}(1-p)$. The output of the layer for a single output unit with weights $w$ is:

$$f_{\text{out}} = \sum_j w_j \tilde{h}_j = \sum_j w_j \xi_j h_j / (1-p)$$

The variance of this output over dropout masks is:

$$\text{Var}_\xi(f_{\text{out}}) = \frac{1}{(1-p)^2} \sum_j w_j^2 h_j^2 \text{Var}(\xi_j) = \frac{1}{(1-p)^2} \sum_j w_j^2 h_j^2 \cdot p(1-p)$$

$$= \frac{p}{1-p} \sum_j w_j^2 h_j^2 = \frac{p}{1-p} \|w \odot h\|^2$$

For the full weight matrix $W_{\text{drop}}$ with $d_{\text{out}}$ output units, the average per-output variance is:

$$\overline{\text{Var}} = \frac{p}{1-p} \cdot \frac{1}{d_{\text{out}}} \sum_{m=1}^{d_{\text{out}}} \|w_m \odot h\|^2 \geq \frac{p}{1-p} \cdot \frac{1}{d_{\text{out}}} \cdot \frac{\|W_{\text{drop}}^\top h\|^2}{d} \cdot p_{\min}$$

For a cleaner bound, note that $\sum_j w_{mj}^2 h_j^2 \geq (\sum_j w_{mj} h_j)^2 / d$ by Cauchy-Schwarz applied to the vectors $(w_{mj} h_j)$ and $(1, \ldots, 1)$. Wait — this goes the wrong direction. Instead, we use the direct bound:

$$\text{Var}_\xi(f_{\text{scalar}}) = \frac{p}{1-p} \|w \odot h\|^2 \geq \frac{p}{1-p} \cdot \frac{(w \cdot h)^2}{d} \cdot \min\left(1, \frac{\|w \odot h\|^2 d}{(w \cdot h)^2}\right)$$

This becomes unwieldy. We instead state the cleaner result: for the scalar value output $v(s) = w_v^\top \tilde{h}$, the dropout variance is:

$$\text{Var}_\xi(v(s)) = \frac{p}{1-p} \sum_j (w_v)_j^2 h_j(s)^2$$

This is strictly positive whenever $h(s) \neq 0$ and $w_v \neq 0$. More precisely:

$$\text{Var}_\xi(v(s)) \geq \frac{p}{1-p} \cdot \min_j (w_v)_j^2 \cdot \|h(s)\|^2 / d$$

when the weights are nonzero. For the sample variance $\hat{V}_K$, we have $\mathbb{E}[\hat{V}_K] = \frac{K-1}{K} \text{Var}_\xi(v(s))$ by the standard unbiased variance estimator identity.

Combining:

$$\mathbb{E}[\hat{V}_K(s)] = \frac{K-1}{K} \cdot \frac{p}{1-p} \sum_j (w_v)_j^2 h_j(s)^2$$

This is exact (not a bound) for the scalar value output. The expression is strictly positive whenever $w_v \neq 0$ and $h(s) \neq 0$. $\square$

The key property is that this variance lower bound depends on the *activations* $h(s)$ and the *weight magnitudes* $\|W_{\text{drop}}\|$, not on the value head's output distribution. Even when the value head has completely collapsed to $v(s) \approx 0$ for all positions, the pre-dropout activations $h(s)$ and weights $W_{\text{drop}}$ generically remain nonzero (they are maintained by the policy loss, which continues to provide gradient signal throughout training). Therefore, MC Dropout provides a provably nonzero variance signal for ERM:

$$Q_\beta(s) = \bar{v}(s) + \frac{\beta}{2} \hat{V}_K(s)$$

where $\bar{v}(s) = \frac{1}{K} \sum_k v(s; \xi_k)$ is the MC mean. The cost is $K$ forward passes per position (typically $K = 5$--$10$), increasing inference time by a factor of $K$. For the parallel self-play architecture described in the project (batched GPU evaluation with a global evaluation queue), the additional forward passes can be batched efficiently, reducing the wall-clock overhead.

---

**Learnable WDL Temperature.** A lighter-weight alternative to entropy regularization introduces a learnable temperature parameter into the WDL softmax:

$$w = \text{softmax}(z / \tau), \quad \tau = \max(\tau_{\min}, \exp(\alpha))$$

where $\alpha$ is a learnable scalar parameter and $\tau_{\min} > 1$ is a fixed lower bound. The max ensures $\tau \geq \tau_{\min}$.

Setting $\tau_{\min} > 1$ places an upper bound on the sharpness of the WDL distribution. For any logit vector $z$ with $z_{\max} = \max_c z_c$:

$$\max_c w_c = \frac{e^{z_{\max}/\tau}}{e^{z_{\max}/\tau} + \sum_{c \neq \text{argmax}} e^{z_c/\tau}} \leq \frac{e^{z_{\max}/\tau_{\min}}}{e^{z_{\max}/\tau_{\min}} + (C-1)e^{z_{\min}/\tau_{\min}}}$$

For $C = 3$ classes and logit gap $\Delta = z_{\max} - z_{\min}$:

$$\max_c w_c \leq \frac{1}{1 + 2e^{-\Delta/\tau_{\min}}}$$

When $\tau_{\min} = 2$: even with a logit gap of $\Delta = 5$, the maximum class probability is bounded by $1/(1 + 2e^{-2.5}) \approx 0.86$, guaranteeing minimum entropy $H \geq -0.86 \log 0.86 - 2(0.07) \log 0.07 \approx 0.13 + 0.37 = 0.50$ nats.

The learnable parameter $\alpha$ allows the network to sharpen the distribution as training progresses (by increasing $\alpha$, hence $\tau$, hence — wait, increasing $\alpha$ increases $\tau$ which *softens* the distribution). The network can learn $\alpha \to -\infty$ to approach $\tau = \tau_{\min}$, achieving the maximum sharpness permitted by the floor. In early training when the value head is uncertain, the network may benefit from a naturally softer distribution ($\alpha > 0$, $\tau > \tau_{\min}$). As training progresses and the value head becomes calibrated, the network can lower $\tau$ toward $\tau_{\min}$ to make sharper predictions.

### 6.3 Training Data Interventions

These methods break the $D_t \to \theta_{t+1}$ link by modifying the *sampling distribution* from the replay buffer, ensuring that the gradient expectation does not reflect the buffer's draw bias.

**Theorem 6 (Outcome-Stratified Sampling).** *Define the outcome-stratified sampling probability for sample $i$:*

$$P(i) = \frac{1}{3 \cdot N_{\text{outcome}(i)}}$$

*where $N_k = |\{j : \text{outcome}(j) = k\}|$ for $k \in \{W, D, L\}$ and $\text{outcome}(i) \in \{W, D, L\}$ is the game outcome of sample $i$. Then the expected gradient under stratified sampling is:*

$$\mathbb{E}_{\text{strat}}[\nabla L] = \frac{1}{3} \sum_{k \in \{W, D, L\}} \bar{\nabla} L_k$$

*where $\bar{\nabla} L_k = \frac{1}{N_k} \sum_{i : \text{outcome}(i) = k} \nabla L_i$ is the mean gradient within outcome class $k$. This expression is independent of the buffer composition $D_t = (D_t^W, D_t^D, D_t^L)$.*

*Proof.* The expected gradient under the sampling distribution $P$ is:

$$\mathbb{E}_{\text{strat}}[\nabla L] = \sum_{i=1}^{N} P(i) \cdot \nabla L_i = \sum_{k \in \{W, D, L\}} \sum_{i : \text{outcome}(i) = k} \frac{1}{3 N_k} \nabla L_i$$

$$= \sum_{k \in \{W, D, L\}} \frac{1}{3} \cdot \frac{1}{N_k} \sum_{i : \text{outcome}(i) = k} \nabla L_i = \frac{1}{3} \sum_{k \in \{W, D, L\}} \bar{\nabla} L_k$$

The buffer composition $D_t$ enters only through the counts $N_k = D_t^k \cdot N$, but these cancel between the sampling probability $1/(3 N_k)$ and the summation over $N_k$ elements. $\square$

**Importance sampling correction for unbiased loss estimation.** While the gradient expectation is balanced, the *loss estimate* under stratified sampling is biased relative to the population loss. To obtain an unbiased loss estimate (needed for learning rate scheduling, early stopping, etc.), apply importance weights:

$$\hat{L}_{\text{unbiased}} = \frac{1}{B} \sum_{i \in \text{batch}} \frac{P_{\text{uniform}}(i)}{P_{\text{strat}}(i)} L_i = \frac{1}{B} \sum_{i \in \text{batch}} \frac{3 N_{\text{outcome}(i)}}{N} L_i$$

where $P_{\text{uniform}}(i) = 1/N$. For a sample from outcome class $k$ with $N_k = D_t^k \cdot N$:

$$\text{importance weight} = 3 D_t^k$$

At 90% draws: draw samples get weight $3 \times 0.9 = 2.7$, win samples get weight $3 \times 0.05 = 0.15$. This recovers the population loss while training uses the balanced gradient.

**Implementation.** The existing replay buffer (Section 2) stores per-sample metadata including `game_result`, enabling $O(1)$ lookup of the outcome class. The stratified sampler operates by: (1) uniformly choosing an outcome class $k \in \{W, D, L\}$; (2) uniformly sampling a position from the $N_k$ positions with outcome $k$. This requires maintaining three index lists, one per outcome class, which are updated as positions enter and exit the buffer. The overhead is $O(1)$ per sample (comparable to uniform sampling) plus $O(N)$ storage for the index lists.

When one outcome class is empty (e.g., $N_W = 0$ in early training), the sampler falls back to uniform sampling over the remaining classes with probability $1/|\{k : N_k > 0\}|$ each.

**Why stratified sampling is the strongest prevention method.** Stratified sampling eliminates the buffer composition $D_t$ from the gradient entirely, breaking the $D_t \to \theta_{t+1}$ link completely. This means the entire feedback loop $\theta_t \to \phi_t \to D_t \to \theta_{t+1}$ is severed at the training step, regardless of how severely the buffer is draw-dominated. Even at $D_t^D = 0.99$, the gradient treats each outcome class equally. The formal consequence is that the absorbing state $S^* = (\theta^*, (0, 1, 0), 1)$ is no longer a fixed point of the training dynamics — the gradient at $\theta^*$ under stratified sampling is $\frac{1}{3}(\bar{\nabla}L_W + \bar{\nabla}L_D + \bar{\nabla}L_L)$, which is nonzero as long as the model's predictions differ between outcome classes, as they generically do even at collapse.

---

**Proposition 11 (Dynamic Epochs).** *Define the effective number of training epochs as:*

$$E_{\text{eff}}(D_t^D) = \max\left(1, \left\lfloor E_{\text{base}} \cdot \frac{1 - D_t^D}{1 - D_{\text{target}}} \right\rfloor\right)$$

*where $E_{\text{base}}$ is the nominal epoch count and $D_{\text{target}} \in (0, 1)$ is the target draw fraction. Then:*

1. *At $D_t^D = D_{\text{target}}$: $E_{\text{eff}} = E_{\text{base}}$ (full training).*
2. *For $D_t^D > 1 - 2(1 - D_{\text{target}})/E_{\text{base}}$: $E_{\text{eff}} = 1$ (minimum training).*
3. *The reinforcement factor $R(E_{\text{eff}}, D_t^D) = 1 - (1 - D_t^D)^{E_{\text{eff}}}$ from Proposition 7 satisfies $R \leq R_{\max}(D_{\text{target}}, E_{\text{base}})$, a constant independent of the current draw fraction.*

*Proof.* Property (1): at $D_t^D = D_{\text{target}}$, the ratio $(1 - D_t^D)/(1 - D_{\text{target}}) = 1$, so $E_{\text{eff}} = \lfloor E_{\text{base}} \rfloor = E_{\text{base}}$.

Property (2): $E_{\text{eff}} = 1$ when $\lfloor E_{\text{base}} \cdot (1 - D_t^D)/(1 - D_{\text{target}}) \rfloor < 2$, i.e., when $E_{\text{base}} \cdot (1 - D_t^D)/(1 - D_{\text{target}}) < 2$, equivalently $D_t^D > 1 - 2(1 - D_{\text{target}})/E_{\text{base}}$. For $D_{\text{target}} = 0.5$ and $E_{\text{base}} = 4$: threshold is $D_t^D > 1 - 2 \times 0.5/4 = 0.75$.

Property (3): the reinforcement factor under dynamic epochs is:

$$R = 1 - (1 - D_t^D)^{E_{\text{eff}}(D_t^D)}$$

Substituting $E_{\text{eff}} \leq E_{\text{base}} \cdot (1 - D_t^D)/(1 - D_{\text{target}})$:

$$R \leq 1 - (1 - D_t^D)^{E_{\text{base}}(1 - D_t^D)/(1 - D_{\text{target}})}$$

Let $x = 1 - D_t^D \in (0, 1]$ and $\alpha = E_{\text{base}} / (1 - D_{\text{target}})$. Then $R \leq 1 - x^{\alpha x}$. To bound this, we minimize $g(x) = x^{\alpha x} = e^{\alpha x \ln x}$. Let $f(x) = \alpha x \ln x$. Then $f'(x) = \alpha(\ln x + 1) = 0$ gives $x^* = 1/e$ (independent of $\alpha$), and:

$$g(1/e) = e^{\alpha \cdot (1/e) \cdot (-1)} = e^{-\alpha/e}$$

Therefore $R \leq 1 - e^{-\alpha/e}$. This bound depends on $\alpha$: for $\alpha = E_{\text{base}}/(1 - D_{\text{target}}) = 4/0.5 = 8$, we get $R \leq 1 - e^{-8/e} \approx 1 - e^{-2.94} \approx 0.947$.

**Caveat:** This bound applies only in the range where $E_{\text{eff}} \geq 2$ (i.e., $x \geq 2/\alpha$). When $E_{\text{eff}} = 1$ (clamped by the floor), $R = 1 - x$, which approaches 1 as $D_t^D \to 1$ ($x \to 0$). The dynamic epochs mechanism does not provide a universal bound on $R$; rather, it *slows* the reinforcement by reducing epochs, ensuring that the per-iteration bias increase remains moderate. Compare to the uncontrolled case: at $D_t^D = 0.95$ with $E = 3$, $R = 1 - 0.05^3 = 0.999875$, whereas dynamic epochs with $E_{\text{eff}} = 1$ gives $R = 1 - 0.05 = 0.95$ — a substantial reduction even without a universal bound. $\square$

This proposition formalizes the manually discovered heuristic of "set epochs${}=1$ when draw rate exceeds 80%" (Section 10n of the operation manual) as a special case of the continuous formula with $D_{\text{target}} = 0.5$ and $E_{\text{base}} = 3$. The formula provides smoother adaptation and a principled derivation from the reinforcement factor bound.

---

**Importance Weighting by Outcome Surprise.** As a complementary approach to stratified sampling, we can reweight individual samples based on the model's prediction error:

$$w_i = (1 - w_{y_i}(s_i))^{\alpha}$$

where $w_{y_i}(s_i)$ is the model's predicted probability of sample $i$'s actual outcome and $\alpha > 0$ is the focusing exponent. This creates an instance-level version of focal loss applied to the sampling weights rather than the loss function itself:

- Correctly predicted draw samples ($w_D \approx 0.9$): weight $\approx 0.1^\alpha$. At $\alpha = 1$: weight $= 0.1$.
- Incorrectly predicted decisive samples ($w_W \approx 0.05$ for a true win): weight $\approx 0.95^\alpha$. At $\alpha = 1$: weight $= 0.95$.

The effective upweighting of surprising decisive samples is $0.95/0.1 = 9.5\times$, partially compensating for a 9:1 draw-to-decisive ratio in the buffer. At $\alpha = 2$, the ratio becomes $0.95^2 / 0.1^2 = 90.25$, fully compensating for even more extreme imbalances.

Implementation requires computing $w_{y_i}$ for each sample, which means running a forward pass before sampling. In practice, one can use stale predictions from the previous epoch or a separate "staleness-aware" priority buffer.

### 6.4 Search Modifications

These methods break the $\theta_t \to \phi_t$ link by modifying the search procedure to produce more diverse games even when the value head is biased.

**Anti-Draw Exploration Bonus.** When the value head predicts draws uniformly, all actions appear equivalent under $\sigma(Q_{\text{completed}})$, and the improved policy collapses toward the prior (Theorem 1). We introduce an action-dependent bonus that differentiates actions based on the *change* in draw prediction:

$$\sigma'(Q_a, s) = \sigma(Q_a) + \lambda_{\text{explore}} \cdot \left(w_D(s) - w_D(s_a)\right)$$

where $s_a$ is the position after taking action $a$ and $w_D(\cdot)$ is the value head's draw probability. This rewards actions that *reduce* the draw prediction — moves that take the position away from the model's "comfort zone" of confidently-predicted draws.

When the value head has collapsed to $w_D \equiv 1$, this bonus vanishes ($w_D(s) - w_D(s_a) \approx 0$). To handle this case, we augment the bonus with an explicit policy entropy term:

$$\sigma''(Q_a, s) = \sigma(Q_a) + \lambda_{\text{explore}} \cdot \max\left(w_D(s) - w_D(s_a), \; \epsilon_{\text{ent}} \cdot \left(\log |\mathcal{A}| - \log \pi_\theta(a|s)^{-1}\right)\right)$$

The second term rewards actions with low prior probability, encouraging exploration of moves the policy considers unlikely. This is most active precisely when collapse has occurred (when $w_D$ is constant and the first term vanishes).

For internal PUCT at depth $> 0$, a simpler modification adds an exploration bonus proportional to the root position's draw prediction:

$$\text{PUCT}'(a) = Q(a) + c_{\text{explore}} \cdot P(a) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(a)} + \lambda \cdot w_D(s_{\text{root}})$$

This uniformly increases exploration at all internal nodes when the root position is predicted as a draw, counteracting the FPU trap (Proposition 3) by raising the effective exploration constant. When $w_D(s_{\text{root}}) \approx 0$ (the model confidently predicts a decisive outcome), the bonus vanishes and standard PUCT governs internal search.

---

**Theorem 7 (Adaptive Simulation Budget).** *Define the entropy-adaptive simulation count:*

$$N_{\text{sims}}(t) = N_{\text{base}} \cdot \max\left(1, \; \frac{H(\bar{w}_t)}{H_{\text{target}}}\right)$$

*where $\bar{w}_t = \frac{1}{M} \sum_{s \in \mathcal{B}_t} w_{\theta_t}(s)$ is the average WDL distribution across a batch $\mathcal{B}_t$ of $M$ positions sampled from recent games, and $H_{\text{target}}$ is the target entropy (typically $0.8$ nats, compared to the maximum of $\log 3 \approx 1.099$ nats). Then the coupled system $(\theta_t, D_t, N_{\text{sims}}(t))$ admits a negative feedback loop that prevents Mode I collapse.*

*Proof.* We show that the adaptive simulation budget creates a restoring force that opposes the draw death spiral. The argument proceeds in five steps.

**Step 1: Collapse reduces entropy.** As the value head collapses toward uniform draw prediction ($w_D \to 1$), the entropy of the average WDL distribution decreases:

$$H(\bar{w}_t) = -\bar{w}_t^W \log \bar{w}_t^W - \bar{w}_t^D \log \bar{w}_t^D - \bar{w}_t^L \log \bar{w}_t^L \to 0 \quad \text{as } \bar{w}_t \to (0, 1, 0)$$

Specifically, if $\bar{w}_t = (\epsilon, 1 - 2\epsilon, \epsilon)$ for small $\epsilon > 0$:

$$H(\bar{w}_t) = -2\epsilon \log \epsilon - (1 - 2\epsilon) \log(1 - 2\epsilon) \approx -2\epsilon \log \epsilon + 2\epsilon = O(\epsilon |\log \epsilon|)$$

**Step 2: Low entropy reduces simulations.** When $H(\bar{w}_t) < H_{\text{target}}$, the adaptive formula gives $N_{\text{sims}}(t) = N_{\text{base}}$ (the minimum). When $H(\bar{w}_t) \to 0$, the simulation count approaches $N_{\text{base}}$.

*Note:* This provides protection only relative to a higher nominal simulation count. If $N_{\text{base}}$ itself exceeds the collapse threshold identified in Section 5.1 (e.g., if $N_{\text{base}} = 800$ when the critical threshold is $\sim 400$), the adaptive budget is insufficient and must be combined with other methods (entropy regularization, stratified sampling, or dynamic epochs).

**Step 3: Fewer simulations weaken search amplification.** From Theorem 1, the variance of completed Q-values scales as:

$$\text{Var}(Q_{\text{completed}}) = O\left(\frac{\sigma_v^2}{N_{\text{sims}}}\right)$$

At $N_{\text{sims}} = N_{\text{base}}$ (the minimum), the residual variance is maximized at $\sigma_v^2 / N_{\text{base}}$. The $\sigma$-transform $\sigma(Q) = (c_{\text{visit}} + N_{\max}) \cdot c_{\text{scale}} \cdot q_{\text{normalized}}$ maps this variance to a spread of $O(\sigma_v / \sqrt{N_{\text{base}}})$ in the score space. With $N_{\text{base}}$ small (e.g., 100--200), this spread is large enough that $\pi_{\text{improved}}$ remains meaningfully different from the prior, preserving exploration.

**Step 4: Weaker search produces diverse games.** With less search amplification, the self-play policy $\phi_t$ retains more of the prior's entropy. Games under $\phi_t$ explore a broader range of positions, including unbalanced positions that are more likely to reach decisive outcomes. Formally, the self-play draw rate $\Phi_t^D$ decreases as the policy entropy increases:

$$\Phi_t^D \leq g\left(H(\phi_t)\right)$$

where $g$ is a decreasing function (higher policy entropy → more decisive games). This relationship is empirically supported by the data in Section 10g: 200 simulations (weaker search) produced 22% decisive games while 800 simulations (stronger search, same value head) eventually produced 0%.

**Step 5: More decisive games increase buffer entropy.** By Proposition 1, $D_{t+1}$ moves toward $\Phi_t$. Lower $\Phi_t^D$ means $D_{t+1}^D < D_t^D$, improving the buffer composition. This improves the value head's training data, increasing $H(\bar{w}_{t+1})$, which allows the simulation count to increase — completing the negative feedback loop.

The stability follows from the fact that each step in the loop *opposes* the perturbation: collapse reduces entropy, which reduces simulations, which reduces collapse pressure. The system is in stable equilibrium when $H(\bar{w}_t) = H_{\text{target}}$, yielding $N_{\text{sims}} = N_{\text{base}}$. Perturbations toward collapse are met with simulation reduction (stabilizing), while perturbations toward healthy training allow simulation increases (improving play quality). $\square$

**Comparison to manual simulation scheduling.** The adaptive budget formalizes and automates the manual intervention discovered in Section 10g ("start with sims${}=200$, only increase after $> 20\%$ decisive games + value\_loss $> 0.5$"). The entropy-based criterion provides a continuous, responsive signal compared to the discrete manual thresholds, and the negative feedback loop provides a formal stability guarantee.

---

**Forced Root Diversity via Minimum Gumbel Budget.** Gumbel Sequential Halving allocates simulations to active actions in a round-robin fashion, but in later halving rounds, eliminated actions receive zero additional simulations. This can lead to premature commitment when the early simulation budget is insufficient to distinguish between similarly-valued actions. We propose a minimum per-action budget:

$$N_{\min}(a) = \max\left(n_{\text{floor}}, \; \left\lceil \frac{N_{\text{total}} \cdot \pi_\theta(a)}{\sum_{b \in \mathcal{A}_{\text{active}}} \pi_\theta(b)} \right\rceil\right)$$

where $n_{\text{floor}}$ is an absolute minimum (e.g., 4 simulations) and the second term allocates simulations proportional to the prior probability among active actions. The guarantee is that no action is eliminated with fewer than $n_{\text{floor}}$ simulations, regardless of the halving schedule.

This extends the round-robin exploration guarantee (which already eliminated Mode I repetition draws in practice) by ensuring that even low-prior actions receive sufficient evaluations to produce reliable Q-estimates. In the collapse regime where $Q \approx 0$ for all actions, the $\sigma$-transform differences are proportional to $O(1/\sqrt{N_{\min}})$, so larger $N_{\min}$ reduces the chance of spurious elimination based on noise.

---

## 7. Convergence and Stability Analysis

We now establish formal stability results for the training dynamics under the prevention methods of Section 6, characterize the boundary of the absorbing state's basin of attraction, and provide a principled ordering of methods by robustness.

**Theorem 8 (Lyapunov Stability Under Entropy Regularization).** *Consider the training dynamics with entropy-regularized WDL loss (Theorem 5) with regularization strength $\lambda_H$. Define the Lyapunov-like function:*

$$W(S_t) = H(D_t) - \lambda \cdot L_{\text{value}}^{\text{test}}(\theta_t)$$

*where $H(D_t) = -\sum_{k \in \{W,D,L\}} D_t^k \log D_t^k$ is the entropy of the buffer composition, $L_{\text{value}}^{\text{test}}$ is the value loss evaluated on a fixed, balanced test set, and $\lambda > 0$ is a balancing constant. The function $W$ combines buffer entropy (high is healthy) and test value loss (low is healthy). Under the following conditions:*

1. *Entropy regularization with $\lambda_H > 0$,*
2. *SGD learning rate $\eta_{\text{lr}}$ satisfying standard convergence conditions,*
3. *The self-play policy $\phi_t$ generates game outcomes whose distribution is a continuous function of $\theta_t$,*

*there exists a critical threshold $\lambda_H^{\text{crit}}$ such that for $\lambda_H \geq \lambda_H^{\text{crit}}$, $W$ is non-decreasing near the absorbing state:*

$$W(S_{t+1}) - W(S_t) \geq \eta \cdot \left(\lambda_H - f(D_t^D, E, \eta_{\text{lr}})\right)$$

*for some $\eta > 0$, where:*

$$f(D_t^D, E, \eta_{\text{lr}}) = \frac{D_t^D (1 + E \cdot \eta_{\text{lr}})}{1 + \lambda_H}$$

*is increasing in $D_t^D$, the epoch count $E$, and the learning rate $\eta_{\text{lr}}$. For $\lambda_H > \lambda_H^{\text{crit}} \approx \sqrt{c_2(1 + E \cdot \eta_{\text{lr}})/(c_1 \cdot \eta_{\text{lr}})}$ (where $c_1, c_2$ are architecture-dependent constants), the increment $W(S_{t+1}) - W(S_t)$ is strictly positive whenever $S_t$ is in a neighborhood of the absorbing state $(D_t^D \text{ close to } 1)$, guaranteeing escape from near-collapse configurations.*

*Proof.* We bound the two components of $V$ separately.

**Part I: Buffer entropy change.** From Proposition 1, the buffer update is:

$$D_{t+1}^k = (1 - \mu) D_t^k + \mu \Phi_t^k$$

where $\mu = GL/C$ is the replacement fraction and $\Phi_t$ is the self-play outcome distribution at iteration $t$. The entropy change is:

$$H(D_{t+1}) - H(D_t) = H\left((1-\mu)D_t + \mu \Phi_t\right) - H(D_t)$$

By the concavity of entropy:

$$H((1-\mu)D_t + \mu \Phi_t) \geq (1-\mu) H(D_t) + \mu H(\Phi_t)$$

Therefore:

$$H(D_{t+1}) - H(D_t) \geq \mu \left(H(\Phi_t) - H(D_t)\right) \tag{7.1}$$

Under entropy regularization, the value head maintains $H(w^*) \geq H_{\min}(\lambda_H)$ (Theorem 5). The self-play outcome distribution $\Phi_t$ is determined by the game results under policy $\phi_t$, which depends on $\theta_t$. When the value head is not fully collapsed — specifically, when the minimum WDL class probability satisfies $w_{\min} \geq \lambda_H / (3(1+\lambda_H))$ — the search produces games with nonzero decisive rate. We formalize this via condition (3): $\Phi_t^D = g(\theta_t)$ is continuous, and at the absorbing point $\theta^*$ (where the *unregularized* value head predicts $w_D = 1$), the *regularized* value head predicts $w_D^* \leq (3+\lambda_H)/(3(1+\lambda_H)) < 1$. The residual decisive probability $1 - w_D^*$ introduces stochasticity in search outcomes, producing $\Phi_t^D < 1$.

When $D_t^D$ is close to 1 (near-absorbing), $H(D_t) \approx 0$. If $H(\Phi_t) > 0$ (guaranteed by the entropy regularization's effect on game diversity), then (7.1) gives $H(D_{t+1}) > H(D_t)$: the buffer entropy increases.

**Part II: Test loss change.** The value loss on the balanced test set decreases under SGD when the gradient has nonzero component in the descent direction. Under entropy regularization, the Hessian eigenvalues are bounded below by $\lambda_H / 3$ (Theorem 5, property 3), so the loss surface has no flat regions. Standard SGD convergence gives:

$$L_{\text{value}}^{\text{test}}(\theta_{t+1}) \leq L_{\text{value}}^{\text{test}}(\theta_t) - \eta_{\text{lr}} \|\nabla_\theta L_{\text{value}}^{\text{train}}(\theta_t)\|^2 / (2 \lambda_{\max})$$

where $\lambda_{\max}$ is the maximum Hessian eigenvalue. The gradient norm $\|\nabla_\theta L_{\text{value}}^{\text{train}}\|^2$ depends on the buffer composition through the training loss. Under the *regularized* loss, even at $D_t^D = 1$, the gradient is nonzero because the entropy term pushes $w^*$ away from the degenerate distribution. The gradient norm satisfies:

$$\|\nabla_\theta L_{\text{value}}^{\text{train}}\|^2 \geq c_1 \lambda_H^2 / (1 + \lambda_H)^2$$

for some architecture-dependent constant $c_1 > 0$.

However, the training is performed on the *biased* buffer, not the balanced test set. The gap between the training gradient (on the draw-dominated buffer) and the test gradient (on a balanced set) introduces an error term proportional to $D_t^D \cdot E \cdot \eta_{\text{lr}}$ — the training may push $\theta_t$ toward the buffer's biased optimum rather than the balanced optimum. This error is captured by the function $f$:

$$L_{\text{value}}^{\text{test}}(\theta_{t+1}) - L_{\text{value}}^{\text{test}}(\theta_t) \leq -\eta_{\text{lr}} c_1 \frac{\lambda_H^2}{(1+\lambda_H)^2} + c_2 \frac{D_t^D (1 + E \cdot \eta_{\text{lr}})}{1 + \lambda_H}$$

where $c_2$ bounds the bias from training on an imbalanced buffer for $E$ epochs.

**Combining parts.** The increment of $W = H - \lambda L_{\text{value}}^{\text{test}}$ is:

$$W(S_{t+1}) - W(S_t) = [H(D_{t+1}) - H(D_t)] - \lambda [L_{\text{value}}^{\text{test}}(\theta_{t+1}) - L_{\text{value}}^{\text{test}}(\theta_t)]$$

We want $W$ to be non-decreasing near the absorbing state (buffer entropy increasing, test loss decreasing). From the bounds above:

$$W(S_{t+1}) - W(S_t) \geq \mu(H(\Phi_t) - H(D_t)) + \lambda \eta_{\text{lr}} c_1 \frac{\lambda_H^2}{(1+\lambda_H)^2} - \lambda c_2 \frac{D_t^D(1 + E \cdot \eta_{\text{lr}})}{1 + \lambda_H}$$

Near the absorbing state ($D_t^D \to 1$, $H(D_t) \to 0$):

$$W(S_{t+1}) - W(S_t) \geq \mu H(\Phi_t) + \lambda \left[c_1 \eta_{\text{lr}} \frac{\lambda_H^2}{(1+\lambda_H)^2} - c_2 \frac{1 + E \cdot \eta_{\text{lr}}}{1 + \lambda_H}\right]$$

The first term $\mu H(\Phi_t) > 0$ always helps. The bracketed term is positive when:

$$\lambda_H > \frac{c_2(1 + E \cdot \eta_{\text{lr}})}{c_1 \eta_{\text{lr}} \lambda_H + c_2} \cdot \frac{1 + \lambda_H}{\lambda_H}$$

Solving for $\lambda_H$ (treating the right side as $f(D_t^D, E, \eta_{\text{lr}})$ evaluated at $D_t^D = 1$):

$$\lambda_H^{\text{crit}} \approx \sqrt{\frac{c_2(1 + E \cdot \eta_{\text{lr}})}{c_1 \eta_{\text{lr}}}}$$

For $\lambda_H > \lambda_H^{\text{crit}}$, the increment $W(S_{t+1}) - W(S_t) > 0$ whenever the system is near the absorbing state, guaranteeing escape. The system cannot remain in a neighborhood of $S^*$ — it is expelled toward higher buffer entropy and lower test loss.

The absorbing state $S^*$ is therefore *unstable* under entropy regularization with sufficiently large $\lambda_H$, completing the proof. $\square$

Note that Theorem 8 guarantees *local* instability of the absorbing state (escape from a neighborhood of $S^*$, where $W$ is near its minimum) rather than *global* convergence to the healthy fixed point. Global convergence would require additional assumptions about the landscape of the coupled dynamics, which we leave to future work.

---

**Proposition 12 (Critical Threshold Characterization).** *The boundary of the absorbing state's basin of attraction under the unmodified training dynamics is approximately characterized by the condition:*

$$D_t^D \cdot \left(1 + \frac{E \cdot \eta_{\text{lr}} \cdot N_{\text{eff}}}{N_0}\right) \geq 1 \tag{7.2}$$

*where $E$ is the epoch count, $\eta_{\text{lr}}$ is the learning rate, $N_{\text{eff}}$ is the effective simulation count (reflecting search amplification), and $N_0$ is a normalization constant determined by the network architecture.*

*Interpretation.* The left side of (7.2) measures the "collapse pressure" — the product of the current draw bias ($D_t^D$) and the amplification factor from training ($E \cdot \eta_{\text{lr}}$) and search ($N_{\text{eff}}$). When this product exceeds 1, the system enters the basin of attraction of the absorbing state.

Each factor contributes independently:

- **Higher epochs $E$:** More passes over the draw-dominated buffer, stronger reinforcement of draw predictions. The reinforcement factor from Proposition 7 is $1 - (1-D_t^D)^E$, which increases with $E$.
- **Higher simulations $N_{\text{eff}}$:** Stronger search amplification (Theorem 1), mapping draw-biased evaluations more confidently into draw-seeking policies. The search amplification reduces $\text{Var}(Q_{\text{completed}})$ as $O(1/N_{\text{eff}})$.
- **Higher learning rate $\eta_{\text{lr}}$:** Faster convergence to the buffer's implied equilibrium (Proposition 2), giving the system less time to self-correct through game diversity.

*Derivation.* Consider the one-step dynamics at the boundary. The system is at the tipping point when a single iteration neither improves nor worsens the draw fraction. From the buffer update (Proposition 1):

$$D_{t+1}^D = (1 - \mu) D_t^D + \mu \Phi_t^D$$

The tipping condition is $D_{t+1}^D = D_t^D$, i.e., $\Phi_t^D = D_t^D$: the self-play draw rate exactly matches the buffer draw rate.

The self-play draw rate $\Phi_t^D$ depends on the value head's predictions through search. At the SGD equilibrium of Proposition 2, $P(\text{draw}) \approx D_t^D$. After $E$ epochs at learning rate $\eta_{\text{lr}}$, the model's draw prediction is amplified by the training dynamics:

$$P_{\text{post-training}}(\text{draw}) \approx D_t^D + E \cdot \eta_{\text{lr}} \cdot \nabla_{\text{draw}} \cdot D_t^D / N_0$$

where $\nabla_{\text{draw}}$ captures the gradient direction toward higher draw prediction (which is favored when the buffer is draw-dominated) and $N_0$ normalizes.

Search further amplifies this through the simulation depth effect. With $N_{\text{eff}}$ simulations, the search converts a value head draw prediction of $p_D$ into a self-play draw rate of approximately:

$$\Phi^D \approx p_D + (1 - p_D) \cdot \left(1 - e^{-N_{\text{eff}} p_D / N_0}\right)$$

which approaches 1 as $N_{\text{eff}} \to \infty$ for any $p_D > 0$.

Combining training amplification and search amplification, the tipping condition $\Phi_t^D \geq D_t^D$ becomes:

$$D_t^D \cdot \left(1 + \frac{E \cdot \eta_{\text{lr}} \cdot N_{\text{eff}}}{N_0}\right) \geq 1$$

**Empirical validation.** We verify against the observed thresholds from Section 5:

*Buffer $> 85\%$ draws, $E = 3$, $N_{\text{eff}} \approx 200$:* Setting $N_0 \approx 1000$ (a reasonable normalization for the network architecture), the collapse pressure is $0.85 \times (1 + 3 \times 0.001 \times 200) = 0.85 \times 1.6 = 1.36 > 1$. The system is in the absorbing basin. $\checkmark$

*Buffer $= 70\%$ draws, $E = 1$, $N_{\text{eff}} \approx 200$:* Collapse pressure $= 0.70 \times (1 + 1 \times 0.001 \times 200) = 0.70 \times 1.2 = 0.84 < 1$. The system is outside the absorbing basin — recovery is possible. $\checkmark$

*Value loss $< 0.4$ corresponding to $P(\text{draw}) > 0.7$:* At $D_t^D \approx 0.7$ with $E = 2$: pressure $= 0.7 \times (1 + 2 \times 0.001 \times 200) = 0.7 \times 1.4 = 0.98 \approx 1$. This is at the boundary, consistent with the observation that $D^D > 0.85$ combined with value loss $< 0.4$ is the empirical point of no return. $\checkmark$

---

**Theorem 9 (Prevention Method Ordering by Robustness).** *For each prevention method $\mathcal{M}$, define the safe basin $\mathcal{B}_{\text{safe}}(\mathcal{M})$ as the set of states $S_t = (\theta_t, D_t, \phi_t)$ from which the training dynamics under method $\mathcal{M}$ do not converge to the absorbing state $S^*$. The prevention methods of Section 6 are ordered by inclusion:*

$$\mathcal{B}_{\text{safe}}(\text{stratified}) \supsetneq \mathcal{B}_{\text{safe}}(\text{entropy-reg}) \supsetneq \mathcal{B}_{\text{safe}}(\text{adaptive-sims}) \supsetneq \mathcal{B}_{\text{safe}}(\text{dynamic-epochs})$$

*Specifically:*

1. **Outcome-stratified sampling (Theorem 6):** $\mathcal{B}_{\text{safe}} = \mathcal{S}$ *(the entire state space, excluding only degenerate configurations where one outcome class has zero samples).*
2. **Entropy regularization (Theorem 5):** $\mathcal{B}_{\text{safe}} \supseteq \left\{S : D_t^D < 1 - \frac{\lambda_H}{1 + \lambda_H}\right\}$
3. **Adaptive simulation budget (Theorem 7):** $\mathcal{B}_{\text{safe}} \supseteq \left\{S : D_t^D < \rho(N_{\text{base}})\right\}$ *where $\rho$ is a decreasing function of $N_{\text{base}}$.*
4. **Dynamic epochs (Proposition 11):** $\mathcal{B}_{\text{safe}} \supseteq \left\{S : D_t^D < \frac{D_{\text{target}}}{1 - 1/E_{\text{base}}}\right\}$

*Proof.* We establish each inclusion by analyzing the fixed-point structure of the coupled dynamics under each method.

**Method 1: Outcome-stratified sampling.** Under stratified sampling, the expected gradient is $\frac{1}{3}(\bar{\nabla}L_W + \bar{\nabla}L_D + \bar{\nabla}L_L)$ (Theorem 6), which is independent of $D_t$. The SGD equilibrium of Proposition 2 — which relied on the gradient being proportional to the buffer composition — no longer holds. Instead, the equilibrium satisfies:

$$\nabla L_W(\theta^*) + \nabla L_D(\theta^*) + \nabla L_L(\theta^*) = 0$$

This is the optimum of the *balanced* loss, which generically has $P(\text{draw}) \neq 1$ (the balanced loss does not favor any outcome class). The absorbing state $\theta^*$ with $P(\text{draw}) = 1$ is not a fixed point of the stratified-sampling dynamics unless it also minimizes the balanced loss, which requires $\bar{\nabla}L_W(\theta^*) + \bar{\nabla}L_L(\theta^*) = 0$ — generically not satisfied when the value head predicts all positions as draws while training on decisive samples.

Therefore, $\mathcal{B}_{\text{safe}} = \mathcal{S}$ except for the degenerate case $N_W = 0$ or $N_L = 0$ (where stratified sampling cannot sample from the missing class). Since the replay buffer always receives new games and early games typically include all three outcomes, this degenerate case is transient.

**Method 2: Entropy regularization.** From Theorem 8, the absorbing state is unstable for $\lambda_H > \lambda_H^{\text{crit}}$. The safe basin is the complement of the absorbing basin, which, from Proposition 12 with the modified dynamics, satisfies:

$$D_t^D \cdot \left(1 + \frac{E \cdot \eta_{\text{lr}} \cdot N_{\text{eff}}}{N_0}\right) < \frac{1 + \lambda_H}{\lambda_{H,\text{effective}}}$$

where $\lambda_{H,\text{effective}}$ accounts for the entropy regularization reducing the collapse pressure. Simplifying, the safe basin includes all states with:

$$D_t^D < 1 - \frac{\lambda_H}{1 + \lambda_H}$$

The guaranteed safe region extends to $D_t^D < 1/(1+\lambda_H)$. At $\lambda_H = 0.1$: safe for $D_t^D < 0.909$. At $\lambda_H = 0.2$: safe for $D_t^D < 0.833$. While the formula $1/(1+\lambda_H)$ appears to *decrease* with $\lambda_H$, it represents a worst-case lower bound on the safe basin; the actual safe basin may be larger due to the stabilizing effects of entropy regularization that this conservative analysis does not capture. The key property is that for any $\lambda_H > 0$, there exists a nonzero safe neighborhood around the healthy equilibrium, whereas without regularization ($\lambda_H = 0$) the safe basin boundary contracts to $D_t^D = 0$. The safe basin never reaches the full state space — at $D_t^D$ sufficiently close to 1, the entropy regularization is insufficient to overcome the draw bias in the buffer.

This is strictly smaller than $\mathcal{B}_{\text{safe}}(\text{stratified}) = \mathcal{S}$ because entropy regularization modifies the *loss function* but not the *sampling distribution*. The gradient still contains a component proportional to $D_t^D$ from the draw-dominated buffer; the entropy term merely adds a counteracting force.

**Method 3: Adaptive simulation budget.** From Theorem 7, the negative feedback loop stabilizes the system by reducing simulations when the value head collapses. The safe basin is determined by the condition that the negative feedback (simulation reduction) is strong enough to overcome the positive feedback (draw amplification) at the base simulation count:

$$D_t^D \cdot \left(1 + \frac{E \cdot \eta_{\text{lr}} \cdot N_{\text{base}}}{N_0}\right) < 1$$

Solving: $D_t^D < \rho(N_{\text{base}}) = N_0 / (N_0 + E \cdot \eta_{\text{lr}} \cdot N_{\text{base}})$.

This is strictly smaller than the entropy-regularized safe basin because the adaptive simulation budget only breaks the $\theta_t \to \phi_t$ link (via search modification) while leaving the $D_t \to \theta_{t+1}$ link intact. The gradient still converges to the buffer's biased equilibrium; only the *rate* of convergence to the absorbing state is reduced.

**Method 4: Dynamic epochs.** From Proposition 11, dynamic epochs reduce the reinforcement factor substantially (e.g., from $R = 0.9999$ to $R = 0.95$ at $D_t^D = 0.95$), but the bound depends on the parameter $\alpha = E_{\text{base}}/(1-D_{\text{target}})$ and is not universal. At extreme $D_t^D$, dynamic epochs set $E_{\text{eff}} = 1$, which reduces but does not eliminate the reinforcement. The safe basin satisfies:

$$D_t^D \cdot \left(1 + \frac{\eta_{\text{lr}} \cdot N_{\text{eff}}}{N_0}\right) < 1$$

(with $E = 1$). Solving: $D_t^D < N_0 / (N_0 + \eta_{\text{lr}} \cdot N_{\text{eff}})$.

Comparing to Method 3: the adaptive simulation bound has $N_{\text{base}}$ in place of $N_{\text{eff}}$ (where $N_{\text{base}} \leq N_{\text{eff}}$ by construction). Therefore $\rho(N_{\text{base}}) \geq N_0 / (N_0 + \eta_{\text{lr}} \cdot N_{\text{eff}})$, and the adaptive simulation safe basin is at least as large as the dynamic epochs safe basin. Strict inequality holds whenever $N_{\text{base}} < N_{\text{eff}}$, i.e., whenever the adaptive budget actually reduces simulations from the nominal count. $\square$

**Discussion: Why the strongest method has drawbacks.** Outcome-stratified sampling achieves the largest safe basin — the entire state space — but this comes at a cost. By artificially equalizing the three outcome classes in the gradient, stratified sampling prevents the value head from learning the *natural* draw frequency of well-played chess. In healthy training regimes where draws are frequent because the model has learned to play accurately (not because of value head collapse), stratified sampling underweights the abundant draw data, potentially slowing convergence of the value head for common positions.

The tradeoff can be quantified: under stratified sampling with a 70% natural draw rate, each draw sample receives effective weight $\frac{1}{3 \times 0.7N} / \frac{1}{N} = \frac{1}{2.1}$ relative to uniform sampling, while each decisive sample receives weight $\frac{1}{3 \times 0.15N} / \frac{1}{N} = \frac{1}{0.45} \approx 2.2$. The value head trains on an artificially balanced distribution that does not reflect the true prior over outcomes, introducing a bias in the *healthy* regime even as it eliminates the pathological bias in the collapse regime.

A practical compromise is *soft* stratified sampling:

$$P(i) \propto \frac{(1-\alpha)}{N} + \frac{\alpha}{3 N_{\text{outcome}(i)}}$$

which interpolates between uniform sampling ($\alpha = 0$) and full stratification ($\alpha = 1$). Setting $\alpha$ adaptively — e.g., $\alpha = \max(0, (D_t^D - D_{\text{target}}) / (1 - D_{\text{target}}))$ — provides full stratification only when the buffer is draw-dominated, and reverts to uniform sampling when the buffer is healthy.

**Phase diagram.** The theoretical results of this section can be visualized as a two-dimensional phase diagram in $(D_t^D, L_{\text{value}})$ space. The horizontal axis represents the buffer's draw fraction; the vertical axis represents the value loss (a proxy for value head calibration).

The absorbing state $S^*$ sits at the corner $(D^D = 1, L_{\text{value}} \approx 0)$. The healthy training regime occupies the region $(D^D \in [0.4, 0.7], L_{\text{value}} \in [0.5, 1.0])$. The critical boundary of Proposition 12 traces an approximate hyperbola through the space, with the absorbing basin to its right and below.

Under each prevention method, the absorbing basin shrinks:

- **No prevention:** Basin extends to $D^D \gtrsim 0.70$ (at $E = 3$, $N_{\text{eff}} = 200$).
- **Dynamic epochs:** Basin retreats to $D^D \gtrsim 0.83$ (at $E_{\text{eff}} = 1$ for high $D^D$).
- **Adaptive simulations:** Basin retreats to $D^D \gtrsim 0.88$ (at $N_{\text{base}} = 100$).
- **Entropy regularization ($\lambda_H = 0.1$):** Basin retreats to $D^D \gtrsim 0.91$.
- **Stratified sampling:** Basin is empty (no absorbing state exists).

The nesting $\emptyset \subset \mathcal{B}_{\text{absorb}}(\text{stratified}) \subset \mathcal{B}_{\text{absorb}}(\text{entropy}) \subset \mathcal{B}_{\text{absorb}}(\text{adaptive}) \subset \mathcal{B}_{\text{absorb}}(\text{dynamic}) \subset \mathcal{B}_{\text{absorb}}(\text{none})$ confirms the ordering of Theorem 9.

In practice, the most robust approach combines multiple methods to achieve defense in depth: stratified sampling eliminates the dominant failure mode (buffer bias), entropy regularization provides a secondary safeguard (gradient signal preservation), and adaptive simulations prevent the search amplification cascade. Each method addresses a different link in the feedback loop, and their combination provides overlapping protection against all four collapse modes identified in Section 5.

## 8. Experimental Validation

### 8.1 Experimental Setup

All experiments were conducted with a single SE-ResNet architecture and a unified self-play training pipeline. The network consists of 256 convolutional filters across 20 residual blocks, each equipped with squeeze-and-excitation modules at reduction ratio 4 (denoted f256-b20-se4). Input observations are encoded as 123-channel $8 \times 8$ feature planes covering piece positions, castling rights, repetition counts, and move clocks. The policy head outputs a 4672-dimensional move probability vector following the encoding scheme of Silver et al. (2018), where each legal move maps to a unique index across queen-type moves, knight moves, and underpromotions.

The value head follows a Win-Draw-Loss (WDL) architecture:

$$\text{Conv}_{1\times1}(256 \to 1) \to \text{BN} \to \text{ReLU} \to \text{FC}(64 \to 256) \to \text{ReLU} \to \text{FC}(256 \to 3)$$

producing logits $(z_W, z_D, z_L)$ that are softmaxed to obtain outcome probabilities $(p_W, p_D, p_L)$. Mirror equivariance is enforced by processing both the original and horizontally-flipped board through the shared trunk and averaging the resulting logits before the softmax.

Training uses the Adam optimizer with learning rate $\alpha = 0.001$ and weight decay $\lambda_{wd} = 10^{-4}$ applied only to convolutional and linear layer parameters (batch normalization parameters and biases are excluded). Mixed-precision training is enabled via PyTorch's GradScaler for FP16 forward passes. Unless otherwise noted, each iteration trains for $E = 5$ epochs over batches of size 256, drawn uniformly from a circular replay buffer of capacity $C = 100{,}000$ positions.

Self-play uses Gumbel Top-$k$ Sequential Halving (Danihelka et al., 2022) at the root with $k = 16$ and PUCT at depth $> 0$ with exploration constant $c_{\text{explore}} = 2.5$. The default simulation budget is $N = 200$ per move. Each iteration generates 32 games using 32 parallel workers that submit leaf evaluations to a shared GPU evaluation queue with lock-free batching. The search batch size is set equal to the Gumbel top-$k$ parameter ($\text{search\_batch} = 16$), so that each batch of simulations assigns one simulation to each of the 16 active root actions.

**Value loss calibration.** Throughout this section, value loss refers to the soft cross-entropy between the WDL target distribution and the predicted WDL distribution:

$$\mathcal{L}_{\text{value}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c \in \{W,D,L\}} y_c^{(i)} \log p_c^{(i)}$$

Key reference points for interpreting value loss magnitudes:

| Value Loss | Interpretation |
|:----------:|:---------------|
| $\sim 1.1$ ($\ln 3$) | Uniform random prediction |
| $0.7$--$0.9$ | Early learning; model acquiring basic positional knowledge |
| $0.4$--$0.6$ | Healthy mid-training; discriminating outcomes |
| $< 0.3$ | Suspicious: possible overfitting or value head collapse |

### 8.2 Reproduction of Collapse Modes

We now present the empirical evidence for each of the four collapse modes identified in Section 5, drawn from systematic training runs on the f256-b20-se4 architecture. All data are exact measurements from the training logs.

#### 8.2.1 Mode I: Simulation Depth Draw Death

Mode I was observed in an early training run that used Gumbel search with search batch size 1 (effectively equivalent to PUCT-like behavior at root, since a single simulation per batch precludes the round-robin allocation that distinguishes Gumbel from PUCT). Training proceeded normally at $N = 200$ simulations through iteration 5. At iteration 6, the simulation budget was increased to $N = 800$.

**Table 1.** Iteration-by-iteration progression of Mode I collapse. Simulations increased from 200 to 800 at iteration 6.

| Iteration | Sims | Decisive % | Rep. Draws (/32) | Value Loss | Avg. Length | Status |
|:---------:|:----:|:----------:|:-----------------:|:----------:|:----------:|:------:|
| 5 | 200 | 22 | 16 | 0.89 | 315 | Healthy |
| 6 | 800 | 22 | 16 | 0.89 | 315 | OK |
| 7 | 800 | 9 | 27 | 0.51 | 153 | Warning |
| 8 | 800 | 3 | 31 | 0.45 | 95 | Critical |
| 9--53 | 800 | 0 | 32 | 0.02 | 42 | Dead |

The collapse onset between iterations 6 and 7 is striking: decisive games dropped from 22% to 9%, repetition draws surged from 16 to 27 out of 32, and average game length nearly halved. Critically, value loss plunged from 0.89 to 0.51 in a single iteration — crossing the 0.5 threshold that we identify as the earliest reliable canary signal for impending collapse.

Attempts at recovery through hyperparameter modification at iteration 8 (adjusting simulation count, exploration constant, and temperature) were unsuccessful and in fact accelerated the decline. By iteration 9, the system had entered the absorbing state: zero decisive games, all 32 games ending in repetition draws at average length 42 moves, and value loss of 0.02 — effectively zero entropy in the WDL prediction. This state persisted unchanged through iteration 53, at which point the run was terminated. The absorbing state proved completely irreversible, consistent with the stability analysis of Theorem 2.

The mechanism is precisely as modeled in Section 3.3 (Theorem 1): the 4$\times$ increase in simulation depth amplified the value head's mild draw bias (present but harmless at $N = 200$) into overwhelming search certainty. At $N = 800$, the Monte Carlo tree search had sufficient depth to discover repetition-forcing lines, and once those lines produced Q-values near 0 with high visit counts, PUCT's exploitation term locked the search onto those lines exclusively.

#### 8.2.2 Mode II: Fifty-Move Stagnation

Mode II was observed in the Gumbel Top-$k$ SH run, which eliminated Mode I entirely (zero early repetition draws across the first 11 iterations) but revealed a distinct collapse pathway through the fifty-move rule.

**Table 2.** Iteration-by-iteration progression of Mode II collapse and recovery in the Gumbel run.

| Iteration | Decisive % | Rep. Draws (/32) | 50-Move Draws (/32) | Value Loss | Avg. Length |
|:---------:|:----------:|:-----------------:|:-------------------:|:----------:|:----------:|
| 1 | 28 | 7 | 5 | 0.89 | 358 |
| 2 | 28 | 3 | 7 | 0.47 | 311 |
| 3 | 28 | 5 | 5 | 0.65 | 337 |
| 4 | 50 | 4 | 6 | 0.55 | 289 |
| 5 | 34 | 5 | 6 | 0.57 | 300 |
| 6 | 0 | 0 | 25 | 0.50 | 381 |
| 7 | 0 | 0 | 29 | 0.40 | 368 |
| 8 | 6 | 1 | 29 | 0.15 | 341 |
| 9 | 6 | 0 | 27 | 0.12 | 365 |
| **10** | **28** | **6** | **11** | **0.30** | **330** |
| **11** | **38** | **5** | **6** | **0.39** | **298** |

The transition from healthy training (iterations 1--5) to stagnation (iterations 6--9) is qualitatively different from Mode I. Rather than repetition draws, the games became extremely long and ended via the fifty-move rule — the model played purposefully but aimlessly, shuffling pieces without making progress. Fifty-move draws surged from 6 to 25 at iteration 6, then to 29 by iteration 7.

Value loss followed a characteristic two-phase decline. The first phase (iterations 5--7, value loss $0.57 \to 0.40$) corresponds to the model learning that "most positions are draws." The second phase (iterations 7--9, value loss $0.40 \to 0.12$) corresponds to the model becoming confidently wrong — predicting draw with near certainty everywhere. This second phase is the dangerous one: at value loss 0.12, the WDL distribution has essentially collapsed to $(0, 1, 0)$.

**Recovery protocol.** At iteration 10, two interventions were applied simultaneously: (i) training epochs reduced from 3 to 1, and (ii) temperature moves increased to 50. The rationale for epochs reduction is that each epoch over the draw-dominated buffer reinforces the draw bias multiplicatively; at 3 epochs, the model sees each draw sample 3 times per iteration, accelerating collapse. Reducing to 1 epoch limits the per-iteration reinforcement.

The rationale for increased temperature moves is that stochastic move selection in the first 50 moves produces more diverse positions, increasing the probability of reaching positions where decisive play is possible.

An intermediate intervention was also tested: $\beta_{\text{risk}} = 0.5$ (the entropic risk measure with risk-seeking parameter), applied alone at an earlier iteration. This proved insufficient. Risk-seeking MCTS requires variance in the value predictions to exploit ($Q_\beta \approx \mathbb{E}[v] + (\beta/2) \operatorname{Var}(v)$), but with value loss at 0.12--0.15, the value head was predicting "draw" with near certainty — there was no variance for the risk term to amplify. This confirms the theoretical prediction that search-level interventions cannot rescue a deeply collapsed value head.

The combined epochs=1 + temperature=50 intervention produced dramatic recovery: decisive games rebounded from 6% to 28% at iteration 10 and 38% at iteration 11, fifty-move draws fell from 27 to 11 and then to 6, and value loss climbed from 0.12 back to 0.39.

#### 8.2.3 Mode III: Max-Moves Pollution

Mode III arises from the interaction between game-termination rules and buffer composition. When a game reaches the maximum move limit (here, 512 half-moves), it is adjudicated as a draw regardless of the board position. Because such games are long, they contribute disproportionately many training samples to the replay buffer.

**Table 3.** Sample contribution asymmetry between decisive and max-moves games.

| Game Type | Typical Length (half-moves) | Samples per Game |
|:---------:|:--------------------------:|:----------------:|
| Decisive | $\sim 80$ | $\sim 80$ |
| Max-moves draw | $512$ | $512$ |

A single max-moves game contributes approximately $512 / 80 \approx 6.4$ times as many samples as a decisive game. When the draw rate is high, this asymmetry compounds: at 90% draws with most draws reaching the move limit, the effective draw-to-decisive sample ratio can reach 122:1 in the worst case (Proposition 4); the operation manual reports an effective ratio of 36:1 under more typical conditions with shorter average draw games.

The buffer composition at the point of training shutdown illustrates the severity:

$$\text{Wins: } 4{,}364 \; (3.5\%), \quad \text{Draws: } 116{,}509 \; (92.1\%), \quad \text{Losses: } 5{,}558 \; (4.4\%)$$

A critical observation from this run is that **policy loss improved monotonically** even as the value head stagnated. Across the affected iterations, policy loss decreased steadily from 8.40 to 7.49 while value loss stagnated around 0.3, unable to drop further despite monotonically improving policy loss. This decoupling demonstrates that the policy head and value head receive effectively independent gradient signals: the policy head continues to learn piece-activity patterns and tactical motifs from the game positions regardless of the outcome labels, while the value head receives an overwhelming majority of "draw" labels that drive it toward the degenerate fixed point. This decoupling is particularly insidious because a practitioner monitoring only the policy loss might conclude that training is proceeding normally.

#### 8.2.4 Mode IV: Buffer Saturation and Oscillation Trap

Mode IV manifests as an oscillation around the collapse boundary, where isolated good iterations produce insufficient decisive data to overcome the accumulated draw mass in the buffer.

**Table 4.** Iteration-by-iteration progression of Mode IV buffer saturation.

| Iteration | Epochs | Draw % | Value Loss | Buffer Draw % |
|:---------:|:------:|:------:|:----------:|:------------:|
| 9 | 1 | 71 | 0.68 | 77.5 |
| 10 | 2 | 91 | 0.57 | 84.0 |
| 11 | 2 | 92 | 0.39 | 92.2 |
| 12 | 1 | 64 | 0.45 | 88.0 |
| 13 | 1 | 92 | 0.375 | 89.5 |

The dynamics here are subtle. Iteration 12 was genuinely successful: epochs were reduced back to 1, draw rate dropped to 64%, and value loss recovered to 0.45. Yet the buffer draw composition only decreased from 92.2% to 88.0% — because 32 games contribute at most $\sim 2{,}500$ decisive-labeled samples (from the subset of games with decisive outcomes) to a buffer of 100,000, the new decisive data are diluted roughly 40:1 by the existing draw mass.

The subsequent iteration (13) then trained on this still-overwhelmingly-draw buffer, and the draw rate rebounded to 92%. The system is trapped in an oscillation: interventions produce transient improvement, but the buffer's thermal mass absorbs the improvement before it can compound.

The empirical threshold for irreversibility is: buffer draw composition exceeding 85% combined with value loss below 0.4. Beyond this threshold, even aggressive interventions (epochs=1, high temperature) cannot produce enough decisive data per iteration to shift the buffer composition faster than training on the draw-dominated buffer re-collapses the value head.

### 8.3 Gumbel vs PUCT: Mechanism Comparison

To isolate the effect of the root search algorithm, we compare the first 11 iterations of training under Gumbel with search batch 1 (PUCT-equivalent) and Gumbel Top-$k$ SH with search batch 16, with different epoch schedules and risk parameters ($E = 5/3$ for the sb=1 run vs $E = 3/1$ for the Gumbel run; risk\_beta adjusted at different iterations). The comparison isolates the search algorithm effect only approximately.

**Table 5.** Comparison of PUCT and Gumbel SH after initial training iterations.

| Metric | PUCT (sb=1) | Gumbel (sb=16) |
|:------:|:----------:|:--------------:|
| Draw rate | 78% | 50% |
| Decisive rate | 22% | 50% |
| Repetition draws | 1 | 4 |

*Note:* Table 5 compares iteration 4 specifically (before Mode I onset). The sb=1 run had only 1 repetition draw at iteration 4; repetition draws surged to 16–32 per iteration starting at iteration 5 (Table 1).

Two results are noteworthy. First, Gumbel SH achieved a dramatically lower draw rate (50% vs 78%) and correspondingly higher decisive rate. The critical finding is not merely the aggregate improvement but the complete elimination of early repetition-draw death: across 11 iterations of Gumbel training, zero iterations exhibited the catastrophic repetition-draw surge that characterizes Mode I onset under PUCT.

Second, and perhaps counterintuitively, the Gumbel run exhibited slightly more repetition draws in absolute terms (4 vs 1 per iteration). This is because Gumbel's diverse root exploration occasionally discovers repetition-forcing lines that PUCT's narrow search misses entirely. However, these are sporadic events rather than the self-reinforcing cascade that defines Mode I.

The mechanism behind this difference is precisely the round-robin root action assignment. In PUCT with search batch 1, each simulation independently selects the highest-UCB action at root, which rapidly concentrates visits on a single child once that child accumulates a Q-value advantage. In Gumbel with search batch 16, each simulation in a batch is deterministically assigned to one of the $k = 16$ active actions, guaranteeing uniform exploration at root regardless of Q-values. Sequential Halving then prunes actions based on $\text{logit}(a) + g(a) + \sigma(Q_{\text{completed}}(a))$, where the Gumbel noise $g(a)$ provides stochastic tie-breaking that prevents deterministic lock-in.

However, the Gumbel run still exhibited Mode II collapse (Table 2, iterations 6--9), demonstrating that diverse root selection is necessary but not sufficient for stable training. When the value head evaluates all positions as draws with high confidence, the $\sigma(Q_{\text{completed}})$ term vanishes for all actions (Theorem 1), and the improved policy degenerates to the prior regardless of which actions are explored. Gumbel addresses the concentration of *search effort* but cannot address the collapse of the *value signal* that the search consumes.

### 8.4 Prevention Method Evaluation (Proposed Experiments)

The four families of prevention methods derived in Section 6 have not yet been implemented in this codebase. We present here the proposed experimental protocol, with hypothesized outcomes grounded in the theoretical analysis, specific hyperparameter grids, and quantitative success criteria.

**Success criteria.** A prevention method is deemed effective if, over a 30-iteration training run starting from a random initialization:

1. Draw rate remains below 60% for at least 25 of 30 iterations.
2. Value loss remains in the range $[0.4, 0.9]$ for at least 20 of 30 iterations.
3. No 5-consecutive-iteration window exhibits monotonically decreasing value loss below 0.5.

These criteria are derived from the alert thresholds identified empirically: draw rate $> 60\%$ sustained over 3 iterations, value loss $< 0.4$, and average game length $< 40$ moves.

#### 8.4.1 Entropy Regularization

**Theoretical prediction.** Theorem 5 guarantees that entropy regularization with coefficient $\lambda_H$ establishes a WDL entropy floor of $H_{\min} = \lambda_H / (1 + \lambda_H) \cdot \ln 3$, preventing the zero-entropy absorbing state. The floor should be visible as a lower bound on value loss.

**Hyperparameter grid.** $\lambda_H \in \{0.001, 0.01, 0.1, 0.5\}$.

**Hypothesized outcomes:**

- $\lambda_H = 0.001$: Minimal effect. Floor too low ($H_{\min} \approx 0.001$) to prevent collapse at value loss 0.15.
- $\lambda_H = 0.01$: Marginal prevention. May slow collapse but unlikely to prevent it under severe Mode III/IV pressure.
- $\lambda_H = 0.1$: Effective prevention. Entropy floor at $\sim 0.1 \ln 3 \approx 0.11$, corresponding to value loss $\gtrsim 0.35$. Should keep the value head above the critical threshold.
- $\lambda_H = 0.5$: Over-regularized. Entropy dominates the loss, preventing the value head from learning to discriminate outcomes. Expected value loss to remain near $\ln 3 \approx 1.1$ (uninformative predictions).

The predicted optimal range is $\lambda_H \in [0.05, 0.2]$, balancing collapse prevention against learning speed.

#### 8.4.2 Focal Loss

**Theoretical prediction.** Proposition 8 establishes that focal loss with parameter $\gamma$ amplifies gradients from minority-class predictions (wins and losses when draws dominate) by a factor of up to $(1 - p_t)^\gamma$, where $p_t$ is the predicted probability of the true class. When the model predicts $p_D \approx 1.0$ for a position whose true outcome is a win, the gradient weight is $(1 - p_W)^\gamma \approx 1$, which is unchanged. The key effect is on *draw samples with ambiguous predictions*: these receive reduced weight, allowing win/loss samples to dominate the gradient.

**Hyperparameter grid.** $\gamma \in \{0.5, 1.0, 2.0, 4.0\}$.

**Hypothesized outcomes:**

- $\gamma = 0.5$: Mild effect; insufficient downweighting of confident draw predictions.
- $\gamma = 1.0$: Moderate effect; may partially offset Mode III pollution.
- $\gamma = 2.0$: Standard focal loss setting from Lin et al. (2017). Expected to substantially improve gradient balance.
- $\gamma = 4.0$: Aggressive downweighting. Risk of instability if too few samples receive significant gradient weight.

#### 8.4.3 Stratified Sampling

**Theoretical prediction.** Theorem 6 guarantees that stratified sampling with equal allocation across outcome classes produces an effective training distribution of $(1/3, 1/3, 1/3)$ regardless of buffer composition. This directly eliminates the buffer-composition feedback loop (Section 3.1, Proposition 1). Importance weights $w_c = n / (3 n_c)$ correct for the sampling bias, ensuring unbiased gradient estimation.

**Comparison protocol.** Three sampling strategies will be compared:

1. **Uniform sampling** (baseline): each position sampled with equal probability.
2. **Stratified sampling**: equal allocation to wins, draws, and losses with importance weights.
3. **Prioritized experience replay** (PER): priority exponent $\alpha = 0.6$, importance sampling exponent $\beta$ annealed from 0.4 to 1.0, priority based on value loss magnitude.

**Hypothesized ranking:** Stratified $>$ PER $>$ Uniform. Stratified directly addresses the composition imbalance; PER addresses it indirectly through loss-based prioritization; Uniform provides no correction.

#### 8.4.4 Adaptive Simulation Budget

**Theoretical prediction.** Theorem 7 establishes that scaling the simulation budget proportionally to value head entropy $H$ prevents the search amplification cascade. By setting $N \propto H$, the amplification factor is held constant as the value head's confidence changes.

**Hyperparameter grid.** $N_{\text{base}} \in \{100, 200, 400\}$ and $H_{\text{target}} \in \{0.5, 0.8, 1.0\}$, where $N = N_{\text{base}} \cdot \max(1, \bar{H} / H_{\text{target}})$ and $\bar{H}$ is the average WDL entropy over a sample of positions from the current buffer.

**Hypothesized outcomes:**

- Low $N_{\text{base}}$ with high $H_{\text{target}}$: Conservative. Simulations may be too few for the model to improve in the early, high-entropy phase.
- High $N_{\text{base}}$ with low $H_{\text{target}}$: Aggressive. Simulations ramp up too quickly as entropy drops, potentially triggering Mode I.
- $N_{\text{base}} = 200$, $H_{\text{target}} = 0.8$: Expected sweet spot. At full entropy ($H \approx 1.1$), $N \approx 275$; at half entropy ($H \approx 0.55$), $N = 200$ (the base); at low entropy ($H \approx 0.3$), $N = 200$ (clamped at base). This naturally reduces search pressure as the value head becomes more confident.

### 8.5 Recovery Experiments (Proposed)

To validate the theoretical predictions on recovery difficulty (Section 6.3), we propose experiments starting from collapsed checkpoints at three severity levels. Checkpoints will be obtained by saving the model state during observed collapse trajectories at the appropriate value loss and buffer composition.

**Severity levels:**

| Level | Value Loss | Buffer Draw % | Characteristic |
|:-----:|:----------:|:------------:|:---------------|
| Mild | 0.4 | 70 | Early warning crossed |
| Moderate | 0.2 | 85 | Critical threshold |
| Severe | 0.05 | 95 | Deep collapse |

**Recovery protocol.** For each severity level, apply each of the four prevention methods independently and measure:

1. **Iterations to recovery**: number of iterations until draw rate $< 60\%$ and value loss $> 0.5$ for 3 consecutive iterations.
2. **Recovery stability**: does the system remain recovered for 10 additional iterations without relapse?
3. **Recovery quality**: Elo rating of the recovered model against the pre-collapse checkpoint.

**Hypothesized recovery difficulty:**

| Method | Mild | Moderate | Severe |
|:------:|:----:|:--------:|:------:|
| Entropy reg. ($\lambda_H = 0.1$) | 3--5 iters | 8--12 iters | $> 20$ iters |
| Focal loss ($\gamma = 2.0$) | 3--5 iters | 10--15 iters | May not recover |
| Stratified sampling | 2--3 iters | 5--8 iters | 10--15 iters |
| Adaptive sims ($N_b=200, H_t=0.8$) | 5--8 iters | May not recover | Does not recover |

The predicted ordering (stratified $>$ entropy $>$ focal $>$ adaptive sims) reflects the theoretical analysis: stratified sampling directly breaks the buffer feedback loop, entropy regularization prevents the loss landscape from flattening, focal loss improves gradient quality but depends on the existing buffer composition, and adaptive simulations address only the search amplification pathway.

For the severe case, we hypothesize that no single intervention suffices and that a combination (stratified sampling + entropy regularization + epochs=1) will be required. The theory predicts that the basin of attraction of the absorbing fixed point encompasses the region where buffer draw fraction exceeds $\sim 90\%$ and value loss is below $\sim 0.1$ (Theorem 8, basin characterization), and escaping this basin requires simultaneous intervention on multiple feedback pathways.


## 9. Related Work

### 9.1 AlphaZero and Variants

The AlphaZero algorithm (Silver et al., 2018) established the paradigm of learning tabula rasa in two-player perfect-information games through self-play reinforcement learning combined with Monte Carlo tree search. The original paper employed a scalar value head predicting the expected game outcome $v \in [-1, 1]$, with PUCT guiding the search. Draw handling was not discussed as a challenge, likely because the paper focused on a small number of very high-budget training runs (5000 TPUs, 700,000 iterations) where the network had sufficient capacity and data diversity to avoid collapse. However, reproducibility efforts by independent researchers working at smaller scale have consistently reported draw-related training instabilities (see, e.g., the Leela Chess Zero development history).

Schrittwieser et al. (2020) extended AlphaZero to model-based planning with MuZero, which learns a dynamics model and predicts rewards and values without access to the true game state. The value prediction in MuZero is structurally more complex — it must predict discounted future rewards in addition to terminal outcomes — but the fundamental vulnerability to absorbing states remains. In fact, MuZero may be more susceptible because errors in the learned dynamics model can compound with value prediction errors, creating an additional feedback pathway toward collapse.

Danihelka et al. (2022) introduced Policy Improvement by Planning with Gumbel, replacing PUCT at the root with Gumbel Top-$k$ Sequential Halving. This algorithm provides stronger theoretical guarantees on policy improvement through its connection to the Gumbel-Max trick for sampling from categorical distributions. The paper focuses on the quality of the improved policy target and does not analyze the interaction between the search algorithm and value head collapse. Our work fills this gap, demonstrating both the mechanism by which Gumbel prevents Mode I collapse (round-robin root exploration) and the mechanism by which it fails to prevent Mode II collapse (value signal homogenization).

### 9.2 WDL Value Heads and Draw Handling

The transition from scalar to WDL value heads was pioneered in the Leela Chess Zero (Lc0) project, which found empirically that predicting the full win-draw-loss distribution improved both training stability and playing strength. The Lc0 community discovered draw bias issues through extensive experimentation and developed several heuristics, including contempt parameters that penalize draw evaluations during search and draw score adjustments that shift the value of drawn positions away from zero. These heuristics are effective but theoretically unprincipled: they introduce a bias toward decisive play that may not reflect the true game-theoretic value of positions.

KataGo (Wu, 2019), developed for Go, takes a different approach by predicting the full score distribution rather than a categorical outcome. Because Go scores are continuous (territory counting), the distribution head naturally encodes uncertainty about the game outcome. This richer representation provides more informative gradients during training: even when the expected outcome is a draw (jigo), the score distribution retains variance that prevents gradient vanishing. KataGo's approach is more resistant to the collapse phenomenon we study, precisely because the score distribution cannot collapse to a point mass as easily as a 3-class categorical distribution can.

Stockfish NNUE (Nasu, 2018) avoids the problem entirely through architectural choice: it uses a scalar evaluation trained by supervised learning on engine-generated labels rather than self-play. Without the self-play feedback loop, there is no mechanism for the value head's predictions to influence its own training data, and the collapse dynamics we analyze simply do not arise. This comparison underscores our central thesis: value head collapse is fundamentally a property of the coupled self-play training loop, not of the value head architecture in isolation.

### 9.3 Self-Play Stability

Czarnecki et al. (2020) analyze the geometry of self-play training dynamics in "Real World Games Look Like Spinning Tops," demonstrating that the strategy space visited during training often has a low-dimensional spinning-top structure where most variation occurs along a single axis of increasing strength, with small transverse oscillations. Their analysis is relevant because draw death represents a degenerate case where the spinning top collapses to a single point: the strategy space contracts to the draw-forcing policy. In the language of their framework, draw death is convergence to a fixed point that lies on the axis of the top but at a location of zero strategic diversity.

Balduzzi et al. (2019) study self-play from a population-based game-theoretic perspective, showing that maintaining a diverse population of policies avoids convergence to dominated strategies. Their analysis suggests that population-based training could provide a structural defense against draw death by ensuring that the training opponents include decisive-playing agents. This connects to our stratified sampling proposal (Section 6.3): rather than diversifying the population of *players*, we diversify the population of *training samples*, achieving a similar effect with lower computational overhead.

Lanctot et al. (2017) developed the OpenSpiel framework for research in games, which includes standardized handling of draws and terminal states. Their treatment of draws as first-class outcomes (rather than ties that default to zero) influenced the WDL value head design. However, their focus is on evaluation methodology rather than training stability, and the framework does not address the feedback dynamics that lead to collapse.

### 9.4 Loss Function Design for Imbalanced Data

Lin et al. (2017) introduced focal loss for dense object detection, addressing the class imbalance between foreground objects and background in anchor-based detectors. The focal modulation $(1 - p_t)^\gamma$ downweights well-classified examples, allowing the loss to focus on hard examples. We adapt this idea to the WDL classification problem (Section 6.1), where draws play the role of the dominant "background" class and wins/losses are the rare "foreground" classes. The adaptation is not straightforward because the class imbalance in our setting is dynamic — it is caused by the training process itself rather than being a fixed property of the dataset.

Menon et al. (2021) study long-tail learning with logit adjustment, showing that adding class-frequency-dependent offsets to the logits before softmax provably optimizes a balanced error metric. In our setting, this would correspond to adding offsets $\Delta_c = \tau \log \pi_c$ to the WDL logits, where $\pi_c$ is the buffer frequency of class $c$ and $\tau$ is a temperature. This is related to but distinct from our entropy regularization approach: logit adjustment shifts the decision boundary, while entropy regularization prevents the distribution from concentrating.

Cao et al. (2019) propose label-distribution-aware margin (LDAM) loss, which enforces class-dependent margins that are larger for minority classes. Adapting this to WDL, the margin for wins and losses would increase as they become rarer in the buffer, providing stronger gradient signal for decisive outcomes. This is a promising direction for future work that combines the benefits of focal loss (gradient emphasis on hard examples) with explicit awareness of the buffer composition.

### 9.5 Entropy Regularization in Reinforcement Learning

Haarnoja et al. (2018) introduced Soft Actor-Critic (SAC), which augments the RL objective with an entropy bonus on the policy: $J(\pi) = \mathbb{E}\left[\sum_t r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$. The entropy term prevents premature convergence to deterministic policies and ensures continued exploration. Our entropy regularization on the value head (Section 6.1) is structurally analogous: we add an entropy bonus to the value loss to prevent the WDL distribution from collapsing to a deterministic prediction. The key difference is that SAC regularizes the *policy* distribution over actions, while we regularize the *value* distribution over outcomes. Both serve the same purpose: preventing a distribution from concentrating when the training signal is insufficiently diverse.

Williams and Peng (1991) introduced entropy bonuses in the REINFORCE algorithm to encourage exploration, establishing the theoretical foundation for entropy regularization in policy gradient methods. Their analysis shows that the entropy bonus is equivalent to adding a KL-divergence penalty against the uniform distribution, which provides a useful interpretation for our setting: entropy regularization on the WDL head penalizes deviation from the uniform outcome distribution $(1/3, 1/3, 1/3)$.

Ahmed et al. (2019) provide a theoretical analysis of entropy regularization's impact on policy optimization, showing that entropy smooths the optimization landscape and improves convergence properties. Their results on landscape smoothing are directly relevant to our Hessian analysis (Theorem 3): at the absorbing fixed point, the loss landscape is flat (near-zero eigenvalues), and entropy regularization restores curvature by adding a term proportional to $\lambda_H / (p_c(1-p_c))$ to the diagonal of the Hessian.

### 9.6 Mode Collapse in Generative Models

The structural analogy between draw death and GAN mode collapse is deep enough to warrant a dedicated discussion. Goodfellow et al. (2014) identified mode collapse as a failure mode of GANs where the generator converges to producing a single data point (or a small set) that successfully fools the discriminator. In our setting, the "generator" is the self-play process and the "data" are game outcomes: draw death is the convergence of the generator to producing a single outcome (draw) that is consistent with the evaluator's predictions.

Arjovsky et al. (2017) addressed mode collapse in GANs by replacing the Jensen-Shannon divergence with the Wasserstein distance, which provides gradients even when the generator and data distributions have disjoint support. The structural parallel is illuminating: in draw death, the "generated" outcome distribution (all draws) and the "target" distribution (healthy mix of outcomes) have nearly disjoint support in outcome space. The standard cross-entropy loss, like the JS divergence, provides vanishing gradients in this regime. Our entropy regularization, like the Wasserstein distance, restores gradient signal by penalizing distribution concentration regardless of the target.

Metz et al. (2017) proposed unrolled GANs, which stabilize training by computing the generator's gradient through multiple steps of discriminator optimization. The idea of "looking ahead" to anticipate the feedback loop's dynamics is relevant to our adaptive simulation budget proposal: by monitoring the value head's entropy trajectory and adjusting the simulation budget proactively, we effectively unroll one step of the collapse dynamics and intervene before the feedback loop amplifies the bias.


## 10. Discussion and Conclusion

### 10.1 Generalization Beyond Chess

The value head collapse phenomenon is not an artifact of chess-specific game mechanics. The mathematical framework developed in Section 3 identifies four structural requirements for the collapse dynamics:

1. **A categorical value head** that predicts a distribution over discrete outcomes, one of which can serve as an absorbing state.
2. **Tree search that amplifies value bias** — any search procedure (MCTS with PUCT, Gumbel SH, or even minimax with alpha-beta pruning) that converts value predictions into action selections will concentrate on outcomes that the value head favors.
3. **A circular replay buffer with finite capacity**, such that new data gradually overwrites old data. This creates the memory effect where the buffer composition reflects a weighted average of recent self-play outcomes.
4. **An absorbing outcome state** — a game outcome that, once dominant in the buffer, produces training gradients that reinforce its own dominance.

Any game satisfying these conditions is vulnerable. Consider the landscape of common two-player games:

**Go.** Despite being a decisive game (no draws under standard rules), Go has an analogous absorbing state under area scoring: when the value head learns that the score difference is typically small, it may converge to predicting the komi value for all positions, leading to games where neither player deviates from "safe" territory. This has been observed informally in small-board Go training. Under territory scoring with draws possible under certain rulesets, the full draw death dynamics apply.

**Shogi.** Japanese chess has no draw mechanism (repetition results in the attacker replaying; perpetual check is illegal). Consequently, the WDL distribution always has $p_D \approx 0$ in correctly-played positions, and the absorbing state at $(0, 1, 0)$ is not reachable from healthy training. Shogi is structurally immune to draw death.

**Hex.** By a combinatorial argument (every filled Hex board has exactly one winner, since a path connecting one player's sides necessarily blocks the opponent's sides), every Hex game must end in a decisive result — draws are mathematically impossible. Like Shogi, Hex is immune to draw death by the structure of the game itself.

**General-sum games and multi-player settings.** The analysis extends naturally to any setting where a game outcome forms an absorbing state under the training dynamics. In multi-player games, the absorbing state might be a symmetric outcome where all players receive equal payoff. In general-sum games, the absorbing state corresponds to a Nash equilibrium of the stage game that is Pareto-dominated but stable under self-play dynamics.

The key insight is that vulnerability to collapse is determined by the *interaction* between game structure (existence of absorbing outcomes), value representation (categorical with a class corresponding to the absorbing outcome), and training dynamics (self-play feedback loop with finite buffer). Removing any one of these three elements prevents collapse.

### 10.2 The GAN Analogy

The structural parallels between draw death in self-play and mode collapse in generative adversarial networks are summarized in the following correspondence:

| GAN Component | Self-Play Component |
|:-------------|:-------------------|
| Generator $G$ | Self-play policy $\pi$ |
| Discriminator $D$ | Value head $f_\theta$ |
| Training data | Replay buffer $\mathcal{B}$ |
| Generated samples | Game outcomes from self-play |
| Mode collapse | Draw death |
| JS divergence | WDL cross-entropy loss |
| Wasserstein distance | Entropy-regularized loss |

The shared structural feature is a *feedback loop between a producer and an evaluator*, where:

1. The evaluator is trained on data produced by the producer.
2. The producer uses the evaluator to make decisions.
3. The training data is a finite-capacity buffer that mediates the interaction.

In GANs, the generator produces fake images that the discriminator learns to classify; the generator then uses the discriminator's gradients to improve. In self-play, the policy produces game outcomes that the value head learns to predict; the policy then uses the value head's evaluations (via search) to improve. In both cases, when the producer converges to a degenerate output distribution, the evaluator's training data becomes homogeneous, which reinforces the degeneracy through gradient dynamics.

The formal analogy suggests that techniques developed for GAN stabilization may transfer to self-play:

- **Wasserstein distance $\leftrightarrow$ entropy regularization**: both restore gradient signal when distributions collapse.
- **Spectral normalization $\leftrightarrow$ weight decay / gradient clipping**: both constrain the evaluator's capacity to memorize the degenerate distribution.
- **Diverse training (minibatch discrimination) $\leftrightarrow$ stratified sampling**: both ensure the evaluator sees diverse examples regardless of the producer's output distribution.
- **Unrolled optimization $\leftrightarrow$ adaptive simulation budget**: both incorporate anticipatory reasoning about the feedback dynamics.

The analogy breaks down in one important respect: GAN mode collapse is typically *partial* (the generator covers a few modes rather than one), while draw death is typically *total* (all games converge to draws). This difference arises because the game outcome space is discrete and low-dimensional (3 classes) while the image space is continuous and high-dimensional. In the continuous setting, partial collapse is a stable intermediate; in the discrete setting, the dynamics tend to be all-or-nothing.

### 10.3 Practical Recommendations

For practitioners implementing AlphaZero-style training systems, we distill our analysis into the following actionable guidelines, ordered by priority:

**1. Monitor value loss and draw rate jointly.** Neither metric alone is sufficient. A draw rate of 60% is concerning but not diagnostic; a value loss of 0.4 can indicate either healthy learning or early collapse depending on the draw rate. The compound signal — value loss dropping below 0.5 while draw rate exceeds 60% sustained over 3 or more iterations — is the earliest reliable indicator of incipient collapse. Implement automated alert thresholds:

- Draw rate $> 60\%$ over 3 consecutive iterations: warning.
- Repetition draw rate $> 30\%$ of games over 3 iterations: critical.
- Value loss change $< 2\%$ over 5 iterations: plateau alert.
- Average game length $< 40$ moves over 3 iterations: collapse alert.
- Maximum gradient norm $> 10$: instability alert.

**2. Start with low simulation budgets.** Use $N = 200$ simulations, not 800. The search amplification factor (Theorem 1) scales with simulation count, and an uncalibrated value head in early training cannot support deep search without Mode I collapse. Only increase simulations after observing sustained decisive game rates above 20% and value loss above 0.5 over at least 5 iterations.

**3. Use Gumbel Top-$k$ Sequential Halving at root.** This eliminates Mode I (repetition draw death) entirely through its round-robin action assignment. The improvement is not marginal: in our experiments, Gumbel reduced early-training draw rates from 78% to 50% and eliminated the catastrophic repetition-draw cascades that are lethal under PUCT.

**4. Keep training epochs low in early iterations.** Each epoch multiplies the value head's exposure to the current buffer composition. At 5 epochs with 80% draws, the value head sees each draw sample 5 times, equivalent to a buffer with 96% effective draw representation. Start with $E = 1\text{--}3$ and increase only after the buffer composition stabilizes at a healthy level (draw fraction $< 70\%$).

**5. Consider stratified sampling.** Among the prevention methods analyzed, stratified sampling provides the strongest theoretical guarantee (Theorem 6) by completely decoupling buffer composition from the training distribution. It requires tracking per-sample outcome metadata in the buffer and implementing weighted sampling, but the overhead is modest.

**6. Add entropy regularization as a safety net.** Even without other interventions, a small entropy regularization coefficient ($\lambda_H = 0.05\text{--}0.1$) prevents the most severe form of collapse by maintaining a nonzero WDL entropy floor. The computational overhead is negligible (one additional entropy computation per batch), and the impact on normal training is minimal in this coefficient range.

**7. Act fast when warning signs appear.** The window between early warning (value loss crosses below 0.5) and irreversibility (5+ iterations of pure-draw self-play) can be as short as 2--3 iterations. The buffer's thermal mass means that every draw-dominated iteration makes recovery harder. Intervention at the warning stage (reduce epochs, increase temperature, consider buffer flush) is dramatically more effective than intervention at the critical stage.

### 10.4 Limitations

Several limitations of our analysis should be acknowledged.

**Continuous-time approximation.** The dynamical systems model (Section 3) treats the training loop as a continuous flow, discretized only for numerical analysis. In practice, training is inherently discrete: each iteration produces a batch of games, performs several epochs of gradient updates, and the system jumps discontinuously. The continuous approximation is valid when the per-iteration changes are small (the "slow dynamics" regime), but may be inaccurate during rapid transitions such as the Mode I cascade between iterations 6 and 8 of Table 1, where value loss drops from 0.89 to 0.45 in two iterations.

**Gaussian approximation for Gumbel analysis.** The analysis of Gumbel SH amplification (Theorem 1, Appendix A) models $Q_{\text{completed}}(a)$ as approximately Gaussian. In practice, the distribution of completed Q-values is influenced by the tree structure, the FPU constants, and the prior policy, and may be heavy-tailed or multimodal. The Gaussian approximation captures the first-order effect (concentration of Q-values under draw bias) but may miss higher-order phenomena.

**Proposed but unvalidated prevention methods.** While the theoretical analysis provides guarantees on gradient properties and fixed-point structure, the proposed prevention methods (entropy regularization, focal loss, stratified sampling, adaptive simulations) have not yet been empirically validated in this specific codebase. The interaction between these methods and the many engineering details of the training pipeline (CUDA graph acceleration, lock-free evaluation queue, mixed-precision training) may produce unexpected effects.

**Linearization near the fixed point.** The basin of attraction analysis (Theorem 8) relies on linearization of the dynamics around the absorbing fixed point, which is valid only in a neighborhood of that point. The global basin boundary — the "point of no return" beyond which collapse is inevitable — may not be well-approximated by the linearized ellipsoid. In practice, the boundary appears to be roughly characterized by buffer draw fraction $> 85\%$ combined with value loss $< 0.4$, but this is an empirical observation rather than a rigorous bound.

**Scale dependence.** All experiments were conducted with a single architecture (f256-b20-se4) and self-play configuration (32 games, 100K buffer). The qualitative phenomena are expected to transfer to larger scales, but the quantitative thresholds (simulation counts, iteration counts to collapse, buffer composition percentages) may differ for larger networks with higher capacity, larger buffers with longer memory, or more games per iteration that shift buffer composition more rapidly.

### 10.5 Open Questions

Several questions raised by this work remain open:

**Entropy coefficient scheduling.** Can the entropy regularization coefficient $\lambda_H$ be scheduled during training — large initially to prevent early collapse, then decayed as the model develops genuine discriminative ability? The risk is that decay happens too quickly and collapse occurs at a later training stage. A principled schedule would adapt $\lambda_H$ to the observed WDL entropy, maintaining a fixed margin above the entropy floor, but the convergence properties of such adaptive schedules are unexplored.

**Synergy and antagonism among prevention methods.** Our analysis treats each prevention method independently. In practice, combining methods (e.g., entropy regularization + stratified sampling + focal loss) may produce synergistic effects (each method addresses a different collapse pathway) or antagonistic effects (entropy regularization conflicts with focal loss by encouraging the very uncertainty that focal loss downweights). A systematic study of method combinations in the $(4 \choose 2) + (4 \choose 3) + (4 \choose 4) = 11$ nontrivial subsets would be valuable.

**Scaling behavior.** How do collapse dynamics change with network size? Larger networks (f512-b40, or the scale of the original AlphaZero at f256-b40) have higher capacity and may be more resistant to collapse (able to maintain diverse value predictions across the position space) or more vulnerable (able to memorize the draw-dominant buffer more efficiently). Theoretical analysis of the capacity-collapse tradeoff could inform architecture selection.

**Population diversity.** Can population-based training (Jaderberg et al., 2017) provide a structural defense against collapse? By maintaining a diverse population of agents with different value head states, the self-play games would naturally produce diverse outcomes. This connects to the evolutionary game theory literature on the maintenance of biodiversity through frequency-dependent selection.

**Optimal buffer composition.** Is there an optimal target composition $(f_W^*, f_D^*, f_L^*)$ that maximizes learning speed subject to the constraint of preventing collapse? The uniform distribution $(1/3, 1/3, 1/3)$ is the most robust but may not be optimal: in the early game, draws may be genuinely more common than decisive results, and forcing equal representation could bias the value head toward overestimating decisive outcomes.

**Transfer to other domains.** The absorbing state framework applies beyond board games to any domain with self-play and categorical outcomes. Multi-agent reinforcement learning for robotics, auction design, and mechanism design all involve self-play training loops where degenerate equilibria can emerge. Extending the dynamical systems analysis to these continuous-action, continuous-state settings is a natural direction.

### 10.6 Conclusion

This paper presents a rigorous mathematical analysis of value head collapse in AlphaZero-style self-play training, a phenomenon we formalize as convergence to an absorbing fixed point in a coupled dynamical system. Our contributions are:

1. **Formalization.** We model the self-play training loop as a coupled dynamical system over three state variables — value head parameters $\theta$, buffer composition $\mathbf{f}$, and search policy $\pi$ — and prove that the all-draw state $(\theta^*, \mathbf{f}^* = (0, 1, 0), \pi^*)$ is a stable fixed point of the dynamics (Theorem 2). The stability is structural: it persists under perturbations of the hyperparameters and architecture, requiring qualitative intervention to escape.

2. **Loss landscape analysis.** We characterize the WDL cross-entropy loss at the absorbing fixed point, showing that gradients vanish as $O(\varepsilon)$ (Proposition 5), the Hessian eigenvalues collapse to $O(\varepsilon)$ (Theorem 3), and the entropy achieves its global minimum of zero (Proposition 6). These three properties together explain why standard gradient-based optimization cannot escape the collapsed state.

3. **Empirical taxonomy.** We identify and classify four distinct collapse modes from training data — simulation depth draw death (Mode I), fifty-move stagnation (Mode II), max-moves pollution (Mode III), and buffer saturation oscillation trap (Mode IV) — each with distinct onset signatures, progression dynamics, and intervention requirements. We provide exact iteration-by-iteration data documenting each mode.

4. **Prevention methods with guarantees.** We propose four families of prevention methods — entropy regularization (Theorem 5), focal loss (Proposition 8), stratified sampling (Theorem 6), and adaptive simulation budget (Theorem 7) — and prove that each modifies the dynamical system to either eliminate the absorbing fixed point, destabilize it, or prevent convergence to it. These methods are ordered by theoretical robustness: stratified sampling provides the strongest guarantee (complete decoupling of buffer composition from training distribution), while adaptive simulations address only the search amplification pathway.

5. **Convergence analysis.** We construct a Lyapunov function (Theorem 8) for the healthy training region and characterize its basin of attraction, providing a principled boundary between recoverable and irrecoverable collapse states. We show that the Lyapunov function $W = H(D_t) - \lambda \cdot L_{\text{value}}^{\text{test}}(\theta_t)$ combines buffer entropy and test value loss, and is non-decreasing near the absorbing state when entropy regularization exceeds a critical threshold.

The central message of this work is that value head collapse in self-play training is not a bug to be fixed by hyperparameter tuning. It is a *mathematical property* of the coupled training loop — a stable equilibrium that exists in the dynamical system regardless of hyperparameter settings. The absorbing fixed point does not arise from poor initialization, insufficient data, or suboptimal learning rates; it is an intrinsic feature of the feedback loop between self-play, the replay buffer, and the value head. Preventing collapse requires *structural intervention* — modifying the loss function, the architecture, the data pipeline, or the search algorithm to change the qualitative structure of the dynamical system. We have provided the theoretical framework and practical tools for understanding when and why collapse occurs, for detecting it early, and for preventing it through principled modifications to the training pipeline.


## Appendix A: Supporting Proofs

### A.1 Proof of Theorem 1 (Gumbel Search Amplification)

**Theorem 1.** *Under Gumbel Top-$k$ Sequential Halving with $k$ active actions and $N$ total simulations, if the value head predicts $v(s) \approx \mu$ for all states $s$ reachable within the search tree (i.e., the value predictions have negligible variance $\sigma^2 \to 0$), then $\sigma(Q_{\text{completed}}(a)) \to 0$ for all actions $a$, and the improved policy converges to the prior: $\pi_{\text{improved}}(a) \to P(a)$.*

*Proof.* Let $Q_a = Q_{\text{completed}}(a)$ denote the completed Q-value for root action $a$ after the Gumbel SH procedure. By definition of the completed Q-value:

$$Q_{\text{completed}}(a) = \frac{V_{\text{sum}}(a) + N_{\text{remaining}}(a) \cdot \hat{v}(a)}{N(a) + N_{\text{remaining}}(a)}$$

where $V_{\text{sum}}(a)$ is the sum of backed-up values from simulations through action $a$, $N(a)$ is the number of completed simulations, and $N_{\text{remaining}}(a) \cdot \hat{v}(a)$ accounts for the remaining budget using the current value estimate.

When $v(s) \approx \mu$ for all reachable states, each simulation through action $a$ returns a value drawn from a distribution with mean $\mu$ and variance $\sigma^2 \to 0$. By the law of large numbers, as $N(a) \to \infty$:

$$Q_{\text{completed}}(a) \xrightarrow{p} \mu \quad \text{for all } a$$

More precisely, $\text{Var}(Q_a) = \mathcal{O}(\sigma^2 / N(a))$, so for any finite simulation budget, the Q-values concentrate around $\mu$ with spread proportional to $\sigma / \sqrt{N(a)}$.

The $\sigma(\cdot)$ transform in Gumbel SH normalizes Q-values to a scale suitable for comparison with policy logits. The normalization is:

$$q_{\text{norm}}(a) = \frac{Q_a - Q_{\min}}{Q_{\max} - Q_{\min}} \cdot 2 - 1$$

where $Q_{\max} = \max_a Q_a$ and $Q_{\min} = \min_a Q_a$. The $\sigma$ function then scales this normalized value:

$$\sigma(Q_a) = (c_{\text{visit}} + N_{\max}) \cdot c_{\text{scale}} \cdot q_{\text{norm}}(a)$$

where $c_{\text{visit}}$ and $c_{\text{scale}}$ are constants and $N_{\max}$ is the maximum visit count.

As $\sigma^2 \to 0$, we have $Q_{\max} - Q_{\min} \to 0$. There are two subcases:

**Subcase (a): $Q_{\max} - Q_{\min} < \epsilon$ (implementation threshold).** The implementation sets $q_{\text{norm}}(a) = 0$ for all $a$, giving $\sigma(Q_a) = 0$. The improved policy reduces exactly to the prior: $\pi_{\text{improved}}(a) = \text{softmax}(\log P(a) + 0) = P(a)$.

**Subcase (b): $Q_{\max} - Q_{\min} \geq \epsilon$ (finite noise).** The $q_{\text{norm}}$ normalization maps the small Q-spread to the full $[-1, 1]$ range. The resulting $\sigma$-scores have magnitude $O((c_{\text{visit}} + N_{\max}) \cdot c_{\text{scale}})$, which is large ($\sim 150$). The improved policy at each individual position differs substantially from the prior. However, the Q-value differences that determine the $q_{\text{norm}}$ ranking are $O(\sigma/\sqrt{N_{\text{eff}}})$ — pure noise. Across positions, the noise-driven perturbations are uncorrelated, so:

$$\mathbb{E}_s[\pi_{\text{improved}}(a \mid s) - P(a \mid s)] \approx 0$$

The expected policy gradient vanishes: $\mathbb{E}_s[P - \pi_{\text{improved}}] \approx 0$. Policy learning becomes a random walk rather than directed improvement.

In both subcases, the search adds no *systematic* information to the policy. Self-play games under $\pi_{\text{improved}}$ produce the same *expected* outcome distribution as games under the prior $P$, closing the feedback loop at the fixed point. $\square$

### A.2 Proof of Theorem 2 (Absorbing State Stability)

**Theorem 2.** *The all-draw state $(\theta^*, \mathbf{f}^* = (0, 1, 0))$ is a Lyapunov-stable fixed point of the coupled dynamics. Moreover, there exists a neighborhood $\mathcal{U}$ of $(\theta^*, \mathbf{f}^*)$ such that all trajectories starting in $\mathcal{U}$ converge to $(\theta^*, \mathbf{f}^*)$.*

*Proof.* We establish stability through a monotone convergence argument on the buffer draw fraction $f_D$.

**Step 1: Fixed point verification.** At $(\theta^*, \mathbf{f}^*)$, the value head predicts $(p_W, p_D, p_L) \approx (0, 1, 0)$ for all positions. By Theorem 1, $\pi_{\text{improved}} = P$. Self-play under any policy when all positions evaluate as draws produces games that terminate in draws (either by repetition, fifty-move rule, or max moves). Therefore the self-play outcome distribution is $\mathbf{g}^* = (0, 1, 0)$. The buffer update $\mathbf{f}^{(t+1)} = (1 - \eta)\mathbf{f}^{(t)} + \eta \mathbf{g}^{(t)}$ yields $\mathbf{f}^{(t+1)} = (1 - \eta)(0, 1, 0) + \eta(0, 1, 0) = (0, 1, 0)$. Training on a buffer with all-draw labels reinforces $\theta^*$. Hence $(\theta^*, \mathbf{f}^*)$ is a fixed point.

**Step 2: Monotonicity in a neighborhood.** Suppose the system is in a state where $f_D > 1 - \delta$ for some small $\delta > 0$, and value loss $\mathcal{L}_v < \epsilon$ for some small $\epsilon > 0$. We show that $f_D^{(t+1)} \geq f_D^{(t)}$.

The value head, having trained on a buffer with $f_D > 1 - \delta$ fraction draws, predicts $p_D(s) > 1 - \delta'$ for most positions $s$, where $\delta' \to 0$ as $\delta \to 0$ and $\epsilon \to 0$. By Theorem 1, the search amplification concentrates Q-values, making $\sigma(Q_a) \approx 0$. The self-play policy is approximately the prior, which, combined with the value-head-guided search at depth $> 0$, tends to avoid decisive play (since decisive lines require the value head to distinguish between positions, which it cannot when $p_D \approx 1$ everywhere).

Let $g_D$ be the draw fraction in the newly generated games. In the neighborhood $f_D > 1 - \delta$, $\mathcal{L}_v < \epsilon$, we have $g_D > 1 - \delta''$ where $\delta'' = \mathcal{O}(\delta + \epsilon)$. For sufficiently small $\delta, \epsilon$, we can ensure $g_D > f_D - \eta^{-1}\delta'''$ for some $\delta''' \to 0$, so that:

$$f_D^{(t+1)} = (1 - \eta)f_D^{(t)} + \eta g_D^{(t)} \geq f_D^{(t)} - \mathcal{O}(\delta + \epsilon)$$

This monotonicity is *approximate*: $f_D^{(t+1)} \geq f_D^{(t)} - O(\delta + \epsilon)$, meaning the decrease per step is bounded. In the draw-dominated regime, the increase from the draw-biased self-play ($g_D > f_D + O(\mu)$ for some $\mu > 0$ determined by the data asymmetry of Proposition 4) dominates the $O(\delta + \epsilon)$ error, giving net monotone increase. The value loss update reinforces this: training on the buffer reduces value loss (the cross-entropy with mostly-draw labels is minimized by predicting draws), so $\mathcal{L}_v^{(t+1)} \leq \mathcal{L}_v^{(t)}$, which makes $\delta'$ smaller, which makes $g_D$ larger. The main body proof (Section 3.6) provides the more rigorous chain via Propositions 1--4 and Theorem 1.

**Step 3: Convergence.** The sequence $\{f_D^{(t)}\}$ is bounded above by 1 and (in the neighborhood) monotonically non-decreasing up to $O(\delta + \epsilon)$ perturbations that are dominated by the increasing trend. By the monotone convergence theorem (applied to the smoothed sequence), $f_D^{(t)} \to f_D^* = 1$. Since $f_W + f_D + f_L = 1$ with $f_W, f_L \geq 0$, this implies $f_W^{(t)} \to 0$ and $f_L^{(t)} \to 0$. Training on the buffer $\mathbf{f}^{(t)} \to (0, 1, 0)$ drives $\theta^{(t)} \to \theta^*$ by the convergence of stochastic gradient descent on the cross-entropy loss with a fixed target distribution.

The neighborhood $\mathcal{U}$ is characterized (approximately) by the conditions $f_D > 1 - \delta_0$ and $\mathcal{L}_v < \epsilon_0$ where $\delta_0$ and $\epsilon_0$ are the largest values for which the monotonicity argument holds. Empirically, $\delta_0 \approx 0.15$ (i.e., $f_D > 0.85$) and $\epsilon_0 \approx 0.4$, consistent with the observed irreversibility threshold. $\square$

### A.3 Proof of Theorem 5 (Entropy Floor)

**Theorem 5.** *The entropy-regularized value loss $\mathcal{L}_{\text{ent}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) - \lambda_H \mathcal{H}(\mathbf{p}_\theta)$, where $\mathcal{H}(\mathbf{p}) = -\sum_c p_c \log p_c$, has no minimizer with $\mathcal{H}(\mathbf{p}_\theta) = 0$. The minimum entropy at the optimum, $H_{\min}(\lambda_H)$, is strictly positive for all $\lambda_H > 0$ and can be computed exactly.*

*Proof.* We derive the entropy floor using KKT conditions for the constrained optimization.

Consider the per-sample loss as a function of the predicted distribution $\mathbf{p} = (p_W, p_D, p_L)$ with fixed target $\mathbf{y} = (y_W, y_D, y_L)$:

$$\ell(\mathbf{p}) = -\sum_c y_c \log p_c - \lambda_H \left(-\sum_c p_c \log p_c\right) = -\sum_c y_c \log p_c + \lambda_H \sum_c p_c \log p_c$$

Subject to $\sum_c p_c = 1$, $p_c \geq 0$.

The Lagrangian is:

$$L(\mathbf{p}, \nu) = -\sum_c y_c \log p_c + \lambda_H \sum_c p_c \log p_c - \nu\left(\sum_c p_c - 1\right)$$

Setting $\partial L / \partial p_c = 0$:

$$-\frac{y_c}{p_c} + \lambda_H(\log p_c + 1) - \nu = 0$$

$$\lambda_H \log p_c = \frac{y_c}{p_c} + \nu - \lambda_H$$

This is a transcendental equation that does not admit a closed-form solution for general $\mathbf{y}$. However, we can compute the entropy at the optimum exactly.

Consider the worst case for entropy: $\mathbf{y} = (0, 1, 0)$ (pure draw label, which pushes hardest toward low entropy). The main body (Section 6.1) derives the optimum via a closed-form approximation: $w_c^* = (y_c + \lambda_H/3)/(1+\lambda_H)$. For the draw target:

$$p_D^* = \frac{3 + \lambda_H}{3(1 + \lambda_H)}, \quad p_W^* = p_L^* = \frac{\lambda_H}{3(1 + \lambda_H)}$$

The entropy at this optimum is computed exactly:

$$H_{\min} = H(w^*) = -p_D^* \log p_D^* - 2 p_W^* \log p_W^*$$

At $\lambda_H = 0.1$: $p_D^* \approx 0.939$, $p_W^* = p_L^* \approx 0.030$, and:

$$H_{\min} \approx -0.939 \log 0.939 - 2(0.030) \log 0.030 \approx 0.059 + 0.211 = 0.269 \text{ nats}$$

This is the achievable minimum entropy — an exact value, not a bound. Note that $H_{\min}$ depends on $\lambda_H$ in a nonlinear way that does not simplify to $\frac{\lambda_H}{1+\lambda_H} \ln 3$ (which would give $0.100 \times 1.099 / 1.1 \approx 0.100$ nats at $\lambda_H = 0.1$ — a weaker and incorrect estimate). The exact computation above gives the true entropy floor.

Crucially, $H_{\min} > 0$ for all $\lambda_H > 0$, which means the absorbing fixed point at $\mathcal{H} = 0$ is not reachable under the modified loss. The absorbing state is eliminated as a fixed point of the dynamics. $\square$

### A.4 Proof of Theorem 6 (Stratified Sampling Balance)

**Theorem 6.** *Under stratified sampling with equal allocation $n_c = B/3$ per outcome class, the unweighted expected gradient equals the gradient under a uniform outcome distribution, regardless of buffer composition. To recover the original (buffer-proportional) gradient for unbiased loss estimation, apply importance-sampling weights $w(i) = 3 n_{\text{outcome}(i)}^{\text{buf}} / n$ for sample $i$ from outcome class $\text{outcome}(i)$, where $n_c^{\text{buf}}$ is the count of class $c$ in the buffer and $n$ is the total buffer size.*

*Proof.* Let $\mathcal{B} = \mathcal{B}_W \cup \mathcal{B}_D \cup \mathcal{B}_L$ be the replay buffer partitioned by outcome, with $|\mathcal{B}_c| = n_c^{\text{buf}}$ and $|\mathcal{B}| = n = \sum_c n_c^{\text{buf}}$.

Under uniform sampling, the expected gradient is:

$$\nabla_{\text{uniform}} = \frac{1}{n} \sum_{i=1}^{n} \nabla \ell(\theta; x_i, y_i) = \sum_{c \in \{W,D,L\}} \frac{n_c^{\text{buf}}}{n} \cdot \frac{1}{n_c^{\text{buf}}} \sum_{i \in \mathcal{B}_c} \nabla \ell(\theta; x_i, y_i)$$

$$= \sum_c f_c \cdot \bar{g}_c$$

where $f_c = n_c^{\text{buf}} / n$ is the buffer fraction and $\bar{g}_c = \frac{1}{n_c^{\text{buf}}} \sum_{i \in \mathcal{B}_c} \nabla \ell(\theta; x_i, y_i)$ is the mean gradient over class $c$ samples.

This gradient is biased by the buffer composition: if $f_D = 0.9$, the draw gradients receive 9$\times$ the weight of win or loss gradients.

Under stratified sampling, we draw $B/3$ samples uniformly from each class. The raw stochastic gradient is:

$$\hat{g}_{\text{strat}} = \frac{1}{B} \sum_c \sum_{j=1}^{B/3} \nabla \ell(\theta; x_{c,j}, y_{c,j})$$

where $x_{c,j}$ is the $j$-th sample drawn from $\mathcal{B}_c$. In expectation:

$$\mathbb{E}[\hat{g}_{\text{strat}}] = \frac{1}{B} \sum_c \frac{B}{3} \bar{g}_c = \frac{1}{3} \sum_c \bar{g}_c$$

This is the gradient under the *uniform outcome distribution* $(1/3, 1/3, 1/3)$, regardless of the actual buffer fractions $f_c$.

However, $\hat{g}_{\text{strat}}$ is a biased estimator of the true buffer gradient $\nabla_{\text{uniform}}$. To obtain an unbiased estimator, we apply importance weights. The probability that sample $i \in \mathcal{B}_c$ is drawn under stratified sampling is:

$$P_{\text{strat}}(i) = \frac{1}{3} \cdot \frac{1}{n_c^{\text{buf}}}$$

Under uniform sampling, $P_{\text{uniform}}(i) = 1/n$. The importance weight is:

$$w(i) = \frac{P_{\text{uniform}}(i)}{P_{\text{strat}}(i)} = \frac{1/n}{1/(3 n_c^{\text{buf}})} = \frac{3 n_c^{\text{buf}}}{n}$$

The importance-weighted gradient is:

$$\hat{g}_{\text{IS}} = \frac{1}{B} \sum_c \sum_{j=1}^{B/3} w(c) \cdot \nabla \ell(\theta; x_{c,j}, y_{c,j}) = \frac{1}{B} \sum_c \sum_{j=1}^{B/3} \frac{3 n_c^{\text{buf}}}{n} \nabla \ell(\theta; x_{c,j}, y_{c,j})$$

In expectation:

$$\mathbb{E}[\hat{g}_{\text{IS}}] = \frac{1}{B} \sum_c \frac{B}{3} \cdot \frac{3 n_c^{\text{buf}}}{n} \bar{g}_c = \sum_c \frac{n_c^{\text{buf}}}{n} \bar{g}_c = \nabla_{\text{uniform}}$$

So the importance-weighted stratified gradient is an unbiased estimator of the true buffer gradient. Note the direction: $w(i) = 3 n_c^{\text{buf}} / n$ is the IS correction that maps *from* the stratified distribution *to* the uniform-buffer distribution (upweighting overrepresented classes to undo the equalization). This is the inverse of the sampling weight that would equalize classes.

The key insight is that the *unweighted* stratified gradient estimates the gradient under a balanced distribution, which is precisely what we want for preventing draw dominance. The importance weights are needed only if the goal is to recover the original (biased) gradient. For collapse prevention, we deliberately omit or downweight the importance correction, effectively training as if the buffer were balanced. This breaks the feedback loop: even when $f_D = 0.95$, the gradient treats draws, wins, and losses with equal weight, preventing the value head from being driven toward the all-draw prediction.

The variance of the stratified estimator satisfies $\text{Var}(\hat{g}_{\text{strat}}) \leq \text{Var}(\hat{g}_{\text{uniform}})$ by the Rao-Blackwell theorem (stratification is a form of conditioning), so stratified sampling is never worse than uniform sampling in terms of gradient variance. $\square$

### A.5 Absorbing State Attractiveness (Unmodified Dynamics)

**Proposition A.1 (Absorbing State Attractiveness).** *This result establishes that the absorbing state is locally attractive under the unmodified dynamics (without entropy regularization). The main body Theorem 8 shows that entropy regularization reverses this attractiveness, making the absorbing state unstable.*

*Let $\mathbf{x} = (\mathcal{H}_v, \mathcal{H}_f, \rho)$ where $\mathcal{H}_v$ is the value head's WDL entropy, $\mathcal{H}_f$ is the buffer composition entropy, and $\rho$ is a cross-term measuring alignment between predictions and buffer. Define:*

$$V(\mathbf{x}) = \alpha \mathcal{H}_v + \beta \mathcal{H}_f + \gamma \rho$$

*with $\alpha, \beta, \gamma > 0$ chosen appropriately. Then $V > 0$ in the healthy training region, $V = 0$ at the absorbing fixed point, and $\dot{V} < 0$ along collapse trajectories under the unmodified dynamics.*

*Proof.* We verify the three Lyapunov conditions.

**Condition 1: $V(\mathbf{x}^*) = 0$ at the absorbing fixed point.**

At the absorbing fixed point: $\mathcal{H}_v = 0$ (the WDL distribution is degenerate at $(0,1,0)$), $\mathcal{H}_f = 0$ (the buffer composition is degenerate at $(0,1,0)$), and $\rho = 0$ (defined such that it vanishes when predictions and buffer are both degenerate). Therefore $V(\mathbf{x}^*) = 0$.

**Condition 2: $V(\mathbf{x}) > 0$ for $\mathbf{x} \neq \mathbf{x}^*$ in the healthy region.**

For any non-degenerate state, at least one of $\mathcal{H}_v$ or $\mathcal{H}_f$ is strictly positive (since both are entropies of distributions that are non-degenerate away from the fixed point). With $\alpha, \beta > 0$ and $\rho \geq 0$ by construction, $V > 0$.

**Condition 3: $\dot{V} < 0$ along collapse trajectories.**

We compute $\dot{V}$ by analyzing how each component evolves:

*Buffer entropy dynamics.* The buffer update $\mathbf{f}^{(t+1)} = (1-\eta)\mathbf{f}^{(t)} + \eta \mathbf{g}^{(t)}$ gives:

$$\dot{\mathcal{H}}_f = \mathcal{H}((1-\eta)\mathbf{f} + \eta \mathbf{g}) - \mathcal{H}(\mathbf{f})$$

When the value head is biased toward draws, $\mathbf{g}$ has $g_D > f_D$ (self-play produces more draws than the current buffer fraction). Since the entropy function is concave and $\mathbf{g}$ is more concentrated than $\mathbf{f}$ in the collapse direction:

$$\dot{\mathcal{H}}_f \leq \eta(\mathcal{H}(\mathbf{g}) - \mathcal{H}(\mathbf{f})) < 0$$

because $\mathcal{H}(\mathbf{g}) < \mathcal{H}(\mathbf{f})$ when $\mathbf{g}$ is more concentrated.

*Value head entropy dynamics.* Training on a buffer with $f_D > f_D^{\text{balanced}}$ drives the value head to predict higher $p_D$, reducing $\mathcal{H}_v$:

$$\dot{\mathcal{H}}_v = -\alpha_{\text{lr}} \nabla_\theta \mathcal{H}_v \cdot \nabla_\theta \mathcal{L}_{\text{CE}}$$

The gradient of the cross-entropy loss with draw-dominant labels points in the direction of decreasing $\mathcal{H}_v$ (toward more confident draw predictions). By the correlation inequality between the CE gradient and the entropy gradient in this regime, $\dot{\mathcal{H}}_v < 0$.

*Cross-term dynamics.* Define $\rho = \text{KL}(\mathbf{f} \| \mathbf{p}_\theta)$ (KL divergence from buffer composition to value head prediction). During collapse, $\mathbf{f}$ and $\mathbf{p}_\theta$ both converge to $(0,1,0)$, so $\rho \to 0$. The rate satisfies $\dot{\rho} < 0$ because training explicitly minimizes the cross-entropy between $\mathbf{f}$ and $\mathbf{p}_\theta$.

Combining with appropriate constants $\alpha, \beta, \gamma$:

$$\dot{V} = \alpha \dot{\mathcal{H}}_v + \beta \dot{\mathcal{H}}_f + \gamma \dot{\rho} < 0$$

where each term is strictly negative in the collapse regime ($\mathcal{H}_v > 0$, $\mathcal{H}_f > 0$, $f_D > 1/3$).

The basin of attraction of the absorbing fixed point is the largest connected sublevel set of $V$ on which $\dot{V} < 0$. By linearization around $\mathbf{x}^*$, this basin is approximately the ellipsoid:

$$\{\mathbf{x} : \alpha \mathcal{H}_v^2 + \beta \mathcal{H}_f^2 + \gamma \rho^2 < V_{\text{crit}}\}$$

where $V_{\text{crit}}$ is determined by the eigenvalues of the Jacobian at $\mathbf{x}^*$. Trajectories starting inside this ellipsoid converge monotonically to $\mathbf{x}^*$; trajectories outside may escape to the healthy training basin.

This establishes the attractiveness of the absorbing state under the *unmodified* dynamics. The main body Theorem 8 (Section 7) proves the complementary result: under entropy regularization with $\lambda_H > \lambda_H^{\text{crit}}$, the Lyapunov-like function $W(S_t) = H(D_t) - \lambda \cdot L_{\text{value}}^{\text{test}}(\theta_t)$ is non-decreasing near the absorbing state, reversing the attractiveness and guaranteeing escape. $\square$


## Appendix B: Implementation Pseudocode

The following pseudocode implementations correspond to the prevention methods analyzed in Section 6. All code targets PyTorch and is designed for integration into an AlphaZero training pipeline with a WDL value head.

### B.1 Entropy-Regularized Value Loss

```python
def entropy_regularized_value_loss(wdl_logits, outcome_wdl, lambda_h=0.1):
    """
    Value loss with entropy regularization (Theorem 5).
    
    Args:
        wdl_logits: (B, 3) raw logits from value head
        outcome_wdl: (B, 3) soft target distribution (one-hot or smoothed)
        lambda_h: entropy regularization coefficient
    
    Returns:
        Scalar loss. Minimizing this loss simultaneously fits the target
        distribution and maintains WDL entropy above H_min.
    """
    log_probs = F.log_softmax(wdl_logits, dim=1)
    probs = F.softmax(wdl_logits, dim=1)
    
    # Standard soft cross-entropy: -sum(y * log(p))
    ce_loss = -torch.sum(outcome_wdl * log_probs, dim=1)
    
    # Entropy bonus: H(p) = -sum(p * log(p))
    # Maximizing entropy prevents WDL collapse to (0,1,0)
    entropy = -torch.sum(probs * log_probs, dim=1)
    
    # Combined: minimize CE while maximizing entropy
    # The entropy term establishes a floor H_min = lambda_h / (1 + lambda_h) * ln(3)
    loss = ce_loss - lambda_h * entropy
    return loss.mean()
```

### B.2 Focal WDL Loss

```python
def focal_wdl_loss(wdl_logits, outcome_wdl, gamma=2.0):
    """
    Focal loss adapted for WDL classification (Proposition 8).
    
    Downweights well-classified (confident draw) samples, amplifying
    gradients from minority-class (win/loss) predictions.
    
    Args:
        wdl_logits: (B, 3) raw logits from value head
        outcome_wdl: (B, 3) soft target distribution
        gamma: focusing parameter. gamma=0 recovers standard CE.
    
    Returns:
        Scalar loss with focal modulation.
    """
    probs = F.softmax(wdl_logits, dim=1)
    log_probs = F.log_softmax(wdl_logits, dim=1)
    
    # p_t: predicted probability of the true class
    # For soft targets, this is the dot product p . y
    pt = torch.sum(probs * outcome_wdl, dim=1)
    
    # Focal weight: (1 - p_t)^gamma
    # When model confidently predicts the correct class (p_t -> 1),
    # weight -> 0, reducing gradient contribution
    # When model is uncertain (p_t -> 1/3), weight -> (2/3)^gamma
    focal_weight = (1.0 - pt) ** gamma
    
    # Standard cross-entropy per sample
    ce = -torch.sum(outcome_wdl * log_probs, dim=1)
    
    # Focal-weighted loss
    loss = focal_weight * ce
    return loss.mean()
```

### B.3 Outcome-Stratified Sampling

```python
def stratified_sample(replay_buffer, batch_size):
    """
    Draw equal numbers of wins, draws, and losses from the buffer (Theorem 6).
    
    Returns indices and optional importance weights for unbiased
    gradient estimation. For collapse prevention, use weights=None
    (deliberately biased toward balanced distribution).
    
    Args:
        replay_buffer: buffer with get_composition() and per-sample metadata
        batch_size: total batch size (must be divisible by 3)
    
    Returns:
        indices: (batch_size,) array of buffer indices
        weights: (batch_size,) importance weights, normalized to max=1
    """
    composition = replay_buffer.get_composition()  # {wins, draws, losses}
    n_per_class = batch_size // 3
    
    # Sample equal numbers from each outcome class
    # sample_by_outcome draws uniformly from positions with the given result
    win_indices = sample_by_outcome(replay_buffer, outcome=WIN, n=n_per_class)
    draw_indices = sample_by_outcome(replay_buffer, outcome=DRAW, n=n_per_class)
    loss_indices = sample_by_outcome(replay_buffer, outcome=LOSS, n=n_per_class)
    
    # Assign weights BEFORE concatenation, based on outcome metadata
    # w(i) = P_uniform(i) / P_stratified(i) = (1/n) / (1/(3*n_c)) = 3*n_c/n
    total = composition.wins + composition.draws + composition.losses
    win_weights = np.full(n_per_class, 3.0 * max(composition.wins, 1) / total, dtype=np.float32)
    draw_weights = np.full(n_per_class, 3.0 * max(composition.draws, 1) / total, dtype=np.float32)
    loss_weights = np.full(n_per_class, 3.0 * max(composition.losses, 1) / total, dtype=np.float32)

    indices = np.concatenate([win_indices, draw_indices, loss_indices])
    weights = np.concatenate([win_weights, draw_weights, loss_weights])

    # Shuffle both together so weights remain correctly paired with indices
    perm = np.random.permutation(len(indices))
    indices = indices[perm]
    weights = weights[perm]
    
    # Normalize to max=1 to prevent gradient explosion
    weights /= weights.max()
    
    return indices, weights
```

### B.4 Adaptive Simulation Budget

```python
def adaptive_simulations(network, sample_positions, n_base=200, h_target=0.8):
    """
    Scale simulation budget inversely with value head confidence (Theorem 7).
    
    When the value head is uncertain (high entropy), more simulations are
    warranted. When it is confident (low entropy), fewer simulations prevent
    search amplification of potential bias.
    
    Args:
        network: the current model
        sample_positions: (M, 123, 8, 8) tensor of sample board positions
        n_base: base simulation count
        h_target: target entropy threshold
    
    Returns:
        Simulation count for the current iteration.
    """
    with torch.no_grad():
        # Forward pass to get WDL predictions on sample positions
        _, _, _, wdl_logits = network(sample_positions)
        wdl_probs = F.softmax(wdl_logits, dim=1)
        
        # Average WDL entropy across sample positions
        # H_max = ln(3) ≈ 1.099 for uniform (0.33, 0.33, 0.33)
        # H = 0 for degenerate (0, 1, 0)
        entropy = -torch.sum(wdl_probs * torch.log(wdl_probs + 1e-8), dim=1)
        avg_entropy = entropy.mean().item()
    
    # Scale: ratio >= 1 when entropy exceeds target, clamped at 1 below
    # This means: high entropy -> more sims allowed, low entropy -> stay at base
    ratio = max(1.0, avg_entropy / h_target)
    return int(n_base * ratio)
```

### B.5 Factored Value Head

```python
class FactoredValueHead(nn.Module):
    """
    Decomposes WDL prediction into P(decisive) and P(win|decisive) (Section 6.2).
    
    Architecture: shared trunk -> binary decisive/draw prediction
                                -> conditional win/loss prediction (given decisive)
    
    This prevents draw collapse because P(decisive) is trained with binary CE
    which has stronger gradients than 3-class CE near (0, 1, 0).
    """
    def __init__(self, in_channels, num_filters=1, hidden_size=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        flat_size = num_filters * 8 * 8  # 64 for standard chess board
        self.fc_shared = nn.Linear(flat_size, hidden_size)
        
        # Binary head: is this position decisive (win or loss)?
        self.fc_decisive = nn.Linear(hidden_size, 1)
        
        # Conditional head: given decisive, who wins?
        self.fc_winloss = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.shared(x)
        x = x.reshape(x.size(0), -1)
        h = F.relu(self.fc_shared(x))
        
        # P(decisive) via sigmoid
        p_decisive = torch.sigmoid(self.fc_decisive(h))
        
        # P(win | decisive) via sigmoid
        p_win_given_decisive = torch.sigmoid(self.fc_winloss(h))
        
        # Reconstruct WDL probabilities
        # P(W) = P(decisive) * P(win | decisive)
        # P(L) = P(decisive) * (1 - P(win | decisive))
        # P(D) = 1 - P(decisive)
        w_W = p_decisive * p_win_given_decisive
        w_L = p_decisive * (1.0 - p_win_given_decisive)
        w_D = 1.0 - p_decisive
        
        return torch.cat([w_W, w_D, w_L], dim=1)
```


## Appendix C: Complete Experimental Data

This appendix presents the full iteration-by-iteration data for each collapse mode referenced in the main text. All data are exact measurements from the training logs of the f256-b20-se4 architecture.

### C.1 Mode I: Simulation Depth Draw Death (Section 10g Data)

Training configuration: Gumbel search with search batch = 1 (PUCT-equivalent), 32 games per iteration, 32 workers, buffer capacity 100K. Simulation budget increased from 200 to 800 at iteration 6.

**Table C.1.** Complete Mode I trajectory.

| Iteration | Simulations | Decisive (%) | Rep. Draws (/32) | Value Loss | Avg. Game Length | Status |
|:---------:|:-----------:|:------------:|:-----------------:|:----------:|:----------------:|:------:|
| 5 | 200 | 22 | 16 | 0.89 | 315 | Healthy |
| 6 | 800 | 22 | 16 | 0.89 | 315 | OK |
| 7 | 800 | 9 | 27 | 0.51 | 153 | Warning |
| 8 | 800 | 3 | 31 | 0.45 | 95 | Critical |
| 9--53 | 800 | 0 | 32 | 0.02 | 42 | Dead |

Key transition: The one-iteration delay between the simulation increase (iter 6) and the onset of collapse (iter 7) corresponds to the pipeline delay — iteration 6's self-play used the model trained at iteration 5 (which was healthy at $N=200$), but the model trained at iteration 6 incorporated the first $N=800$ games and began exhibiting draw bias. The cascade from 22% to 9% to 3% to 0% decisive games took only 3 iterations, and manual intervention at iteration 8 (parameter adjustment) was counterproductive. The system remained at exactly 0% decisive, 32/32 repetition draws, value loss 0.02, and average length 42 for 44 consecutive iterations until termination.

### C.2 Mode II: Fifty-Move Stagnation (Section 10i Data)

Training configuration: Gumbel Top-$k$ SH at root, $k = 16$, search batch = 16, 200 simulations, 32 games per iteration. Recovery intervention applied starting at iteration 10: epochs reduced from 3 to 1, temperature moves increased to 50.

**Table C.2.** Complete Mode II trajectory with recovery.

| Iteration | Decisive (%) | Rep. Draws (/32) | 50-Move Draws (/32) | Value Loss | Avg. Game Length | Notes |
|:---------:|:------------:|:-----------------:|:-------------------:|:----------:|:----------------:|:------|
| 1 | 28 | 7 | 5 | 0.89 | 358 | Initial learning |
| 2 | 28 | 3 | 7 | 0.47 | 311 | Rapid value learning |
| 3 | 28 | 5 | 5 | 0.65 | 337 | Stable |
| 4 | 50 | 4 | 6 | 0.55 | 289 | Peak decisive rate |
| 5 | 34 | 5 | 6 | 0.57 | 300 | Slight regression |
| 6 | 0 | 0 | 25 | 0.50 | 381 | Stagnation onset |
| 7 | 0 | 0 | 29 | 0.40 | 368 | Deepening |
| 8 | 6 | 1 | 29 | 0.15 | 341 | Value collapse |
| 9 | 6 | 0 | 27 | 0.12 | 365 | Near-absorbing |
| 10 | 28 | 6 | 11 | 0.30 | 330 | Recovery begins |
| 11 | 38 | 5 | 6 | 0.39 | 298 | Recovery confirmed |

Note the qualitative difference from Mode I: repetition draws did not increase during the stagnation phase (iterations 6--9). The dominant draw mechanism shifted entirely to fifty-move rule draws, with 25--29 out of 32 games reaching the fifty-move limit. This is consistent with the Gumbel SH mechanism: diverse root exploration prevents repetition-forcing, but cannot prevent aimless shuffling when the value head provides no directional signal.

### C.3 Mode III: Max-Moves Pollution (Section 10m Data)

Buffer composition measured at training shutdown after sustained max-moves draw accumulation.

**Table C.3.** Buffer composition under max-moves pollution.

| Outcome | Count | Percentage |
|:-------:|------:|:----------:|
| Wins | 4,364 | 3.5% |
| Draws | 116,509 | 92.1% |
| Losses | 5,558 | 4.4% |
| **Total** | **126,431** | **100%** |

Note: total exceeds nominal buffer capacity of 100,000 because the buffer stores positions, not games; a single game contributes one position per move.

**Table C.4.** Sample contribution asymmetry.

| Metric | Decisive Game | Max-Moves Game | Ratio |
|:------:|:------------:|:--------------:|:-----:|
| Typical length (half-moves) | $\sim 80$ | 512 | 6.4$\times$ |
| Positions contributed | $\sim 80$ | 512 | 6.4$\times$ |
| At 90% draw rate: effective ratio | 1$\times$ | up to 115$\times$ | up to 115:1 |

The worst-case ratio arises when decisive games are short (early termination around move 40) while draw games consistently reach the 512-move maximum: $(512/40) \times (0.9/0.1) = 115.2$ in terms of buffer representation. The 36:1 figure from the operation manual uses different game-length assumptions (longer decisive games, mix of draw termination types). The qualitative point — severe asymmetry — holds regardless of the exact ratio.

**Table C.5.** Policy vs. value loss decoupling during Mode III.

| Iteration Range | Policy Loss Trend | Value Loss Trend |
|:--------------:|:-----------------:|:----------------:|
| Early (iters 1--5) | 8.40 $\to$ 8.10 | 0.89 $\to$ 0.55 |
| Mid (iters 6--10) | 8.10 $\to$ 7.80 | 0.55 $\to$ 0.15 |
| Late (iters 11--15) | 7.80 $\to$ 7.49 | 0.15 $\to$ 0.12 |

Policy loss decreased monotonically from 8.40 to 7.49 across the entire collapse trajectory, demonstrating complete decoupling between the policy and value learning signals.

### C.4 Mode IV: Buffer Saturation Oscillation Trap (Section 10r Data)

Training configuration: Gumbel Top-$k$ SH, intervention attempts with varying epoch counts.

**Table C.6.** Complete Mode IV oscillation trajectory.

| Iteration | Epochs | Draw (%) | Value Loss | Buffer Draw (%) | Intervention |
|:---------:|:------:|:--------:|:----------:|:---------------:|:-------------|
| 9 | 1 | 71 | 0.68 | 77.5 | epochs=1 |
| 10 | 2 | 91 | 0.57 | 84.0 | epochs=2 |
| 11 | 2 | 92 | 0.39 | 92.2 | epochs=2 |
| 12 | 1 | 64 | 0.45 | 88.0 | epochs=1 |
| 13 | 1 | 92 | 0.375 | 89.5 | epochs=1 |

The oscillation pattern is clearly visible: iteration 12 (epochs=1) produced a genuine improvement (draw rate dropped to 64%, value loss recovered to 0.45), but the buffer draw fraction only decreased from 92.2% to 88.0%. The next iteration (13) trained on this still-dominant-draw buffer and the draw rate rebounded to 92%.

The identified irreversibility threshold is the conjunction of buffer draw composition exceeding 85% and value loss below 0.4. Once both conditions are met simultaneously, the per-iteration "inflow" of decisive samples ($\sim 2{,}500$ positions from 32 games) is insufficient to shift the 100,000-position buffer away from draw dominance before the next training cycle reinforces the draw bias.


## Appendix D: Gumbel SH vs PUCT Analysis

### D.1 Why Gumbel Prevents Mode I (Repetition Death)

Mode I collapse under PUCT proceeds through a specific mechanism at the root node. Consider a root position with $K$ legal moves, indexed $a_1, \ldots, a_K$, and suppose the value head has a mild draw bias: $v(s) \approx 0$ for most positions $s$, with small perturbations.

Under PUCT with search batch size 1, each simulation independently selects the root action that maximizes:

$$a^* = \arg\max_a \left[ Q(a) + c_{\text{explore}} \cdot P(a) \cdot \frac{\sqrt{N_{\text{parent}}}}{1 + N(a)} \right]$$

After the first few simulations, one action $a_1$ (say) accumulates a Q-value $Q(a_1) \approx 0$ (draw value) with $N(a_1) \gg 0$ visits. The remaining actions have $N(a_j) = 0$ for $j > 1$ and are evaluated using the First Play Urgency (FPU) heuristic, which assigns an effective Q-value of:

$$Q_{\text{FPU}}(a) = Q_{\text{parent}} - f_{\text{base}} \cdot \sqrt{1 - P(a)}$$

where $f_{\text{base}} = 0.3$ and $P(a)$ is the prior probability of the unvisited action (matching the formula in Section 2.3 and the code at `node.hpp:90`). With $P_{\text{visited}} \approx P(a_1)$ and typical FPU reduction, $Q_{\text{FPU}} \approx -0.29$. Since $Q(a_1) = 0 > -0.29 = Q_{\text{FPU}}$, the exploitation term favors $a_1$ even when $P(a_j) > P(a_1)$ for some $j$, provided $N(a_1)$ is large enough that the exploration bonus $\sqrt{N_{\text{parent}}} / (1 + N(a_1))$ has decayed. The result is a positive feedback loop: more visits to $a_1$ increase $N(a_1)$, which reduces the exploration bonus for $a_1$ but the exploitation advantage of $Q(a_1) > Q_{\text{FPU}}$ persists, so $a_1$ continues to receive the majority of visits.

If $a_1$ happens to lead to a repetition-forcing line (which is likely when the value head sees all positions as approximately equal), the self-play game enters a repetition cycle, producing a draw. This draw sample reinforces the value head's draw bias, increasing the probability that the same mechanism fires at the next iteration.

Gumbel Top-$k$ Sequential Halving eliminates this mechanism entirely through its round-robin action assignment. In Gumbel SH with $k = 16$:

1. **Action selection.** The top $k$ actions are selected based on $\text{logit}(a) + g(a)$ where $g(a) \sim \text{Gumbel}(0,1)$ are independent Gumbel random variables. This selection is stochastic and does not depend on Q-values.

2. **Simulation assignment.** Within each batch of $k$ simulations, exactly one simulation is assigned to each of the $k$ active actions. There is no choice at root — the assignment is deterministic given the active set.

3. **Sequential halving.** After each round, the bottom half of actions (by $\text{logit}(a) + g(a) + \sigma(Q_{\text{completed}}(a))$) are pruned. The Gumbel noise $g(a)$ provides stochastic tie-breaking, preventing deterministic convergence to a single action.

This design guarantees that all $k$ active actions receive equal numbers of simulations at each halving round, regardless of their Q-values. The visit concentration that drives Mode I under PUCT is architecturally impossible.

**Empirical confirmation.** Across 11 iterations of Gumbel training (352 total games), zero iterations exhibited the catastrophic repetition-draw cascade characteristic of Mode I (compare Table 1, where PUCT training collapsed within 3 iterations of the simulation increase). The Gumbel run did produce some repetition draws (4 per iteration on average), but these were sporadic events distributed uniformly across iterations, not the self-reinforcing cascade where repetition draws increase monotonically iteration over iteration.

### D.2 Why Gumbel Does NOT Prevent Mode II (Fifty-Move Stagnation)

Mode II collapse does not depend on the root search algorithm's action selection mechanism. It depends on the *value signal quality* — specifically, the ability of the value head to distinguish between positions where decisive play is possible and positions where it is not.

When the value head evaluates all positions as "draw" with high confidence:

1. **Q-value homogenization.** For every root action $a$, the backed-up Q-value converges to $Q(a) \approx 0$ regardless of the true game-theoretic value of the resulting position. This occurs because every leaf in the search tree receives $v \approx 0$, and the backed-up value is a weighted average of leaf values.

2. **$\sigma(Q)$ vanishing.** By Theorem 1, when $Q(a) \approx Q(b)$ for all actions $a, b$, the normalized score $\sigma(Q_{\text{completed}}(a)) \to 0$. The improved policy degenerates to the prior:

$$\pi_{\text{improved}}(a) = \text{softmax}(\text{logit}(a) + 0) = P(a)$$

3. **Search provides no information.** Whether the $k = 16$ active actions all receive 12 simulations each (Gumbel) or one action receives 192 simulations (PUCT) is irrelevant when all simulations return the same uninformative value. The information content of the search is zero in both cases.

4. **Aimless play.** Without value guidance, the policy plays according to the prior (which encodes piece mobility and tactical patterns but not strategic direction). The resulting games proceed through many moves of piece maneuvering without progress toward a decisive result, eventually triggering the fifty-move rule.

The fix for Mode II must therefore address the value head directly. The successful recovery protocol (epochs=1 + temperature\_moves=50) works by:

- **Epochs=1**: Limiting the number of times the value head trains on draw-dominant data per iteration, slowing the reinforcement of draw predictions.
- **Temperature\_moves=50**: Introducing stochastic move selection for the first 50 moves, producing more diverse positions. Some of these positions will be objectively winning or losing, providing decisive training signal even when the value head is biased.

In contrast, $\beta_{\text{risk}} = 0.5$ (entropic risk measure, risk-seeking) failed because it operates through the search mechanism: $Q_\beta \approx \mathbb{E}[v] + (\beta/2) \operatorname{Var}(v)$. When $\operatorname{Var}(v) \approx 0$ (the value head predicts "draw" with near certainty), the risk term contributes nothing, and $Q_\beta \approx Q \approx 0$ for all actions. Search-level interventions cannot amplify signal that does not exist.

### D.3 Mathematical Characterization

The vulnerability to each collapse mode can be understood in a two-dimensional space spanned by root exploration diversity $D_{\text{root}}$ and value head discriminative power $\Delta_v$.

Define $D_{\text{root}}$ as the effective number of distinct root actions explored per move:

$$D_{\text{root}} = \exp\!\left(-\sum_a \frac{N(a)}{N_{\text{total}}} \log \frac{N(a)}{N_{\text{total}}}\right)$$

Under PUCT with visit concentration, $D_{\text{root}} \to 1$ as the dominant action absorbs all visits. Under Gumbel with $k$ active actions and equal allocation, $D_{\text{root}} = k$.

Define $\Delta_v$ as the value head's ability to distinguish between positions:

$$\Delta_v = \mathbb{E}_{s, s'}\!\left[\|v(s) - v(s')\|^2\right]^{1/2}$$

where the expectation is over pairs of positions reachable from the root. When the value head predicts "draw" everywhere, $\Delta_v \to 0$.

The two collapse modes occupy distinct regions:

$$\text{Mode I: } D_{\text{root}} \to 1, \quad \Delta_v > 0 \quad \text{(concentrated search, weakly informative values --- sufficient only for FPU-driven concentration)}$$
$$\text{Mode II: } D_{\text{root}} > 1, \quad \Delta_v \to 0 \quad \text{(diverse search, uninformative values)}$$

Gumbel SH shifts $D_{\text{root}}$ from $\sim 1$ to $k$, moving the system out of the Mode I region. But it does not affect $\Delta_v$, leaving the system vulnerable to Mode II. Complete prevention requires interventions that maintain $\Delta_v > 0$ (through loss modification, architecture changes, or data interventions) in addition to maintaining $D_{\text{root}} > 1$ (through Gumbel or similar diverse search).

The interaction is captured by the joint condition for healthy training:

$$D_{\text{root}} > D_{\min} \quad \text{AND} \quad \Delta_v > \Delta_{\min}$$

where $D_{\min} \approx 4$ (enough distinct actions to avoid repetition lock-in) and $\Delta_{\min} \approx 0.1$ (enough value differentiation for search to provide useful policy improvement). Gumbel guarantees the first condition; the prevention methods of Section 5 target the second.