Self-Simulating Computational Manifold (SCM v1.3)

Symbol / Term	Definition

$n$	Number of qubits in the system
$T$	Total number of time steps in the simulation
$\Delta t$	Size of a single time step
$\rho$	Density matrix representing the state of the quantum system
$\rho_{ij}$	Reduced density matrix for the subsystem of qubits $i$ and $j$
$S_2(\rho)$	2-Rényi entropy of state $\rho$, defined as $S_2(\rho) = -\log \left[ \mathrm{Tr}(\rho^2) \right]$
$S(\rho)$	von Neumann entropy, $S(\rho) = -\mathrm{Tr}(\rho \log \rho)$
$\mathcal{C}$	Cost functional minimized during optimization
$\eta$	Learning rate / gradient descent step size
$I(A:B)$	Mutual information between subsystems $A$ and $B$
$K_{ij}$	SCM curvature between qubits $i$ and $j$
$\epsilon$	Small regularization constant to prevent division by zero
$\text{Purity}(\rho)$	Defined as $\mathrm{Tr}(\rho^2)$, measures state mixedness
$\theta_t$	Control parameters at time step $t$
$\psi_0$	Initial pure state of the system



---

1. Abstract

Structured Correlation Mapping (SCM) is a computational framework for analyzing the evolution of correlations and geometric structures in quantum systems. It quantifies relationships between qubits by tracking pairwise entropies, mutual information, and a derived curvature metric, offering insights into quantum information flow and topology.


---

2. Introduction

SCM combines principles from quantum information theory, geometry, and optimization. It models a multi-qubit system, applies an entropy-based cost functional, and evolves system parameters through projected gradient descent. The outputs — pairwise purities, mutual information matrices, and curvature maps — reveal emergent structures and patterns in quantum dynamics.


---

3. Scientific Context

The 2-Rényi entropy provides a computationally efficient way to measure subsystem purity. By minimizing neighborhood entropies across all qubit pairs, SCM favors low-entanglement structures that still preserve essential correlations. The derived curvature metric, $K_{ij}$, parallels geometric curvature concepts, mapping quantum correlation space into an interpretable topology.


---

4. Core Results

4.1 Purity Evolution

The mean pair purity over time is given by:

\text{Purity}(\rho_{ij}) = \mathrm{Tr}(\rho_{ij}^2)

For all pairs $(i,j)$, the mean is:

\overline{\mathcal{P}} = \frac{1}{\binom{n}{2}} \sum_{i<j} \mathrm{Tr}(\rho_{ij}^2)

4.2 Mutual Information

Mutual information between subsystems $A$ and $B$:

I(A : B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})

where:

S(\rho) = -\mathrm{Tr}(\rho \log \rho)

4.3 SCM Curvature

Defined as:

K_{ij} = \frac{1}{\epsilon + I_{ij}}

where $\epsilon > 0$ is a small constant.


---

5. Detailed Analysis

5.1 Observation: Purity Increase Phase

The mean pair purity increased from $\approx 0.404$ to $\approx 0.443$ during the first six time steps, followed by a stable plateau.
This behavior indicates that the optimizer effectively reduced entanglement across the system while preserving key correlations.

5.2 Observation: Mutual Information Decay

Average mutual information values between non-neighboring qubits decreased over time, suggesting that the system’s correlation graph became more localized.

5.3 Observation: Curvature Stabilization

SCM curvature values converged to stable magnitudes, indicating a balanced correlation geometry after optimization.


---

6. Methodology & Simulation Details

6.1 Simulation Environment

Language: Python 3.11

Libraries:

numpy — numerical operations

scipy.linalg — matrix exponentials

qutip — quantum objects and dynamics

matplotlib — visualization

random / numpy.random — reproducibility



6.2 System Parameters

Parameter	Value	Description

$n$	5	Number of qubits
$T$	10	Total time steps
$\Delta t$	0.2	Time step size
$\eta$	0.08	Gradient descent step size
seed	3	Random seed


6.3 State Initialization

Initial pure state:

|\psi_0\rangle = \frac{1}{\sqrt{\sum_i |a_i|^2}} \sum_i a_i |i\rangle

with $a_i$ drawn from a complex Gaussian distribution.
Density matrix:

\rho_0 = |\psi_0\rangle\langle\psi_0|

6.4 Cost Functional

2-Rényi entropy:

S_2(\rho) = -\log \left[ \mathrm{Tr}(\rho^2) \right]

SCM cost functional:

\mathcal{C} = \sum_{\text{pairs }(i,j)} S_2(\rho_{ij})

6.5 Optimization Algorithm

1. Compute $\nabla \mathcal{C}(\theta_t)$


2. Project onto valid density matrices


3. Update:



\theta_{t+1} = \theta_t - \eta \nabla \mathcal{C}(\theta_t)

4. Repeat for $T$ steps



6.6 Purity Calculation

\text{Purity}(\rho_{ij}) = \mathrm{Tr}(\rho_{ij}^2)

6.7 Mutual Information Calculation

I(A : B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})

6.8 Curvature Calculation

K_{ij} = \frac{1}{\epsilon + I_{ij}}

6.9 Reproducibility Checklist

1. Set seed = 3


2. Use parameters in 6.2


3. Implement optimizer in 6.5


4. Match outputs to Section 4 metrics




---

Got it — I’ll start Section 7: Equations & Mathematical Derivations in your locked-in SCM v1.3 GitHub format, so all math renders correctly and is consistent with Sections 1–6.

Here’s the full draft:


---

7. Equations & Mathematical Derivations

This section contains the complete mathematical framework for the Self-Simulating Computational Manifold (SCM v1.3), including all primary equations referenced in earlier sections. Equations are formatted for GitHub-compatible LaTeX rendering.


---

7.1 Initial State Definition

The initial pure state is defined as:

|\psi_0\rangle = \frac{1}{\sqrt{\sum_i |a_i|^2}} \sum_i a_i |i\rangle

where  are complex coefficients drawn from a Gaussian distribution:

a_i \sim \mathcal{N}(0,1) + i \,\mathcal{N}(0,1)

The corresponding density matrix is:

\rho_0 = |\psi_0\rangle \langle \psi_0|


---

7.2 Reduced Density Matrix

For a subsystem  of the full quantum state , the reduced density matrix is:

\rho_A = \mathrm{Tr}_{\bar{A}}(\rho)

where  denotes the partial trace over all degrees of freedom not in .


---

7.3 2-Rényi Entropy

The neighborhood 2-Rényi entropy is defined as:

S_2(\rho) = -\log \left[ \mathrm{Tr}(\rho^2) \right]

For SCM, the total cost functional is:

\mathcal{C} = \sum_{\text{pairs } (i,j)} S_2(\rho_{ij})


---

7.4 Gradient Descent Update Rule

The optimization step is given by:

\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{C}(\theta_t)

where  is the gradient step size ( in Section 6.2).


---

7.5 Purity

For a bipartite subsystem :

\text{Purity}(\rho_{ij}) = \mathrm{Tr}(\rho_{ij}^2)

Mean pair purity is computed as:

\overline{\text{Purity}} = \frac{1}{N_\text{pairs}} \sum_{i<j} \mathrm{Tr}(\rho_{ij}^2)


---

7.6 Mutual Information

For subsystems  and :

I(A : B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})

where the von Neumann entropy is:

S(\rho) = -\mathrm{Tr}(\rho \log \rho)


---

7.7 Curvature

SCM curvature between nodes  and  is:

K_{ij} = \frac{1}{\epsilon + I_{ij}}

where  is a small regularization constant () to avoid division by zero.


---

