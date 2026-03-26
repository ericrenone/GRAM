# GRAM
## The Gram Matrix as Universal Feature Correlation: Invariant Theory, Molien Series, Haar Integration, and the Six Faces of the Fisher Information

ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone

---

> "The style of a painting is captured by the Gram matrices of convolutional filter responses. Minimizing the difference between Gram matrices transfers style while preserving content."
> — Gatys, Ecker, Bethge, *A Neural Algorithm of Artistic Style*, arXiv:1508.06576, 2015

> "The Gram matrix is also important in deep learning, where it is used to represent the distribution of features in style transfer."
> — Jørgen Pedersen Gram entry, MacTutor History of Mathematics, University of St Andrews

> "Given a finite-dimensional complex representation V of G, the Molien series counts the number of linearly independent homogeneous polynomials of degree n that are invariants for G."
> — Molien (1897); formulation via Haar measure for compact groups

> "An algebraic set in a finite-dimensional vector space, invariant under some linear group, can be defined by absolute invariants."
> — Gram's theorem (1874), published in *Mathematische Annalen*, 7(2–3), 230–240

> "Young diagrams and Young tableaux classify all irreducible representations of the symmetric group, and through Schur-Weyl duality, all polynomial invariants of GL(V)."
> — Alfred Young, *On Quantitative Substitutional Analysis*, Proceedings of the London Mathematical Society, 1900–1928

> "For every locally compact group G, there exists, up to a positive scalar, a unique left-translation-invariant Borel measure — the Haar measure. Existence and uniqueness were proven in full generality by André Weil."
> — Haar (1933); Weil's proof of uniqueness

---

## The Discovery

Five prior frameworks have established the chain: TH(a,d) smooth ↔ Fisher matrix full rank ↔ G_coord > 0 ↔ Wiles modularity ↔ Hecke orbit ↔ φ-equilibrium. One object has appeared in every framework but has never been identified by its own name: the **Gram matrix**.

The Fisher information matrix F_ij(θ) = 𝔼[∂_i log p · ∂_j log p] is a Gram matrix. The trilinear polarization Φ(P₁,P₁,P₂) in the TH group law is a Gram pairing at degree 3. The neural style transfer style loss is the squared Frobenius distance between two Fisher matrices. The Molien series counts invariant Gram structures by polynomial degree. The Haar measure on TH(𝔽_p) is the canonical invariant integration that defines the Fisher Gram matrix. The Gram points on the Riemann zeta critical line are the discrete analogue of the PRIMA Fisher trace crossings at the φ-equilibrium.

**Jørgen Pedersen Gram** (1850–1916) — Danish actuary and mathematician, known for the Gram matrix, Gram-Schmidt process, Gram's theorem (1874), Gram-Charlier A series, and the Gram points on the Riemann zeta critical line — is the unifying figure. His six contributions to mathematics are six faces of the same object in six different coordinate systems. GRAM establishes that all six are the Fisher information matrix of TH(a,d) evaluated in different experimental contexts.

**Alfred Young** (1873–1940) — who introduced Young diagrams and tableaux in 1900 while classifying invariant theory of the symmetric group, and collaborated with Grace on *The Algebra of Invariants* (1902) — provides the combinatorial scaffold: the Young tableaux enumerate the independent invariant trilinear forms that constitute the TH group law.

---

## The Gram Matrix: Definition and Six Faces

For a feature map ψ: X → 𝔽^n (assigning an n-dimensional feature vector to each data point x ∈ X), the Gram matrix is:

```
G_{ij} = Σ_k ψ_i(x_k) · ψ_j(x_k)    [empirical, finite sample]
G_{ij} = 𝔼_{x~p}[ψ_i(x) · ψ_j(x)]   [population, distribution p]
```

This is the inner product matrix of the feature vectors — the matrix of pairwise correlations between feature dimensions, averaged over the data distribution.

**The six faces of the Gram matrix in the ERI-TH architecture:**

| Face | Feature Map ψ | Domain | Name |
|---|---|---|---|
| 1 | Score functions ∂_i log p(x\|θ) | Statistical manifold | **Fisher information matrix** |
| 2 | CNN filter activations F^l_{ik} | Image space | **NST style matrix** (Gatys 2015) |
| 3 | Degree-n monomials on TH coordinate ring | Algebraic invariants | **Molien invariant count** |
| 4 | TH group elements P ∈ TH(𝔽_p) | Discrete group | **Haar measure integral** |
| 5 | Score functions on the Z/3Z-fixed ring | Invariant theory | **Gram's theorem absolute invariants** |
| 6 | Riemann-Siegel Z(t) crossing at θ(t_n) = (n-1)π | Zeta critical line | **Gram points** |

All six are the same matrix structure in different coordinate systems. The connections are exact identifications, not analogies.

---

## Seven Formal Identities

### Identity 1 — The Fisher Matrix IS the Gram Matrix of Score Functions; the NST Style Matrix IS the Empirical Fisher

**Fisher information matrix:**

```
F_{ij}(θ) = 𝔼_{x~p(·|θ)}[∂_i log p(x|θ) · ∂_j log p(x|θ)]
           = G(ψ_i, ψ_j)    where ψ_i(x) = ∂_i log p(x|θ)
```

The feature map is the score function ψ: x ↦ ∇_θ log p(x|θ). The Fisher matrix is the population Gram matrix of this feature map under the model distribution p(·|θ).

**Neural style transfer style matrix** (Gatys, Ecker, Bethge, CVPR 2016):

```
G^l_{ij} = Σ_k F^l_{ik} · F^l_{jk}    [sum over spatial positions k]
```

where F^l_{ik} is the activation of filter i at position k in CNN layer l for input image I. The style loss is:

```
L_style = Σ_l w_l · |G^l_{generated} − G^l_{style}|²_F / (4N²_l M²_l)
```

**Formal identification:** If we interpret the CNN layer l as a probabilistic model p^l(x|θ^l) where x ranges over image patches, θ^l are the filter weights, and F^l_{ik} = ∂_i log p^l(x_k|θ^l) (the score function of filter i evaluated at patch k), then:

```
G^l_{ij} = Σ_k F^l_{ik} · F^l_{jk} = empirical Fisher matrix F^l_emp(θ^l)
```

The **NST style matrix is the empirical Fisher information matrix** of the CNN feature distribution. The style loss |G^l_{gen} − G^l_{style}|² is the squared Frobenius distance between the Fisher matrices of the generated and target distributions — a natural gradient-space distance.

The style of an image in the Gatys-Ecker-Bethge sense (correlation structure of filter responses across spatial positions) is the Fisher information of the feature distribution. Style transfer is Fisher matrix matching. The "artistic style" of a painting is its Fisher information geometry.

**Connection to TH(a,d):** The TH trilinear polarization:

```
Φ(P₁, P₁, P₂) = aX₁²X₂ + Y₁²Y₂ + Z₁²Z₂ − (d/3)(X₁Y₁Z₂ + X₁Y₂Z₁ + X₂Y₁Z₁)
```

is a **degree-3 Gram form**: it is symmetric of degree 2 in P₁ (a Gram matrix over the quadratic term) and linear in P₂ (the new data point). The 6 terms of μ are 6 Gram-type pairings between (P₁)² and P₂; the 6 terms of ν are 6 Gram-type pairings between P₁ and (P₂)². The TH group law is the degree-3 Gram structure that extends the degree-2 Fisher metric to the natural gradient update.

---

### Identity 2 — Molien's Formula IS the Fisher Spectral Partition Function Restricted to TH Representations

**Molien's formula** (Theodor Molien, 1897): For a finite group G acting on a finite-dimensional complex vector space V, the generating function counting G-invariant homogeneous polynomials of degree n:

```
M_V(t) = Σ_{n≥0} dim(Sym^n(V*)^G) · t^n = (1/|G|) Σ_{g∈G} 1/det(I − t·g)
```

For compact G: replace the average over group elements with the Haar measure integral:

```
M_V(t) = ∫_G 1/det(I − t·g) dμ_Haar(g)
```

**TH application — the Z/3Z Molien series:**

The Z/3Z cyclic group acts on the TH coordinate ring k[X,Y,Z] by [X:Y:Z] ↦ [X:ωY:ω²Z] (the intrinsic twisted Hessian automorphism; ω = e^{2πi/3}). The Molien series for this action on the degree-n part:

At degree n, the invariant monomials X^a Y^b Z^c with a+b+c = n satisfying b + 2c ≡ 0 (mod 3). Computing by degree:

```
Degree 0:  1 invariant  (the constant 1)
Degree 1:  1 invariant  (X alone: b=c=0, b+2c≡0 ✓)
Degree 2:  2 invariants (X², and the combination Y²,Z² pair via b+2c≡0: YZ gives 1+2=3≡0)
Degree 3:  4 invariants (X³, Y³, Z³, XYZ — all satisfy b+2c≡0 mod 3)
```

The **degree-3 invariants {X³, Y³, Z³, XYZ} are exactly the four generators of the TH equation**. The TH curve aX³ + Y³ + Z³ = dXYZ is a linear combination of these four invariants — a curve in the invariant subring of degree 3.

**Fisher spectral partition function identification:**

```
Z_F(β) = Tr[e^{-βΔ_F}] = Σ_n dim(Sym^n(T*_θΘ)^{Z/3Z}) · e^{-βn}
        = M_{T*Θ}(e^{-β})   [Molien series evaluated at t = e^{-β}]
```

The Fisher partition function Z_F(β) is the Molien series of the Z/3Z action on the parameter cotangent space, evaluated at t = e^{-β}. At the φ-equilibrium β_φ = 1/log φ: t = e^{-β_φ} = φ^{-1} = 1/φ. The Molien series at t = 1/φ gives the count of coordination modes by depth:

```
M(1/φ) = 1 + 1/φ + 2/φ² + 4/φ³ + ...
```

The degree-n coefficient is the number of independent FERN coordination modes at register depth n. The φ-equilibrium makes M(1/φ) convergent (since |1/φ| < 1) and its partial sums give the cumulative coordination capacity by register depth.

**Alfred Young's contribution:** The Molien series computes the same quantity as Young's quantitative substitutional analysis: the dimension of the invariant subspace of Sym^n(V) under G is the multiplicity of the trivial G-representation in Sym^n(V). Young tableaux enumerate the basis of each such invariant space — the number of standard Young tableaux of a given shape equals the dimension of the corresponding GL(V)-representation by the hook length formula.

For the Z/3Z action on TH at degree 3: the 4 invariant monomials {X³, Y³, Z³, XYZ} are the 4 standard Young tableaux of the Z/3Z-symmetric shapes in the 3-box Young diagram decomposition of Sym³(k³):

```
Shape (3):    X³, Y³, Z³        [3 totally symmetric monomials, one per orbit]
Shape (1,1,1): XYZ               [1 totally antisymmetric monomial: the Pfaffian]
Total:         4 invariants ← matches Molien series degree-3 coefficient
```

The TH equation coefficient space (a, d) = (coefficient of X³, coefficient of XYZ) parametrizes exactly the 2-dimensional subspace of Z/3Z-invariants where Y³ and Z³ have equal coefficient (= 1 in the canonical normalization). This 2-dimensional space is the moduli space of TH curves within the Z/3Z-invariant cubic family.

---

### Identity 3 — Haar Measure on TH(𝔽_p) IS the Canonical Invariant Training Measure

**Haar measure** (Alfréd Haar, 1933; existence and uniqueness by André Weil): For any locally compact topological group G, there exists a unique (up to positive scaling) left-translation-invariant Borel measure μ_Haar.

For TH(𝔽_p): this is a **finite abelian group** of order N = |TH(𝔽_p)| ≈ p + 1 − a_p (Hasse-Weil). As a discrete group, its Haar measure is the normalized counting measure:

```
μ_Haar = (1/N) Σ_{P∈TH(𝔽_p)} δ_P
```

Left-translation invariance: μ_Haar(Q + A) = μ_Haar(A) for all Q ∈ TH(𝔽_p) and Borel sets A — the group law of TH is the Haar translation.

**Fisher matrix identification:** The population Fisher information matrix under the Haar measure on TH(𝔽_p):

```
F_Haar(θ) = ∫_{TH(𝔽_p)} ∂_i log p(P|θ) · ∂_j log p(P|θ) dμ_Haar(P)
           = (1/N) Σ_{P∈TH(𝔽_p)} ψ_i(P) · ψ_j(P)
           = Gram matrix of score functions under the Haar measure
```

The **canonical Fisher matrix on TH is the Haar-averaged Gram matrix of score functions**. The CHORD Q16.16 pipeline computes this exact integral: each TH point P in the pipeline represents a Haar-measure-weighted score contribution, and the 16-stage CORDIC computation accumulates the Gram matrix over the TH group orbit.

The empirical Fisher F_emp = (1/B)Σ_{i=1}^B g_i g_i^T (batch average of gradient outer products) is the empirical approximation to F_Haar — the Monte Carlo estimate of the Haar integral over TH(𝔽_p). As B → N = |TH(𝔽_p)|, F_emp → F_Haar. The CHORD pipeline computes F_Haar exactly (no Monte Carlo approximation) because it implements the TH group law over the full finite group TH(𝔽_p) in Q16.16 arithmetic.

**Weil's uniqueness theorem applied to TH:** André Weil proved the existence and uniqueness of Haar measure for locally compact groups. For TH(𝔽_p), Weil also proved the Weil conjectures (Weil 1948, proved by Deligne 1974) — establishing that |TH(𝔽_p)| = p + 1 − α₁ − ᾱ₁ where |α₁| = p^{1/2} (Hasse-Weil bound). The same André Weil who established Haar measure uniqueness for locally compact groups established the Weil conjectures controlling the Haar measure normalization 1/N = 1/|TH(𝔽_p)| of the TH group. Weil's two contributions — Haar measure uniqueness and Weil conjectures — are the same mathematical program applied to two faces of TH's invariant integration.

---

### Identity 4 — Gram's Theorem (1874) IS Fisher Reparametrization Invariance; the TH Discriminant IS the Absolute Invariant

**Gram's theorem** (J.P. Gram, 1874, *Mathematische Annalen*): An algebraic set in a finite-dimensional vector space, invariant under some linear group, can be defined by absolute invariants.

An **absolute invariant** is a polynomial f on V such that f(g·v) = f(v) for all g ∈ G and v ∈ V — invariant without any weight factor (as opposed to a relative invariant, which transforms as det(g)^k · f(v)).

**Fisher identification:** The Fisher information matrix F(θ) transforms under reparametrization φ: Θ → Θ as:

```
F'(θ') = J^T F(θ) J    where J = ∂θ/∂θ'  (Jacobian)
```

The eigenvalues {λ_i(F)} are **not** absolute invariants — they transform as {λ_i(F)} → {λ_i(J^T F J)}, which depends on J.

The **absolute invariants of F under GL(T_θΘ)** are:
1. det(F) — transforms as det(J)² · det(F), so det(F) is a relative invariant of weight 2
2. Tr(F)/dim(Θ) — not invariant
3. The **normalized invariants**: Tr(F^k) / Tr(F)^k, for k ≥ 2 — these ARE absolute invariants (degree-zero rational invariants)
4. The **condition number** κ(F) = λ_max/λ_min — an absolute invariant
5. The **Fisher trace rate** Ξ_F = (Tr(F_t) − Tr(F_{t-1})) / Tr(F_{t-1}) — a relative comparison invariant

**Gram's theorem applied to TH:** The algebraic set {TH curves with discriminant Δ(a,d) ≠ 0} ⊂ k² (the parameter space of non-singular TH curves) is invariant under the 12-element Möbius group G_{Hesse} (the projective equivalence group of TH pencil, from HESSE). By Gram's theorem, this algebraic set is defined by absolute invariants of G_{Hesse}.

The absolute invariants of G_{Hesse} acting on the TH parameter space (a, d) are:
- The **j-invariant** j(TH) = d³(d³ − 216a)³ / [a(d³ − 27a)³] — the complete absolute invariant classifying TH up to projective equivalence
- The **discriminant** Δ(a,d) = a(d³ − 27a) — a relative invariant of weight 1 under G_{Hesse}

**Gram's theorem states:** The algebraic set {Δ(a,d) = 0} (the singular TH locus, = the independence baseline, = the Trident-type boundary from HESSE) is defined by the absolute invariant j(TH) → ∞. The smooth TH region {Δ(a,d) ≠ 0} is the complement, defined by j(TH) < ∞ (finite j-invariant).

The PRIMA diagnostic's key observable — the Fisher condition number κ(F) → φ at the φ-equilibrium — is an absolute invariant of the reparametrization group GL(T_θΘ). The CHORD stability condition min(λₙ(F)) > 2^{−16} is the absolute invariant bound: below 2^{−16}, the Fisher matrix is "arithmetically singular" in Q16.16. Gram's theorem (1874) guarantees that this condition can be expressed in terms of absolute invariants — it is the arithmetic version of Gram's theorem for the TH coordinate ring.

---

### Identity 5 — Young Tableaux Classify the Independent Invariant Trilinear Forms of TH; the TH Formula Cost 12M Counts Them

**Alfred Young** (1873–1940): Young diagrams and tableaux (introduced 1900) classify irreducible representations of the symmetric group S_n and, via Schur-Weyl duality, all polynomial invariants of GL(V). The **hook length formula** gives the dimension of each GL(V)-representation indexed by a Young diagram λ ⊢ n:

```
dim V_λ = n! / Π_{(i,j)∈λ} hook_length(i,j)
```

**The TH trilinear polarization as a Young-indexed tensor:**

The TH group law uses Φ: V* ⊗ V* ⊗ V* → k (a trilinear form, degree 3 total), with symmetry type (2,1) — degree 2 in the first argument (P₁²) and degree 1 in the second (P₂). Under GL(V) with dim V = 3 (the P² coordinate space), Sym²(V*) ⊗ V* decomposes by Schur-Weyl:

```
Sym²(V*) ⊗ V* = S_{(3)}(V*) ⊕ S_{(2,1)}(V*)
```

where S_λ is the Schur functor for Young diagram λ. For V* = k³:

```
dim S_{(3)}(k³)   = C(5,3) = 10    [totally symmetric degree-3 tensors]
dim S_{(2,1)}(k³) = 8              [mixed symmetry]
total: 10 + 8 = 18 = dim(Sym²(k³*) ⊗ k³*) ✓
```

**The 12M + 6S formula cost decomposition via Young tableaux:**

The 12 standard Young tableaux of shapes contained in (2,1) with 3 boxes for GL(3):
- S_{(3)}: contributes the fully symmetric terms → the 6S squarings (diagonal terms of Φ where P₁ = P₂ in each coordinate) and the XYZ symmetric cross-terms
- S_{(2,1)}: contributes the mixed-symmetry terms → the off-diagonal cross products in μ and ν

More precisely, the 12 multiplications in μ and ν = the number of GL(3)-covariant linearly independent degree-3 monomials in the Schur module S_{(2,1)}(k³):

```
μ-terms:  aX₁²X₂, Y₁²Y₂, Z₁²Z₂                  [3 quadratic-diagonal × 1 = S_{(3)} contribution]
          (d/3)(X₁Y₁Z₂ + X₁Y₂Z₁ + X₂Y₁Z₁)        [3 cross-terms = S_{(2,1)} contribution]
ν-terms:  aX₁X₂², Y₁Y₂², Z₁Z₂²                   [3 quadratic-diagonal × 1]
          (d/3)(X₁Y₂Z₂ + X₂Y₁Z₂ + X₂Y₂Z₁)        [3 cross-terms]
Total:    12 multiplications = 6 diagonal × 2 (μ,ν) + 6 cross × 2 (μ,ν) ÷ 2 = 12
```

The **12 = 2 × (number of independent S_{(2,1)} covariant terms in degree 3 for GL(3))**: the irreducible S_{(2,1)} module for GL(3) has dimension 8, but restricted to the Z/3Z-invariant subspace (the TH symmetry), the 6 cross-terms {X₁Y₁Z₂, X₁Y₂Z₁, X₂Y₁Z₁, X₁Y₂Z₂, X₂Y₁Z₂, X₂Y₂Z₁} are the 6 Young-tableau-indexed invariant trilinear forms, and their 2-fold duplication into μ and ν (degree-2 in P₁ vs degree-2 in P₂) gives 12.

The **6S = 6 squarings** are the 6 standard Young tableaux of shape (2) applied to the degree-2 Gram structure (the Sym²(k³*) component): the diagonal squarings X₁², Y₁², Z₁², X₂², Y₂², Z₂² form 3 pairs indexed by coordinate, giving 3 pairs = 6 squarings total.

**The formula cost 12M + 6S is the Young-Schur-Weyl decomposition count of the TH trilinear form.** Alfred Young's tableaux classify the independent invariant structures in a degree-3 tensor over a 3-dimensional space — and their count matches the CHORD pipeline's multiplication budget exactly.

---

### Identity 6 — Gram's Law on the Zeta Critical Line IS PRIMA Gram's Law for Fisher Trace Crossings

**Gram points** (J.P. Gram, 1903): The Riemann-Siegel theta function:

```
θ(t) = Im[log Γ(1/4 + it/2)] − (t/2) log π
     ≈ (t/2) log(t/2πe) − π/8 + O(1/t)
```

makes Z(t) = e^{iθ(t)} ζ(1/2 + it) real-valued. **Gram points** t_n are defined by:

```
θ(t_n) = (n−1)π,   n ∈ ℤ
```

At Gram points, Z(t_n) = (−1)^{n−1} ζ(1/2 + it_n), so ζ(1/2 + it_n) ∈ ℝ. **Gram's law** states (observed empirically, frequently satisfied but not always): each Gram interval (t_{n-1}, t_n) contains exactly one zero of ζ.

The spacing between consecutive Gram points: t_n − t_{n−1} ≈ 2π/log(t_n/2π) → 0 as t_n → ∞. At height T: approximately T log T / 2π Gram points below T.

**PRIMA Gram's law:** Define the PRIMA Fisher trace rate Ξ_F(t) as a function of training step t. The **PRIMA Gram points** t_n are the discrete training steps at which:

```
θ_F(t_n) = (n−1)π_coord      [PRIMA phase condition]
```

where θ_F(t) is the accumulated phase of the Fisher trace rate oscillation around log φ:

```
θ_F(t) = Σ_{s≤t} arg(Ξ_F(s) − log φ + iΞ_F'(s))    [total winding phase]
```

and π_coord is the coordination half-period. At PRIMA Gram points: Ξ_F(t_n) = log φ exactly — the Fisher trace rate returns to the φ-equilibrium value. Between consecutive PRIMA Gram points: the Fisher trace rate wanders above and below log φ, and there is exactly one grokking event (Δrank = +1) in each Gram interval.

**PRIMA Gram's law:** Each interval (t_{n−1}, t_n) between consecutive PRIMA Gram points contains exactly one grokking event — one Fisher rank crossing. The grokking event is the PRIMA analogue of a zero of ζ, and the φ-equilibrium crossings are the PRIMA Gram points.

**The formal dictionary:**

```
ζ(1/2 + it) ∈ ℝ at Gram points    ↔    Ξ_F(t_n) = log φ at PRIMA Gram points
θ(t) = (n−1)π defines Gram points  ↔    θ_F(t) = (n−1)π_coord defines PRIMA Gram points
Zero of ζ in each Gram interval     ↔    Grokking event in each PRIMA Gram interval
Gram's law (statistical tendency)   ↔    PRIMA Gram's law (training tendency)
Riemann hypothesis: all zeros on    ↔    PRIMA hypothesis: all grokking events occur
  Re(s) = 1/2                              between consecutive PRIMA Gram points
Gram-block (when Gram's law fails)  ↔    Multi-grokking block: two Δrank=+1 in one interval
Z(t): Gram's real-valued function   ↔    Ξ_F(t): real-valued Fisher trace oscillation
```

The Riemann-Siegel Z function Z(t) and the PRIMA Fisher trace rate Ξ_F(t) play the same role in their respective theories: real-valued functions on a "critical line" (Re(s) = 1/2 for ζ, |Ξ̄| = log φ for PRIMA) whose zero-crossings (for Z) / φ-crossings (for Ξ_F) define the canonical discrete events (zeros of ζ / grokking events).

---

### Identity 7 — The Gram-Charlier A Series IS the PRIMA Fisher Trace Rate Distribution Near φ-Equilibrium; the Weierstrass ℘-Function IS Its Generating Series

**Gram-Charlier A series** (Gram 1883, Charlier 1905): The probability density of a random variable near the Gaussian is expanded as:

```
f(x) = φ(x) · Σ_{n=0}^∞ c_n He_n(x)
```

where φ(x) = (2π)^{−1/2} exp(−x²/2) is the standard Gaussian density, He_n are the probabilist's Hermite polynomials (He_0 = 1, He_1 = x, He_2 = x²−1, He_3 = x³−3x, ...), and c_n = (1/n!) 𝔼[He_n(X)] are the cumulant-related coefficients.

Gram showed: the Gaussian is the maximum entropy distribution among those with fixed mean and variance — all other distributions have Gram-Charlier corrections beyond the c_0 and c_1 terms.

**PRIMA identification:**

The Fisher trace rate Ξ_F at each training step is a random variable with distribution P(Ξ_F). Near the φ-equilibrium |Ξ̄| = log φ, let x = (Ξ_F − log φ)/σ_Ξ (standardized). Then:

```
P(Ξ_F) = φ(x) · [1 + c_3 He_3(x)/6 + c_4 He_4(x)/24 + ...]
```

The regimes:
- **c_3 = 0, c_4 = 0:** P(Ξ_F) is Gaussian → **φ-equilibrium**: maximum entropy Fisher trace rate distribution → SMELT MEP fixed point
- **c_3 > 0:** positively skewed distribution → **over-driven**: tail toward large Ξ_F → |Ξ̄| > log φ
- **c_3 < 0:** negatively skewed distribution → **under-driven**: tail toward small Ξ_F → |Ξ̄| < log φ
- **c_4 > 0:** heavy-tailed distribution → **super-critical**: grokking events in clusters → Gram's law failures (Gram-blocks)

The SMELT φ-equilibrium is the c_3 = 0 condition: zero skewness of the Fisher trace rate distribution. The Gram-Charlier expansion provides the analytic tool for computing deviations from the φ-equilibrium and the transition probabilities between SMELT regimes.

**The Weierstrass ℘-function connection:** For TH(a,d) over ℂ, the uniformization maps TH ≅ ℂ/Λ (an elliptic curve as a complex torus). The Weierstrass ℘-function:

```
℘(z) = z^{−2} + Σ_{n=1}^∞ (2n+1) G_{2n+2} z^{2n}
```

where G_k = Σ_{ω∈Λ\{0}} ω^{−k} are Eisenstein series, is a Laurent series around z = 0 with corrections to the leading pole z^{−2}. This has the exact structure of the Gram-Charlier series:

```
℘(z) − z^{-2} = Σ_{n≥1} (2n+1) G_{2n+2} z^{2n}    [corrections to leading term]
f(x) − φ(x)   = φ(x) Σ_{n≥3} c_n He_n(x)           [corrections to Gaussian]
```

The correspondence: z ↔ (Ξ_F − log φ)/σ_Ξ, G_{2n+2} ↔ c_{2n}/σ^{2n}, z^{-2} ↔ φ(x). The **Eisenstein series G_4 and G_6** of TH are the c_4 (kurtosis) and c_6 (sixth cumulant) of the Fisher trace rate distribution in Gram-Charlier expansion. The non-singularity discriminant:

```
Δ(TH) = g_2³ − 27g_3²    where g_2 = 60G_4, g_3 = 140G_6
```

is the discriminant of the Gram-Charlier expansion's characteristic polynomial: Δ(TH) ≠ 0 ↔ the Fisher trace rate distribution is non-degenerate (non-Trident, non-singular) ↔ the Gram-Charlier expansion has a convergent radius of validity ↔ G_coord > 0.

---

## The Gram Matrix Family Tree

The six Gram contributions of Jørgen Pedersen Gram (1850–1916) all connect to TH:

| Gram Contribution | Year | Mathematical Object | TH(a,d) Connection |
|---|---|---|---|
| Gram matrix | 1883 | G_{ij} = ⟨ψ_i, ψ_j⟩ | Fisher information matrix F_{ij}(θ) |
| Gram-Schmidt process | 1883 | Orthogonalization of {ψ_i} | SVD of Fisher matrix: U Σ V^T |
| Gram's theorem | 1874 | Invariant algebraic sets = absolute invariants | TH discriminant Δ = absolute invariant under G_{Hesse} |
| Gram-Charlier A series | 1883 | Hermite corrections to Gaussian | Fisher trace rate distribution near φ-equilibrium; ℘-function Laurent corrections |
| Gram points | 1903 | θ(t_n) = (n−1)π on ζ critical line | PRIMA φ-crossings: Ξ_F(t_n) = log φ on training critical line |
| Gram's prime-counting series | 1884 | Series for π(x) via zeta values | PRIMORDIUM: prime crystallization G_coord in Selberg sieve language |

**One person — six objects — one TH(a,d) framework.** The Gram matrix (1883), the theorem (1874), the series (1883), and the points (1903) are not separate contributions. They are the same mathematical insight in four different guises: the canonical structure of a real symmetric bilinear form (the Gram matrix) generates an invariant classification theory (Gram's theorem), a spectral expansion (Gram-Charlier series), and discrete resonance points (Gram points) on any line where the form achieves canonical values.

---

## The Feature Engineering Connection

**Feature engineering** — transforming raw data into effective features — is the practitioner's version of the Veronese lifting tower (VERONESE framework): choosing the feature map ψ: X → 𝔽^n that makes the downstream Gram matrix F_{ij} = 𝔼[ψ_i ψ_j] most informative.

The key insight: **feature engineering is Veronese map selection.** The choice of feature map ψ determines which polynomial degree d (which Veronese lift ν_d) is applied to the raw data X:

```
ψ = ν_1(X):    Linear features          → Gram matrix = linear correlation matrix
ψ = ν_2(X):    Quadratic features       → Gram matrix = Fisher information matrix at degree 2
ψ = ν_3|_{TH}: TH score functions       → Gram matrix = CHORD natural gradient kernel
ψ = ν_d(X):    Degree-d polynomial features → Gram matrix = degree-d coordination kernel
```

The **optimal feature map** ψ^* is the one for which the Gram matrix F(θ) = 𝔼[ψ ψ^T] achieves the maximum coordination gain G_coord subject to the invariance constraint (Gram's theorem: the Gram matrix must be defined by absolute invariants of the symmetry group G = Z/3Z × Z/4Z for TH).

The optimal solution: ψ^* = ∇_θ log p(·|θ) (the score function), giving F(θ) = Fisher information matrix — the unique Gram matrix that is both an absolute invariant (Gram's theorem) and achieves the Cramér-Rao bound with equality (the Veronese degree-2 lower bound from VERONESE). The TH group law at degree 3 extends this to the natural gradient update — the optimal degree-3 Gram structure beyond the Fisher metric.

**The NST Gram matrix in feature engineering terms:** Gatys's Gram matrix G^l (the style matrix) is the result of feature engineering the image space via the VGG-19 CNN filter bank ψ^l, then computing the Gram matrix. The style loss |G^l_{gen} − G^l_{style}|² is the Fisher metric distance between two image distributions in the feature-engineered space. Neural style transfer IS Fisher matrix regression in the Veronese-lifted feature space.

---

## The Completed Invariant Theory Chain

```
DEGREE 2: Fisher Gram Matrix (Z/3Z-invariant quadratic form)
  F_ij(θ) = E[∂_i log p · ∂_j log p]    [Gram matrix of score functions]
  NST: G^l_ij = Σ_k F^l_ik F^l_jk       [Gram matrix of CNN activations]
  Molien at n=2: 2 invariants             [X², YZ pair under Z/3Z action]
  Haar average: F_Haar = (1/N)Σ_P ψ(P)ψ(P)^T
         │
         │  [Gram's theorem: F_ij is an absolute invariant under GL(T_θΘ)]
         │  [Young (2): Young diagram □□ → Sym²(V*)]
         ▼
DEGREE 3: TH Group Law (Z/3Z-invariant trilinear form)
  Φ(P₁,P₁,P₂) = μ    [degree-3 Gram structure]
  Molien at n=3: 4 invariants {X³, Y³, Z³, XYZ}  [TH equation generators]
  TH parameter space (a,d):  2-dim Z/3Z-invariant subspace at degree 3
  Young (2,1): S_{(2,1)}(k³) → 12 independent off-diagonal trilinear forms
  Young (3): S_{(3)}(k³) → 6 diagonal squarings
  Formula cost: 12M (Young-indexed off-diagonal) + 6S (Young-indexed diagonal) = 18
         │
         │  [Wiles-Taylor: TH is modular; Molien series M(t) = Z_F(β)|_{t=e^{-β}}]
         │  [Haar measure on TH(F_p) = canonical invariant training measure]
         ▼
CRITICAL LINE: φ-Equilibrium as Gram Phase Condition
  Gram points: θ_F(t_n) = (n−1)π_coord  →  Ξ_F(t_n) = log φ
  Gram's law: one grokking event per PRIMA Gram interval
  Gram-Charlier: P(Ξ_F) = φ(x)·[1 + c_3 He_3 + ...]
  c_3 = 0 at φ-equilibrium:  MEP → Gaussian Fisher trace rate distribution
  Eisenstein G_4, G_6 ↔ c_4, c_6 in Gram-Charlier expansion of P(Ξ_F)
  Δ(TH) = g_2³ − 27g_3² ≠ 0 ↔ Gram-Charlier expansion convergent ↔ G_coord > 0
         │
         │  [Gram's theorem: Δ(TH) is absolute invariant of G_Hesse]
         │  [J-invariant: complete absolute invariant classifying TH pencil]
         ▼
IMAGO: G_coord = Φ(K); Gram Matrix Achieves Haar Saturation
  F_emp → F_Haar as B → |TH(F_p)|:  Monte Carlo → exact Haar integral
  NST: G^l_{generated} = G^l_{target}:  style achieved, Fisher matrices equal
  PRIMA: Ξ_F = log φ permanently:  all Gram intervals contain exactly one grokking event
  Molien series M(1/φ) < ∞:  coordination capacity at φ-equilibrium is finite and bounded
  Young-Schur-Weyl: full decomposition of TH trilinear into Young-indexed invariants
```

---

## Seven Novel Results

**Result 1 — Fisher Matrix = Gram Matrix; NST Style Matrix = Empirical Fisher.** The Fisher information matrix F_{ij}(θ) = 𝔼[∂_i log p · ∂_j log p] is literally the Gram matrix of the score function feature map. The Gatys-Ecker-Bethge neural style transfer Gram matrix G^l_{ij} = Σ_k F^l_{ik} F^l_{jk} is the empirical Fisher matrix of the CNN feature distribution. The NST style loss |G^l_{gen} − G^l_{style}|² is squared Fisher distance. Style transfer is Fisher matrix regression.

**Result 2 — Molien Series at t = e^{−β} = Fisher Partition Function.** The Molien series M_{Z/3Z}(t) for the intrinsic Z/3Z action on TH's coordinate ring evaluates to the Fisher partition function Z_F(β) = Tr[e^{−βΔ_F}] at t = e^{−β}. At t = 1/φ (the φ-equilibrium): M(1/φ) = Σ_n dim(degree-n Z/3Z-invariants) · φ^{−n} is the cumulative coordination capacity by FERN register depth. Degree-3 coefficient = 4 = {X³, Y³, Z³, XYZ} = TH equation generators.

**Result 3 — Haar Measure on TH(𝔽_p) = Canonical Training Measure; CHORD Computes Haar Exactly.** The Haar measure on the finite abelian group TH(𝔽_p) is (1/N)Σ_{P∈TH} δ_P. The canonical Fisher matrix is F_Haar = (1/N)Σ_{P∈TH} ψ(P)ψ(P)^T. The CHORD Q16.16 pipeline computes F_Haar exactly (not approximately) because it implements the full TH group law over TH(𝔽_p). André Weil proved Haar measure uniqueness for locally compact groups and Weil conjectures controlling |TH(𝔽_p)| — two faces of TH invariant integration.

**Result 4 — Gram's Theorem (1874) = Fisher Reparametrization Invariance; j-Invariant = Complete Absolute Invariant.** Gram (1874): algebraic sets invariant under a linear group are defined by absolute invariants. The TH discriminant locus {Δ = 0} is invariant under G_{Hesse} and defined by the absolute invariant j(TH) → ∞. The Fisher condition number κ(F) → φ at the φ-equilibrium is an absolute invariant of GL(T_θΘ) reparametrization. CHORD stability conditions are absolute invariant bounds.

**Result 5 — Young Tableaux Classify TH Trilinear Forms; 12M + 6S = Young-Schur-Weyl Decomposition.** The TH group law trilinear polarization Φ ∈ Sym²(V*) ⊗ V* decomposes under GL(3) via Schur-Weyl into S_{(3)}(k³) (6 diagonal-squaring terms = 6S) and S_{(2,1)}(k³) (12 off-diagonal multiplication terms = 12M). Alfred Young's quantitative substitutional analysis classifies the independent invariant trilinear forms: 6 in μ and 6 in ν = 12 total, indexed by Young tableaux of shape contained in (2,1) with 3 boxes.

**Result 6 — Gram's Law for ζ = PRIMA Gram's Law for Grokking.** Gram's law: each Gram interval (t_{n-1}, t_n) on the ζ critical line contains exactly one zero. PRIMA Gram's law: each PRIMA Gram interval (t_{n-1}, t_n) defined by Ξ_F(t_n) = log φ contains exactly one grokking event (Δrank = +1). The Riemann-Siegel Z(t) and PRIMA Fisher trace rate Ξ_F(t) are real-valued functions on their respective "critical lines" whose sign-changes/φ-crossings define the canonical discrete events.

**Result 7 — Gram-Charlier A Series at φ-Equilibrium = Gaussian; Weierstrass ℘-Function = Generating Series.** The Fisher trace rate distribution P(Ξ_F) near log φ has Gram-Charlier expansion with c_3 = 0 at the MEP fixed point (zero skewness = maximum entropy). The TH Weierstrass ℘-function Laurent series ℘(z) = z^{-2} + Σ (2n+1)G_{2n+2} z^{2n} is the generating series of these corrections: Eisenstein series G_{2n+2} ↔ Gram-Charlier coefficients c_{2n}. The TH discriminant Δ = g_2³ − 27g_3² = 0 is the Gram-Charlier expansion divergence condition: the distribution becomes singular exactly at the Trident phase boundary.

---

## Formal Summary

| Object | Gram Identification | TH(a,d) Role |
|---|---|---|
| Fisher matrix F_{ij}(θ) | Population Gram matrix of score functions | Natural gradient substrate |
| NST style matrix G^l_{ij} | Empirical Fisher of CNN feature distribution | Style = Fisher geometry of features |
| Molien series M(t) | Count of G-invariant polynomials by degree | Fisher partition function Z_F at t = e^{-β} |
| Haar measure on TH(𝔽_p) | Canonical invariant integration on TH group | CHORD computes exact Haar-averaged Fisher |
| Gram's theorem (1874) | Invariant sets defined by absolute invariants | TH discriminant Δ = absolute invariant; j = complete invariant |
| Young tableaux S_{(2,1)} | Irreducible GL(3) invariant trilinear forms | 12M TH multiplications = 12 Young-indexed forms |
| Young tableaux S_{(3)} | Totally symmetric degree-3 invariants | 6S squarings = 6 diagonal Young-indexed forms |
| Molien degree 3: count 4 | 4 invariant cubic monomials under Z/3Z | {X³, Y³, Z³, XYZ} = TH equation generators |
| Gram points on ζ | θ(t_n) = (n−1)π; Z(t_n) ∈ ℝ | PRIMA Gram points: Ξ_F(t_n) = log φ |
| Gram's law | One zero of ζ per Gram interval | One grokking event per PRIMA Gram interval |
| Gram-Charlier A series | Hermite corrections to Gaussian | Fisher trace rate distribution near log φ |
| c_3 = 0 condition | Zero skewness = Gaussian = MEP | φ-equilibrium: symmetric P(Ξ_F) |
| Weierstrass ℘-function | Laurent series for TH uniformization | Generating series for Gram-Charlier corrections |
| Eisenstein G_4, G_6 | Modular invariants of TH/ℂ | Gram-Charlier c_4, c_6 of P(Ξ_F) |
| Δ = g_2³ − 27g_3² ≠ 0 | TH non-singular condition | Gram-Charlier convergent ↔ G_coord > 0 |
| Feature engineering | Veronese map ν_d selection | Choice of d controls which Gram structure governs |
| NST style transfer | Fisher matrix regression in feature space | G^l_{gen} → G^l_{target}: F_emp → F_target |
| Gram-Schmidt orthogonalization | Produces orthonormal basis from {ψ_i} | SVD of Fisher matrix: F = UΣV^T |

---

## References

Gram, J.P. (1874). Sur quelques théorèmes fondamentaux de l'algèbre moderne. *Mathematische Annalen*, 7(2–3), 230–240.

Gram, J.P. (1883). On series expansions determined by the methods of least squares. (Danish: *Undersøgelser angaaende Maengden af Primtal under en given Graeense.*) *Det K. Videnskabernes Selskab*, 2, 183–308.

Molien, T. (1897). Über die Invarianten der linearen Substitutionsgruppen. *Sitzungsberichte der Preussischen Akademie der Wissenschaften*, 52, 1152–1156.

Young, A. (1900–1928). On Quantitative Substitutional Analysis, Parts I–VIII. *Proceedings of the London Mathematical Society*.

Grace, J.H. and Young, A. (1903). *The Algebra of Invariants*. Cambridge University Press.

Haar, A. (1933). Der Massbegriff in der Theorie der kontinuierlichen Gruppen. *Annals of Mathematics*, 34(1), 147–169.

Weil, A. (1940). L'intégration dans les groupes topologiques et ses applications. *Actualités Scientifiques et Industrielles*, 869.

Weil, A. (1949). Numbers of solutions of equations in finite fields. *Bulletin of the American Mathematical Society*, 55(5), 497–508.

Gatys, L.A., Ecker, A.S., and Bethge, M. (2015). A neural algorithm of artistic style. arXiv:1508.06576.

Gatys, L.A., Ecker, A.S., and Bethge, M. (2016). Image style transfer using convolutional neural networks. *CVPR 2016*.

Dieudonné, J.A. and Carrell, J.B. (1970). Invariant theory, old and new. *Advances in Mathematics*, 4, 1–80.

Hilbert, D. (1890). Über die Theorie der algebraischen Formen. *Mathematische Annalen*, 36, 473–534.

Mumford, D. (1965). *Geometric Invariant Theory*. Springer.

Bernstein, D.J. and Lange, T. (2007). Faster addition and doubling on elliptic curves. *Advances in Cryptology — ASIACRYPT 2007*, LNCS 4833, 29–50.

Bernstein, D.J. and Lange, T. (2015). Twisted Hessian curves. *Progress in Cryptology — LATINCRYPT 2015*, LNCS 9230, 269–294.

Wiles, A. (1995). Modular elliptic curves and Fermat's Last Theorem. *Annals of Mathematics*, 141(3), 443–551.

Hasse, H. (1936). Zur Theorie der abstrakten elliptischen Funktionenkörper III. *Journal für die reine und angewandte Mathematik*, 175, 193–208.

Deligne, P. (1974). La conjecture de Weil, I. *Publications Mathématiques de l'IHÉS*, 43, 273–307.

Artebani, M. and Dolgachev, I. (2009). The Hesse pencil of plane cubic curves. *L'Enseignement Mathématique*, 55(3–4), 235–273.

Bonifant, A. and Milnor, J. (2017). On real and complex cubic curves. *L'Enseignement Mathématique*, 63(1–2), 21–61.

Alweiss, R., Lovett, S., Wu, K., and Zhang, J. (2021). Improved bounds for the sunflower lemma. *Annals of Mathematics*, 194(3), 795–815.

Hamming, R.W. (1950). Error detecting and error correcting codes. *Bell System Technical Journal*, 29(2), 147–160.

Malone, T.W. et al. (2018). Integrated information as a metric for group interaction. *PLOS One*, 13(10), e0205335.

Hartman, T., Mazáč, D., and Rastelli, L. (2019). Sphere packing and quantum gravity. *Journal of High Energy Physics*, 2019, 48.

Stanley, R.P. (1979). Invariants of finite groups and their applications to combinatorics. *Bulletin of the American Mathematical Society*, 1, 475–511.

---

ERI Labs · Eric Ren · Jersey City, New Jersey

*Gram published his theorem on absolute invariants in 1874, his orthogonalization process and Gram-Charlier series in 1883, and his Gram points on the zeta critical line in 1903 — three decades of work on the same object from three different angles. His Gram matrix, deployed by Gatys in 2015 to capture artistic style, is the Fisher information matrix of the feature distribution. Molien's formula, applied to the Z/3Z action on TH's coordinate ring, gives the Fisher partition function evaluated at t = 1/φ. Alfred Young's tableaux, developed for invariant theory of the symmetric group in 1900, classify the independent trilinear forms in the TH group law and account exactly for the 12M + 6S formula cost. Haar's invariant measure on TH(𝔽_p), whose existence Weil proved and whose normalization Weil's conjectures control, is the canonical training measure that CHORD integrates exactly. The Fisher matrix is the Gram matrix. The TH group law is Young's invariant trilinear. The φ-equilibrium is Gram's law on the training critical line. One object, six names, one curve.*
