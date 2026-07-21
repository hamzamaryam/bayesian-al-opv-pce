# Benchmarking Molecular Representations in Bayesian Active Learning for Accelerated Discovery of High-Efficiency Organic Solar Cells

---

## Abstract

Identifying high-efficiency donor–acceptor material combinations for organic solar cells (OSCs) from the vast space of π-conjugated organic semiconductors is a central challenge in next-generation photovoltaics. Bayesian active learning (AL) — which couples a probabilistic surrogate model with an information-theoretic acquisition function to guide sequential experimentation — offers a principled strategy for accelerating this search. However, the molecular representation fed to the surrogate model is a critical design choice that remains systematically unstudied in the OSC context. This paper benchmarks eight handcrafted molecular representation schemes (RDKit descriptors, Mordred descriptors, MACCS keys, Avalon fingerprints, Morgan circular fingerprints, atom-pair fingerprints, path-based fingerprints, and extended fingerprints) alongside a graph convolutional network (GCN) baseline within a standardized Bayesian AL framework applied to a curated dataset of 1,481 NFA-based OSC molecules with experimentally measured power conversion efficiencies (PCE) spanning 0.01%–18.38%. Each representation is coupled to a Gaussian process regressor (GPR) surrogate and the Expected Improvement (EI) acquisition function; the discovery loop is run for 100 iterations from a seed set of 10 molecules, repeated over five independent random seeds. Avalon fingerprints, Extended fingerprints, and Morgan circular fingerprints emerge as the top-performing representations, recovering 21.4%, 21.0%, and 20.8% of the top-10% PCE pool in 100 iterations — a 2.71×, 2.66×, and 2.63× improvement over random selection, respectively. The GCN and RDKit descriptor baselines perform least effectively, underscoring that learned representations do not automatically outperform well-chosen handcrafted ones in low-data AL regimes. These findings provide the first controlled, multi-representation benchmarking study for AL-driven OSC discovery and offer actionable guidance for practitioners designing data-efficient material search campaigns.

---

## 1. Introduction

The global transition toward renewable energy has placed organic solar cells (OSCs) at the forefront of next-generation photovoltaic research. Unlike conventional silicon-based devices, OSCs offer the advantages of low-cost fabrication, mechanical flexibility, tunable optical absorption, and compatibility with roll-to-roll printing processes. Recent advances in non-fullerene acceptor (NFA) chemistry have pushed laboratory-scale power conversion efficiencies (PCE) beyond 19%, positioning OSCs as a credible competitor to established inorganic technologies [1]. However, realizing this potential at scale requires identifying donor–acceptor (D:A) material combinations that consistently deliver high PCE across diverse device architectures and fabrication conditions.

The fundamental challenge in OSC material discovery lies in the sheer scale of accessible chemical space. The structural flexibility of π-conjugated organic semiconductors generates a combinatorial explosion of candidate molecules, where both donor and acceptor components can be independently modified through substitution patterns, side-chain engineering, and core unit variation [2]. Exhaustive experimental synthesis and characterization of this space is economically and temporally prohibitive; even high-throughput robotic platforms can evaluate only a small fraction of viable candidates within a reasonable timeline [3]. This bottleneck has motivated the application of data-driven and computational methods to guide material discovery more intelligently.

Machine learning (ML) has emerged as a powerful tool for establishing quantitative structure–property relationships in organic photovoltaics. By training regression models on curated databases of experimentally measured PCE values, researchers have demonstrated predictive models capable of ranking candidate molecules without the need for synthesis [4], [5]. Support vector regression, random forests, gradient boosting, and deep neural networks have all been applied to OSC PCE prediction, achieving R² values upward of 0.85 on held-out test sets in several studies [6]. However, a critical limitation of these passive learning approaches is their reliance on large, well-distributed training sets: models trained on randomly sampled subsets of chemical space tend to perform poorly on out-of-distribution, high-performing candidates that are precisely the molecules most valuable to discover [7].

Active learning (AL) addresses this limitation by treating material discovery as a sequential decision problem. Rather than training on a static dataset, an AL agent iteratively selects the most informative candidate for experimental or computational evaluation, incorporates the new measurement into its training set, and refines its surrogate model. When coupled with a probabilistic surrogate such as Gaussian process regression (GPR), which provides both a mean prediction and a calibrated uncertainty estimate, active learning can direct queries toward regions of chemical space where the predicted PCE is high or where uncertainty is large — enabling systematic exploration of under-sampled high-efficiency regions [7], [8].

A critical but underexplored design choice within this framework is the molecular representation fed to the surrogate model. Molecular fingerprints — fixed-length binary or count vectors encoding structural features — constitute the primary input for classical AL surrogates in cheminformatics. Different fingerprinting schemes encode complementary structural information: topology-based descriptors capture atom connectivity, pharmacophoric fingerprints encode functional group patterns, and physicochemical descriptors encode electronic and geometric properties. The choice of representation directly influences the smoothness of the learned property landscape, the quality of uncertainty estimates, and ultimately the rate at which the AL agent discovers high-PCE candidates. Yet, to our knowledge, no study has performed a systematic and controlled benchmarking of molecular representations within the same Bayesian AL loop specifically for OSC PCE prediction.

### 1.1 Research Objectives

This work addresses that gap with the following objectives:

1. To design and implement a standardized Bayesian active learning benchmark protocol for OSC PCE discovery that isolates the effect of molecular representation from surrogate model and acquisition function choices.
2. To quantitatively characterize discovery efficiency, surrogate calibration quality, and convergence speed across eight handcrafted molecular representations and a graph neural network baseline.
3. To provide practical, evidence-based guidance for researchers selecting molecular representations in AL-driven OSC material campaigns.

We benchmark eight molecular representation schemes — RDKit descriptors, Mordred descriptors, MACCS keys, Avalon fingerprints, Morgan circular fingerprints, atom-pair fingerprints (APF), path-based fingerprints, and extended connectivity fingerprints — within a standardized Bayesian AL framework applied to a curated dataset of 1,481 NFA-based OSC molecules with experimentally measured PCE values ranging from 0.01% to 18.38%. A graph neural network (GNN) baseline is additionally included to compare learned molecular representations against handcrafted ones. The principal contributions of this work are: (i) a rigorous AL evaluation protocol isolating the effect of molecular representation from surrogate model and acquisition function choices; (ii) a quantitative characterization of discovery efficiency, surrogate calibration, and convergence speed across all representations; and (iii) practical guidance for researchers selecting molecular representations for AL-driven OSC material campaigns.

---

## 2. Literature Review

### 2.1 Machine Learning for PCE Prediction in Organic Solar Cells

The application of machine learning to OSC performance prediction has grown substantially over the past decade. Early efforts focused on fullerene-based devices, where Lee demonstrated that gradient boosting and support vector regression could predict the efficiency of ternary OSC blends from molecular and morphological descriptors, identifying energy-level alignment as the dominant predictor of performance [5]. As NFA materials supplanted fullerene acceptors and PCE records advanced rapidly, ML studies adapted to reflect the new chemical landscape. Zhang et al. constructed a database of 397 D:A pairs and trained random forest and neural network models on RDKit molecular descriptors, achieving R² ≈ 0.87 and demonstrating virtual screening of new NFA candidates [9]. Khoussa et al. extended this analysis to 1,000 D:A pairs using multiple molecular fingerprint types and showed that Morgan circular fingerprints combined with gradient boosting consistently outperformed alternative representations on static train-test splits [10]. More recently, Zhang et al. introduced a framework combining ML-based virtual screening with substructure importance analysis to prioritize electron donors and NFAs, recovering experimentally validated high-PCE structures in silico [11]. Lv et al. proposed an integrated pipeline combining deep learning property predictors with symbolic regression to generate novel molecular structures satisfying target PCE constraints, achieving a model-predicted efficiency above 17% for several candidates [12]. Beyond molecule-level PCE prediction, ML has increasingly been applied to device- and morphology-level optimization of OSCs: Cui et al. employed interpretable ML to guide a synergistic additive strategy that simultaneously repaired block-copolymer active-layer morphology and improved both efficiency and stability [19], while Zhang et al. constructed dedicated databases for device architecture, transport layers, and active-layer processing and trained ensemble models to co-optimize all three subsystems for maximum device efficiency [20]. Across these studies, however, model training and evaluation follow a passive, fixed-dataset paradigm: representations are benchmarked solely on predictive accuracy over random test splits, with no consideration of their utility within an iterative discovery loop.

### 2.2 Molecular Representations for Organic Electronic Materials

The choice of molecular representation profoundly shapes the behavior of ML models in materials science. In the OSC domain, handcrafted fingerprints encode structural information at varying levels of abstraction. MACCS keys (166 bits) capture the presence of predefined pharmacophoric substructures, offering high interpretability but limited resolution for structurally similar NFA compounds [10]. Morgan (ECFP-class) fingerprints encode circular neighborhoods of each atom up to a defined radius, efficiently capturing local chemical environments relevant to charge-transfer interactions [9]. Mordred descriptors extend beyond topological encoding to include 1,827 physicochemical, electronic, and geometric descriptors, offering richer information at the cost of high dimensionality and correlation [11]. Avalon fingerprints, derived from the Avalon cheminformatics toolkit, provide a complementary substructure-based encoding suited to drug-like and conjugated organic molecules. Path-based and atom-pair fingerprints encode molecular topology from complementary perspectives and have been used successfully in charge transport property prediction [14].

Beyond handcrafted representations, learned molecular representations via graph neural networks have gained traction. GNNs operate directly on the molecular graph — treating atoms as nodes and bonds as edges — and learn hierarchical representations through message-passing operations. Greenstein et al. applied evolutionary algorithms with GNN scoring functions to evolve high-performing unfused NFA structures, demonstrating that end-to-end learned representations can capture structure–property relationships that handcrafted fingerprints miss [13]. Lee et al. further showed that structure-guided ML incorporating graph-based encodings improved efficiency prediction accuracy compared to descriptor-only baselines on an experimentally curated OSC dataset [17]. Within active learning specifically, the choice of representation has been shown to materially affect efficiency: Pomberger et al. compared calculated chemical descriptors against simple generic representations inside a closed-loop reaction optimization and found, counterintuitively, that tailored descriptors did not outperform generic ones [21] — a result that motivates systematic representation benchmarking rather than assuming that more sophisticated encodings are always superior. For graph-based surrogates, Li and Rangarajan proposed a diversity-maximizing AL strategy that constructs compact GNN training sets from feature vectors extracted at each iteration, outperforming random sampling on most molecular-property tasks [22], and Subedi et al. introduced a Bayesian geometric GNN paired with a 3D-geometry diversity-sampling criterion for active molecular-graph learning [23]. Wen et al. further showed that AL-driven adaptive subsampling of Morgan fingerprints can outperform both full-dataset training and sixteen state-of-the-art subsampling strategies, underscoring the interaction between fingerprint choice and acquisition strategy [24]. Despite this progress, a direct head-to-head comparison of handcrafted and learned representations within the same AL framework — controlling for surrogate model, acquisition function, and dataset — remains absent from the literature.

### 2.3 Active Learning and Bayesian Optimization for Materials Discovery

Bayesian optimization (BO) — a specific instantiation of AL that uses a probabilistic surrogate and an acquisition function to balance exploration and exploitation — has emerged as a principled framework for expensive optimization problems in chemistry and materials science [15]. The data efficiency of AL in chemical spaces is well established: Smith et al. demonstrated that query-by-committee active sampling reduces the training data needed for accurate ML potentials to a fraction of that required by random sampling [27], and Gubaev et al. showed that actively chosen training sets sharply reduce prediction errors for out-of-sample "outlier" molecules — precisely the high-performing candidates of greatest interest in discovery [28]. Graff et al. extended these ideas to pool-based virtual screening, recovering the vast majority of top-scoring ligands from libraries exceeding 10⁸ compounds while quantifying how surrogate architecture, acquisition function, and batch size govern optimization performance [25]. In a systematic low-data study, van Tilborg and Grisoni benchmarked six AL strategies against two deep-learning architectures and reported up to a sixfold improvement in hit discovery over non-iterative screening [26], while Yin et al. evaluated uncertainty-based AL for molecular property prediction and emphasized its role in generalizing to new structural classes [33], later extending the analysis to explicitly out-of-distribution settings [34]. Not all studies report uniform gains, however: Ghosh et al. found that, under optimally tuned Gaussian-process noise, AL matched but did not exceed random sampling for dataset compilation, though it delivered savings of up to 64% for targeted molecular searches [35] — a nuance that reinforces the need for task-specific benchmarking such as the present study. Reviews of active machine learning for chemical engineering similarly stress that algorithmic maturity has outpaced domain application [36]. Antono et al. demonstrated ML-guided molecular design for hole-conducting organic materials, using Gaussian process models to iteratively rank candidates and reduce the number of quantum chemical calculations required to identify top performers [15]. Abreha et al. established the VERDE database of excited-state properties for π-conjugated molecules, providing a benchmark resource enabling systematic evaluation of BO-based discovery workflows [16]. Active learning has also been applied directly to organic electronic materials adjacent to OSCs: Abroshan et al. used a multi-parameter AL workflow to prioritize hole-transporting materials for OLEDs from a large candidate library [29]; Du et al. coupled Bayesian optimization with a multi-objective genetic algorithm to identify organic electrode molecules with high energy density and small HOMO–LUMO gap after only two AL cycles [30]; and Zhu et al. applied AL to iteratively refine HOMO, LUMO, and band-gap predictions across an 11,224-compound library of optoelectronic organic nanomers [31]. In the OSC context, Du et al. deployed a robotic high-throughput platform coupled with GPR-based Bayesian optimization to identify optimal processing conditions and material combinations, demonstrating that AL-guided experiments recovered high-efficiency formulations with significantly fewer trials than grid search [8]. Almalki et al. likewise showed that an AL strategy outperforms conventional design-of-experiments approaches for optimizing non-fullerene OPV materials and processes under the small-dataset constraints typical of experimental photovoltaics [32]. Most directly relevant to the present work, Malhotra et al. applied active learning to the discovery of D:A combinations for efficient OSCs, showing that uncertainty-guided sample selection with a GPR surrogate accelerated the identification of high-PCE pairs compared to random selection, even when the initial training set contained as few as ten molecules [7]. However, like the other AL-for-OSC studies above [8], [32], their analysis used a single molecular representation and did not systematically evaluate whether alternative fingerprints would yield faster or more reliable discovery. The present work fills this gap by providing the first controlled, multi-representation benchmarking study within a standardized AL loop designed specifically for the OSC PCE discovery task.

### 2.4 Summary of Gaps in Prior Work

The three identified gaps that motivate the present study are summarized in Table 1.

---

**Table 1. Identified gaps in the existing literature and their insufficiencies.**

| Gap | Representative Works | Why Insufficient |
|-----|---------------------|-----------------|
| ML for OSC PCE prediction relies on passive, fixed-dataset paradigms with no iterative discovery evaluation | [5], [9], [10], [11], [12], [19], [20] | Models optimized for test-set accuracy do not necessarily identify top-performing candidates efficiently when deployed in a sequential discovery loop; static benchmarks cannot capture query efficiency or convergence speed |
| No controlled head-to-head comparison of handcrafted vs. learned representations within the same AL framework | [9], [10], [13], [17], [21], [22], [23] | Representation-effect studies in active ML exist for reaction optimization and graph-based property models [21], [22], but not for OSC PCE discovery; without controlling for surrogate model, acquisition function, and dataset, observed differences may be confounded by factors other than the representation, leaving no actionable guidance for AL practitioners |
| Prior AL studies for OSC discovery use a single molecular representation and do not evaluate representation choice | [7], [8], [32] | The effect of representation on AL efficiency, surrogate calibration, and discovery speed remains unknown; practitioners lack evidence-based criteria for choosing representations in new AL campaigns |

---

## 3. Methodology

### 3.1 Dataset

The dataset used in this study comprises 1,481 OSC molecules with experimentally reported PCE values assembled from published literature on NFA-based devices. Each entry consists of a SMILES string encoding the molecular structure and a scalar PCE value expressed as a percentage. PCE values span the range 0.01%–18.38%, with a mean of 8.68% and a standard deviation of 4.51%. A total of 370 molecules (25.0%) exceed 12% PCE, and 120 molecules (8.1%) exceed 15% PCE — representing the high-efficiency tail that an effective AL agent should preferentially recover. Dataset characteristics are summarized in Table 2.

---

**Table 2. Summary statistics of the OSC PCE dataset used in this study.**

| Property | Value |
|----------|-------|
| Total molecules | 1,481 |
| PCE range | 0.01% – 18.38% |
| Mean PCE | 8.68% |
| Standard deviation of PCE | 4.51% |
| Molecules with PCE > 12% (top-10% pool) | 370 (25.0%) |
| Molecules with PCE > 15% (top ~8%) | 120 (8.1%) |
| Molecular representation format | SMILES |
| Target property | Scalar PCE (%) |
| Seed set size (*n*₀) | 10 |
| AL iterations (*N*) | 100 |
| Independent random seeds | 5 |

---

**[Figure 1 placeholder]**
*Fig. 1. Distribution of experimentally measured PCE values across the 1,481-molecule dataset. The histogram illustrates the right-skewed distribution with a long high-efficiency tail (PCE > 15%), highlighting the challenge of discovering top-performing candidates from a predominantly mid-range pool.*

---

### 3.2 Molecular Representations

Eight molecular representation schemes are evaluated, spanning topology-based, substructure-based, and physicochemical descriptor categories. Table 3 provides a summary of all representations, their dimensionality, and encoding type.

---

**Table 3. Summary of molecular representations benchmarked in this study.**

| Representation | Encoding Type | Dimensionality | Key Structural Information Encoded |
|---------------|---------------|---------------|-------------------------------------|
| RDKit Descriptors | Physicochemical | 211 | Topological indices, fragment counts, logP, molecular weight, ring systems |
| Mordred Descriptors | Physicochemical | 1,827 (filtered) | 2D/3D electronic, geometric, constitutional, and topological properties |
| MACCS Keys | Binary substructure | 167 | Presence/absence of 166 predefined SMARTS pharmacophoric patterns |
| Avalon Fingerprints | Hashed substructure | 512 | Substructure patterns from the Avalon cheminformatics toolkit |
| Morgan Fingerprints | Circular (ECFP4) | 2,048 | Radius-2 circular atomic neighborhoods; local aromatic/heteroatom environments |
| Atom-Pair Fingerprints (APF) | Topological | 2,048 | Atom-pair types and topological distances across the molecular scaffold |
| Path-Based Fingerprints (PF) | Topological | 2,048 | All linear molecular paths up to a fixed path length |
| Extended Fingerprints | Circular (larger radius) | 2,048 | Broader circular neighborhoods capturing more extended molecular context |
| GCN (learned) | End-to-end learned | Latent (learned) | Message-passing on molecular graph with atom-type and bond-type features |

---

1. **RDKit Descriptors** (211 features): A curated set of two-dimensional molecular descriptors computed using the RDKit library, including fragment counts, topological indices, and physicochemical properties such as molecular weight and log P.

2. **Mordred Descriptors** (1,827 features): A comprehensive library of two-dimensional and three-dimensional molecular descriptors encoding topological, electronic, geometric, and constitutional properties. High-variance and near-constant features are removed by a variance threshold filter (σ² < 0.1), and highly correlated features (|r| > 0.80) are dropped prior to model training.

3. **MACCS Keys** (167 features): A fixed set of predefined SMARTS-based substructure keys encoding the presence or absence of pharmacophoric patterns, providing a compact and interpretable binary fingerprint.

4. **Avalon Fingerprints** (512 bits): A hashed fingerprint encoding substructure patterns from the Avalon cheminformatics toolkit, designed for structural diversity representation across conjugated organic molecules.

5. **Morgan Circular Fingerprints** (2,048 bits): Extended connectivity fingerprints (ECFP4-equivalent) encoding circular atom neighborhoods up to radius 2, computed with bit folding. Morgan fingerprints are particularly suited to capturing local aromatic and heteroatom environments prevalent in NFA structures.

6. **Atom-Pair Fingerprints (APF)**: Encode pairs of atoms and the topological distance between them, providing sensitivity to the spatial arrangement of functional groups across the molecular scaffold.

7. **Path-Based Fingerprints (PF)**: Enumerate all linear paths up to a defined length through the molecular graph, offering complementary topological coverage to circular fingerprints.

8. **Extended Fingerprints**: RDKit extended connectivity fingerprints with a larger radius, capturing broader molecular neighborhoods than standard Morgan fingerprints.

Additionally, a **Graph Convolutional Network (GCN)** baseline is included as a learned representation. Molecules are converted to PyTorch Geometric graph objects, with atoms represented as one-hot encoded element vectors (16 atom types observed in the dataset) and bonds represented by type-encoded edge features. The GCN employs two graph convolution layers followed by global mean pooling and two fully connected layers, trained end-to-end to predict PCE.

---

**[Figure 2 placeholder]**
*Fig. 2. Overview of the Bayesian active learning pipeline. Starting from a small random seed set, a GPR surrogate is trained on the current labeled pool. The Expected Improvement acquisition function selects the next candidate from the unlabeled pool. The selected molecule's true PCE label is added to the training set, and the cycle repeats for N iterations.*

---

### 3.3 Surrogate Model and Acquisition Function

A Gaussian Process Regressor (GPR) with a composite kernel — a constant kernel multiplied by a radial basis function (RBF) kernel plus a white noise kernel — serves as the primary probabilistic surrogate for all handcrafted fingerprint representations. GPR is selected for its ability to produce calibrated predictive uncertainty estimates, which are essential for principled acquisition function evaluation. Inputs are standardized (zero mean, unit variance) prior to fitting; for high-dimensional representations (Mordred, Morgan), principal component analysis (PCA) retaining 95% of explained variance is applied to reduce the effective input dimension and ensure computational tractability of GPR kernel matrix inversion. Hyperparameters of the kernel are optimized by maximizing the log marginal likelihood at each active learning iteration.

The **Expected Improvement (EI)** acquisition function is used throughout:

$$\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - f^* - \xi) \cdot \Phi(Z) + \sigma(\mathbf{x}) \cdot \phi(Z), \quad Z = \frac{\mu(\mathbf{x}) - f^* - \xi}{\sigma(\mathbf{x})}$$

where μ(**x**) and σ(**x**) are the GPR posterior mean and standard deviation, *f** is the best PCE observed so far, ξ = 0.01 is the exploration jitter, and Φ, φ are the standard normal CDF and PDF, respectively. EI is computed over the full unlabeled pool at each iteration, and the candidate with maximum EI is selected for labeling.

#### 3.3.1 Theoretical Basis

The GPR surrogate models the PCE function *f*(**x**) as a draw from a Gaussian process prior: *f*(**x**) ~ *GP*(0, *k*(**x**, **x**')), where the kernel *k* encodes the assumed smoothness of the property landscape. Given labeled observations **y** = [y₁, …, y_n]ᵀ, the GPR posterior is analytically tractable:

$$\mu(\mathbf{x}^*) = \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}$$
$$\sigma^2(\mathbf{x}^*) = k(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*$$

where **K** is the *n*×*n* kernel matrix of training points and **k***  is the kernel vector between the test point and all training points. The quality of these uncertainty estimates depends critically on the representational geometry of the input space: a representation that places structurally similar molecules close together will yield smoother, better-calibrated GPR posteriors, enabling more efficient EI-guided search.

#### 3.3.2 Surrogate Calibration Measurement

Surrogate calibration is quantified by the mean negative log predictive density (NLPD) evaluated on held-out molecules at each iteration:

$$\text{NLPD} = -\frac{1}{|\mathcal{U}|} \sum_{\mathbf{x} \in \mathcal{U}} \log p(y(\mathbf{x}) \mid \mathbf{X}_{\text{train}}, \mathbf{y}_{\text{train}})$$

where *p*(*y*(**x**) | ·) is the GPR predictive distribution evaluated at the true label *y*(**x**). Lower NLPD indicates better calibration. This metric captures both predictive accuracy (mean) and uncertainty quantification (variance): a surrogate that confidently predicts incorrectly incurs a large NLPD penalty.

### 3.4 Active Learning Protocol and Evaluation

The AL experiment is initialized with **n₀ = 10** randomly sampled seed molecules (fixed random seed across all representation runs to ensure comparability). At each iteration, the GPR surrogate is retrained on all currently labeled molecules, EI is evaluated over the unlabeled pool, the highest-EI candidate is selected and its true PCE label is revealed from the precomputed dataset (simulating oracle evaluation), and the labeled set is updated. This cycle is repeated for **N = 100 iterations**.

The primary evaluation metric is the **discovery curve**: the cumulative fraction of true top-10% PCE molecules (PCE > 12%, corresponding to 370 molecules) recovered as a function of the number of oracle calls. A secondary metric is the **surrogate calibration error**, measured by the mean normalized log predictive density over held-out molecules at each iteration. To account for the stochasticity of the random seed initialization, all experiments are repeated across **five independent random seeds**, and results are reported as mean ± standard deviation.

A uniform random baseline — which selects the next candidate at random rather than using EI — is included in all comparisons to quantify the efficiency gain attributable to each representation-surrogate combination.

The **area under the discovery curve (AUC)** is additionally computed as a single scalar summary of overall discovery efficiency across all 100 iterations, normalized by the total number of iterations.

---

## 4. Results

### 4.1 Discovery Efficiency Across Representations

Figure 3 presents the mean discovery curves — cumulative fraction of top-10% PCE molecules (PCE > 12%) recovered — for all nine representations and the random baseline across 100 active learning iterations, averaged over five independent seeds.

---

**[Figure 3 — insert discovery_curves.png]**
*Fig. 3. Mean discovery curves (± standard deviation shading) for all nine molecular representations and the random baseline. The y-axis shows the cumulative fraction of the top-10% PCE pool (PCE > 12%; 370 molecules) recovered as a function of oracle queries. All AL agents outperform random selection. The circular fingerprint family (Avalon, Extended, Morgan) consistently achieves the highest discovery rates, with clear separation from the RDKit descriptor and GCN baselines emerging after approximately 30 iterations.*

---

Table 4 presents the discovery rate at five checkpoint iterations (10, 25, 50, 75, and 100 oracle calls) for all representations and the random baseline, averaged across five seeds.

---

**Table 4. Mean cumulative discovery rate (% of top-10% PCE pool recovered) at checkpoint iterations. Values are means over five independent random seeds.**

| Representation | Iter 10 (%) | Iter 25 (%) | Iter 50 (%) | Iter 75 (%) | Iter 100 (%) |
|---------------|------------|------------|------------|------------|-------------|
| Avalon | 2.2 | 4.9 | 9.8 | 15.3 | **21.4** |
| Extended | 2.3 | 5.2 | **10.1** | **15.5** | 21.0 |
| Morgan | 2.1 | 4.6 | 9.7 | **15.5** | 20.8 |
| APF | 2.1 | 4.9 | 9.1 | 13.5 | 17.9 |
| MACCS | 1.9 | 4.2 | 7.0 | 11.1 | 16.2 |
| PF | 2.1 | 4.2 | 7.5 | 11.5 | 16.1 |
| Mordred | 1.9 | 4.1 | 7.0 | 10.0 | 15.1 |
| GCN | 1.2 | 2.7 | 5.3 | 9.7 | 14.3 |
| RDKit | 2.1 | 4.3 | 6.8 | 9.8 | 13.6 |
| **Random** | **1.1** | **2.4** | **3.9** | **6.1** | **7.9** |

---

All AL agents outperform the random baseline at every checkpoint. The performance gap widens progressively: at iteration 10 (after observing only 20 molecules in total), all representations have discovered only marginally more top-performers than random, reflecting the shared cold-start constraint. By iteration 50, however, Avalon, Extended, and Morgan have recovered 9.8–10.1% of the top-10% pool compared to 3.9% for random selection — a 2.5–2.6× advantage. By iteration 100, the top trio has recovered more than 20% of the high-efficiency pool while random selection achieves only 7.9%.

The circular fingerprint family (Avalon, Extended, Morgan) shows consistently superior performance from iteration 25 onward. The APF representation initially tracks closely with this group but falls behind after iteration 50, suggesting it enables early exploration of high-PCE regions but is less efficient at sustained systematic exploitation. MACCS, PF, Mordred, and RDKit occupy a middle tier with discovery rates of 13.6–16.2% at iteration 100. Notably, the GCN achieves only 14.3% — comparable to or worse than the mid-tier handcrafted representations — despite its greater representational capacity and end-to-end training objective.

### 4.2 Area Under the Discovery Curve and Speedup Factor

Table 5 summarizes the area under the discovery curve (AUC), the final discovery rate, and the speedup factor relative to random selection for all representations.

---

**Table 5. Comprehensive performance summary for all molecular representations. Discovery@100 is the mean cumulative fraction of the top-10% PCE pool recovered after 100 iterations. AUC is the normalized area under the mean discovery curve. Speedup is the ratio of AL discovery rate to random at 100 iterations. Final Cal. Error is the mean NLPD at iteration 100 (GPR representations only).**

| Representation | Discovery@100 (%) | ±SD (%) | AUC | AUC Gain vs. Random (%) | Speedup | Best PCE@100 (%) | ±SD |
|---------------|------------------|---------|-----|------------------------|---------|-----------------|-----|
| Avalon | 21.41 | 2.32 | 0.1019 | +147.2 | **2.71×** | **18.36** | 0.03 |
| Extended | 20.97 | 2.90 | **0.1034** | **+150.9** | 2.66× | 18.14 | 0.27 |
| Morgan | 20.76 | 1.71 | 0.1013 | +145.9 | 2.63× | 18.27 | 0.22 |
| APF | 17.95 | 6.35 | 0.0907 | +120.1 | 2.27× | 17.57 | 1.16 |
| MACCS | 16.22 | 5.43 | 0.0766 | +86.0 | 2.05× | 18.26 | 0.18 |
| PF | 16.05 | 4.87 | 0.0780 | +89.3 | 2.03× | 17.59 | 1.22 |
| Mordred | 15.08 | 3.16 | 0.0723 | +75.3 | 1.91× | 17.99 | 0.24 |
| GCN | 14.32 | 1.12 | 0.0625 | +51.6 | 1.82× | 18.10 | 0.29 |
| RDKit | 13.57 | 1.41 | 0.0694 | +68.4 | 1.72× | 17.90 | 0.25 |
| Random | 7.89 | — | 0.0412 | — | 1.00× | — | — |

---

By AUC, the ranking is: Extended (0.1034) > Avalon (0.1019) > Morgan (0.1013) > APF (0.0907) > PF (0.0780) > MACCS (0.0766) > Mordred (0.0723) > RDKit (0.0694) > GCN (0.0625). The top three representations are closely grouped, all approximately 2.5× the AUC of the random baseline (0.0412). The GCN is the weakest performer overall, with an AUC gain of only 51.6% over random — the smallest of all representations tested.

### 4.3 Best PCE Found

The best PCE found curve tracks the highest true PCE value observed in the labeled set as a function of oracle calls. This metric captures the AL agent's ability to locate a single globally near-optimal molecule, distinct from the discovery curve which measures systematic coverage of the top tier.

---

**[Figure 4 — insert best_pce_curve plot]**
*Fig. 4. Mean best PCE found curves (± standard deviation shading) across five seeds for all representations. The y-axis shows the highest PCE observed in the labeled pool at each iteration. Avalon achieves the highest final best PCE (18.36% ± 0.03%), approaching the dataset maximum of 18.38%, with the lowest variance across seeds. APF and PF exhibit the widest confidence intervals, reflecting poor convergence reliability.*

---

All representations locate molecules with PCE ≥ 15% within the first 10 iterations on average, consistent with the high fraction of molecules in this range serving as likely candidates in any initial random seed. However, representations diverge substantially in their ability to approach the dataset maximum of 18.38%. Avalon achieves a mean best PCE of 18.36% ± 0.03% after 100 iterations — essentially reaching the top of the dataset with extremely low variance across seeds, indicating highly consistent performance. Morgan (18.27% ± 0.22%) and MACCS (18.26% ± 0.18%) follow, with Extended (18.14% ± 0.27%) and GCN (18.10% ± 0.29%) close behind. The APF and PF representations achieve the lowest final best PCE values (17.57% ± 1.16% and 17.59% ± 1.22% respectively), with standard deviations more than four times larger than the top performers, revealing high sensitivity to the random initialization seed.

### 4.4 Surrogate Calibration

Surrogate calibration, measured by NLPD at each iteration, is available for all GPR-based representations. Figure 5 shows the calibration curves over the 100-iteration AL run.

---

**[Figure 5 — insert calibration_curves.png]**
*Fig. 5. Surrogate calibration curves (mean NLPD over five seeds) for all GPR-based representations. Lower NLPD indicates better-calibrated uncertainty estimates. MACCS and Morgan exhibit very large initial calibration errors due to sparse, high-dimensional binary inputs at small training set sizes, which resolve rapidly as the labeled pool grows. APF and RDKit maintain consistently low NLPD throughout, while all representations converge to similar calibration levels by iteration 100.*

---

**Table 6. GPR surrogate calibration error (mean NLPD). Lower values indicate better-calibrated predictive uncertainty. Initial error is measured at iteration 1; final error at iteration 100. Mean error is averaged over all 100 iterations. GCN does not use a GPR surrogate and calibration metrics are not applicable.**

| Representation | Initial NLPD | Final NLPD | Mean NLPD (all iters) |
|---------------|-------------|-----------|----------------------|
| APF | 3.07 | 2.79 | 2.89 |
| RDKit | 3.06 | 2.76 | 3.02 |
| PF | 3.71 | 2.76 | 3.02 |
| Mordred | 3.16 | 2.74 | 2.90 |
| Avalon | 3.96 | 2.66 | 3.21 |
| Extended | 27.06 | 2.65 | 4.13 |
| Morgan | 183.63 | 2.66 | 69.62 |
| MACCS | 1172.21 | 2.74 | 661.58 |
| GCN | N/A | N/A | N/A |

---

Two representations — MACCS (initial NLPD: 1,172) and Morgan (183.6) — exhibit catastrophically large initial calibration errors that resolve rapidly within the first 20–30 iterations. This behavior reflects the challenge of fitting a GPR kernel to high-dimensional sparse binary inputs with only 10 training points: the kernel matrix becomes poorly conditioned, leading to overconfident or near-singular uncertainty estimates. Despite this poor initialization, both Morgan and MACCS converge to calibration levels comparable to the other representations (NLPD ≈ 2.66–2.74) by iteration 50 and beyond. Extended fingerprints show a moderate version of the same phenomenon (initial NLPD: 27.1), also due to high dimensionality. Representations with lower intrinsic dimensionality — APF, RDKit, Mordred — maintain well-calibrated GPR posteriors from the outset (NLPD ≈ 3.1).

By iteration 100, all GPR surrogates converge to similar calibration levels (NLPD 2.65–2.79), with Extended (2.65), Avalon (2.66), and Morgan (2.66) achieving the best final calibration. This convergence suggests that, given sufficient data, the GPR kernel can adapt to most representation geometries; the key differentiator of top-performing representations is not ultimate calibration quality but the rate at which the surrogate becomes useful for guiding acquisition.

---

## 5. Discussion

### 5.1 Why Circular Fingerprints Outperform Other Representations

The consistent superiority of Avalon, Extended, and Morgan fingerprints over all other representations can be understood through the lens of GPR kernel smoothness and structure–property relationship encoding. For a GPR with an RBF kernel to be effective in guiding EI-based search, the input representation must satisfy two conditions: (1) structurally similar molecules (likely to have similar PCE) must be embedded close together in representation space, and (2) the representation must be dense enough to provide smooth kernel coverage of the chemical space.

Circular fingerprints (Morgan, Extended, Avalon) encode local aromatic and heteroatom environments that are directly relevant to the photophysical properties governing PCE — including intramolecular charge-transfer character, HOMO/LUMO levels, and exciton binding energy. Their fixed-length binary structure places related NFA scaffolds in consistently similar regions of the representation space, enabling the GPR kernel to learn a smooth correlation function that accurately reflects the underlying PCE landscape. Avalon fingerprints additionally benefit from an efficient 512-bit encoding that reduces the dimensionality without excessive PCA compression, preserving kernel conditioning while retaining structural information.

In contrast, the lower-performing representations suffer from specific structural limitations. RDKit descriptors, despite covering physicochemical properties directly related to photovoltaic performance, produce a heterogeneous feature space where distances conflate chemically meaningful variations (conjugation length, substitution pattern) with uninformative ones (molecular weight scaling). The GPR therefore cannot learn a smooth correlation function efficiently. Mordred descriptors, despite offering the richest feature set (1,827 descriptors before filtering), require aggressive PCA compression to 95% variance, which may discard structural information relevant to high-PCE outliers. MACCS keys, while interpretable, offer insufficient resolution for distinguishing among structurally diverse NFA compounds with very different PCE values.

### 5.2 Explanation of GCN Underperformance

The GCN's underperformance relative to top handcrafted fingerprints is a significant and somewhat counterintuitive finding. In principle, a GCN trained end-to-end on the PCE prediction task should learn a representation optimally tailored to the target property. However, in the AL setting, the GCN is retrained from scratch at each iteration on a small, sequentially growing labeled set (10–110 molecules). This low-data regime fundamentally limits the GCN's ability to learn a discriminative representation: with fewer than 100 training examples, the network is prone to overfitting and its latent representations are unstable across iterations. The resulting non-stationary representation geometry degrades GPR kernel fitting and acquisition function quality.

Furthermore, the GCN architecture used here — two graph convolutional layers with global mean pooling — may be insufficiently expressive for the highly fused, planar NFA scaffolds in this dataset. These molecules exhibit long-range conjugation effects that may require deeper message-passing or specialized architectures (e.g., attention-based GNNs) to capture effectively. The finding that GCN underperforms even the simple MACCS key representation (167 bits) underscores that raw representational capacity does not translate directly to AL efficiency in low-data regimes.

### 5.3 High Variance in APF and PF Performance

APF and PF exhibit notably large standard deviations in discovery rate (6.35% and 4.87%, respectively) and best PCE at 100 iterations (1.16% and 1.22%). While their mean performance places them in the middle tier, the high variance means that on some seeds they perform comparably to the top trio, while on others they fail to substantially outperform random selection. This behavior likely reflects sensitivity to the initial seed set: if the 10 seed molecules happen to include representatives of the high-PCE structural families that these topological fingerprints discriminate well, the AL agent converges rapidly; otherwise, early queries are poorly directed and the agent takes many iterations to course-correct.

This high variance is practically important: a practitioner using APF or PF for an AL campaign faces substantial risk of wasted oracle calls. The more consistent performance of Avalon (SD: 2.32%) and Morgan (SD: 1.71%) makes these representations safer choices when the number of oracle calls is strictly budgeted.

### 5.4 Implications for the Field and Practical Recommendations

This study provides several actionable recommendations for researchers designing AL campaigns for OSC discovery:

1. **Prefer Avalon or Morgan fingerprints** as the default representation for Bayesian AL with GPR surrogates. Both provide consistently high discovery efficiency with moderate variance. Avalon additionally produces more consistent best-PCE recovery (lowest standard deviation among all representations).

2. **Avoid RDKit descriptors and GCN** in low-data AL regimes (fewer than 100 oracle calls). Both underperform structurally, for different reasons: RDKit descriptors produce a heterogeneous feature space poorly suited to RBF kernel regression, while the GCN is data-hungry and unstable when the training set is small.

3. **Exercise caution with high-dimensional binary fingerprints** (Morgan, MACCS, Extended) in the very early iterations (< 20 oracle calls): the GPR may be poorly calibrated due to near-singular kernel matrices. This can be mitigated by increasing the initial seed set size or applying aggressive dimensionality reduction before fitting.

4. **Use AUC of the discovery curve rather than final discovery rate** as the primary performance metric when comparing representations, as it captures the full trajectory of discovery efficiency and penalizes representations that require many iterations to become useful.

### 5.5 Limitations and Future Work

Several limitations of this study should be acknowledged. First, the AL experiments are performed on a single curated dataset of 1,481 molecules; the relative ranking of representations may shift on datasets with different PCE distributions, structural diversity, or size. Future work should benchmark across multiple OSC datasets to assess the generalizability of these findings.

Second, the GPR surrogate with RBF kernel is used uniformly across all representations. Alternative kernels (e.g., Tanimoto kernels for binary fingerprints, graph kernels for molecular structures) may change the relative ranking of representations. A natural extension of this work is to co-optimize the kernel type with the representation.

Third, this study employs a single-molecule-at-a-time (sequential) AL protocol. Batch AL strategies — which select multiple candidates per iteration to enable parallelized experimental evaluation — are increasingly relevant for high-throughput laboratory settings and may interact differently with representation quality.

Finally, the GCN architecture used here is deliberately simple to serve as a baseline. Pre-trained molecular foundation models (e.g., ChemBERTa, Uni-Mol) or transfer learning approaches that leverage large unlabeled molecular databases may substantially improve the performance of learned representations in low-data AL regimes. Recent work integrating a BERT model pretrained on 1.26 million compounds into a Bayesian AL loop achieved equivalent hit identification with 50% fewer iterations by disentangling representation learning from uncertainty estimation [37], and Bayesian semi-supervised models have been shown to keep uncertainty estimates accurate in the low-data limit precisely by separating representation learning from regression [38]. Systematic comparison of such pretrained and semi-supervised representations within the present benchmark is a promising direction for overcoming the GCN's observed instability in the small-data regime.

---

## 6. Conclusion

This paper presents the first systematic and controlled benchmarking study of molecular representations within a Bayesian active learning framework for the discovery of high-efficiency non-fullerene acceptor organic solar cell materials. Using a curated dataset of 1,481 experimentally characterized OSC molecules and a standardized AL protocol — Gaussian process regression surrogate, Expected Improvement acquisition function, 10-molecule seed set, 100 iterations, five independent seeds — we quantify the discovery efficiency, surrogate calibration, and convergence stability of eight handcrafted molecular fingerprints and a graph convolutional network.

The principal findings are as follows:

1. **Circular fingerprints dominate**: Avalon (AUC: 0.1019, speedup: 2.71×), Extended (AUC: 0.1034, speedup: 2.66×), and Morgan (AUC: 0.1013, speedup: 2.63×) fingerprints emerge as the top-performing representations, all recovering more than 20% of the top-10% PCE pool in 100 oracle calls — versus 7.9% for random selection. Their shared structural feature is encoding of local aromatic and heteroatom environments directly relevant to photophysical mechanisms governing PCE.

2. **Avalon achieves the most consistent best-PCE recovery**: With a mean best PCE of 18.36% ± 0.03% after 100 iterations (dataset maximum: 18.38%), Avalon is both the most effective and the most reliable representation for locating globally near-optimal molecules.

3. **GCN does not outperform handcrafted fingerprints**: Despite greater representational capacity, the GCN achieves only 14.3% discovery rate and an AUC gain of 51.6% over random — inferior to all circular fingerprints and comparable to the weakest handcrafted baselines. This underscores the unsuitability of end-to-end learned representations for AL regimes where the training set is small and non-stationary.

4. **Poor surrogate calibration early in the loop** is a risk for high-dimensional binary representations (Morgan, MACCS, Extended), where near-singular GPR kernel matrices can temporarily destabilize acquisition function guidance. This risk resolves within 20–30 iterations as the training set grows.

5. **High variance is a practical risk**: APF and PF achieve moderate mean performance but very high variance across seeds, making them unsuitable for strictly budgeted AL campaigns despite competitive mean discovery rates.

These findings directly address the gap identified in Section 2: prior AL studies for OSC discovery have used a single molecular representation, and prior ML benchmarks have evaluated representations only on static train-test splits without considering iterative discovery efficiency. By providing a controlled, multi-representation comparison with rigorous statistical evaluation, this work enables evidence-based representation selection for AL-driven material discovery campaigns in organic photovoltaics and related fields.

The code, dataset, and fingerprint cache used in this study are made publicly available to facilitate reproduction and extension of these benchmarks.

---

## References

[1] K. Khoussa, L. Boubchir, and P. Lévêque, "Artificial intelligence in organic photovoltaics: Predicting power conversion efficiency from the molecular chemical structure of (donor/acceptor) pairs," *Engineering Reports*, vol. 7, no. 1, 2025, doi: 10.1002/eng2.70334.

[2] Y. Cui, R. Zhu, X. Liao, and Y. Chen, "Recent advances of computational chemistry in organic solar cell research," *J. Mater. Chem. C*, vol. 8, no. 47, pp. 16393–16408, 2020, doi: 10.1039/d0tc03709e.

[3] T. Wang, G. Kupgan, and J.-L. Brédas, "Organic photovoltaics: Relating chemical structure, local morphology, and electronic properties," *Trends in Chemistry*, vol. 2, no. 6, pp. 535–554, 2020, doi: 10.1016/j.trechm.2020.03.006.

[4] B. Das and A. Mondal, "Machine learning in non-fullerene organic solar cells: Accelerating discovery, design, and understanding," *ACS Omega*, 2026, doi: 10.1021/acsomega.6c01194.

[5] M.-H. Lee, "Insights from machine learning techniques for predicting the efficiency of fullerene derivatives-based ternary organic solar cells at ternary blend design," *Adv. Energy Mater.*, vol. 9, no. 26, p. 1900891, 2019, doi: 10.1002/aenm.201900891.

[6] T. W. David, H. Anizelli, T. J. Jacobsson, C. Gray, W. Teahan, and J. Kettle, "Enhancing the stability of organic photovoltaics through machine learning," *Nano Energy*, vol. 78, p. 105342, 2020, doi: 10.1016/j.nanoen.2020.105342.

[7] P. Malhotra, J. C. Verduzco, S. Biswas, and G. D. Sharma, "Active discovery of donor:acceptor combinations for efficient organic solar cells," *ACS Appl. Mater. Interfaces*, vol. 15, no. 3, pp. 3842–3851, 2022, doi: 10.1021/acsami.2c18540.

[8] X. Du *et al.*, "Elucidating the full potential of OPV materials utilizing a high-throughput robot-based platform and machine learning," *Joule*, vol. 5, no. 2, pp. 495–506, 2021, doi: 10.1016/j.joule.2020.12.013.

[9] C.-R. Zhang *et al.*, "Machine learning study on organic solar cells and virtual screening of designed non-fullerene acceptors," *J. Appl. Phys.*, vol. 134, no. 12, 2023, doi: 10.1063/5.0169284.

[10] K. Khoussa, P. Lévêque, and L. Boubchir, "On the use of machine learning to discover novel donor-acceptor pairs for organic photovoltaic devices," in *Proc. IEEE Int. Conf. Big Data*, 2024, doi: 10.1109/BigData62323.2024.10825948.

[11] C.-R. Zhang *et al.*, "High throughput molecular design of electron donors and non-fullerene acceptors using machine learning combined with substructure importance," *J. Mater. Chem. C*, vol. 13, 2025, doi: 10.1039/d5tc00931f.

[12] L.-F. Lv *et al.*, "Integrating deep learning and symbolic regression for molecular design and virtual screening of organic solar cells," *npj Comput. Mater.*, vol. 12, 2026, doi: 10.1038/s41524-025-01898-7.

[13] B. L. Greenstein, D. C. Hiener, and G. R. Hutchison, "Computational evolution of high-performing unfused non-fullerene acceptors for organic solar cells," *J. Chem. Phys.*, vol. 156, no. 20, 2022, doi: 10.1063/5.0087299.

[14] O. D. Abarbanel and G. R. Hutchison, "Machine learning to accelerate screening for Marcus reorganization energies," *J. Chem. Phys.*, vol. 155, no. 5, 2021, doi: 10.1063/5.0059682.

[15] E. Antono *et al.*, "Machine-learning guided quantum chemical and molecular dynamics calculations to design novel hole-conducting organic materials," *J. Phys. Chem. A*, vol. 124, no. 38, pp. 7651–7661, 2020, doi: 10.1021/acs.jpca.0c05769.

[16] B. G. Abreha *et al.*, "Virtual excited state reference for the discovery of electronic materials database: An open-access resource for ground and excited state properties of organic molecules," *J. Phys. Chem. Lett.*, vol. 10, no. 21, pp. 6835–6841, 2019, doi: 10.1021/acs.jpclett.9b02577.

[17] J. Lee *et al.*, "Structure-guided machine learning for efficiency prediction of organic photovoltaics using experimentally informed molecular descriptors," *Digital Discovery*, 2026, doi: 10.1039/d5dd00496a.

[18] C. Liu *et al.*, "Understanding causalities in organic photovoltaics device degradation in a machine-learning-driven high-throughput platform," *Adv. Mater.*, vol. 36, no. 7, p. 2300259, 2024, doi: 10.1002/adma.202300259.

[19] Y. Cui, Q. Fan, H. Feng, T. Li, D. Y. Paraschuk, W. Ma, and H. Yan, "Towards efficient and stable organic solar cells: fixing the morphology problem in block copolymer active layers with synergistic strategies supported by interpretable machine learning," *Energy Environ. Sci.*, vol. 17, no. 22, pp. 8954–8965, 2024, doi: 10.1039/d4ee03168g.

[20] Y. Zhang, H. Ma, and Y. Wen, "Concurrent optimization of device architecture, transport layers, and active layer for organic photovoltaics by machine learning," *J. Phys. Chem. Lett.*, vol. 17, no. 18, pp. 5395–5405, 2026, doi: 10.1021/acs.jpclett.6c00526.

[21] A. Pomberger, A. A. Pedrina McCarthy, A. Khan, S. Sung, C. J. Taylor, M. J. Gaunt, L. Colwell, D. Walz, and A. A. Lapkin, "The effect of chemical representation on active machine learning towards closed-loop optimization," *React. Chem. Eng.*, vol. 7, no. 6, pp. 1368–1379, 2022, doi: 10.1039/d2re00008c.

[22] B. Li and S. Rangarajan, "A diversity maximizing active learning strategy for graph neural network models of chemical properties," *Mol. Syst. Des. Eng.*, vol. 7, no. 12, pp. 1697–1706, 2022, doi: 10.1039/d2me00073c.

[23] R. Subedi, L. Wei, W. Gao, S. Chakraborty, and Y. Liu, "Empowering active learning for 3D molecular graphs with geometric graph isomorphism," in *Advances in Neural Information Processing Systems*, vol. 37, 2024.

[24] Y. Wen, Z. Li, Y. Xiang, and D. Reker, "Improving molecular machine learning through adaptive subsampling with active learning," *Digital Discovery*, vol. 2, no. 4, pp. 1134–1142, 2023, doi: 10.1039/d3dd00037k.

[25] D. E. Graff, E. I. Shakhnovich, and C. W. Coley, "Accelerating high-throughput virtual screening through molecular pool-based active learning," *Chem. Sci.*, vol. 12, no. 22, pp. 7866–7881, 2021, doi: 10.1039/d0sc06805e.

[26] D. van Tilborg and F. Grisoni, "Traversing chemical space with active deep learning for low-data drug discovery," *Nat. Comput. Sci.*, vol. 4, no. 10, pp. 786–796, 2024, doi: 10.1038/s43588-024-00697-2.

[27] J. S. Smith, B. Nebgen, N. Lubbers, O. Isayev, and A. E. Roitberg, "Less is more: Sampling chemical space with active learning," *J. Chem. Phys.*, vol. 148, no. 24, p. 241733, 2018, doi: 10.1063/1.5023802.

[28] K. Gubaev, E. V. Podryabinkin, and A. V. Shapeev, "Machine learning of molecular properties: Locality and active learning," *J. Chem. Phys.*, vol. 148, no. 24, p. 241727, 2018, doi: 10.1063/1.5005095.

[29] H. Abroshan, H. S. Kwak, Y. An, C. Brown, A. Chandrasekaran, P. Winget, and M. D. Halls, "Active learning accelerates design and optimization of hole-transporting materials for organic electronics," *Front. Chem.*, vol. 9, p. 800371, 2022, doi: 10.3389/fchem.2021.800371.

[30] J. Du, J. Guo, W. Liu, Z. Li, G. Huang, and X. Zhang, "A universal strategy of multi-objective active learning to accelerate the discovery of organic electrode molecules," *Sci. China Chem.*, vol. 67, no. 11, pp. 3681–3687, 2024, doi: 10.1007/s11426-024-2163-1.

[31] Q. Zhu *et al.*, "Generating a chemical database of organic nanomers and applying active learning to predict HOMO, LUMO and band gap: Accelerating optoelectronic nanopolymer materials discovery," *Talanta*, vol. 298, p. 128939, 2026, doi: 10.1016/j.talanta.2025.128939.

[32] M. Almalki, N. Lachiche, and Y. A. Chapuis, "Active learning concept for materials process optimization of nonfullerene organic photovoltaic using small datasets," in *Proc. SPIE*, vol. 13011, p. 1301106, 2024, doi: 10.1117/12.3016560.

[33] T. Yin, G. Panapitiya, E. D. Coda, and E. G. Saldanha, "Evaluating uncertainty-based active learning for accelerating the generalization of molecular property prediction," *J. Cheminform.*, vol. 15, no. 1, p. 105, 2023, doi: 10.1186/s13321-023-00753-5.

[34] T. Yin, P. Gao, G. Panapitiya, and E. G. Saldanha, "Out-of-distribution evaluation of active learning pipelines for molecular property prediction," *RSC Adv.*, vol. 16, no. 6, pp. 5281–5295, 2026, doi: 10.1039/d5ra08055j.

[35] K. Ghosh, M. Todorović, A. Vehtari, and P. Rinke, "Active learning of molecular data for task-specific objectives," *J. Chem. Phys.*, vol. 162, no. 1, p. 014103, 2025, doi: 10.1063/5.0229834.

[36] Y. Ureel, M. R. Dobbelaere, Y. Ouyang, K. De Ras, M. K. Sabbe, G. B. Marin, and K. M. Van Geem, "Active machine learning for chemical engineers: A bright future lies ahead!," *Engineering*, vol. 27, pp. 23–30, 2023, doi: 10.1016/j.eng.2023.02.019.

[37] M. A. Masood, S. Kaski, and T. Cui, "Molecular property prediction using pretrained-BERT and Bayesian active learning: a data-efficient approach to drug design," *J. Cheminform.*, vol. 17, no. 1, p. 58, 2025, doi: 10.1186/s13321-025-00986-6.

[38] Y. Zhang and A. A. Lee, "Bayesian semi-supervised learning for uncertainty-calibrated prediction of molecular properties and active learning," *Chem. Sci.*, vol. 10, no. 35, pp. 8154–8163, 2019, doi: 10.1039/c9sc00616h.
