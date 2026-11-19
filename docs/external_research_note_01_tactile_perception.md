# External Research Note 01 — Tactile Perception, Spatial Representation and Few-shot Adaptation

> Source: Bilibili lecture “脑子有话说 第5期｜机器为什么开始学着‘触摸’？” (BV1SLME6WE38), delivered by Zhu Lixin (Peking University School of Psychological and Cognitive Sciences, as introduced in the recording).
> Source material: user-provided Vidleaf AI-generated Chinese transcript, 00:00:01–00:29:42. The transcript contains speech-recognition errors; technical names and claims that may depend on exact wording require verification against the original slides/papers before formal citation.
> Purpose: external conceptual reference only. This note does not add data, labels, modalities, or experimental results to the STAG study.

## 1. Executive assessment

The lecture is useful to the thesis mainly as motivation for (i) treating tactile measurements as spatially organized observations rather than unrelated scalar channels, and (ii) studying how a learned representation may transfer or adapt when the contacted object/task changes.

It does **not** establish that a particular GNN architecture is superior, nor does it present Reptile/MAML/ANIL as the method used in the described demonstrations. The lecture's robot-control, optical tactile sensing, visual-tactile fusion, and simulation systems are outside the current STAG pressure-only scope.

The appropriate use is therefore conceptual: motivate testable questions, while relying exclusively on our own STAG experiments for empirical conclusions.

## 2. Lecture content, organized by its own narrative

### 2.1 Why touch matters (approximately 00:02–00:05)

The speaker begins with an example in which disrupting touch and part of motor ability makes an otherwise familiar action (striking a match) much harder despite visual information remaining available. The lecture emphasizes that human touch includes multiple signals such as normal force, tangential force, and temperature, and argues that touch is important for robotic manipulation.

**Relevance:** motivates the importance of tactile information. It does not directly establish statistical dependence between STAG sensor channels.

### 2.2 Tactile sensor development (approximately 00:05–00:11)

The lecture reviews the group's development from early piezoelectric glove sensors and capacitive sensing toward optical/vision-based tactile sensors. The described optical sensor uses a camera to observe deformation of an elastic membrane and reconstruct contact-surface information.

**Relevance:** sensor design, spatial resolution, and physical layout affect what can be measured. This is background motivation for treating sensor coordinates and spatial topology as meaningful metadata. These optical devices are not the same as STAG pressure-only measurements.

### 2.3 Touch resolves ambiguity during interaction (approximately 00:11–00:16)

A cabinet may look visually similar while admitting different ways of opening. The lecture argues that visual input alone may be ambiguous, while contact feedback can reveal the object's constraints. It describes a feedback loop: establish contact, move, detect tactile deviation, recover, and maintain a stable contact state. It also discusses following irregular trajectories.

**Relevance:** tactile signals can encode information about interaction state and constraints. For STAG, this motivates studying whether early pressure patterns contain information about later sensor-pair relationships. Our current target remains a statistical future Pearson correlation, not a direct estimate of physical constraints or causal interaction.

### 2.4 Multi-sensor integration and dexterous grasping (approximately 00:16–00:20)

The lecture describes tactile sensors integrated into a robotic hand, the importance of retaining individual sensor quality after integration, object displacement during grasping, and replanning based on tactile feedback. It also discusses continuous representations of grasp strategies rather than relying only on manually specified categories.

**Relevance:** motivates investigating sensor coordination, spatial representation, and object heterogeneity. The lecture does not specify our k-NN graph, message-passing architecture, or pairwise Pearson target.

### 2.5 Simulation and scaling (approximately 00:20–00:25)

The lecture discusses coupled soft-/rigid-body simulation, parallel simulation, and the simulation-to-real gap.

**Relevance:** broad context for generalization across conditions. The simulation stack itself is not relevant to our present pressure-only statistical study and should not be implemented.

### 2.6 Visual-tactile sensing and learning from demonstrations (approximately 00:25–00:30)

The final section describes a sensor intended to retain both tactile sensing and visual access, followed by robot learning from human demonstrations. The speaker gives an example involving roughly half an hour of training and around 100 human-view demonstrations for some tasks.

**Relevance:** illustrates the practical value of data-efficient transfer/adaptation. It is not evidence that Reptile, MAML, or ANIL was used, and the demonstration count is not directly comparable to our Support Recording count.

## 3. Mapping lecture ideas to the thesis

| Lecture idea | Thesis interpretation | Status |
|---|---|---|
| Tactile information matters for interaction | Motivation for studying pressure-derived relationships | Background only |
| Sensor layout and tactile spatial detail | Motivation for explicit spatial coordinates/topology | Tested empirically in G0–G2 |
| Multi-sensor integration | Motivation for sensor-pair relation prediction | Central task framing |
| Object/interaction differences | Motivation for object-level validation and task heterogeneity analysis | Already part of protocol |
| Learning from limited demonstrations | Motivation for few-shot adaptation across object tasks | Tested with Reptile/ANIL; no reliable gain so far |
| Visual-tactile fusion | Outside present modality scope | Excluded |
| Robot control and simulation | Outside present task scope | Excluded |

## 4. Implications for GNN experiments

The lecture supports a *motivation* for considering spatial structure, but it does not prove GNN superiority. Our G0–G2 evidence must be interpreted independently:

- G0: the 548 active sensors form a connected spatial k-NN graph; the legacy random 96-node subset yields a much coarser neighborhood scale.
- G1: on the legacy random 96-node subset, Node-MLP was slightly better than the tested Spatial-GNN variants in a single exploratory 5-fold repeat.
- G2: on all 548 nodes, the k=6 Spatial-GNN showed a small MAE improvement over matched Node-MLP (0.122150 vs 0.122504; approximately 0.29%), while Spearman was nearly unchanged. A degree-preserving rewired graph performed worse (MAE 0.123158). The comparison used three split seeds and 15 outer folds, but only five training epochs; convergence was not established and the same 26 objects were reused across repeats.

**Interpretation:** retain GNN as a candidate, not as an established winning method. Complete k=4/k=8 sensitivity, training-budget/convergence analysis, and multiple independent rewiring controls before deciding whether the spatial effect is robust enough to justify integration with meta-learning.

## 5. Implications for meta-learning

The lecture's demonstrations motivate data-efficient adaptation at a conceptual level. In our study, the analogous task definition is one physical object = one meta-task, with recordings from that object split into Support and Query.

However, our own experiments currently do not show reliable meta-learning gains:

- Reptile 200-iteration × 3-seed validation did not outperform matched ordinary MLP on MAE for the four Meta-validation objects.
- Reptile adaptation-step and conservative learning-rate diagnostics did not reveal a robust adaptation gain.
- First-order ANIL trained with Query-aware outer updates, but did not outperform ordinary pooled MLP on validation MAE; head-only adaptation also did not show stable benefit.
- The current 26-object collection is small at the independent-task level. Repeated episodes or sensor pairs do not create additional independent object tasks.
- The existing 18/4/4 split and repeated object-level CV have already informed research decisions; the current validation/test subsets must not be described as untouched confirmatory data.

**Interpretation:** keep meta-learning as a research question required by the thesis plan, but do not claim it works. The negative/boundary results are important evidence. If additional independent pressure-only object tasks cannot be obtained, the thesis must state the limited task count and exploratory nature of its conclusions.

## 6. Integrated research questions

The thesis can be organized around two linked but distinct questions:

**RQ1 — Spatial representation:** Does explicitly modeling the physical spatial organization of pressure sensors provide incremental information for predicting future sensor-pair correlation, beyond a matched non-graph node model?

**RQ2 — Cross-object adaptation:** Does a task-level learned initialization improve few-shot adaptation to an unseen object, compared with an ordinary model trained and adapted under a matched data/compute protocol?

A broader synthesis question is:

> Can spatially structured pressure representations improve cross-object generalization and few-shot adaptation?

This is a research question, not a result or an assumed conclusion.

## 7. Updated execution plan

### Priority 1 — Finish G2 robustness before combining GNN with meta-learning

1. Full-548 k=4 and k=8 sensitivity using the same three object split seeds.
2. Increase training budget and use training-only inner validation to assess convergence and select epochs.
3. Generate multiple independent degree-preserving rewired graphs per split; report variability across graph realizations.
4. Keep Node-MLP as the matched control and compare at the object level.
5. Decide whether the spatial graph effect is stable, small-but-credible, or unsupported.

### Priority 2 — Resolve the meta-learning evidence boundary

1. Do not continue broad Reptile/ANIL tuning on the same four validation objects.
2. Audit whether additional STAG pressure-only recordings can provide genuinely new independent object tasks. The previously audited blindfolded, weight, and hand-pose packages do not add new physical object identities; the weight package may add repeated grasp episodes but must not be counted as new object tasks.
3. Continue public pressure-only dataset assessment only where object identity, sensor schema, permissions, and task compatibility can be verified.
4. If new independent object tasks are unavailable, keep Reptile/ANIL results exploratory and report the absence of reliable gains honestly.

### Priority 3 — Conditional spatial/meta-learning integration

Only if Priority 1 supports a stable graph contribution, compare:
- Node-MLP + ordinary adaptation;
- Spatial-GNN + ordinary adaptation;
- Node-MLP + meta-learning;
- Spatial-GNN + meta-learning.

Use matched task splits, Support/Query recordings, initialization controls, optimization budgets, and object-level inference. Avoid a large factorial search with only 26 objects. Start with a small pre-specified comparison and state the limited independent-task count.

## 8. Scope and reporting rules

- Keep STAG pressure-only as the core dataset. Do not introduce video, images, visual labels, robot control, or simulation into the experiments.
- The lecture may be cited as background only if a stable public bibliographic record is verified; prefer citing the underlying peer-reviewed tactile-sensing papers for technical claims.
- Do not describe the lecture as evidence for GNN, Reptile, ANIL, or MAML performance.
- Do not treat sensor pairs, frames, or episodes as independent objects.
- Report MAE/RMSE/Spearman separately; ranking gains do not imply absolute-error gains.
- Keep all conclusions bounded to the observed objects and data protocol.

## 9. Source transcript anchors

- Touch and motor disruption example; tactile modalities: 00:02–00:05.
- Sensor development and optical tactile sensing: 00:05–00:11.
- Visual ambiguity, tactile feedback and contact recovery: 00:11–00:16.
- Integrated tactile hand, object displacement and grasp strategies: 00:16–00:20.
- Simulation and scaling: 00:20–00:25.
- Visual-tactile sensor and learning from demonstrations: 00:25–00:30.

The source transcript is AI-generated and may contain recognition errors. Verify names, exact paper titles, quantitative claims, and technical terms against primary publications before using them in the thesis bibliography.
