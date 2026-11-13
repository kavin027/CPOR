# Primary sources checked for Section 2.3

| Source | Statement checked |
| --- | --- |
| [Mahdavi, Jin and Yang, JMLR 2012](https://www.jmlr.org/papers/v13/mahdavi12a.html) | Long-term functional constraints may be violated in individual rounds. Regret and aggregated violation are bounded under the paper's regularity assumptions. |
| [Jenatton, Huang and Archambeau, ICML 2016](https://proceedings.mlr.press/v48/jenatton16.html) | The parameter trades cumulative regret against cumulative violation of long-term constraints. |
| [Yu and Neely, JMLR 2020](https://www.jmlr.org/papers/v21/16-494.html) | Assumptions 1–4 allow an order-one long-term violation bound with square-root regret. The constraint itself need not hold at every decision. |
| [Mannor, Tsitsiklis and Yu, JMLR 2009](https://www.jmlr.org/papers/v10/mannor09a.html) | The reward-in-hindsight benchmark for sample-path average constraints is generally unattainable. This is a different criterion from CPOR's every-prefix contract. |
| [Wu, Shariff, Lattimore and Szepesvári, ICML 2016](https://proceedings.mlr.press/v48/wu16.html) | In the adversarial setting the scalar reward lies in [0,1] and cumulative reward is compared with the default arm throughout. The stochastic setting separately assumes bounded means and sub-Gaussian noise. |
| [Amani, Alizadeh and Thrampoulidis, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/09a8a8976abcdfdee15128b4cc02f33a-Abstract.html) | The action has an unknown linear safety constraint and observations contain noise. This is stage-wise action safety. |
| [Chaudhary and Kalathil, AAAI 2022](https://doi.org/10.1609/aaai.v36i6.20566) | Multiple unknown linear constraints apply to the current action, with high probability. Assumption 1 bounds cost gradients. |
| [Gibbs and Candès, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | The guarantee is long-run coverage frequency of prediction sets, measured by a binary miscoverage indicator. |
| [Bernasconi de Luca et al., ECML PKDD 2021](https://re.public.polimi.it/handle/11311/1183563) | Conservative projection is a meta-algorithm for bounded convex losses preserving cumulative loss relative to a default and the base learner's regret order under the paper's assumptions. |

The manuscript's comparator obstruction is derived directly in Section 4.3. The chance experiment used in the proof has two bounded outcomes; the comparator's forecasts must still satisfy the full nonnegative-outcome envelope on its own trajectory. The result covers arbitrary hindsight sequences. It makes no impossibility claim about the best fixed mixture or a comparator with bounded path variation.
