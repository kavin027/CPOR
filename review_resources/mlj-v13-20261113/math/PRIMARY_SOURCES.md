# Original sources checked for the comparator and calibration revisions

| Publication | Specific support |
|---|---|
| [Zinkevich, ICML 2003](https://www.martin.zinkevich.org/publications/ICML03.pdf) | Section 2.2, Definitions 6–7 and Theorem 2 define the path length of a sequence of decisions and bound the dynamic regret of greedy projection for a comparator with bounded movement. |
| [Besbes, Gur and Zeevi, Operations Research 2015](https://doi.org/10.1287/opre.2015.1408) | Variation budget for changing loss functions in stochastic optimization. This is a restriction on the environment. |
| [Zhang, Lu and Zhou, NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/file/10a5ab2db37feedfdeaab192ead4ac0e-Paper.pdf) | Theorems 2–3 give matching lower and upper orders of general dynamic regret in terms of comparator path length under the paper's regularity assumptions. |
| [Gibbs and Candès, ACI extended paper](https://arxiv.org/pdf/2106.00170) | Appendix A explicitly derives the nominal miscoverage-level update from an online pinball-loss subgradient. The coverage error supplies the observed update signal. |
| [Amani, Alizadeh and Thrampoulidis, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/09a8a8976abcdfdee15128b4cc02f33a-Abstract.html) | The existing safe-linear-bandit text and bibliography use the same reference. |

The lower bound in the main paper is proved for CPOR's own every-prefix admissibility requirement in Supplementary Note S17. The cited dynamic-regret papers place the strength of the comparator in context.
