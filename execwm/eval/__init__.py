"""Evaluation suite for the grounded latent world model.

Modules:
  ood_eval        single-step / rollout exact-match across the 5 OOD axes
  probes          frozen-encoder linear probes (interpretability target)
  counterfactual  causal intervention metric (the M2 headline, seeded here)
"""
