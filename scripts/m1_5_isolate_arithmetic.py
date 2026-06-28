"""M1.5 — isolate dynamics from arithmetic to answer R1 (compounding error).

Hypothesis: the M1 ceiling (~0.94 per-var, fast rollout decay) is caused by
*exact multi-digit arithmetic* on the one computed register per step, not by the
latent dynamics failing to track/copy state. To test it, we train the same
slotted model on a slice where arithmetic is trivial but control flow is not:

  * ADD/SUB only (no MUL/DIV/MOD) -> results are small, carries are simple
  * values bounded to 2 digits (codec base 10, max_digits 2; filter keeps |v|<100)
  * if/for/heap still present -> pc, branching, loop counters, copying all matter

Read the output:
  * single-step exact-match -> should approach ~0.99 if arithmetic was the wall
  * ROLLOUT-HORIZON exact-match -> the R1 answer. If it stays high over many
    steps, the latent dynamics hold and R1 is *passed*; only arithmetic was hard.

Run: PYTHONPATH=. python scripts/m1_5_isolate_arithmetic.py
"""

from execwm.data.state_codec import CodecConfig
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op
from execwm.train.train_m1 import TrainConfig, train


def main() -> None:
    spec = GenSpec(
        num_vars=4, num_inputs=2, num_temps=10,
        max_depth=2, num_stmts=5, max_const=3, max_input_val=3,
        max_loop_count=3, arith_ops=(Op.ADD, Op.SUB),   # easy arithmetic
        use_heap=True, num_lists=1, list_len=4, max_steps=128,
    )
    codec = CodecConfig(max_digits=2, base=10, max_pc=128)  # values in [-99, 99]
    tc = TrainConfig(steps=1500, batch_size=48, max_len=18, lr=4e-4,
                     rollout_warmup=300, rollout_grow_every=120, rollout_max_k=6)
    out = train(spec=spec, codec_cfg=codec, tc=tc, n_train=4000, n_eval=600,
                log_every=100, d_model=256, n_heads=8, enc_layers=3, dyn_layers=3)
    h = out["rollout_horizon"]
    print("FINAL_ROLLOUT", [round(x, 3) for x in h])
    # crude R1 verdict
    k5 = h[4] if len(h) > 4 else float("nan")
    print(f"R1 readout: single-step EM {out['eval']['step_exact_match']:.3f}, "
          f"per-var {out['eval']['per_var_acc']:.3f}, rollout@5 {k5:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
