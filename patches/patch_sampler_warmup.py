"""Apply the SGLANG_HCU_SAMPLER_WARMUP patch to sglang/srt/layers/sampler.py (re-runnable:
backs the pristine file up to sampler.py.bak_20261023 once and always patches that copy).
usage (inside glm5.3-pd): python3 patch_sampler_warmup.py"""
import os
import shutil

P = "/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/sampler.py"
BAK = P + ".bak_20261023"
if not os.path.exists(BAK):
    shutil.copy2(P, BAK)
src = open(BAK).read()  # always patch the pristine copy, so re-running updates the patch
assert "_hcu_warmup_sampling_kernels" not in src, "backup is not pristine"

FUNC = '''

# --- HCU patch (SGLANG_HCU_SAMPLER_WARMUP=1) ---------------------------------------------
# Server and bench warmups are greedy, so the T>0 sampling kernels (multinomial: max/min/sum/
# exponential_/div/argmax; top-p: sort/cumsum/masked fill/gather) were first launched -- and
# their HIP code objects lazily loaded -- on the first real request of each DP rank, where
# some ranks intermittently died with hipErrorNoBinaryForGpu ("no kernel image").  Run the
# same code paths once when the sampler is created, so the first use happens at startup with
# no other work in flight, and retry a failed load instead of failing a request later.
_hcu_sampler_warmup = get_bool_env_var("SGLANG_HCU_SAMPLER_WARMUP")


def _hcu_warmup_sampling_kernels(vocab: int = 155648, tries: int = 3):
    dev = torch.device("cuda", torch.cuda.current_device())
    for attempt in range(1, tries + 1):
        try:
            for bs in (1, 8):
                # next_token_logits are always float32 here (logits_processor casts them)
                logits = torch.randn(bs, vocab, device=dev, dtype=torch.float32)
                temps = torch.full((bs, 1), 0.7, device=dev, dtype=torch.float32)
                logits.div_(temps)
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits
                torch.log(probs)
                sampling_from_probs_torch(probs)
                top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    torch.full((bs,), TOP_K_ALL, device=dev, dtype=torch.int32),
                    torch.full((bs,), 0.95, device=dev, dtype=torch.float32),
                    torch.zeros((bs,), device=dev, dtype=torch.float32),
                    False,
                    None,
                    None,
                )
                torch.argmax(probs, -1)
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()  # runs before KV-pool sizing; give the scratch back
            if attempt > 1:
                logger.warning("HCU sampler warmup succeeded on attempt %d", attempt)
            return
        except RuntimeError as e:  # torch.AcceleratorError derives from RuntimeError
            if attempt == tries:
                raise
            logger.warning("HCU sampler warmup attempt %d failed, retrying: %s",
                           attempt, str(e).splitlines()[0])
'''

anchor = '''    raise ValueError(
        f"Unknown sampling backend '{backend}'. Register it via register_sampler_backend()."
    )
'''
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + FUNC)

init_anchor = '''        self.output_logprob_processor = OutputLogprobProcessor()
'''
assert src.count(init_anchor) == 1
src = src.replace(init_anchor, init_anchor + '''        if _hcu_sampler_warmup and torch.cuda.is_available():
            _hcu_warmup_sampling_kernels()
''')
open(P, "w").write(src)
print("patched", P)
