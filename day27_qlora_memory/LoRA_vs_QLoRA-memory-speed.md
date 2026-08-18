# LoRA / QLoRA memory accounting

A reproducible measurement of *where* memory goes when fine-tuning an LLM, and
which of those costs LoRA and QLoRA actually remove.

The claim being tested is narrower than the usual one. "LoRA saves memory" is
true but uninformative — it does not say whether the saving comes from weights,
gradients, optimizer state, or activations. The answer determines when the
technique helps and when it does not.

**Model:** Qwen2.5-0.5B-Instruct · **Hardware:** single T4 (15 GB) · **Runtime:** a few minutes

---

## Results

Run `lora_qlora_memory_benchmark.ipynb` and paste the output table here.

| Arm | Load (GB) | Peak (GB) | Mem saved | Trainable | ms/step | Rel. time |
|---|---|---|---|---|---|---|
| Full FT (fp32) | | | — | | | 1.00x |
| LoRA (fp16) | | | | | | |
| QLoRA (nf4) | | | | | | |

### Reading the table

The interesting column is not `Peak (GB)` but the gap between `Load (GB)` and
`Peak (GB)`. That gap is gradients plus optimizer state — the part LoRA removes.
In the Full FT arm it should be roughly 3x the load footprint (2 GB weights →
2 GB gradients → 4 GB AdamW state). In the LoRA and QLoRA arms it should be
close to zero, because 99.5% of parameters have no gradient and therefore no
optimizer state.

---

## Design decisions

Written in assumption → alternative → choice → limitation form, so each can be
challenged independently.

### 1. Full FT uses fp32 while the other arms use fp16/nf4

- **Assumption:** a reader wants to know what they would actually observe, not
  what a laboratory-clean comparison would show.
- **Alternative:** run all three arms in fp16 for a strictly controlled test.
- **Choice:** fp32 for Full FT. Pure fp16 full fine-tuning is numerically
  unstable and nobody runs it; an fp32 baseline is the honest comparison.
- **Limitation:** this conflates two variables (precision and trainable-parameter
  count) in the Full-FT-vs-LoRA delta. A fourth fp16 full-FT arm would separate
  them; the weight and gradient rows should halve.

### 2. Gradient checkpointing is disabled everywhere

- **Assumption:** the question is what quantization alone costs and saves.
- **Alternative:** leave `prepare_model_for_kbit_training` at its default, which
  enables checkpointing.
- **Choice:** disabled. Otherwise the QLoRA arm gains an unrelated memory saving
  and an unrelated slowdown simultaneously, and neither number can be attributed.
- **Limitation:** these QLoRA numbers are *not* a production QLoRA configuration,
  where checkpointing is normally on. Expect lower memory and worse throughput
  in real use.

### 3. Memory is measured after a training step, not after loading

- **Assumption:** optimizer state is the dominant term and must be captured.
- **Alternative:** measure immediately after `from_pretrained`, which is simpler.
- **Choice:** run one untimed warm-up step first. AdamW allocates `exp_avg` and
  `exp_avg_sq` lazily on the first `step()`, so a load-time measurement misses
  the largest allocation entirely — Full FT would appear to use ~2 GB instead of ~8 GB.
- **Limitation:** peak memory now depends on batch size and sequence length
  through activations. Both are pinned, but the absolute numbers are not
  transferable to other shapes.

### 4. Random token ids instead of a real dataset

- **Assumption:** memory and throughput are independent of input content.
- **Alternative:** use a real tokenized corpus.
- **Choice:** random ids. Loss is meaningless, which is fine because no quality
  claim is being made.
- **Limitation:** real data has variable sequence lengths and padding, which
  changes activation memory. This measures a fixed-shape best case.

---

## Extrapolation

Section 7 of the notebook projects to 7B / 13B / 70B from bytes-per-parameter
arithmetic rather than renting larger hardware. The projection excludes
activations and is therefore a lower bound. Its purpose is to show *where the
crossover is* — the point at which QLoRA stops being a marginal optimization and
becomes the difference between fitting on one card and not.

---

## What this does not measure

- **Quality.** Whether a QLoRA-trained adapter matches a LoRA-trained one on a
  downstream metric. Quantization error is real; this repo only shows it is cheap.
- **The merge path.** A QLoRA adapter cannot be merged cleanly into its own 4-bit
  base — re-quantizing after the addition discards most of the learned delta. The
  common workaround merges into a freshly loaded fp16 base, which creates a
  mismatch between training and deployment conditions. That penalty is unmeasured
  here and, as far as I can tell, not well characterized at this model scale.
- **Optimizer choice.** AdamW's 8 bytes/param dominates the Full FT arm. SGD or
  8-bit Adam would change the ranking materially.

## Open questions

1. Does the fp16-base merge mismatch show up on a task metric at 0.5B, or only at
   larger scale?
2. LoRA's backward pass still traverses every frozen layer to reach the earliest
   adapters. At what parameter count does its throughput advantage stop growing?

---

## Environment

Tested on `transformers` 5.x. Two v5 changes affect this notebook:

- `torch_dtype` renamed to `dtype` (old name works, warns).
- dtype default changed to `auto` — models load in the checkpoint's precision.
  Qwen2.5-0.5B ships as bf16, so omitting an explicit dtype would load the Full FT
  arm in bf16 and halve its weight and gradient rows without any error message.
  All three arms pin dtype explicitly.

After `pip install -U`, restart the runtime. `pip` replaces files on disk but
cannot evict modules already resident in `sys.modules`.

## Running it

```
Colab → File → Upload notebook → lora_qlora_memory_benchmark.ipynb
Runtime → Change runtime type → T4 GPU
```

Run cells individually rather than Run All; the install cell requires a restart
before the rest will import correctly.
