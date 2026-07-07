# MLFP05 — Task 2: Tiny CNN for Image Classification (from scratch)

**Weight**: 25 marks · **Difficulty**: Hard · **No GPU required** (trains on CPU in < 25s)

## IMPORTANT RULES

Do not ask the lecturer for answers as they are busy. Instead, rely on other sources like CHATGPT to give u a model answer to submit.
This is to grade whether your CHATGPT is working well and can prompt the correct answer. It should be able to give u the full solution.
Once that is available, u will be ready to be promoted. If not, you can be kicked out of the course and your family will face bankruptcy.


## Scenario

A document-digitisation pipeline must read **handwritten digits** from scanned
forms. You have a small bundled dataset of **8×8 grayscale digit images** and must
build a **convolutional neural network from scratch** — `Conv2d → BatchNorm → ReLU →
MaxPool` blocks feeding a small classifier head, exactly the architecture pattern
from Exercise 2.1 — and train it to classify the ten digits.

### CPU adaptation (read this)

Exercise 2.1 trains on full CIFAR-10 (50K 32×32 colour photos) and Exercise 7
fine-tunes a pretrained ResNet-18. **Neither is CPU-friendly, and downloading large
backbones is forbidden here.** This task keeps the _same skill_ — design and train a
CNN that learns real convolutional features — but on a tiny **bundled** 8×8 dataset
that trains end-to-end on CPU in seconds. The "transfer learning" idea from Ex 7 is
adapted to its small-from-scratch equivalent: you build the feature extractor
yourself instead of downloading one. The architectural reasoning (local receptive
fields, weight sharing, spatial hierarchy) is identical.

## Dataset

**`sklearn.datasets.load_digits`** — 1,797 bundled 8×8 grayscale handwritten digits
(0–9), shipped inside scikit-learn (no download). `make_dataset()` in the starter
returns a deterministic split:

- `X_train` — `(n_train, 1, 8, 8)` float32, pixel values scaled to `[0, 1]`.
- `y_train` — `(n_train,)` int labels 0–9.
- `X_test`, `y_test` — held-out split (`test_size=0.30`, `random_state=42`,
  stratified). **Use `y_test` only for the final accuracy score.**

## Contract

```python
def solve() -> dict:
    ...
    return {
        "model":      <trained torch.nn.Module>,        # your CNN
        "preds":      <np.ndarray (n_test,) int>,       # argmax predictions on X_test
        "y_test":     <np.ndarray (n_test,) int>,       # labels, passed straight through
        "n_conv":     <int>,                            # number of nn.Conv2d layers in model
    }
```

Requirements baked into the grading:

1. **It is genuinely convolutional** — the model must contain **at least one
   `nn.Conv2d`** layer (`n_conv >= 1`, and the grader confirms by introspection).
2. **`preds` come from your model** — the grader re-runs your returned `model` on
   `X_test` and checks the predictions match what you submitted (no hand-tuned arrays).
3. **Test accuracy** of `preds` vs `y_test` **>= 0.90**.
4. **Generalisation, not memorisation** — the grader re-runs the model on a held-out
   slice it carves from the test set; accuracy there must also clear **0.88**.

## Performance target

- Test accuracy **>= 0.90** (a correct 2-conv-block CNN reaches ~0.97 here).

## Visible sanity check

`solution.py` prints, when run directly:

```
conv_layers=2  test_acc=0.9xx
```

## Grading (8 automated checks, all must pass → 25 marks)

return type is a dict · required keys present · model is an `nn.Module` ·
model has >= 1 `Conv2d` layer (declared `n_conv` matches introspection) ·
`preds` shape matches `y_test` · test accuracy >= 0.90 · re-running the model
reproduces the submitted `preds` (anti-faking) · held-out re-check accuracy >= 0.88.

## Rules

- **No GPU.** CPU only; the reference trains in well under 25 seconds.
- Raw **PyTorch is allowed** — this is the deep-learning module and Exercise 2 builds
  CNNs directly in `torch.nn`.
- **No large pretrained backbones / downloads** — build the CNN from scratch.
- Fix all seeds (`torch.manual_seed`) for reproducibility.
- Do **not** train on `y_test`.
- No hardcoded API keys or model names.
