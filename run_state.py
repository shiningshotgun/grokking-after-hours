"""Pure run-state helpers for train.py.

This module holds the trainer's state machine in testable, torch-free form:
checkpoint validation (current + legacy), resume planning, the evaluation
schedule recurrence, save-view normalization, destination policy, and the
training-loop transition logic (parameterized by callbacks so tests can drive
it with inert recorders instead of a model/optimizer).

Serialization (torch.save/torch.load) and tensor handling stay in train.py.
Nothing here performs or imports any workload: no forward/backward passes, no
optimizer steps, no entry points.
"""

import hashlib
import json

CHECKPOINT_SCHEMA_VERSION = 2
# Explicit legacy markers. A checkpoint with NO schema_version field, or with
# schema_version == 1, is routed to the legacy path. Any other integer
# schema_version is unsupported (rejected, never silently treated as legacy).
LEGACY_SCHEMA_VERSIONS = (1,)

EPOCH_CONVENTION = (
    "epoch is 0-based index of epoch in progress/completed; permutation applies "
    "to that epoch; batch_cursor is next minibatch index within epoch"
)

EXPECTED_CSV_HEADER = [
    "step",
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
]

# The original (historical) runs used a fixed schedule of 10 minibatches per
# epoch (4656 rows, batch size 512 -> 9 full + 1 short batch of 48). Legacy
# checkpoints carry no batch-size field, so epoch-boundary arithmetic uses this
# documented constant of the original runs.
ORIGINAL_LEGACY_N_BATCHES = 10
ORIGINAL_LEGACY_BATCH_SIZE = 512

# Model/optimizer/sampler contract that current-schema checkpoints must match
# exactly (kept here so train.py and the tests validate against one copy).
EXPECTED_ARCHITECTURE = {
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 128,
    "dropout": 0.0,
    "max_context_len": 16,
    "vocab_len": 239,
}
EXPECTED_LOSS_MASKING = "answer_plus_EOS_after_="
EXPECTED_OPTIMIZER = {
    "name": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1.0,
    "betas": [0.9, 0.98],
    "eps": 1e-8,
    "warmup_updates": 10,
    "no_annealing": True,
}
EXPECTED_SAMPLER = {
    "batch_size": 512,
    "shuffle_within_epoch": True,
    "eval_cadence": "next=max(int(1.02*e), e+1)",
}
EXPECTED_MODULUS = 97
EXPECTED_TRAIN_PCT = 50
EXPECTED_BATCH_SIZE = 512


class CheckpointError(ValueError):
    """A checkpoint cannot be used as given."""


class UnsupportedCheckpointError(CheckpointError):
    """A checkpoint from an unknown/future schema; never treated as legacy."""


class LegacyCheckpointRefused(CheckpointError):
    """A legacy checkpoint whose identity/boundary evidence is inadequate."""


class DestinationError(CheckpointError):
    """The requested run destination conflicts with retained evidence."""


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------
# Evaluation schedule
# --------------------------------------------------------------------------


def next_eval_after(epoch):
    """Official eval cadence: next = max(int(1.02*e), e+1)."""
    return max(int(1.02 * epoch), epoch + 1)


def first_due_eval_epoch(start_epoch):
    """First epoch of the canonical eval chain that is >= start_epoch.

    The canonical chain is 0, next_eval_after(0), next_eval_after(that), ... .
    Used to normalize stale/missing schedule pointers on legacy resumes: it
    follows the official recurrence deterministically, never skips evaluation,
    and never invents historical metrics. It may differ from what the original
    (pre-rewrite) trainer actually did after its pauses; any such difference is
    reported in the resume notes, never hidden.
    """
    if not _is_int(start_epoch) or start_epoch < 0:
        raise CheckpointError("start_epoch must be a nonnegative int")
    if start_epoch == 0:
        return 0
    e = 0
    while e < start_epoch:
        e = next_eval_after(e)
    return e


# --------------------------------------------------------------------------
# Run identity
# --------------------------------------------------------------------------


def config_prefix_five_field():
    """Training-time config prefix (exact str() serialization, insertion order)."""
    return str({"modulus": 97, "train_pct": 50, "bs": 512, "lr": 1e-3, "wd": 1.0})


def config_prefix_two_field():
    """The older endpoint-script prefix (2 fields) that produced c4625d6b..."""
    return str({"modulus": 97, "train_pct": 50})


def identity_hash(identity):
    """SHA-256 over the identity dict with sort_keys, excluding the hash field."""
    payload = {k: v for k, v in identity.items() if k != "run_identity_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_run_identity(seed, vocab, cfg_hash, tok_hash, optimizer=None):
    """Build the run identity. `optimizer` must be the ACTUAL optimizer
    settings of this invocation; when omitted, the documented defaults are
    used (valid only for a default invocation)."""
    ident = {
        "seed": seed,
        "modulus": EXPECTED_MODULUS,
        "train_pct": EXPECTED_TRAIN_PCT,
        "architecture": dict(EXPECTED_ARCHITECTURE, vocab_len=vocab),
        "loss_masking": EXPECTED_LOSS_MASKING,
        "optimizer": dict(EXPECTED_OPTIMIZER) if optimizer is None else dict(optimizer),
        "sampler": EXPECTED_SAMPLER,
        "config_split_hash": cfg_hash,
        "tokenizer_hash": tok_hash,
    }
    ident["run_identity_hash"] = identity_hash(ident)
    return ident


# --------------------------------------------------------------------------
# Live loop state
# --------------------------------------------------------------------------


class LiveState(object):
    """Mutable live state of one trainer invocation.

    epoch          0-based index of the epoch currently in progress
    current_perm   the epoch's permutation (list), or None before it is drawn
    batch_cursor   number of minibatches of `epoch` already consumed
    global_step    completed updates
    next_eval_epoch schedule pointer: next epoch at whose end an eval is due
    """

    def __init__(
        self,
        epoch,
        current_perm,
        batch_cursor,
        global_step,
        next_eval_epoch,
        steps_this_run=0,
        unsaved_updates=0,
        migrated_legacy=False,
    ):
        self.epoch = epoch
        self.current_perm = current_perm
        self.batch_cursor = batch_cursor
        self.global_step = global_step
        self.next_eval_epoch = next_eval_epoch
        self.steps_this_run = steps_this_run
        self.unsaved_updates = unsaved_updates
        self.migrated_legacy = migrated_legacy


# --------------------------------------------------------------------------
# Save view: live state -> unambiguous checkpoint state
# --------------------------------------------------------------------------


def save_view(epoch, current_perm, batch_cursor, n_batches):
    """Map live loop state onto the two unambiguous checkpoint states.

    (A) epoch completed:  epoch E, cursor == n_batches, permutation optional.
        Saved either just after E's bookkeeping (permutation retained) or
        after advancing into E+1 before any draw (permutation None, in which
        case the view describes E-1... i.e. the completed epoch `epoch - 1`).
    (B) epoch in progress: 0 <= cursor < n_batches, permutation required.

    Returns (epoch_for_save, perm_for_save, cursor_for_save, epoch_completed).
    Raises CheckpointError for impossible states rather than guessing.
    """
    if not _is_int(epoch) or epoch < 0:
        raise CheckpointError("epoch must be a nonnegative int")
    if not _is_int(batch_cursor):
        raise CheckpointError("batch_cursor must be an int")
    if current_perm is None:
        if batch_cursor != 0:
            raise CheckpointError(
                "inconsistent state: no permutation but nonzero batch_cursor"
            )
        if epoch == 0:
            raise CheckpointError(
                "nothing completed to save at epoch 0 with no permutation"
            )
        return epoch - 1, None, n_batches, True
    if batch_cursor < 0 or batch_cursor > n_batches:
        raise CheckpointError("batch_cursor out of range: %r" % (batch_cursor,))
    if batch_cursor == n_batches:
        return epoch, current_perm, n_batches, True
    return epoch, current_perm, batch_cursor, False


# --------------------------------------------------------------------------
# Checkpoint validation (current schema)
# --------------------------------------------------------------------------


def _validate_permutation(perm, n_train, field="epoch_permutation"):
    if not isinstance(perm, (list, tuple)):
        raise CheckpointError("%s must be a list" % field)
    if len(perm) != n_train:
        raise CheckpointError(
            "%s has length %d, expected %d" % (field, len(perm), n_train)
        )
    seen = set()
    for v in perm:
        if not _is_int(v) or v < 0 or v >= n_train:
            raise CheckpointError("%s contains an out-of-range value: %r" % (field, v))
        seen.add(v)
    if len(seen) != n_train:
        raise CheckpointError("%s is not a permutation (duplicates)" % field)


def _validate_mps_rng_policy(ckpt):
    device = ckpt.get("device")
    device_str = "" if device is None else str(device)
    if "mps" not in device_str.lower():
        return
    mps = ckpt.get("mps_rng_state")
    if mps is None:
        raise CheckpointError(
            "checkpoint declares device %r but has no mps_rng_state; an MPS "
            "checkpoint must carry a nonempty saved MPS RNG state" % (device,)
        )
    try:
        if len(mps) == 0:
            raise CheckpointError(
                "checkpoint declares device %r but has an empty mps_rng_state"
                % (device,)
            )
    except TypeError:
        raise CheckpointError("mps_rng_state has no length")


def _validate_rng_fields(ckpt):
    trs = ckpt.get("torch_rng_state")
    if trs is None:
        raise CheckpointError("missing torch_rng_state")
    try:
        if len(trs) == 0:
            raise CheckpointError("empty torch_rng_state")
    except TypeError:
        raise CheckpointError("torch_rng_state has no length")
    mps = ckpt.get("mps_rng_state")
    if mps is not None:
        try:
            if len(mps) == 0:
                raise CheckpointError("empty mps_rng_state")
        except TypeError:
            raise CheckpointError("mps_rng_state has no length")
    nrs = ckpt.get("numpy_rng_state")
    if not isinstance(nrs, (tuple, list)) or len(nrs) != 5:
        raise CheckpointError("numpy_rng_state is not a 5-tuple")
    if nrs[0] != "MT19937":
        raise CheckpointError("unexpected numpy RNG kind %r" % (nrs[0],))
    try:
        if len(nrs[1]) != 624:
            raise CheckpointError("numpy RNG key array has wrong length")
    except TypeError:
        raise CheckpointError("numpy RNG key state has no length")
    prs = ckpt.get("python_rng_state")
    if not isinstance(prs, (tuple, list)) or len(prs) != 3:
        raise CheckpointError("python_rng_state is not a 3-tuple")
    try:
        if len(prs[1]) != 625:
            raise CheckpointError("python RNG state tuple has wrong length")
    except TypeError:
        raise CheckpointError("python RNG state has no length")


def validate_current_checkpoint(ckpt, exp):
    """Strictly validate a schema-2 full checkpoint; return its normalized view.

    exp keys: seed, cfg_hash, tok_hash, identity, n_train, batch_size, n_batches.
    Raises UnsupportedCheckpointError for future/unknown schemas and
    CheckpointError for any missing/inconsistent required state. There are no
    default substitutions that could silently skip an epoch or skip identity
    checks.
    """
    if not isinstance(ckpt, dict):
        raise CheckpointError("checkpoint is not a dict")
    sv = ckpt.get("schema_version")
    if not _is_int(sv):
        raise CheckpointError("missing or non-integer schema_version")
    if sv > CHECKPOINT_SCHEMA_VERSION:
        raise UnsupportedCheckpointError(
            "checkpoint schema_version %d is newer than supported version %d"
            % (sv, CHECKPOINT_SCHEMA_VERSION)
        )
    if sv != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError("unsupported schema_version %r" % (sv,))
    if ckpt.get("checkpoint_type") != "full":
        raise CheckpointError(
            "checkpoint_type must be 'full' (got %r); model-only snapshots are "
            "not resume sources" % (ckpt.get("checkpoint_type"),)
        )
    if ckpt.get("seed") != exp["seed"] or not _is_int(ckpt.get("seed")):
        raise CheckpointError("seed mismatch or missing seed")
    for field, expected in (
        ("architecture", exp["identity"]["architecture"]),
        ("optimizer_settings", exp["identity"]["optimizer"]),
        ("sampler_policy", exp["identity"]["sampler"]),
        ("loss_masking_contract", exp["identity"]["loss_masking"]),
    ):
        if ckpt.get(field) != expected:
            raise CheckpointError("%s mismatch" % field)
    if (
        ckpt.get("modulus") != EXPECTED_MODULUS
        or ckpt.get("train_pct") != EXPECTED_TRAIN_PCT
    ):
        raise CheckpointError("modulus/train_pct mismatch")
    if ckpt.get("epoch_convention") != EPOCH_CONVENTION:
        raise CheckpointError("epoch_convention mismatch or missing")
    epoch = ckpt.get("epoch")
    if not _is_int(epoch) or epoch < 0:
        raise CheckpointError("epoch must be a nonnegative int")
    completed = ckpt.get("epoch_completed")
    if not isinstance(completed, bool):
        raise CheckpointError("epoch_completed must be a bool")
    gs = ckpt.get("global_step")
    cu = ckpt.get("completed_updates")
    if not _is_int(gs) or gs < 0 or not _is_int(cu) or cu != gs:
        raise CheckpointError(
            "global_step/completed_updates missing, negative, or inconsistent"
        )
    nee = ckpt.get("next_eval_epoch")
    if not _is_int(nee) or nee < 0:
        raise CheckpointError(
            "next_eval_epoch missing/invalid (%r); a checkpoint without a valid "
            "schedule pointer is refused (it could silently end all evaluation)"
            % (nee,)
        )
    if completed:
        if not nee > epoch:
            raise CheckpointError(
                "completed-epoch checkpoint must have next_eval_epoch > epoch "
                "(got %d vs epoch %d)" % (nee, epoch)
            )
    else:
        if not nee >= epoch:
            raise CheckpointError(
                "in-progress checkpoint must have next_eval_epoch >= epoch "
                "(got %d vs epoch %d)" % (nee, epoch)
            )
    if ckpt.get("batch_size") != exp["batch_size"]:
        raise CheckpointError("batch_size mismatch")
    if ckpt.get("n_batches") != exp["n_batches"] or not _is_int(ckpt.get("n_batches")):
        raise CheckpointError("n_batches mismatch or missing")
    cursor = ckpt.get("batch_cursor")
    if not _is_int(cursor) or cursor < 0 or cursor > exp["n_batches"]:
        raise CheckpointError("batch_cursor out of range: %r" % (cursor,))
    perm = ckpt.get("epoch_permutation")
    if completed:
        if cursor != exp["n_batches"]:
            raise CheckpointError(
                "completed checkpoint must have batch_cursor == n_batches"
            )
        if perm is not None:
            _validate_permutation(perm, exp["n_train"])
    else:
        if cursor >= exp["n_batches"]:
            raise CheckpointError(
                "in-progress checkpoint must have batch_cursor < n_batches"
            )
        if perm is None:
            raise CheckpointError(
                "in-progress checkpoint is missing epoch_permutation; the epoch "
                "could not be resumed without an extra shuffle"
            )
        _validate_permutation(perm, exp["n_train"])
    # Fixed-schedule geometry: in-progress and completed states must satisfy
    # exact arithmetic; checking only global_step == completed_updates is not
    # enough.
    nb = exp["n_batches"]
    if completed:
        if gs != (epoch + 1) * nb:
            raise CheckpointError(
                "completed checkpoint arithmetic mismatch: global_step %d != "
                "(epoch+1)*n_batches (%d*%d=%d)" % (gs, epoch, nb, (epoch + 1) * nb)
            )
        if cursor != nb:
            raise CheckpointError(
                "completed checkpoint must have batch_cursor == n_batches"
            )
    else:
        if gs != epoch * nb + cursor:
            raise CheckpointError(
                "in-progress checkpoint arithmetic mismatch: global_step %d != "
                "epoch*n_batches + batch_cursor (%d*%d+%d=%d)"
                % (gs, epoch, nb, cursor, epoch * nb + cursor)
            )
    _validate_rng_fields(ckpt)
    _validate_mps_rng_policy(ckpt)
    if ckpt.get("config_split_hash") != exp["cfg_hash"]:
        raise CheckpointError("config_split_hash mismatch")
    if ckpt.get("tokenizer_hash") != exp["tok_hash"]:
        raise CheckpointError("tokenizer_hash mismatch or missing")
    ident = ckpt.get("run_identity")
    if not isinstance(ident, dict):
        raise CheckpointError("missing run_identity")
    if ident != exp["identity"]:
        raise CheckpointError("run_identity does not match this run's configuration")
    if ckpt.get("run_identity_hash") != exp["identity"]["run_identity_hash"]:
        raise CheckpointError("run_identity_hash mismatch")
    # Recompute the canonical identity hash from the checkpoint's own contents
    # rather than trusting the stored hash alone.
    if identity_hash(ident) != ckpt.get("run_identity_hash"):
        raise CheckpointError("run_identity_hash does not match run_identity contents")
    migrated = ckpt.get("migrated_legacy", False)
    if not isinstance(migrated, bool):
        raise CheckpointError("migrated_legacy must be a bool")
    ms = ckpt.get("model_state")
    os_ = ckpt.get("optimizer_state")
    if not isinstance(ms, dict) or len(ms) == 0:
        raise CheckpointError("missing model_state")
    if not isinstance(os_, dict):
        raise CheckpointError("missing optimizer_state")
    return {
        "epoch": epoch,
        "epoch_completed": completed,
        "global_step": gs,
        "next_eval_epoch": nee,
        "current_perm": list(perm) if perm is not None else None,
        "batch_cursor": cursor,
        "migrated_legacy": migrated,
    }


def plan_current_resume(view):
    """Normalized view -> resume plan (epoch/cursor/perm to continue from)."""
    if view["epoch_completed"]:
        return {
            "start_epoch": view["epoch"] + 1,
            "global_step": view["global_step"],
            "next_eval_epoch": view["next_eval_epoch"],
            "current_perm": None,
            "batch_cursor": 0,
            "migrated_legacy": view["migrated_legacy"],
            "notes": [],
        }
    return {
        "start_epoch": view["epoch"],
        "global_step": view["global_step"],
        "next_eval_epoch": view["next_eval_epoch"],
        "current_perm": list(view["current_perm"]),
        "batch_cursor": view["batch_cursor"],
        "migrated_legacy": view["migrated_legacy"],
        "notes": [],
    }


# --------------------------------------------------------------------------
# Legacy checkpoint validation (explicit, evidence-gated path)
# --------------------------------------------------------------------------


def validate_legacy_checkpoint(ckpt, exp, provenance, checkpoint_sha256=None):
    """Validate a legacy (pre-schema) full checkpoint for a boundary resume.

    Legacy format facts (verified against the archived original trainer):
    - fields: model_state, optimizer_state, epoch, global_step, next_eval_epoch,
      batch_idx (ALWAYS hardcoded 10, even for mid-epoch stop saves), device,
      torch/mps/numpy/python rng states, config_split_hash (the 5-field
      training-time definition), lib_versions.
    - no schema_version, no seed, no tokenizer hash, no run identity, and no
      stored epoch permutation.

    Because batch_idx carries no information, an epoch boundary is established
    ONLY by global-step/epoch arithmetic under the original fixed 10-batch
    schedule. Facts absent from the format (seed, run provenance) must come
    from a trusted external provenance record that matches the checkpoint.
    Mid-epoch legacy checkpoints are refused: without the stored permutation an
    in-progress epoch cannot be resumed faithfully. The original checkpoint
    file is never modified.
    """
    if not isinstance(ckpt, dict):
        raise CheckpointError("checkpoint is not a dict")
    sv = ckpt.get("schema_version")
    if _is_int(sv) and sv > 1:
        raise UnsupportedCheckpointError(
            "schema_version %d is not a legacy format and is not supported" % sv
        )
    if (
        isinstance(ckpt.get("model_state"), dict) is False
        or isinstance(ckpt.get("optimizer_state"), dict) is False
        or len(ckpt.get("model_state") or {}) == 0
    ):
        raise LegacyCheckpointRefused(
            "legacy checkpoint lacks full model/optimizer state; model-only "
            "snapshots are not resume sources"
        )
    epoch = ckpt.get("epoch")
    gs = ckpt.get("global_step")
    if not _is_int(epoch) or epoch < 0:
        raise LegacyCheckpointRefused(
            "legacy checkpoint missing or invalid 'epoch'; identity cannot be "
            "established"
        )
    if not _is_int(gs) or gs <= 0:
        raise LegacyCheckpointRefused(
            "legacy checkpoint missing or invalid 'global_step'; identity cannot "
            "be established"
        )
    # Trusted provenance is required for facts absent from the legacy format.
    if not isinstance(provenance, dict) or not provenance:
        raise LegacyCheckpointRefused(
            "legacy checkpoint has no seed/identity fields; refusing migration "
            "without a trusted provenance record (--legacy-provenance JSON with "
            "seed, global_step, epoch recorded from the run's logs)"
        )
    if not _is_int(provenance.get("seed")):
        raise LegacyCheckpointRefused(
            "provenance seed must be an int (got %r)" % (provenance.get("seed"),)
        )
    if provenance.get("seed") != exp["seed"]:
        raise LegacyCheckpointRefused(
            "provenance seed %r does not match requested seed %r"
            % (provenance.get("seed"), exp["seed"])
        )
    if not _is_int(provenance.get("global_step")):
        raise LegacyCheckpointRefused(
            "provenance global_step must be an int (got %r)"
            % (provenance.get("global_step"),)
        )
    if provenance.get("global_step") != gs:
        raise LegacyCheckpointRefused(
            "provenance global_step %r does not match checkpoint global_step %r"
            % (provenance.get("global_step"), gs)
        )
    if not _is_int(provenance.get("epoch")):
        raise LegacyCheckpointRefused(
            "provenance epoch must be an int (got %r)" % (provenance.get("epoch"),)
        )
    if provenance.get("epoch") != epoch:
        raise LegacyCheckpointRefused(
            "provenance epoch %r does not match checkpoint epoch %r"
            % (provenance.get("epoch"), epoch)
        )
    # Trusted provenance must bind its claims to the checkpoint bytes via a
    # SHA-256 over the explicitly selected checkpoint file.
    # Only the canonical digest key is accepted; aliases are rejected to
    # remove ambiguity and ignored contradictions.
    if isinstance(provenance, dict) and any(
        a in provenance for a in ("checkpoint_hash", "sha256", "file_sha256")
    ):
        raise LegacyCheckpointRefused(
            "provenance uses a non-canonical digest key; only "
            "'checkpoint_sha256' is accepted"
        )
    prov_hash = provenance.get("checkpoint_sha256") if isinstance(provenance, dict) else None
    if prov_hash is None:
        raise LegacyCheckpointRefused(
            "trusted provenance record lacks required checkpoint_sha256; a "
            "sidecar repeating seed/epoch/step does not bind claims to bytes"
        )
    if not isinstance(prov_hash, str) or len(prov_hash) != 64 or any(
        c not in "0123456789abcdef" for c in prov_hash.lower()
    ):
        raise LegacyCheckpointRefused(
            "provenance checkpoint digest %r is not a 64-hex SHA-256" % (prov_hash,)
        )
    if checkpoint_sha256 is None:
        raise LegacyCheckpointRefused(
            "checkpoint file hash was not provided for verification; refuse "
            "migration without hashing the selected checkpoint"
        )
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in checkpoint_sha256.lower()
    ):
        raise LegacyCheckpointRefused(
            "supplied checkpoint digest %r is not a 64-hex SHA-256"
            % (checkpoint_sha256,)
        )
    if prov_hash != checkpoint_sha256:
        raise LegacyCheckpointRefused(
            "checkpoint SHA-256 %r does not match provenance record %r"
            % (checkpoint_sha256, prov_hash)
        )
    # Run-identity evidence is REQUIRED for legacy migration (not optional).
    if not isinstance(provenance.get("run_identity"), dict):
        raise LegacyCheckpointRefused(
            "trusted provenance record lacks required run_identity"
        )
    if provenance.get("run_identity") != exp["identity"]:
        raise LegacyCheckpointRefused(
            "provenance run_identity does not match this run's identity"
        )
    if not isinstance(provenance.get("run_identity_hash"), str):
        raise LegacyCheckpointRefused(
            "trusted provenance record lacks required run_identity_hash"
        )
    if provenance.get("run_identity_hash") != exp["identity"]["run_identity_hash"]:
        raise LegacyCheckpointRefused(
            "provenance run_identity_hash does not match this run's identity"
        )
    if identity_hash(provenance.get("run_identity")) != provenance.get("run_identity_hash"):
        raise LegacyCheckpointRefused(
            "provenance run_identity_hash does not match run_identity contents"
        )
    # Epoch-boundary arithmetic under the original fixed 10-batch schedule.
    nb = ORIGINAL_LEGACY_N_BATCHES
    if exp["n_batches"] != nb:
        raise LegacyCheckpointRefused(
            "legacy boundary arithmetic assumes %d batches/epoch, but this "
            "configuration has %d" % (nb, exp["n_batches"])
        )
    bd_implied = gs - epoch * nb
    if bd_implied < 0 or bd_implied > nb:
        raise LegacyCheckpointRefused(
            "legacy global_step/epoch inconsistent with the fixed %d-batch "
            "schedule (implied %d batches done in epoch %d at step %d)"
            % (nb, bd_implied, epoch, gs)
        )
    if bd_implied == 0:
        raise LegacyCheckpointRefused(
            "legacy checkpoint implies 0 batches done in epoch %d; legacy saves "
            "always follow at least one update" % epoch
        )
    if bd_implied != nb:
        raise LegacyCheckpointRefused(
            "legacy checkpoint is mid-epoch (epoch %d, %d/%d batches done at "
            "step %d); the legacy format stores no epoch permutation, so a "
            "faithful mid-epoch resume is impossible. Refusing rather than "
            "skipping or replaying the remaining minibatches."
            % (epoch, bd_implied, nb, gs)
        )
    bi = ckpt.get("batch_idx")
    if bi is not None and bi != nb:
        raise LegacyCheckpointRefused(
            "legacy batch_idx %r contradicts the boundary arithmetic "
            "(expected the hardcoded %d)" % (bi, nb)
        )
    if ckpt.get("config_split_hash") != exp["cfg_hash"]:
        raise LegacyCheckpointRefused(
            "legacy config_split_hash mismatch (the legacy format used the same "
            "5-field training-time definition)"
        )
    try:
        _validate_rng_fields(ckpt)
        _validate_mps_rng_policy(ckpt)
    except LegacyCheckpointRefused:
        raise
    except CheckpointError as e:
        raise LegacyCheckpointRefused("legacy checkpoint RNG state incomplete: %s" % e)
    # Schedule normalization: legacy next_eval_epoch values were recorded
    # pre-update at pause boundaries and can be stale (== epoch) or missing.
    nee = ckpt.get("next_eval_epoch")
    notes = []
    if _is_int(nee) and nee > epoch:
        next_eval_epoch = nee
        notes.append("kept recorded legacy schedule pointer next_eval_epoch=%d" % nee)
    else:
        start_epoch = epoch + 1
        next_eval_epoch = first_due_eval_epoch(start_epoch)
        notes.append(
            "recorded legacy next_eval_epoch=%r is stale/missing (pre-update); "
            "normalized to the canonical official recurrence: next eval at "
            "epoch %d. This may differ from the historical continuation, which "
            "clamped the stale pointer to the resume epoch." % (nee, next_eval_epoch)
        )
    return {
        "start_epoch": epoch + 1,
        "global_step": gs,
        "next_eval_epoch": next_eval_epoch,
        "current_perm": None,
        "batch_cursor": 0,
        "migrated_legacy": True,
        "notes": notes,
    }


def plan_resume_from_checkpoint(ckpt, exp, provenance, checkpoint_sha256=None):
    """Route a loaded checkpoint to the current-schema or legacy path.

    schema_version absent/None/1 -> legacy path; anything else must validate
    strictly as the current schema. Unknown/future versions are rejected, never
    silently treated as legacy.
    """
    if not isinstance(ckpt, dict):
        raise CheckpointError("checkpoint is not a dict")
    sv = ckpt.get("schema_version")
    if sv is not None and not _is_int(sv):
        raise CheckpointError(
            "schema_version must be an int or absent (got %r)" % (sv,)
        )
    if sv is None or sv in LEGACY_SCHEMA_VERSIONS:
        plan = validate_legacy_checkpoint(
            ckpt, exp, provenance, checkpoint_sha256=checkpoint_sha256
        )
        return plan
    view = validate_current_checkpoint(ckpt, exp)
    return plan_current_resume(view)


def fresh_plan():
    """Plan for a fresh run: epoch-0 cadence starts at next_eval_epoch == 0."""
    return {
        "start_epoch": 0,
        "global_step": 0,
        "next_eval_epoch": 0,
        "current_perm": None,
        "batch_cursor": 0,
        "migrated_legacy": False,
        "notes": ["fresh run: epoch-0 evaluation cadence (next=max(int(1.02*e), e+1))"],
    }


# --------------------------------------------------------------------------
# Destination policy (pure checks over an inspection record)
# --------------------------------------------------------------------------


def validate_fresh_destination(insp):
    """Refuse any destination that already carries evidence or conflicts.

    In particular: retained metrics without a checkpoint are NOT a resumable
    run directory and a fresh run must never start from step 0 into them.
    There is no force/append/reset escape hatch.
    """
    if insp.get("latest_exists"):
        raise DestinationError(
            "destination contains latest.pt; a fresh run may not target it "
            "(use it as a resume, or choose an unused run directory)"
        )
    if insp.get("csv_data_rows", 0) > 0:
        raise DestinationError(
            "destination already contains %d logged metric row(s) without a "
            "valid matching checkpoint; refusing a fresh start that would "
            "append to historical evidence. Use a fresh, unused run directory "
            "for a new run." % insp["csv_data_rows"]
        )
    if insp.get("csv_exists") and insp.get("csv_header") is not None:
        if list(insp["csv_header"]) != EXPECTED_CSV_HEADER:
            raise DestinationError(
                "existing metrics.csv header %r does not match the expected "
                "header %r" % (insp["csv_header"], EXPECTED_CSV_HEADER)
            )
    if insp.get("snapshot_files") or insp.get("other_pt_files"):
        raise DestinationError(
            "destination contains conflicting checkpoint artifacts: %r"
            % (insp.get("snapshot_files") + insp.get("other_pt_files"),)
        )


def validate_resume_destination(insp, plan_global_step):
    """A resume must be consistent with the destination's existing metrics."""
    if not insp.get("latest_exists"):
        raise DestinationError("resume requested but latest.pt is missing")
    if insp.get("csv_exists"):
        if (
            insp.get("csv_header") is not None
            and list(insp["csv_header"]) != EXPECTED_CSV_HEADER
        ):
            raise DestinationError(
                "existing metrics.csv header %r does not match the expected "
                "header %r" % (insp["csv_header"], EXPECTED_CSV_HEADER)
            )
        last = insp.get("csv_last_step")
        if last is not None and last > plan_global_step:
            raise DestinationError(
                "checkpoint global_step %d is behind the last logged metrics "
                "row (step %d); refusing to append replayed steps. Fail-closed: "
                "investigate the divergence; if a new lineage is intended, use "
                "a new unused output destination and record it explicitly."
                % (plan_global_step, last)
            )


def reject_existing_outputs(paths):
    """Refuse to overwrite any existing output file (collision protection).

    Used by analyze.py and evaluate_endpoints.py so future writes always go to
    a new, unused output path. Checked before any expensive work or partial
    write; existing files are left untouched.
    """
    existing = [p for p in paths if p and os_path_exists(p)]
    if existing:
        raise DestinationError(
            "refusing to overwrite existing output(s): %s; pass a new, unused "
            "output path" % ", ".join(sorted(existing))
        )


def os_path_exists(path):
    import os

    return os.path.exists(path)


def validate_invocation_args(steps, segment_steps):
    if not _is_int(steps) or steps <= 0:
        raise DestinationError("--steps must be a positive integer (got %r)" % (steps,))
    if segment_steps is not None and (not _is_int(segment_steps) or segment_steps <= 0):
        raise DestinationError(
            "--segment-steps must be a positive integer (got %r)" % (segment_steps,)
        )


# --------------------------------------------------------------------------
# Training-loop transition logic (the single source of truth)
# --------------------------------------------------------------------------


def run_training_loop(
    state,
    env,
    exec_batch,
    draw_perm,
    eval_and_log,
    save_latest,
    write_obligations,
    stop_requested,
):
    """Drive updates/epochs with explicit, testable transition rules.

    state: LiveState. env: dict with n_train, batch_size, n_batches,
    target_steps. Callbacks:
      exec_batch(idx)        performs exactly one update for these indices
      draw_perm()            returns a fresh permutation for a new epoch
      eval_and_log(epoch, step)   full eval + CSV append at an epoch end
      save_latest(view)      persists one full checkpoint from a save_view
      write_obligations()    writes due-but-missing snapshots (idempotent)
      stop_requested()       signal flag

    Invariants maintained here:
    - An epoch is advanced and its permutation discarded ONLY when all its
      minibatches are consumed and its epoch-end bookkeeping is done.
    - A segment/target limit and a signal stop preserve an unfinished epoch
      identically (same mid-epoch save path).
    - Post-update obligations (snapshots) run before a stop can break out, so a
      stop exactly on a snapshot step still writes the snapshot.
    - Due epoch-end evaluation runs before any checkpoint at a boundary; the
      schedule pointer advances exactly once per performed evaluation.
    - latest.pt is written at eval epoch ends (durability parity with the
      historical trainer) and at every exit with unsaved updates — never twice
      for the same state.
    """
    while state.global_step < env["target_steps"] and not stop_requested():
        if state.current_perm is None:
            # Normalize to plain ints so the permutation round-trips through
            # checkpoints (and JSON) exactly, independent of the callback's
            # return type (tensor or list).
            state.current_perm = [int(v) for v in draw_perm()]
            if len(state.current_perm) != env["n_train"]:
                raise CheckpointError("drawn permutation has wrong length")
            state.batch_cursor = 0
        advanced = False
        while (
            state.batch_cursor < env["n_batches"]
            and state.global_step < env["target_steps"]
            and not stop_requested()
        ):
            start = state.batch_cursor * env["batch_size"]
            idx = state.current_perm[start : start + env["batch_size"]]
            exec_batch(idx)
            state.global_step += 1
            state.steps_this_run += 1
            state.batch_cursor += 1
            state.unsaved_updates += 1
            if state.batch_cursor >= env["n_batches"]:
                _epoch_end(state, env, eval_and_log, save_latest)
                write_obligations()
                state.epoch += 1
                state.current_perm = None
                state.batch_cursor = 0
                advanced = True
                break
            write_obligations()
        if advanced:
            continue
        return _exit_save(state, env, save_latest, stop_requested)
    return _exit_save(state, env, save_latest, stop_requested)


def _epoch_end(state, env, eval_and_log, save_latest):
    """Epoch-end bookkeeping: due evaluation, schedule advance, durability save."""
    performed = False
    if state.epoch >= state.next_eval_epoch:
        eval_and_log(state.epoch, state.global_step)
        # Post-evaluation pointer is anchored to the completed epoch, so a
        # stale pointer is normalized to beyond it in one bounded step rather
        # than replaying missed historical epochs.
        state.next_eval_epoch = next_eval_after(state.epoch)
        performed = True
    if performed and state.unsaved_updates > 0:
        save_latest(
            save_view(
                state.epoch, state.current_perm, state.batch_cursor, env["n_batches"]
            )
        )
        state.unsaved_updates = 0
    return performed


def _exit_save(state, env, save_latest, stop_requested):
    reason = "stopped" if stop_requested() else "target_reached"
    if state.unsaved_updates > 0:
        save_latest(
            save_view(
                state.epoch, state.current_perm, state.batch_cursor, env["n_batches"]
            )
        )
        state.unsaved_updates = 0
    return reason
