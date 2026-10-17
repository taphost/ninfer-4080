# Artifact Container Porting Analysis: Version 3

This document analyses porting the NInfer `.ninfer` **version-3** container onto the RTX 4080
engine **without modifying the current version-2 reader**. It is a planning and feasibility study,
not a new authority. The current v2 container remains the active reference in
[`artifact-container.md`](artifact-container.md). This file documents what would change, what is
still unknown, and the recommended implementation order.

## 1. Purpose and scope

The RTX 4080 engine reads a version-2 `.ninfer` artifact. The upstream `Neroued/ninfer` project
publishes a newer **version-3** artifact that carries more runtime facts than v2. This analysis
answers one question:

> Can we add a v3 read path to the 4080 engine while leaving the existing v2 path byte-for-byte
> unchanged?

The scope is the **reader** only. It does not yet decide whether any v3 feature (bindings,
components, proposal head) is executable on the 4080 Op set. That is a later, model-by-model
decision.

## 2. What the current v2 reader actually does

The v2 reader lives in `src/artifact/reader.cpp` and `src/artifact/reader.h`. It is a single
`Reader::Impl` constructor that hard-codes the v2 contract:

- **Magic (8 bytes):** `N I N F E R 0x00 <version>`, where the version byte is `2`.
- **Prefix (16 bytes):** bytes 0-7 magic, bytes 8-15 little-endian `json_bytes`.
- **JSON directory:** placed at offset 16, length `json_bytes`, then the payload is aligned to
  4096 bytes.
- **Root JSON members:** exactly `{ "identity", "objects" }`.
- **Identity members:** exactly `{ "model_id", "weights_id" }`.
- **Tensor members:** `{ "name", "kind", "shape", "format", "layout", "offset", "bytes" }`.
- **Resource members:** `{ "name", "kind", "encoding", "offset", "bytes" }`.
- **Registries:** the format, layout, and encoding registries are closed enums in `reader.h`
  (`NumericFormat`, `StorageLayout`, `ResourceEncoding`). Only `StorageLayout` and
  `ResourceEncoding` carry a `...V1` suffix (e.g. `ContiguousLeV1`, `RawBytesV1`); `NumericFormat`
  values (`BF16`, `FP32`, `Q4G64_F16S`, `NVFP4`, ...) do not.
- **Rejection messages:** `"artifact is shorter than the v2 prefix"` and
  `"artifact magic is not NInfer v2"`.

The reader is a pure directory parser: it validates framing, resolves object offsets, checks
alignment and overlap, and exposes payloads by name. It performs no model execution.

## 3. v2 vs v3: what differs

The upstream `Neroued/ninfer` project ships the v3 writer/reader in `tools/artifact/`
(`framing.py`, `schema.py`, `reader.py`). Reading that code removes the "unknown framing" from
Section 4 and fixes the concrete byte-level differences. The v3 artifact is **not** a v2 file with
the version byte changed: the framing header, the object naming, and the JSON directory schema are
all different.

### 3.1 Framing header

| Aspect | v2 (current) | v3 (upstream) |
|---|---|---|
| Header size | 16 bytes | 32 bytes |
| Header layout | `magic(8) + json_bytes(8)` | `magic(8) + json_bytes(8) + artifact_id(16)` |
| Entry magic | `NINFER\0\2` | `NINFER\0\3` |
| Continuation magic | n/a | `NINPRT\0\3` |

The v3 header is `struct("<8sQ16s")` (little-endian: 8-byte magic, 8-byte `json_bytes`, 16-byte
`artifact_id`), so `HEADER.size = 32`. The JSON directory is placed at offset 32 and the payload is
aligned to 4096 bytes, exactly as v2 aligns at 16.

### 3.2 Object naming

| | v2 | v3 |
|---|---|---|
| Object identity key | `"name"` | `"id"` |
| Tensor members | `name, kind, shape, format, layout, offset, bytes` | `id, kind, shape, format, layout, offset, bytes` |
| Resource members | `name, kind, encoding, offset, bytes` | `id, kind, encoding, offset, bytes` |

### 3.3 JSON directory schema

| | v2 | v3 |
|---|---|---|
| Root members | `{ identity, objects }` | `{ components, objects, bindings, uses, files }` + optional `{ metadata, provenance }` |
| Components | none | `text`, `vision`, `mtp`, `dflash2`, each with `config` and optional `target`, `resources`, `proposal` |
| Bindings | none | maps a logical parameter to one tensor or a list of `(object, [begin, end))` parts |
| Uses | none | `{ parameter, input }` + optional `activation_policy` (A16Only/AllowA8/AllowA4) and `auxiliaries` |
| Files | implicit single file | explicit `files` array; `files[0].path` is null (entry), continuations carry sibling paths |

### 3.4 Multi-part files

The v3 entry may be split into continuation files:

```text
model.ninfer                 (header 32B, magic NINFER\0\3)
model.ninfer.part-0001       (header magic NINPRT\0\3)
model.ninfer.part-0002       ...
```

Every part shares a 16-byte `artifact_id` carried in the entry header. The reader opens parts
lazily (`_file(index)` on first use) and validates each continuation header before reading. The v2
reader is single-file only.

**Empirically, the real artifact is single-file.** The analyzed `Huihui-Qwen3.8-27B-...` artifact
has one `files` entry (`path: null`, ~12.79 GB), so `parse_v3()` for the 4080 needs multi-file
support only defensively; the entry always carries its full payload.

### 3.5 Why this matters for the 4080 engine

The 4080 engine already supports text, vision, MTP, and DFlash via v2. Those capabilities are
baked into the engine's compiled model code, so the v3 `components`/`bindings`/`uses` metadata is
**redundant for us**: we already know the architecture. The v3 metadata exists for portability
(other engines), not for the original author. What we likely need from v3 is only (1) the 32-byte
framing and (2) the weight objects, consumed by our existing binders. The real risk is the object
naming change (`name` -> `id`): if our binders key off object names, a v3 file breaks all lookups
until the key is adapted.

### 3.6 Object resolution is name-based (verified in the 4080 binders)

The 4080 model binders resolve objects by **name**, not by id. The `Binder` in
`src/artifact/binder.cpp` routes every lookup through `reader_.find(name)`:

```cpp
bool Binder::has_object(std::string_view name) const { return reader_.find(name) != nullptr; }
ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto* object = reader_.find(name);   // resolved by name
    ...
}
ObjectHandle Binder::require_tensor(std::string_view name, ...) {
    const auto handle = find_unconsumed(name); // resolved by name
    ...
}
```

The model plan (`src/targets/qwen3_6_27b/impl/load/bindings.cpp`) builds those logical names
dynamically and binds them. Each layer sets `prefix = "text/layers/<n>/"`, and weights are bound
through helpers that resolve the name through the same `find(name)` path:

```cpp
const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
...
target.mlp.down = bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S, {5120, 17408});
```

`bind_weight` forwards `name` to `bind_tensor`, which resolves it via the binder's `find(name)`. The
resolved name (`text/layers/0/mlp/down`) must match the artifact object's `.name`. There is no
literal `"tokenizer.json"` request in this plan; resources follow the same `find(name)` path.

**Consequence:** the binders never see the JSON field `"name"` or `"id"` directly; they only see
the internal `ObjectDescriptor.name`. Therefore the v3 reader can keep the binders unchanged by
mapping the v3 `"id"` field onto `ObjectDescriptor.name`. The binding code does not need to touch
`"id"`. The one open question — whether the v3 `"id"` carries the same logical names the binders
expect (e.g. `"text/layers/0/mlp/down"`), or a different scheme — is now answered by the real v3
artifact. The v3 `bindings` map does **not** expose logical names as keys. Of the 12 names the
4080 binders request, **0 have a direct match** in the v3 `bindings`. The v3 instead maps each
logical parameter to a tensor id through `bindings`, so the reader must resolve names via
`bindings` -> tensor id, not by a direct name match.

**Empirical confirmation on the real v3 artifact.** The above is the general v2-vs-v3 contract.
The real v3 artifact checked today (`Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_S-mtp
.infer`, 12.79 GB) confirms it and adds three facts:

- **Integrity.** All 1422 binding references resolve to a real object (0 missing), and all 375
  multi-part ranges are sorted and non-overlapping (0 bad). The file is self-consistent.
- **Resources.** All 6 resources are present and non-empty (e.g. `resource/text/tokenizer.json`,
  12.8 MB). No binding references a resource; resources load separately.
- **Architecture.** The text config is `Qwen3_5ForCausalLM` with 64 layers (48 `linear_attention`
  + 16 `full_attention`), vocab 248320 — a Qwen3.5 with state, not a plain causal-LM.

**Upstream is now accessible.** The `sergiuszm/ninfer-4090` repository (branch `catchup4`) is
fetchable from this machine and holds the v3 writer/reader in `tools/artifact/`
(`reader.py`, `writer.py`, `schema.py`). The framing claims in Sections 3.1-3.4 (32-byte header,
`NINPRT\0\3` continuation magic, `dflash2` component, `files` array) can therefore be verified
against that code rather than assumed. Only the entry magic (`NINFER\0\3`), the 32-byte header
layout, the `"id"` key, and the root schema were confirmed against the real artifact directly; the
upstream framing should be read before implementing `parse_v3()`.

### 3.7 Format compatibility: why `parse_v3()` over a v3->v2 conversion

The v2 reader validates that a tensor's stored `bytes` equals
`tensor_encoded_size(layout, format, shape)` (`storage_layouts.cpp`). So a v3->v2 conversion is
not a byte copy: it must translate each v3 format/layout to a v2 one and recompute the size.

Verified against the real artifact (1177 tensors):

- **Same physical layout for shared formats.** The 791 tensors that use a v3 format/layout with a
  v2 equivalent convert **byte-exact** (`0` size mismatch): `contiguous_le_v1`
  (`bf16`/`fp32`/`int32`) -> v2 `contiguous-le-v1`, and `row_split_k128_v1`
  (`q*_g64_fp16`, `q8_g32_fp16`) -> v2 `row-split-k128-v1`. The conversion is a name+shape
  translation; no weight rewriting.
- **386/1177 tensors have no v2 equivalent.** All use layout `gguf_blocks_v1` with GGUF formats
  (`gguf_iq4_xs`, `gguf_iq3_s`, `gguf_iq3_xxs`, `gguf_iq2_xs`, `gguf_q2_k`, `gguf_q4_k`,
  `gguf_iq2_xxs`, `gguf_q6_k`, `gguf_iq2_s`, `gguf_iq1_m`). The v2 reader has no format for these.

A full-model v3->v2 conversion would therefore drop ~33% of the weights and produce a v2 file the
reader loads but that is missing most of its parameters. This rules out conversion as the path for
this model and selects **`parse_v3()`**: the reader reads the GGUF formats natively, so there is no
format wall. (Extending the v2 reader with a `gguf_blocks` layout + GGUF formats is the only other
v2-native option, and is a larger change.)

### 3.8 How the real v3 reader works

The upstream v3 reader (`tools/artifact/reader.py`, branch `catchup4`) is a **demand-driven,
multi-file** reader. It is not a v2-style single mmap. Three differences matter for `parse_v3()`:

- **Lazy continuation opening.** `open()` opens only the entry. Continuation files
  (`part-N`) are opened on first use (`_file(index)`), after validating each continuation header
  (`PART_MAGIC`, matching `index`, and the shared `artifact_id`).
- **Lazy validation.** Encoding is validated on first access, not upfront:
  `validate_encoding(obj)` runs once per object id (`self._validated`).
- **Chunked payload access.** `iter_range(offset, length)` splits a logical range across
  continuation files using `bisect_right` on per-file prefixes, with bounded temporary memory
  (`READ_CHUNK_BYTES`).

The reader also validates entry length (`entry_bytes == payload_offset + files[0].payload_bytes`)
and builds `by_id = {obj.id: obj}`. It resolves objects **by id**, not by name. The 4080's
`find(name)` contract does not exist upstream; it must be reconstructed by `parse_v3()`.

```python
# tools/artifact/reader.py (upstream catchup4), abridged
def __init__(self, path):
    fd = os.open(path, os.O_RDONLY)
    magic, json_bytes, self.artifact_id = HEADER.unpack(os.pread(fd, HEADER.size, 0))
    if magic != MAGIC: raise ArtifactError("expected NInfer v3 entry magic")
    self.payload_offset = align_up(HEADER.size + json_bytes, PAYLOAD_ALIGNMENT)
    if os.fstat(fd).st_size != self.payload_offset + self.directory.files[0].payload_bytes:
        raise ArtifactError("entry length does not match declared payload")
    self.by_id = {obj.id: obj for obj in self.directory.objects}
```

### 3.9 The real v3 artifact: what else is in it

Beyond framing and objects, the analyzed `Huihui-Qwen3.8-27B-...` artifact carries four more roots
(`metadata`, `provenance`, `uses`) and structural facts that `parse_v3()` must handle.

- **No `identity` member.** The v3 root has exactly `{ bindings, components, files, metadata,
  objects, provenance, uses }` — **no** `identity`, unlike v2's required `{ identity, objects }`.
  `parse_v3()` must not copy the v2 identity check. Identity comes from `metadata.name`
  (`"huihui-qwen3.8-27b-..."`) or the `components`.
- **Single file.** `files` has one entry (`{ "path": null, "payload_bytes": 12790287104 }`); the
  entry holds its full payload. Multi-file continuation is upstream-only; `parse_v3()` needs it
  only defensively.
- **Multi-part bindings are flat.** Of 1422 bindings, 1047 are single-part and 375 are multi-part;
  **every** multi-part binding has exactly one part, and nesting depth is 1. Payload slicing is a
  single `(offset, bytes)` cut per binding — no concatenation.
- **`uses` is runtime metadata.** 785 entries map a `parameter` to an `input` with an
  `activation_policy` (`AllowA8`: 523, `A16Only`: 262). It drives activation caching, not the reader.
- **`proposal` is the indexed MTP head.** `text.proposal = { "domain": "indexed", "rows": 131072 }`.
  The 4080 already runs MTP via v2; this is architecture metadata, not a new requirement.
- **`metadata`/`provenance` are informational.** `metadata` is a name; `provenance` records the
  converter (`ninfer-v3`), recipe (`qwen3_8_27b_gguf`), and build paths. Neither is read by the
  reader.
- **`components.<comp>.resources` maps logical names to ids**, e.g.
  `"tokenizer.json" -> "resource/text/tokenizer.json"`. Resources resolve through this map, the
  same way `bindings` resolve tensors.

These are all portability extras for other engines. For the 4080, `parse_v3()` recognizes them and
moves on; only framing, objects, `bindings`, and `components.<comp>.resources` feed the existing
binders.

## 4. What we still need before writing code

The v3 schema is now known from the upstream `tools/artifact/` code (Section 3). Only one input
remains before implementing `parse_v3()`:

1. **At least one real v3 artifact** to inspect and test against. The current official downloads
   are v3, so a sample is obtainable independently of this machine.

The remaining risk is not the framing (it is fixed in Section 3), but whether our existing binders
can consume the v3 weight objects as-is. That risk is now **verified closed**:

- **The binders key off `name`, not `id`.** Every artifact lookup in the 4080 routes through
  `reader_.find(name)` (`Binder` in `src/artifact/binder.cpp`, backed by `reader_.index.find(name)`
  in `src/artifact/reader.cpp`); nothing resolves objects by `id`.
- **The binder validates the descriptor.** `require_tensor` checks each tensor's `format`, `layout`,
  and `shape` against the target contract; `require_resource` checks its `encoding`. So `parse_v3()`
  must populate those fields — the format wall (Section 3.7) and the `RawBytesV1` resource mapping.
- **No binder change needed.** Because the v3 `id` is remapped to the logical binding `name` at the
  reader boundary (Section 5.1), callers never see a raw `id`. This closes the remaining risk above.

**Identity is already decided, but must be concretized.** The v2 `Reader` returns an
`ArtifactIdentity` (`model_id`, `weights_id`) that the engine uses for caching and publication, but
v3 has no `identity` member (Section 3.9). The approach is already chosen — identity comes from
`metadata.name` — but the convention is not: how `metadata.name` maps to `model_id`/`weights_id`
(and whether the engine needs both) is a small implementation detail to pin down when `parse_v3()`
is written.

## 5. Recommended architecture: version-dispatch

Keep the v2 path intact and branch on the version byte:

```text
read magic (bytes 0-6)
read version (byte 7)
if version == 2 -> parse_v2()      // existing code, unchanged
else if version == 3 -> parse_v3() // new, isolated
else -> throw "unsupported artifact version"
```

Implementation notes:

- Move the current v2 logic into a `parse_v2()` function called only when `version == 2`. The v2
  framing, registries, and rejection messages stay identical.
- Add a `parse_v3()` function for `version == 3`. It owns its own header layout, JSON schema, and
  resource model. It shares only the low-level `MappedFile` and payload-access helpers.
- The public `Reader` interface (`identity()`, `objects()`, `find()`, `payload()`,
  `read_direct()`) stays the same, so callers do not change.
- Do **not** change the v2 registries or messages. A v3-only registry is added alongside.

This keeps the v2 contract as the active authority and isolates v3 as a new, testable branch.

### 5.1 Name resolution: bindings -> tensor id

The 4080 binders request logical names (`find("text/layers/0/mlp/down")`), but a v3 object carries
a generic id (`"weight/000019"`). The logical name does not live on the object; it lives on a v3
map. `parse_v3()` must therefore rebuild the `find()` lookup by **inverting** the v2 strategy: the
v2 reader keys objects by their own `"name"` field, the v3 reader keys by the map that points at
them.

The v3 artifact exposes two name->id maps:

- **`bindings`** maps a logical tensor parameter to a tensor id, or to a list of `(object,
  [begin, end))` parts.
- **`components.<component>.resources`** maps a logical resource name to a resource id (e.g.
  `"tokenizer.json" -> "resource/text/tokenizer.json"`). No binding references a resource;
  resources resolve through this parallel map.

```jsonc
"bindings": {
  "text/layers/0/mlp/down": { "object": "weight/000019" },
  "text/layers/0/mlp/gate_up": {
    "parts": [
      { "object": "weight/000005", "range": [0, 20000000] },
      { "object": "weight/000006", "range": [0, 14000000] }
    ]
  }
}
```

The resolution algorithm builds a name-keyed map that `find()` consumes:

```python
by_id = {o["id"]: o for o in objects}
name_to_descriptor = {}

# tensors via bindings
for logical_name, spec in bindings.items():
    if "object" in spec:
        obj = by_id[spec["object"]]
        name_to_descriptor[logical_name] = make_descriptor(logical_name, obj)
    else:
        name_to_descriptor[logical_name] = make_parts_descriptor(logical_name, spec["parts"], by_id)

# resources via components.<comp>.resources
for comp in components.values():
    for logical_name, obj_id in comp.get("resources", {}).items():
        name_to_descriptor[logical_name] = make_descriptor(logical_name, by_id[obj_id])
```

**The key:** the `ObjectDescriptor` built here does **not** carry the v3 object `"id"`. Its `name`
is the logical key (`"text/layers/0/mlp/down"`), so `find("text/layers/0/mlp/down")` returns the
right object while the binder never sees the generic `"weight/000019"`. The binding code stays
untouched.

**Multi-part:** when a `bindings` entry has `"parts"`, the descriptor payload is sliced per part:
`offset = object.offset + part.range[0]` and `bytes = part.range[1] - part.range[0]`. Multiple
parts concatenate in order. This is the one behaviour the v2 reader cannot express, because a v2
object is a single contiguous payload.

**Plumbing:** `parse_v3()` fills the same `index` map that `find()` reads and sets
`ObjectDescriptor.name` to the logical key. Offset resolution, alignment, overlap checks, and
payload access are reused from the v2 machinery unchanged. `parse_v2()` keeps its own path; the
version byte selects which reader populates the index.

### 5.2 parse_v3(): replicate vs skip

The concrete v3 reader (`tools/artifact/reader.py`) shows what `parse_v3()` must keep and what it
can drop. The 4080 reads a local, single-file entry and already knows the architecture, so much of
the upstream reader is unnecessary.

- **Replicate (needed by the 4080):**
  - 32-byte framing (`magic + json_bytes + artifact_id`) and 4096 payload alignment.
  - Entry-length validation (`entry_bytes == payload_offset + files[0].payload_bytes`).
   - Identity from `metadata.name` or `components` — the v3 root has **no** `identity` member
     (Section 3.9), so the v2 identity check does not apply.
  - `by_id` object resolution and the `bindings`/`components.<comp>.resources` -> id maps.
  - Name reconstruction: assign the logical binding name to `ObjectDescriptor.name` so the
    existing `find(name)` works (Section 5.1).
  - Chunked, multi-part payload access (`iter_range`-style) only if an artifact is split.

- **Skip (upstream-only concerns, not needed locally):**
  - Lazy continuation opening — the 4080 reads a local entry, not streamed parts.
  - Lazy `validate_encoding` — validate once at parse time.
  - `components`/`bindings`/`uses` as *executable* metadata — the 4080 already knows the model
    from v2 (Section 3.5).
  - `provenance`/`metadata` — informational only.

This keeps `parse_v3()` minimal: framing, name reconstruction, and payload access. Nothing
upstream-specific leaks in.

## 6. Risks

- **Object key naming.** The v3 directory uses `"id"` where v2 uses `"name"`. If the 4080 binders
  resolve objects by name, a v3 file breaks every lookup until the key is adapted at the reader
  boundary. This is the highest-risk change and must be verified before coding (Section 4).
- **Feature vs. format.** Accepting a v3 file is not the same as executing its features. Bindings,
  components, and the proposal head are redundant for the 4080 engine, which already supports
  text, vision, MTP, and DFlash via v2. Only the framing and weight objects are needed. Any v3-only
  feature (e.g. the indexed proposal head) must be checked against the 4080 Op set before runtime
  use.
- **Multi-part files.** Large v3 artifacts may be split into `.part-NNNN` files with a
  `NINPRT\0\3` continuation header. The reader and CLI path must agree on the entry path and keep
  all parts together.
- **No v3 sample on this machine.** A real v3 artifact and the upstream `catchup4` code are now
  available, so the v3 path can be validated against a real artifact on a capable machine. The
  format wall in Section 3.7 (386 GGUF tensors have no v2 equivalent) applies only to a v3->v2
  conversion, not to `parse_v3()`.

## 7. Work plan

1. Obtain one real v3 artifact for inspection and testing.
2. Verify whether the 4080 model binders resolve objects by name. If so, decide how to adapt the
   v3 `id` key at the reader boundary so callers never see a raw `id`.
3. Refactor `Reader::Impl` into `parse_v2()` + `parse_v3()` with version-dispatch; leave v2
   untouched. The v3 path owns the 32-byte header, the richer JSON schema, and multi-part reads.
4. Implement `parse_v3()` for the confirmed schema (Section 3).
5. Compile and validate against the real v3 artifact (on a capable machine), plus regression
   of the existing v2 tests.
6. Decide, per model, which v3 features are executable on the 4080 Op set (text, vision, MTP,
   DFlash already exist via v2; only v3-only features need checking).

## 8. Decision requested

Proceed with the version-dispatch plan in Section 5 (`parse_v3()` reads v3 natively). The v3 schema
is now fixed (Section 3), the upstream reader is accessible, and the object-key verification
(Section 3.6) is answered. Section 3.7 closes the remaining alternative: a v3->v2 conversion is
ruled out for this model because 386/1177 tensors (GGUF block quant) have no v2 equivalent and a
conversion would drop ~33% of the weights. This document does not yet implement anything; it fixes
the location, the approach, the concrete byte-level differences, the format decision, and the next
steps.
