# Dictionary spelling corrections

The **Dictionary** tab has **Add entry**, a search field for spellings and aliases,
and **Change** / **Remove** on every entry. Each change saves immediately after
confirmation in the entry editor. Aliases have separate fields, so punctuation
such as semicolons stays part of the alias. Search ignores case and covers the
whole dictionary; only 50 rows are displayed per page. Reload refreshes changes
made by other clients. Load errors stay in the tab, with editing disabled until
the dictionary loads successfully.

If the running service predates Dictionary support, the tab explains that the
updated service needs restarting. Rebuilding a binary alone does not replace an
already-running process.

Select a word or phrase in the transcript. **Desired spelling** opens above the
highlight (GTK can place it below if there is insufficient room). Enter the
spelling and press Enter or **Save spelling**. Selection with the keyboard and
**Save spelling…** in the text context menu also open the editor.
Leading and trailing whitespace is trimmed before prefilling the spelling and
learning the alias; internal spaces remain part of the phrase. A whitespace-only
selection does not open the editor. Successful saves update the text without a
confirmation message; errors remain visible for correction.

Saving uses the immutable ASR spelling as the alias, even if earlier replacements
have shifted the displayed text. It updates matching text in that speech section,
uses the added rule for subsequent sections of the same recording, and teaches
the global dictionary for new recordings. Capture continues. Other completed
sections and saved Stop copies stay unchanged. To apply the latest dictionary
throughout an older recording, reprocess with `audio: false`.

Click the pill beside **Replacements** to control the result: when off, saving learns
the rule but keeps the displayed/copied transcript original. Hovering a
replacement shows its original wording. Original ASR text and old revision/edit
records are retained.

A selection must cover complete words or phrases. Select a whole replacement,
not only part of its displayed text. If the transcript or replacement state has
changed, reselect it. An alias already assigned to another spelling is rejected;
use **Change** on that entry in the **Dictionary** tab instead. Errors stay in
the spelling editor.

## Programmatic additions

`POST /dictionary/entries` atomically merges one entry into the current dictionary:

```json
{"canonical":"Paraketto","aliases":["paraketo","para keto"]}
```

It preserves unrelated entries and existing aliases, rejects conflicting mappings,
and returns the current dictionary with its `revision`. Repeating the same
addition does not create a new revision. Programmatic additions affect new
recordings; existing recordings keep their pinned rules.

`GET /dictionary` returns the complete list. `PUT /dictionary` replaces it;
include `expected_revision` from the GET response to reject a stale full-list save
with HTTP 409.

`PUT /dictionary/entries/{index}` changes one entry. Send `canonical`, `aliases`
and the required `expected_revision`. `DELETE /dictionary/entries/{index}` removes
it with a JSON body containing `expected_revision`. The index is zero-based in
the latest GET response and is valid only for that revision. Stale writes return
409 before touching any entry. Both routes return the updated dictionary; the
desktop uses them for Change/Remove and preserves unrelated entries. Conflicts
stay in the editor so its contents remain available before cancelling/reloading.

`POST /sessions/{id}/spelling` is the editor operation. Obtain the section ID,
revision and text from `/sessions/{id}/updates`, then send:

```json
{
  "segment_id": 12,
  "revision_id": 18,
  "generation": 1,
  "start": 20,
  "end": 28,
  "selected": "paraketo",
  "canonical": "Paraketto",
  "replacements_enabled": true
}
```

The IDs and offsets above are illustrative. `start`/`end` are half-open **UTF-8
byte offsets in the displayed section**, not character indices or whole-document
offsets. The exact `selected` bytes, section revision, generation and replacement
state are checked before any dictionary write. The server maps the selection
back through existing edits and commits the dictionary plus new section revision
atomically. The response is the updated session metadata; refresh its updates
to obtain the new text. Stale revisions/state return 409; invalid selections or
conflicting rules return 400. Errors contain an `error` string.

## Matching and latency

The service caches immutable prepared dictionaries by revision. A compact
Aho–Corasick trie scans each incoming segment and selects the leftmost longest
complete phrase. Replacements never cascade. ASCII is folded without per-character
allocations; Unicode case folding retains exact source byte boundaries, including
expansions such as `ß → ss`.

Cache hits perform no dictionary JSON parsing, rule rebuilding or dictionaries-table
query. The LRU retains up to four revisions with a 64 MiB target; one larger
active dictionary is retained rather than repeatedly compiled. Limits: 10,000
entries, 1,000 aliases per entry, 4,096 bytes per term, and 4 MiB of terms in total.

Local CPU measurements on 12th Gen Intel(R) Core(TM) i7-12700, in milliseconds:

| Entries | Input words | Previous path | Cached matcher |
| ---: | ---: | ---: | ---: |
| 10 | 100 | 0.210 | 0.004 |
| 10 | 500 | 0.603 | 0.021 |
| 100 | 100 | 0.577 | 0.004 |
| 100 | 500 | 1.671 | 0.022 |
| 1,000 | 100 | 4.450 | 0.004 |
| 1,000 | 500 | 10.566 | 0.021 |
| 10,000 | 100 | 46.171 | 0.004 |
| 10,000 | 500 | 106.054 | 0.021 |

These are dictionary-only medians of 11 samples after three warmups, using the
same dictation-like UTF-8 text. They exclude ASR, database operations and HTTP.
The previous path reparses and prepares rules per segment; the new path reuses
its matcher. All eight output hashes and edit counts agree. Preparation costs
about 33 ms for 10,000 entries, paid when a revision is compiled rather than on
every segment. [Machine-readable results and provenance](dictionary-performance.json).

```sh
make bench-dictionary
make test-session-store test-sessions test-desktop-pane
```

The benchmark reports preparation separately from warmed application. It can
also build against the saved prior implementation with `DICTIONARY_PREPARED=0`;
the local before/after sources, binaries and raw samples are in
`scratch/dictionary-popover/`. CMake provides the `bench-dictionary` target when
`PARAKETTO_BUILD_SERVICE=ON`.

Validation includes 750 randomized comparisons against an independent brute-force
matcher, Unicode boundaries/expansions, overlapping phrases, non-cascading edits,
cache eviction, concurrent additions, an in-flight ASR result after a correction,
stale selections, and unchanged original/publication text. The private GTK test
requires an actually mapped popover and checks its anchor after scrolling.
