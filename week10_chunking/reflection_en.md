# Week 10 notes: evaluating chunking (steps 1–2)

## Week 9 recap

Done: used JQaRA, turned the questions and candidate passages into vectors, ranked them by cosine similarity, then checked where the correct passage ended up.

---

## Week 10 plan

| # | Step | Status |
|---|---|---|
| 0 | Check which model I'm actually using (MiniLM) | ✅ |
| 1 | Prep data (JSQuAD → long docs + labels) | ✅ |
| 2 | Fixed-length chunks × overlap, measure cut rate | ✅ |
| 3 | Hook up retrieval: hit@k, hit@budget | ⬜ ← next |
| 4 | Structure-aware chunking (by sentence / paragraph) | ⬜ |
| 5 | Semantic chunking | ⬜ |
| 6 | Metadata injection (put the title back into chunks) | ⬜ |
| 7 | Staleness (JSQuAD has no dates, so I can only test the mechanism, not prove it helps) | ⬜ |
| 8 | Make the chunker configurable | ⬜ |
| 9 | Ablation table + confidence intervals | ⬜ |

---

## Step 1: turning JSQuAD into a test set

### Why JSQuAD

To evaluate chunking I need three things:

1. long documents
2. questions
3. the exact position of each answer

JQaRA was already chunked, so there was nothing left to cut. JSQuAD has questions and answer positions (`answer_start`), so I switched to that.

> JSQuAD = the Japanese version of SQuAD (Stanford Question Answering Dataset). It's part of JGLUE, made by Yahoo Japan and Waseda University.

### Loading problem

Newer versions of `datasets` don't support script-based datasets anymore, so neither JGLUE nor JaQuAD would load:

```
RuntimeError: Dataset scripts are no longer supported, but found JGLUE.py
```

Workaround: just grab the raw JSON from GitHub.

```python
!wget -O jsquad_train.json "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.3/train-v1.3.json"
```

### Raw format (SQuAD-style)

article (`title`) → paragraph (`context`) → question (`question` + `answers[text, answer_start]`)

```json
{
  "data": [
    {
      "title": "造語",
      "paragraphs": [
        {
          "context": "造語 [SEP] 造語（ぞうご）は、新たに語（単語）を造ることや、既存の語を組み合わせて新たな意味の語を造ること、また、そうして造られた語である。…",
          "qas": [
            {
              "question": "新たに語（単語）を造ることや、既存の語を組み合わせて新たな意味の語を造ること",
              "id": "a1000888p0q0",
              "answers": [
                { "text": "造語", "answer_start": 0 }
              ],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

- the top level only has one key: `data`
- every `context` starts with `title [SEP] ` (I've been calling this the "sticker", it has to go)
- `answer_start` counts from character 0 of `context`, sticker included

### After processing (`jsquad_docs.json`)

```json
[
  {
    "title": "造語",
    "doc": "造語（ぞうご）は、新たに語…\n(paragraph 2 body)\n…",
    "qas": [
      {
        "question": "新たに語（単語）を造ることや、…",
        "answer": "造語",
        "start": 0,
        "end": 2
      }
    ]
  }
]
```

- `doc`: all paragraphs from one article, sticker removed, joined with newlines
- `start` / `end`: where the answer sits in `doc`, so `doc[start:end] == answer`

### How `build_doc` works

1. each paragraph starts with the `title [SEP] ` sticker, strip it
2. subtract the sticker length from the answer position (e.g. `造語 [SEP] ` = 9 chars)
3. negative result = the answer was labeled on the sticker → search the body for it, drop the question if it's not there
4. convert to a position in the long doc: `offset + position in body`
5. after each paragraph, `offset += paragraph length + 1` (the +1 is the newline)
6. join all bodies with newlines → `doc`
7. check: `doc[start:end] == answer`

> ⚠️ Mistakes I need to watch for
> - updating `offset` at the wrong time (it has to wait until every question in that paragraph is done)
> - forgetting the +1

### Sanity check results

| | |
|---|---|
| articles | 710 |
| questions (raw) | 62,697 |
| questions kept | 59,568 (95%) |
| start > 0 | 79.5% |
| start = 0 | 9.3% |
| start < 0 (answer on the sticker) | 11.1% (5% ended up dropped) |
| doc length (chars) | min 48 / median 2,264 / 75th pct 5,048 / max 59,624 |
| failed checks | 0 (all ~60k labels line up) |

### Limitations (honestly more important than the results)

1. The dropped 5% isn't random. It's almost all "the answer is the article title" type questions.
2. Questions were written from a single paragraph each, so nothing needs info from multiple paragraphs. That probably favors small chunks.
3. A lot of questions reuse the wording from the source text, so retrieval is easier than it'd be in real life.
4. It's Wikipedia, not business docs.
5. This isn't how JSQuAD is normally used, I repurposed it. My numbers can't be compared with anyone else's.
6. Long docs have way more questions, so they dominate the overall score. I should report both per-question and per-document averages.

---

## Step 2: cut rate

Cut rate = the share of answers that don't fully fit inside any single chunk, i.e. the knife went right through them.

### Prediction first

$$
\text{cut rate} \approx \frac{\text{answer length} - 1}{\text{chunk length}}
$$

Answer length: median 5 chars, mean 6.1, max 19.

### Results

| L | predicted | actual (no overlap) | actual (15% overlap) |
|---|---|---|---|
| 200 | 2.57% | 2.56% | 0.00% (overlap = 30) |
| 400 | 1.28% | 1.24% | 0.00% (overlap = 60) |
| 800 | 0.64% | 0.59% | 0.00% (overlap = 120) |

### What this means

- prediction and reality basically match, so the reasoning is right and the code is right
- actual is a bit lower than predicted, and the gap grows with L. My guess: docs shorter than L don't have any boundary at all. Haven't checked this yet.
- overlap ≥ 18 (longest answer − 1) guarantees nothing gets cut, so 30 / 60 / 120 are all enough. More than enough, actually.
- overlap isn't free: more chunks, more embeddings, and search results can end up being near-duplicates
- the other thing overlap does (keeping context around the boundary) doesn't show up in cut rate at all. Need step 3 for that.

### Takeaway

On JSQuAD, cutting answers just isn't a problem. This step is a sanity check, nothing more.

But in business / legal / government docs the "answer" could easily be a 150-char clause. Plug that into the formula and you get roughly 75% cut rate at L = 200. Same mechanism, completely opposite conclusion. Need to keep that in mind.

---

## Other stuff I picked up

### Benchmark vs ablation

- benchmark: the test (questions + how you score them)
- ablation: using that test to compare approaches, changing one thing at a time

### Chunking isn't only a RAG thing

Plain search, summarizing long docs, pulling clauses out of contracts, pretraining (nanoGPT's `block_size` is basically chunking too). Different goal, different definition of "good chunking".

### Multimodal chunking

For docs with tables, figures, scanned pages. Three ways people do it:

1. convert to text first, then chunk by layout
2. have a VLM write descriptions and retrieve on those
3. embed the page images directly

Doing this after the text version. Still, it lines up with the JD more than staleness does.

### RAG eval metrics (preview for week 13)

| side | metric | meaning |
|---|---|---|
| retrieval | context precision | how much of what came back is actually relevant |
| retrieval | context recall | did the stuff we needed actually get retrieved |
| generation | faithfulness | is the answer supported by the passages |
| generation | answer relevance | does it actually answer the question |

- RAGAS uses an LLM as the judge, so I should check the judge against a small hand-labeled sample first
- faithfulness ≠ correct. If the passage itself is wrong, a "faithful" answer is still wrong.

### Switching to other data

Metrics and method carry over, but:

- labels have to be rebuilt (this is the slow part)
- what counts as an "answer" might need to change too (long clauses, evidence spread across places, tables)
