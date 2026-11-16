# Week 11 Day 1：bi-encoder vs cross-encoder（re-ranking 的直覺）
## 1. bi-encoder vs cross-encoder 的資料流

**bi-encoder**：會先將自己壓縮成一條向量，所以最後是 query 一個 vector、doc 一個 vector，再比較兩個向量。This is fast.

- 事先算好的只有**文件**向量（存進索引）；**問題**向量要等問題來了才算。
- 缺點是**粗**：壓成一個向量時，字和字的對應（例如「上限」對「最大」）就丟掉了。

**cross-encoder**：looks at every word in both query and docs and compares them.

- 精確說：問題和文件先接成一條序列 `[CLS] query [SEP] doc [SEP]`，每個 token 的 **q** 跟所有 token 的 **k** 做點積（attention），包括問題的字對文件的字，而且**每一層都做**。
- 最後的分數**不是點積**：是取 `[CLS]` 位置的向量，接一個 Linear 層，輸出一個數字。

## 2. 為什麼 cross-encoder 不能事先算好、存進索引

Because it needs cross-encoding and we don't know the query in advance, so we can only calculate once we get the query and the doc at the same time.

- 關鍵差別：bi-encoder 每份文件的計算**跟問題無關** → 可以先算好存起來；cross-encoder 計算的對象是「**這個問題 × 這份文件**」這一對 → 問題沒來，就沒有東西可以先算。
- 注意：bi-encoder 也不能事先「比較」，比較一樣要等問題來；能事先做的只有「把文件變成向量」。

## 3. 兩階段：檢索 → 重排

We don't have to choose one: use bi-encoder first, and for the rest we use cross-encoder to rerank.

- 第一階段（bi-encoder）：依分數排序，**留下前 k 名**。目標是**不要漏掉**對的文件（recall）。它不會判斷「符合 / 不符合」。
- 第二階段（cross-encoder）：只重排這 k 份，目標是把對的**排到最前面**。
- 第 k 名以後的文件，reranker 根本看不到。

## 4. 第 10 週 chunking vs 第 11 週 re-ranking；為什麼 rerank 救不了 chunking

Chunking is chopping paragraphs into small chunks. If you chunk it badly, the meaning gets lost in different pieces, then even reranking can't help you, because the meaning is already lost.

- 後果一：答案沒有完整地在任何一個 chunk 裡 → 每一塊的向量都不太像問題 → **可能根本進不了 top-k**。
- 後果二：就算兩半都被撈到，cross-encoder **一次只看一對（問題, 一個 chunk）**，每塊都只有半個答案，分數都不高；它沒辦法把兩塊拼回去。
- 順序：第 10 週先拉高上限（對的東西撈得到），第 11 週再提高準度（排得好）。量 rerank 的效果時，要**固定 chunking 設定**，一次只改一個東西。

## 5. cross-encoder 裡面的 attention

- 〔補·修正〕cross-encoder 用的**就是 self-attention**，沒有「self cross-attention」這個東西。判斷規則：**Q 和 K、V 來自同一條序列 → self-attention；來自兩條不同的序列 → cross-attention。** cross-encoder 把兩段接成一條，所以是 self-attention。
- **沒有 causal mask**：輸入的一整對都已經在手上，沒有「未來」可言，所以整個矩陣都算。nanoGPT 要遮**上三角**，是因為生成時未來的字還沒產生。

## 6. cross-encoder vs cross-attention

- **cross-encoder** = 把問題和文件用膠帶黏成一張長紙，讓一個人從頭讀到尾（self-attention）。名字裡的「cross」是指兩段文字在同一張紙上交叉。
- **cross-attention** = 兩張紙分開放。Q 從一段來（例如正在寫的英文譯文），K、V 從另一段來（例如日文原文）。attention 矩陣是長方形（譯文長度 × 原文長度），不需要 mask。用在翻譯這類 encoder-decoder 模型；GPT 是 decoder-only，沒有這一層。
- 三者對照：bi-encoder 不互相看；cross-encoder 接成一條，雙向都看；cross-attention 分開放，從一段去查另一段。

## 7. 點積（內積）怎麼算

做內積：`total += a[i] * b[i]`，每個位置相乘，然後 **add them ALL**，得到一個數字。

90° → 0，180° → −1。

- 0°（同方向）→ 1，最像。
- 這個範圍只在向量長度都是 1 時成立，也就是 `normalize_embeddings=True` 的作用。
- 直覺：方向相近時，兩個向量在大部分位置的正負號一樣，`a[i]*b[i]` 多半是正的，加起來就大。
- 實際的文字向量很少真的到 90°：員工餐廳那份文件跟問題無關，分數卻是 0.417。所以 bi-encoder 的分數**不能當成相關的機率**來讀。

## 8. 為什麼向量固定是 1024 維

Because the person who created the model says so.

- 等於 nanoGPT 的 `n_embd`（bge-m3 建在 XLM-RoBERTa-large 上）。
- 不管文字多長，都會被壓成 1024 個數字 → chunk 太長，細節會被稀釋（連回第 4 點）。

---

## 9. 實驗

**環境**
- bi-encoder：`BAAI/bge-m3`（`normalize_embeddings=True`，分數 = 點積 = cosine）
- cross-encoder：`BAAI/bge-reranker-v2-m3`（`CrossEncoder.predict`，這個版本輸出 0–1）

**Run 1** — query：有給休暇の繰越上限は何日？

| 文件 | bi | cross |
|---|---|---|
| 有給休暇は翌年度に最大20日まで繰り越せる。 | 0.749 | 0.993 |
| 2026年度より有給休暇の取得ルールを変更し、特別休暇を新設した。 | 0.573 | 0.001 |
| 社員食堂の営業時間は11時から14時までです。 | 0.417 | 0.000 |

**Run 2** — query：退職金が出ない条件は？

| 文件 | bi | cross |
|---|---|---|
| 勤続3年未満で自己都合退職した場合、退職金は支給されない。 | 0.737 | 0.707 |
| 勤続3年以上の社員には退職金が支給される。 | 0.590 | 0.014 |
| 通勤手当は月額3万円を上限として支給する。 | 0.462 | 0.000 |

**Run 3** — 同上 query，只換第二份文件

| 文件 | bi | cross |
|---|---|---|
| 勤続3年未満で自己都合退職した場合、退職金は支給されない。 | 0.737 | 0.707 |
| 勤続3年未満の場合、退職金は支給される。 | 0.673 | 0.116 |
| 通勤手当は月額3万円を上限として支給する。 | 0.462 | 0.000 |

**Run 4** — query：勤続3年未満だと退職金は出ない？（兩份文件只差「ない」）

| 文件 | bi | cross |
|---|---|---|
| 勤続3年未満の場合、退職金は支給されない。 | 0.880 | 0.999 |
| 勤続3年未満の場合、退職金は支給される。 | 0.841 | 0.998 |
| 通勤手当は月額3万円を上限として支給する。 | 0.521 | 0.000 |

### 發現

- **第一名從來沒有不一樣**：四次實驗，兩個模型排出的第一名都一樣（Run 4 差距太小，等於平手）。差別在**分得多開**：Run 1 正確答案和第二名的差距，bi 是 0.176，cross 是 0.992。
- **預測 vs 實際**：原本預測 bi-encoder 在 Run 1 會排錯、cross-encoder 在 Run 4 分得出「ない」，兩個都猜錯了。只看示意不夠，要自己跑。
- **一題證明不了 precision 比較高**：要說「precision 比較高」，需要一整組評估題和指標（Day 3 的有無 re-rank 對照表）。
- **Run 2：cross 只有 0.707** → 如果看 Run 1 就把拒答門檻設在 0.9，這題的正確答案會被拒答掉。門檻要用一整組評估題校準，不能看一兩題決定（第 12、13 週）。
- **Run 3：用字越接近，bi-encoder 越難分**：第二份文件也有「3年未満」之後，bi 的差距從 0.147 縮到 0.064。
- **Run 4：兩個模型都分不出「ない」**。cross-encoder 沒有做錯：問題是是非題，兩份文件都**直接在回答這個問題**，只是答案相反。reranker 判斷的是「**相關**」，不是「**正確**」。對照 Run 2：問「出ない条件は？」時，「支給される」那份沒有回答，只拿到 0.014。**同一份文件，換一個問法，相關性就不一樣。**
- **所以矛盾的文件要靠其他階段處理**：第 10 週的 metadata（用日期判斷哪份是現行版本），以及第 12 週的引用忠實度（檢查回答是否真的被引用的文件支持）。

### 面試可以講的版本

> "I expected the cross-encoder to separate a negated minimal pair. It scored both at 0.999, because for a yes/no question both documents are relevant: they just answer it differently. That taught me relevance and correctness are different problems, so the reranker can't be my guard against contradictory documents. That's handled by metadata and citation-faithfulness checks downstream."
