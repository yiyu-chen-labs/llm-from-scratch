# Week 10 反思 — Chunking 評估（Step 1–2）

## Week 9 回顧

**Done**：用 JQaRA 把問題和候選段落轉成向量，用 cosine similarity 排序，再看正解排第幾。

---

## Week 10 計畫

| # | 步驟 | 狀態 |
|---|---|---|
| 0 | 確認模型（MiniLM） | ✅ |
| 1 | 準備資料（JSQuAD → 長文件 + label） | ✅ |
| 2 | 固定長度 × overlap，算切斷率 | ✅ |
| 3 | 接上檢索：hit@k、hit@budget | ⬜ ← 下一步 |
| 4 | 結構感知切法（按句子、段落） | ⬜ |
| 5 | 語意切法 | ⬜ |
| 6 | metadata 注入（把標題放回 chunk） | ⬜ |
| 7 | staleness（JSQuAD 沒有日期，只能測機制，不能證明效果） | ⬜ |
| 8 | config 化 | ⬜ |
| 9 | 消融表 + 信賴區間 | ⬜ |

---

## Step 1：把 JSQuAD 改造成考卷

### 為什麼用 JSQuAD

評估 chunking 需要三樣東西：

1. 長文件
2. 問題
3. 答案的確切位置

JQaRA 資料集的段落已經切好了，沒得再切，所以改用 JSQuAD。JSQuAD 有問題和答案位置（`answer_start`）。

> **JSQuAD** = Japanese 版的 SQuAD（Stanford Question Answering Dataset），屬於 Yahoo Japan 和早稻田大學做的 JGLUE。

### 載入時遇到的問題

新版 `datasets` 不支援需要執行腳本的資料集，JGLUE 和 JaQuAD 都載不了：

```
RuntimeError: Dataset scripts are no longer supported, but found JGLUE.py
```

**解法**：直接從 GitHub 下載原始 JSON。

```python
!wget -O jsquad_train.json "https://raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.3/train-v1.3.json"
```

### 原始資料格式（SQuAD 格式）

結構：文章（`title`）→ 段落（`context`）→ 問題（`question` + `answers[text, answer_start]`）

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

- 最外層只有一個 key：`data`
- `context` 的開頭一定是 `標題 [SEP] `（這張「貼紙」要撕掉）
- `answer_start` 是從 `context` 第 0 個字開始數，包含貼紙

### 處理後的格式（`jsquad_docs.json`）

```json
[
  {
    "title": "造語",
    "doc": "造語（ぞうご）は、新たに語…\n（第 2 段本文）\n…",
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

- `doc`：同一篇文章所有段落撕掉貼紙後，用換行接成的長文件
- `start` / `end`：答案在 `doc` 裡的位置，滿足 `doc[start:end] == answer`

### `build_doc` 的邏輯

1. 每段開頭有「標題 [SEP] 」這張貼紙，要撕掉
2. 答案位置要減掉貼紙長度（例如「造語 [SEP] 」= 9 格）
3. 減完是負數 = 答案標在貼紙上 → 去本文找；找不到就丟掉
4. 換算成長文件位置：`offset + 本文位置`
5. 每段處理完，`offset += 段落長度 + 1`（+1 是換行）
6. 用換行把所有本文接成長文件 `doc`
7. 驗證：`doc[start:end] == 答案`

> ⚠️ **常見錯誤**
> - 更新 `offset` 的時機放錯（要等這段所有題目都算完）
> - 忘記 +1

### 檢驗結果

| 項目 | 數值 |
|---|---|
| 文章數 | 710 |
| 原始題數 | 62,697 |
| 保留題數 | 59,568（95%） |
| start > 0 | 79.5% |
| start = 0 | 9.3% |
| start < 0（答案在貼紙上） | 11.1%（其中 5% 被丟掉） |
| 長文件長度（字） | 最短 48｜中位數 2,264｜75% 5,048｜最長 59,624 |
| **驗證失敗** | **0**（6 萬題的 label 全部正確） |

### 限制（非常重要，比結果重要）

1. **被丟掉的 5% 不是隨機的**：幾乎都是「答案 = 標題」型的題目
2. **單段落出題**：出題者每次只看一段，所以沒有跨段落的題目。考卷會偏袒小 chunk
3. **用字重疊高**：很多題目的用字跟原文很像，搜尋比真實情況簡單
4. **Wikipedia 文體**：不是商務文件
5. **非標準用法**：這不是 JSQuAD 的標準用法，是改造。數字不能跟別人比
6. **長文件主導總分**：長文件題目多，要同時報「按題目」和「按文件」的平均

---

## Step 2：切斷率

**切斷率**：答案沒有完整落在任何一塊裡的比例，也就是被刀子切成兩半。

### 先預測

$$
\text{切斷率} \approx \frac{\text{答案長度} - 1}{\text{chunk 長度}}
$$

答案長度：中位數 5 字、平均 6.1 字、最長 19 字。

### 結果

| L | 預測 | 實際（overlap = 0） | 實際（15% overlap） |
|---|---|---|---|
| 200 | 2.57% | 2.56% | 0.00%（overlap = 30） |
| 400 | 1.28% | 1.24% | 0.00%（overlap = 60） |
| 800 | 0.64% | 0.59% | 0.00%（overlap = 120） |

### 解釋

- **預測和實際幾乎一致**：原理對，程式也對
- **實際值略低，L 越大差越多**：推論是短於 L 的文件沒有邊界（尚未驗證）
- **overlap ≥ 18（最長答案 − 1）就保證不切斷**：30 / 60 / 120 都夠，而且偏多
- **overlap 有成本**：塊變多、embedding 變多、搜尋結果可能重複
- **overlap 的另一個作用（保留前後文）切斷率量不到**：要到 Step 3 才看得出來

### 結論

在 JSQuAD 上，切斷不是問題，這一步只是 sanity check。

但在商務／法律／政府文件上，證據可能長達 150 字。代入公式，L = 200 時切斷率約 75%。**同一套機制，結論可能完全相反，所以需要注意。**

---

## 其他學到的概念

### Benchmark vs Ablation

- **Benchmark**：考卷（題目 + 評分方式）
- **Ablation**：用考卷比較不同做法，一次只改一個變數

### Chunking 不只用在 RAG

純搜尋、長文件摘要、合約抽取、預訓練（nanoGPT 的 `block_size` 也是一種切法）。目標不同，「切得好」的定義就不同。

### Multimodal chunking

處理有表格、圖、掃描頁的文件。三種做法：

1. 先轉文字，再按版面切
2. 用 VLM 產生描述來檢索
3. 直接 embed 頁面圖片

排在文字版之後做，但它跟 JD 的連結比 staleness 強。

### RAG 評估指標（Week 13 預習）

| 端 | 指標 | 意思 |
|---|---|---|
| 檢索端 | context precision | 撈回來的有多少是相關的 |
| 檢索端 | context recall | 需要的有沒有被撈到 |
| 生成端 | faithfulness | 答案有沒有被段落支持 |
| 生成端 | answer relevance | 有沒有回答到問題 |

- RAGAS 用 LLM 當評審，要先用人工標的小樣本檢查評審準不準
- **faithfulness ≠ 正確**：段落本身錯了，答案「忠實」也一樣是錯

### 換資料時

指標和方法可以照搬，但：

- **label 要重建**（最花時間）
- **「答案」的定義也可能要擴充**（長條款、多處證據、表格）
