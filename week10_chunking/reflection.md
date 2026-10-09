1. Week 9 回饋與修正
Done: 用JQaRA把問題和候選段落轉成向量，用cosine similarity排序,再看正解排第幾。

2. Week 10 計畫
確認模型 ✅（MiniLM）
準備資料（JSQuAD → 長文件 + label）✅
固定長度 × overlap，算切斷率 ✅
接上檢索：hit@k、hit@budget ← 下一步
結構感知切法（按句子、段落）
語意切法
metadata 注入（把標題放回 chunk）
staleness（JSQuAD 沒有日期，只能測機制，不能證明效果）
config 化
消融表 + 信賴區間

3. Step1: 把JSQuAD改造成考卷
為什麼要用 JSQuAD：評估 chunking 需要三樣東西：長文件、問題、答案的確切位置。
JQaRA資料及因為已經切好了沒得再切了所以用JSQuAD資料集
JSQuAD資料集有問題和答案位置（answer_start）

JSQuAD是JP版的 SQuAD（Stanford Question Answering Dataset），
屬於 Yahoo Japan 和早稻田大學做的 JGLUE
p.s.載入時遇到的問題：新版 datasets 不支援需要執行腳本的資料集，JGLUE 和 JaQuAD 都載不了。解法是直接從 GitHub 下載原始 JSON：
raw.githubusercontent.com/yahoojapan/JGLUE/main/datasets/jsquad-v1.3/train-v1.3.json

資料格式（SQuAD 格式）：文章（title）→ 段落（context）→ 問題（question + answers[text, answer_start]）

build_doc 的邏輯

每段開頭有「標題 [SEP] 」這張貼紙，要撕掉
答案位置要減掉貼紙長度（例如「造語 [SEP] 」= 9 格）
減完是負數 = 答案標在貼紙上 → 去本文找；找不到就丟掉
換算成長文件位置：offset + 本文位置
每段處理完，offset += 段落長度 + 1（+1 是換行）
用換行把所有本文接成長文件 doc
驗證：doc[start:end] == 答案

常見錯誤：更新 offset 的時機放錯（要等這段所有題目都算完）、忘記 +1。

檢驗結果

710 篇文章、62697 題，保留 59568 題（95%）
start > 0：79.5%｜= 0：9.3%｜< 0：11.1%（其中 5% 被丟掉）
長度：最短 48｜中位數 2264｜75% 5048｜最長 59624
驗證失敗 0：6 萬題的 label 全部正確


限制（非常重要比結果重要）

被丟掉的 5% 不是隨機的，幾乎都是「答案 = 標題」型的題目
單段落出題：出題者每次只看一段，所以沒有跨段落的題目。考卷會偏袒小 chunk
很多題目的用字跟原文很像，搜尋比真實情況簡單
Wikipedia 文體，不是商務文件
這不是 JSQuAD 的標準用法，是改造。數字不能跟別人比
長文件題目多，總分會被它們主導：要同時報「按題目」和「按文件」的平均


4. Step 2：切斷率

切斷率：答案沒有完整落在任何一塊裡的比例，也就是被刀子切成兩半。

先預測：切斷率 ≈ (答案長度 − 1) / chunk 長度。答案中位數 5 字、平均 6.1 字、最長 19 字。

結果

L	預測	實際	15% overlap
200	2.57%	2.56%	0.00%
400	1.28%	1.24%	0.00%
800	0.64%	0.59%	0.00%

解釋

預測和實際幾乎一致：原理對，程式也對
實際值略低，L 越大差越多。推論：短於 L 的文件沒有邊界。這還沒驗證
overlap ≥ 18（最長答案 − 1）就保證不切斷，30/60/120 都夠，而且偏多
overlap 有成本：塊變多、embedding 變多、搜尋結果可能重複
overlap 的另一個作用（保留前後文）切斷率量不到，要到 Step 3 才看得出來

結論：在 JSQuAD 上，切斷不是問題，這一步只是 sanity check。
但在商務/法律/政府文件上，證據可能長達 150 字，代入公式，L=200 時切斷率約 75%。同一套機制，結論可能完全相反所以需要注意。

5. 其他學到的概念
Benchmark vs ablation：benchmark 是考卷（題目 + 評分方式），ablation 是用考卷比較不同做法，一次只改一個變數
chunking 不只用在 RAG：純搜尋、長文件摘要、合約抽取、預訓練（nanoGPT 的 block_size 也是一種切法）。目標不同，「切得好」的定義就不同
Multimodal chunking：處理有表格、圖、掃描頁的文件。三種做法：先轉文字再按版面切、用 VLM 產生描述來檢索、直接 embed 頁面圖片。排在文字版之後做，但它跟 JD 的連結比 staleness 強
RAG 評估指標（Week 13 預習）
檢索端：context precision（撈回來的有多少是相關的）、context recall（需要的有沒有被撈到）
生成端：faithfulness（答案有沒有被段落支持）、answer relevance（有沒有回答到問題）
RAGAS 用 LLM 當評審，要先用人工標的小樣本檢查評審準不準
faithfulness ≠ 正確：段落本身錯了，答案「忠實」也一樣是錯
換資料時：指標和方法可以照搬，但 label 要重建（最花時間），「答案」的定義也可能要擴充（長條款、多處證據、表格）
