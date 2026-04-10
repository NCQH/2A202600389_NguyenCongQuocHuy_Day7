# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Công Quốc Huy
**Nhóm:** C401-A2
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector embedding có hướng rất giống nhau, tức là nội dung của hai câu có ý nghĩa gần giống nhau dù cách diễn đạt có thể khác.

**Ví dụ HIGH similarity:**
- Sentence A: “Tôi muốn đặt vé máy bay đi Hà Nội”
- Sentence B: “Tôi cần mua vé máy bay đến Hà Nội”
- Tại sao tương đồng: Hai câu có cùng ý định (intent): đặt/mua vé máy bay đến Hà Nội → vector biểu diễn gần cùng hướng nên cosine similarity cao.

**Ví dụ LOW similarity:**
- Sentence A: “Tôi muốn đặt vé máy bay đi Hà Nội”
- Sentence B: “Trời hôm nay rất đẹp và nhiều nắng”
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau (du lịch vs thời tiết) → vector hướng khác nhau → cosine similarity thấp.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào độ giống về hướng (semantic meaning) và không bị ảnh hưởng bởi độ dài vector, trong khi Euclidean distance bị ảnh hưởng bởi magnitude nên kém ổn định hơn khi so sánh embedding text.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> **Trình bày phép tính:**
>
> - Chunk size = 500  
> - Overlap = 50  
> - Stride = 500 - 50 = 450  
> 
> Số chunks:
> \[
> N = \left\lceil \frac{10000 - 500}{450} \right\rceil + 1
> \]
>
>\[
>N = \left\lceil \frac{9500}{450} \right\rceil + 1
>= \lceil 21.11 \rceil + 1
>= 22 + 1 = 23
>\]
>
>**Đáp án:** **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Overlap = 100 → stride = 500 - 100 = 400 (giảm so với 450)
Khi stride giảm, số chunks sẽ **tăng lên** vì mỗi lần trượt đi ít hơn.
>**Lý do dùng overlap lớn hơn:**
Overlap lớn giúp các chunk **giữ lại nhiều ngữ cảnh liên tục hơn**, giảm mất thông tin ở ranh giới giữa các đoạn. Điều này đặc biệt quan trọng trong RAG để tránh việc câu hoặc ý nghĩa bị cắt đôi giữa các chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hybrid retrieval benchmark (5 tài liệu nội bộ mẫu của giảng viên + 1 tài liệu OCR về hướng dẫn nấu ăn do cá nhân crawl/xử lý).

**Tại sao nhóm chọn domain này?**
> Nhóm dùng 5 tài liệu có sẵn của giảng viên làm baseline để so sánh chunking và retrieval ổn định. Đồng thời, nhóm bổ sung 1 tài liệu OCR do cá nhân tự thu thập để kiểm tra độ bền của pipeline khi gặp dữ liệu nhiễu và cấu trúc không đồng nhất. Cách chọn này giúp đánh giá rõ hiệu quả metadata/filter trong cả điều kiện “sạch” và “thực tế”.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
1   | customer_support_playbook.txt | data/customer_support_playbook.txt | 1692     | source, extension, doc_type, department, language       |
| 2   | rag_system_design.md          | data/rag_system_design.md          | 2391     | source, extension, doc_type, department, language       |
| 3   | vector_store_notes.md         | data/vector_store_notes.md         | 2123     | source, extension, doc_type, department, language       |
| 4   | vi_retrieval_notes.md         | data/vi_retrieval_notes.md         | 1667     | source, extension, doc_type, department, language       |
| 5   | chunking_experiment_report.md | data/chunking_experiment_report.md | 1987     | source, extension, doc_type, department, language       |
| 6   | huong_dan_nau_an.md           | data/huong_dan_nau_an.md           | 195560   | source, doc_type, department, language, domain, cuisine |

### Metadata Schema

| Trường metadata | Kiểu   | Ví dụ giá trị                 | Tại sao hữu ích cho retrieval?                 |
| --------------- | ------ | ----------------------------- | ---------------------------------------------- |
| cuisine         | string | Vietnamese                    | Hữu ích cho retrieval món ăn                   |
| source          | str    | rag_system_design.md          | Truy vết nguồn chunk sau khi retrieve          |
| doc_type        | str    | playbook / notes / design_doc | Lọc theo loại tài liệu cho đúng ngữ cảnh       |
| department      | str    | support / platform            | Giảm nhiễu khi query theo team                 |
| language        | str    | vi / en                       | Tránh lấy sai ngôn ngữ khi câu hỏi có scope rõ |
| domain          | str    | cooking / ai / system_design  | Phân nhóm theo lĩnh vực                        |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
|huong_dan_nau_an.md| FixedSizeChunker (`fixed_size`) |20|484| Trung bình|
|huong_dan_nau_an.md | SentenceChunker (`by_sentences`) |130 |70 | Khá tốt nhưng chunk dài|
|huong_dan_nau_an.md | RecursiveChunker (`recursive`) |253 |35 |Tốt  |
|huong_dan_nau_an.md | SectionChunker (`by_section`) |36 |254 | Tốt nhưng Chunk dài|
|chunking_experiment_report.md| FixedSizeChunker (`fixed_size`) |5|413|Trung bình |
|chunking_experiment_report.md | SentenceChunker (`by_sentences`) |8 |247 |Tốt nhưng Chunk dài |
|chunking_experiment_report.md | RecursiveChunker (`recursive`) |11 |179 |Tốt  |
|chunking_experiment_report.md | SectionChunker (`by_section`) |6 |327 |Tốt nhưng Chunk dài |
|vi_retrieval_notes.md| FixedSizeChunker (`fixed_size`) |4|432|Trung bình |
|vi_retrieval_notes.md | SentenceChunker (`by_sentences`) |7 |237 |Khá tốt |
|vi_retrieval_notes.md | RecursiveChunker (`recursive`) |6 |276 | Tốt|
|vi_retrieval_notes.md | SectionChunker (`by_section`) |4 |416 |Tốt nhưng Chunk dài |

### Strategy Của Tôi

**Loại:** [SectionChunker(custom)]

**Mô tả cách hoạt động:**
> *Chunk theo từng section trong dữ liệu nấu ăn, mỗi section bắt đầu bằng 1 cụm ##, nếu vượt quá max_tokens(cài đặt trước) thì chunk thêm*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Vì phù hợp với cấu trúc của file huong_dan_nau_an.md*

**Code snippet (nếu custom):**
```python
class SectionChunker:
    """
        Split text into sections based on headers (lines starting with ##, ### is not included).
        Max chunk size -> int
    """
    
    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        sections = re.split(r"(?m)^##\s+", text.strip())
        chunks: List[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # If section is too long, split it into smaller fixed-size chunks
                for i in range(0, len(section), self.chunk_size):
                    chunk = section[i:i + self.chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)

        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
|huong_dan_nau_an.md | best baseline (recursive) |253 |35 | OK|
| huong_dan_nau_an.md| **của tôi (sectionChunk)** | 36|254 |OK |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SectionChunker(Custom) | 8/10 | Đưa ra chính xác section cần |                  Số lượng chunk lớn làm tăng thời gian nhúng (embedding) và tìm kiếm ban đầu.|
|Nguyễn Anh Tài                 | RecipeChunker (Custom) | 8/10                  | Giữ trọn vẹn ngữ cảnh từng bước nấu; lọc nhiễu tốt nhờ tách biệt tiêu đề và nội dung. | Số lượng chunk lớn làm tăng thời gian nhúng (embedding) và tìm kiếm ban đầu.         |
| Hoàng Bá Minh Quang  | RecursiveChunker       | 8/10                  | Cân bằng giữa độ ngắn chunk và giữ ngữ cảnh                                           | Nhiều chunk hơn, tốn index hơn                                                       |
| Trần Quang Long      | RecursiveChunker       | 10/10                 | giữ trọn vẹn danh sách nguyên liệu hoặc trọn vẹn một bước nấu ăn trong cùng một chunk | Không có điểm yếu                                                                    |
| Vũ Minh Quân         | RecursiveChunker       | 8/10                  | giữ cửa sổ ngữ cảnh, tránh bị loãng thông tin                                         | nhiều chunk gây tốn thời gian trích xuất và tìm kiếm                                 |
| Đỗ Lê Thành Nhân     | SentenceChunker        | 7/10                  | Đảm bảo tính toàn vẹn về mặt ngữ nghĩa của từng câu đơn lẻ.                           | AI khó liên kết giữa nguyên liệu và hành động nấu nếu chúng nằm ở các câu khác nhau. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Chọn RecursiveChunker vì nó phân tách văn bản theo thứ tự ưu tiên từ lớn đến nhỏ (như xuống dòng rồi mới đến dấu chấm), giúp giữ trọn vẹn các khối thông tin liên quan như danh sách nguyên liệu hay quy trình chế biến trong cùng một đoạn. Chiến lược này tạo ra sự cân bằng tối ưu giữa kích thước chunk và tính toàn vẹn ngữ cảnh, đảm bảo AI luôn truy xuất được các chỉ dẫn nấu ăn hoàn chỉnh thay vì những mảnh vụn rời rạc.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex (?<=[.!?])\s+ để tách câu, dựa trên dấu kết thúc câu (., !, ?) và khoảng trắng phía sau. Cách này giúp giữ lại dấu câu trong câu trước (nhờ lookbehind). Edge case xử lý gồm: nhiều khoảng trắng liên tiếp, xuống dòng (\n), và tránh tách sai khi không có khoảng trắng sau dấu chấm.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán chia văn bản theo thứ tự ưu tiên các separator (\n\n → \n → . → space → char), nếu chunk vẫn quá dài thì tiếp tục đệ quy với separator nhỏ hơn. Base case là khi độ dài chunk ≤ chunk_size hoặc không còn separator để split. Cách này đảm bảo ưu tiên giữ semantic structure trước khi phải cắt thô.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được embed thành vector và lưu dưới dạng record gồm id, content, embedding, metadata (in-memory hoặc ChromaDB). Khi search, query cũng được embed rồi tính cosine similarity với từng vector. Kết quả được sort giảm dần theo similarity và lấy top_k.

**`search_with_filter` + `delete_document`** — approach:
> Filter được áp dụng trước khi tính similarity để giảm không gian tìm kiếm và tăng độ chính xác. Với delete, hệ thống xoá tất cả chunks liên quan đến một document bằng cách match theo id, prefix id, hoặc metadata (doc_id, source). Điều này đảm bảo xoá sạch cả document đã bị chunk ra nhiều phần.

### KnowledgeBaseAgent

**`answer`** — approach:
> Prompt được thiết kế theo dạng: system (instruction) + context (retrieved chunks) + user question. Context được inject trực tiếp vào prompt (thường dạng concatenated text hoặc bullet points). Cách này giúp LLM chỉ dựa vào thông tin đã retrieve (RAG) thay vì hallucinate.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi thích ăn phở|Tôi rất thích ăn phở bò | high | 0.897| Yes |
| 2 | Trời hôm nay nóng|Thời tiết hôm nay oi bức | low |0.61 |No |
| 3 | Tôi học AI| Con mèo đang ngủ| low | 0.369| Yes|
| 4 |Tôi không thích cà phê | Tôi thích cà phê| high |0.857 | No |
| 5 | Tôi thích đọc sách|Tôi thích viết văn | low |0.684 | No|

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp 4, khi hai câu mang nghĩa trái ngược nhưng vẫn có độ tương đồng cao. Điều này cho thấy embeddings thường dựa nhiều vào từ khóa và ngữ cảnh chung hơn là hiểu chính xác yếu tố phủ định. Ngoài ra, các cặp như (2) và (5) cho thấy embeddings đôi khi đánh giá thấp các câu đồng nghĩa hoặc liên quan ngữ nghĩa nhưng khác cách diễn đạt.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| #   | Query                                                                            | Gold Answer                                                                                                 |
| --- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | Các nguyên liệu cần thiết để làm món "Cá ngừ hấp cải rổ" là gì?                  | 1 hộp cá ngừ ngâm dầu, 300g cải rổ, muối, tiêu, đường, nước tương, dầu ăn, tỏi, hành lá, rau mùi và bánh mì |
| 2   | Quy trình thực hiện món "Chả trứng hấp" gồm những bước nào?                      | 1. Trộn tất cả nguyên liệu (trứng, thịt xay, nấm mèo, miến). 2. Hấp chín. 3. Phết lòng đỏ lên mặt.          |
| 3   | Những món ăn nào trong tài liệu sử dụng "nước dừa tươi" làm nguyên liệu?         | Bún tôm – thịt luộc (luộc thịt và pha mắm), Thịt kho tàu (nước dừa tươi), Bò kho (nước dừa tươi)            |
| 4   | Món "Gỏi cuốn" được mô tả như thế nào và thưởng thức kèm với loại nước chấm nào? | Mô tả là món cuốn tươi mát, dễ ăn. Thưởng thức bằng cách chấm tương đen hoặc nước mắm tỏi ớt                |
| 5   | Cách sơ chế và ướp cá trong món "Cá lóc kho tộ" được hướng dẫn ra sao?           | Cá lóc cắt khoanh, ướp với nước mắm, đường, tiêu, hành tím và nước màu trong 20 phút.                       |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Các nguyên liệu cần thiết để làm món "Cá ngừ hấp cải rổ" là gì?|Cá ngừ hấp cải rổ  ### Mô tả  Món ăn đơn giản, giá thành rẻ, chế biến nhanh.  ### Nguyên liệu  - 1 hộp cá ngừ ngâm dầu (... |0.694 |Yes |Các nguyên liệu cần thiết để làm món Cá ngừ hấp cải rổ là: - 1 hộp cá ngừ ngâm dầu (Vissan hoặc Hạ Long) |
| 2 |Quy trình thực hiện món "Chả trứng hấp" gồm những bước nào? | Chả trứng hấp  ### Nguyên liệu  - 2 trứng - 100g thịt xay - Nấm mèo - Miến  ### Cách làm  1. Trộn tất cả nguyên liệu. 2....|0.594 |Yes | Quy trình thực hiện món Chả trứng hấp gồm những bước sau:1. Trộn tất cả nguyên liệu (2 trứng, 100g thịt xay, nấm mèo, miến).2. Hấp chín.3. Phết lòng đỏ lên mặt.|
| 3 | Những món ăn nào trong tài liệu sử dụng "nước dừa tươi" làm nguyên liệu?|Bò trái dừa  ### Nguyên liệu  - 300g thịt bò - Nước cốt dừa - Sả, gừng, tỏi  ### Cách làm  1. Ướp thịt. 2. Xào sơ. 3. Hầ... |0.498 | No|Món ăn trong tài liệu sử dụng nước dừa tươi làm nguyên liệu là "Bò trái dừa". |
| 4 | Món "Gỏi cuốn" được mô tả như thế nào và thưởng thức kèm với loại nước chấm nào?|Gỏi cuốn  ### Mô tả  Món cuốn tươi mát, dễ ăn.  ### Nguyên liệu  - Bánh tráng - Tôm luộc, thịt ba chỉ luộc - Bún tươi - ... |0.687 | Yes| Món Gỏi cuốn được mô tả là một món cuốn tươi mát, dễ ăn. Khi thưởng thức, món này được chấm kèm với tương đen hoặc nước mắm tỏi ớt.|
| 5 | Cách sơ chế và ướp cá trong món "Cá lóc kho tộ" được hướng dẫn ra sao?|Cá lóc kho tộ  ### Mô tả  Món kho đậm đà, hao cơm, đặc trưng miền Nam.  ### Nguyên liệu  - 500g cá lóc (cắt khoanh) - 10... |0.584 |Yes |Cách sơ chế và ướp cá trong món Cá lóc kho tộ được hướng dẫn như sau:1. **Sơ chế cá**: Cá lóc được cắt thành khoanh.2. **Ướp cá**: Cá được ướp với nước mắm, đường, tiêu, hành tím và nước màu trong thời gian 20 phút. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Cách tối ưu hóa Regular Expression để xử lý các trường hợp xuống dòng lạ trong file .txt mà không làm hỏng cấu trúc câu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm bạn có cách tính Eval thực sự rất hay, đáng học hỏi để tự đánh giá model của mình.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ thay đổi gì trong data strategy? Tôi sẽ thêm bước data cleaning (xóa dòng trống, chuẩn hóa dấu câu, xóa kí tự lạ,...) trước khi đưa vào Chunker để các mẩu thông tin sạch sẽ hơn, giúp AI trả lời không bị dính các ký tự lạ.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **90 / 90** |
