# Design Template

## Problem

Xây dựng research assistant có thể nhận câu hỏi phức tạp, tìm thông tin từ web, phân tích và viết câu trả lời cuối cùng có trích dẫn nguồn.

## Why multi-agent?

Single-agent bị giới hạn bởi context window và không có chuyên môn hoá. Tách thành nhiều agent giúp:
- Researcher tập trung tìm kiếm và thu thập nguồn
- Analyst tập trung đánh giá chất lượng thông tin
- Writer tập trung viết câu trả lời rõ ràng có citation
- Mỗi bước có thể trace, debug, và cải thiện độc lập

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Điều phối routing, quyết định agent tiếp theo | ResearchState | route_history | Loop vô hạn → giới hạn bởi max_iterations |
| Researcher | Tìm kiếm web, tóm tắt nguồn | query | sources, research_notes | Search API fail → ghi vào state.errors, supervisor dừng |
| Analyst | Phân tích notes, đánh giá evidence | research_notes | analysis_notes | LLM fail → ghi vào state.errors |
| Writer | Viết câu trả lời có citation | analysis_notes, sources | final_answer | LLM fail → ghi vào state.errors |
| Critic | Fact-check câu trả lời cuối | final_answer | fact_check | LLM fail → ghi vào state.errors |

## Shared state

| Field | Lý do cần |
|---|---|
| `request` | Query gốc của user, tất cả agent cần biết |
| `sources` | Researcher ghi, Writer đọc để cite |
| `research_notes` | Researcher → Analyst handoff |
| `analysis_notes` | Analyst → Writer handoff |
| `final_answer` | Writer → Critic handoff, output cuối |
| `fact_check` | Critic ghi, output bổ sung |
| `route_history` | Supervisor track để biết ai đã chạy |
| `trace` | Log từng bước để debug |
| `errors` | Capture failure, supervisor dùng để fail fast |
| `iteration` | Supervisor dùng để enforce max_iterations |

## Routing policy

```
START → supervisor
supervisor:
  errors != []        → done (fail fast)
  research_notes None → researcher → supervisor
  analysis_notes None → analyst    → supervisor
  final_answer None   → writer     → supervisor
  fact_check None     → fact_checker → supervisor
  else                → done
```

## Guardrails

- Max iterations: `MAX_ITERATIONS=6` trong `.env`
- Timeout: `TIMEOUT_SECONDS=60` trong `.env`
- Retry: chưa implement — hiện tại fail fast qua `state.errors`
- Fallback: Supervisor route `done` ngay khi có error
- Validation: Pydantic schema cho tất cả input/output

## Benchmark plan

| Query | Metric | Expected (baseline) | Expected (multi-agent) |
|---|---|---|---|
| "What is RAG?" | Latency | ~20s | ~40s |
| "What is RAG?" | Cost USD | ~$0.0005 | ~$0.0015 |
| "What is RAG?" | LLM calls | 1 | 3-4 |
| "What is RAG?" | Citation coverage | 0% | 100% |
| "What is RAG?" | Sources | 0 | 5 |
