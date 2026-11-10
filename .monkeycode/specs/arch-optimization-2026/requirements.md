# Strata 架构优化需求规格

版本：1.0
日期：2026-11-09
状态：草案

---

## 1. 背景与目标

### 1.1 现状问题
Strata 当前采用“反应式三级驻留”架构：专家缓存、KV 缓存、内存探测相对独立，同步粒度粗（以 window 为单位），缺乏统一预算仲裁与自动降级机制。在消费级硬件上运行超大 MoE 时表现为：
- 短对话 decode 峰值尚可，但 P99 延迟波动大（随机长停顿）
- 32K+ 长上下文随对话增长 decode 吞吐衰减 >30%
- 多会话并发时 Expert 与 KV 争抢 VRAM 导致雪崩式性能下降
- ZFS ARC / Page Cache 占用被误判为“可用内存”，触发 OOM 或过度 offload
- 缺少 GET /v1/models/{model_id} 等标准端点，影响 VS Code 等客户端集成

### 1.2 目标
在不重写 llama.cpp/ggml 成熟 Kernel 的前提下，将 Strata 重构为：
- **预测式事件驱动执行层**：per-expert / per-layer 细粒度同步，I/O 前移藏入计算
- **韧性资源治理层**：统一预算仲裁、Greedy-Dual 代价驱逐、五级降级链、ZFS/PageCache 感知
使消费级硬件上的超大 MoE 推理“更稳、更可预测、长对话不衰减、多会话不互踩”。

### 1.3 非目标
- 不追求实验室峰值吞吐翻倍
- 不重写现有 CUDA/HIP/Metal/SYCL kernel
- 不引入 Rust 重写（新模块用 C++，保持代码库一致）
- PD 解耦、MTP/ngram 预取实验留 Phase 4 可选

---

## 2. 范围

### 2.1 In Scope
| 编号 | 需求描述 | 优先级 | 验收阶段 |
|------|----------|--------|----------|
| REQ-01 | ZFS ARC / Page Cache 可回收内存识别，修正 `device_free_bytes()` | P0 | Phase 1 |
| REQ-02 | 补齐 GET /v1/models/{model_id} 端点 | P0 | Phase 1 |
| REQ-03 | MemoryBudgetArbiter：Expert 与 KV 统一预算竞争 | P1 | Phase 1 |
| REQ-04 | DegradationStateMachine：五级降级链 | P1 | Phase 2 |
| REQ-05 | ExpertEventSync：逐专家事件同步 | P1 | Phase 2 |
| REQ-06 | KvPipeline：跨层异步 KV 传输 | P1 | Phase 2 |
| REQ-07 | ResidencyManager：Greedy-Dual 代价驱逐 + 锚点保护集 | P1 | Phase 3 |
| REQ-08 | Prefetcher：置信分级 + 全局带宽硬上限 | P1 | Phase 3 |
| REQ-09 | Anchor Cache + 聚类顺序读取 | P2 | Phase 3 |
| REQ-10 | 感知 PCIe 链路差异的非对称层分配 | P2 | Phase 4 (可选) |

### 2.2 Out of Scope
- Kernel 级优化（GEMM、Attention、MoE routing 等）
- 新模型架构支持（仅针对现有 Flash-Next / Qwen 27B）
- 分布式推理（单机多卡已在范围内）
- Web UI 重构

---

## 3. 功能需求（EARS 格式）

### 3.1 内存探测增强
**REQ-01-ZFS-ARC**
- **WHEN** 系统运行在 Linux 且检测到 `/proc/spl/kstat/zfs/arcstats` 存在
- **THEN** `device_free_bytes()` 读取 `size` 字段，计算 `arc_reclaimable = size * 0.8`，并加入可用内存估算
- **WHEN** 系统无 ZFS 或读取失败
- **THEN** 静默回退到现有 `MemAvailable` 逻辑，不报错

**REQ-01-PAGE-CACHE**
- **WHEN** `MemAvailable` 已包含 page cache 可回收部分
- **THEN** 不再额外叠加 page cache 估算，避免重复计算
- **RATIONALE** Linux `MemAvailable` 定义已包含 page cache + slab 可回收部分

### 3.2 API 端点补齐
**REQ-02-MODEL-DETAIL**
- **WHEN** 客户端请求 `GET /v1/models/{model_id}`
- **THEN** 返回单模型详情，字段与 `GET /v1/models` 列表项一致（id、object、status、meta、architecture、aliases）
- **WHEN** `model_id` 不匹配当前服务模型及其 aliases
- **THEN** 返回 404，错误格式遵循 OpenAI 规范

### 3.3 统一预算仲裁
**REQ-03-BUDGET-ARBITER**
- **WHEN** Engine 启动或每 N 个 decode window（可配置，默知 10）
- **THEN** `MemoryBudgetArbiter::tick(free_vram, expert_cache_cost, kv_pages_cost)` 返回：
  - `target_expert_bytes`：本轮允许驻留的专家总字节数
  - `target_kv_pages`：KV pool 目标页数
  - `degradation_level`：0~4 降级等级
- **WHEN** `free_vram` 连续 3 次 tick 低于 `low_water_mark`（默认 10%）
- **THEN** `degradation_level` 单调递增，触发降级链

### 3.4 五级降级链
**REQ-04-DEGRADATION-CHAIN**
| Level | 名称 | 触发动作 |
|-------|------|----------|
| 0 | Normal | 无 |
| 1 | ShrinkPrefetch | 预取窗口缩小 50% |
| 2 | HighConfOnly | 仅执行 confidence > 0.8 的预取 |
| 3 | NoNgram | 关闭 ngram/MTP 预取（保留路由历史启发式） |
| 4 | NoPrefetch | 完全关闭预取，仅 LRU 被动驱逐 |
| 5 | FreezeLowPrio | 冻结 priority < threshold 的会话（不再分配新 KV page） |

- **WHEN** `degradation_level` 变化
- **THEN** 在下一个 window 前原子应用，不中断进行中请求
- **WHEN** 内存压力缓解（`free_vram > high_water_mark` 连续 5 tick）
- **THEN** `degradation_level` 单调递减，逐级恢复

### 3.5 事件驱动同步
**REQ-05-EXPERT-EVENT-SYNC**
- **WHEN** 某层的 routed experts 准备就绪（host->device copy 完成或已在 VRAM）
- **THEN** 发出 `expert_ready[layer][expert]` Event，后续层的 kernel 可立即 `cudaStreamWaitEvent` 而非等待整个 window barrier
- **WHEN** 启用 `--no-event-sync` 或 `FNX_NO_EVENT_SYNC=1`
- **THEN** 退回现有同步 window 模式（`cudaStreamSynchronize` per window）

### 3.6 KV 跨层流水线
**REQ-06-KV-PIPELINE**
- **WHEN** Layer L 的 K/V 投影完成
- **THEN** 在 `stream_kv[L]` 上 `cudaMemcpyAsync` 到 `KvPages`，并 `cudaEventRecord(kv_ready[L])`
- **WHEN** Layer L+1 的 Attention kernel 启动
- **THEN** `cudaStreamWaitEvent(stream_attn[L+1], kv_ready[L])`，实现跨层重叠
- **WHEN** 统一内存设备（APU/Apple Silicon）
- **THEN** 可选启用 `FNX_KV_RAM_TIER=1`，冷 KV page 落到 page-locked host mem

### 3.7 Greedy-Dual 驱逐与锚点保护
**REQ-07-GREEDY-DUAL**
- **WHEN** `ResidencyManager::evict(target_bytes)` 被调用
- **THEN** 计算每专家 `cost = (gate_up_bytes + down_bytes) / estimated_bandwidth`，`score = heat * cost`（heat 来自 `route_counts` 指数衰减）
- **THEN** 驱逐 `score` 最低的专家，直到释放 ≥ `target_bytes`，**但** `protected_set`（锚点专家）永不驱逐
- **WHEN** `protected_set` 由 `FNX_ANCHOR_EXPERTS` 环境变量指定（层:专家 逗号分隔），或自动推断为路由频率 Top-K

### 3.8 置信分级预取
**REQ-08-PREFETCHER**
- **WHEN** `Prefetcher::tick(route_history, probe_counts)` 运行
- **THEN** 计算每专家 `confidence = P(route_next | route_history)`，仅 `confidence > threshold` 入预取队列
- **THEN** 全局带宽 Token Bucket：`max_prefetch_bps`（默认 20 GB/s），每 tick 补充，预取消耗 token
- **WHEN** `degradation_level >= 1`
- **THEN** `threshold` 线性升高，`max_prefetch_bps` 按级衰减

### 3.9 Anchor Cache 与聚类读取
**REQ-09-ANCHOR-CACHE**
- **WHEN** 模型首次加载或 `FNX_CLUSTER_GGUF=1`
- **THEN** 离线工具 `tools/cluster_gguf.cpp` 将同层专家 tensor 聚类写入单文件，索引记录 offset/size
- **WHEN** 运行时加载该层专家
- **THEN** 单次 `mmap` + 顺序 `pread` 读取整个 cluster，减少系统调用与寻道

---

## 4. 非功能需求

| 编号 | 需求 | 指标 |
|------|------|------|
| NFR-01 | 短对话 decode 吞吐 | 相对旧版 +15% ~ +35% |
| NFR-02 | 32K 长上下文 decode 衰减 | 相对旧版衰减后 +20% ~ +60%，P99 波动降低 ≥30% |
| NFR-03 | 多会话聚合吞吐 | 1.3× ~ 1.8× |
| NFR-04 | 正确性 | 贪心解码逐 token 与旧版一致（`tools/greedy_consistency_test.cpp`） |
| NFR-05 | ZFS 误判 | 0 例（自动化测试 `tools/test_zfs_arc.py`） |
| NFR-06 | 回退完整性 | 所有高级特性可通过环境变量单独关闭，旧路径字节级一致 |
| NFR-07 | 构建通过 | CUDA/HIP/SYCL/Metal 四后端 `cmake --build` 通过 |
| NFR-08 | 静态分析 | ASAN/TSAN 无泄漏竞态 |

---

## 5. 约束与假设

- **硬件**：消费级 NVIDIA (RTX 30/40/50)、AMD (RDNA3/4)、Apple Silicon (M1-M4)、Intel Arc
- **OS**：Linux (kernel 6.1+)、Windows 10/11、macOS 13+
- **依赖**：现有 llama.cpp/ggml kernel 不变；CUDA 12+、HIP 6+、Metal 3、SYCL 2020
- **配置**：所有新特性默认开启，但可通过 `FNX_*` 环境变量或 `--set key=value` 逐个关闭
- **向后兼容**：现有 `strata launch` flags、run config JSON、server API 保持不变

---

## 6. 验收测试矩阵

| 测试用例 | 覆盖需求 | 通过标准 |
|----------|----------|----------|
| `tools/test_zfs_arc.py` | REQ-01 | ZFS 池写入 50 GiB 后 `device_free_bytes()` 仍 ≥ 物理内存 * 0.6 |
| `server_test.py::test_model_detail_endpoint` | REQ-02 | GET /v1/models/fnx 返回 200，字段完整 |
| `tools/memory_arbiter_test.cpp` | REQ-03 | 压力脚本并发 8 会话 32K，VRAM 占用 ≤ 物理 * 0.95，无 OOM |
| `tools/degradation_test.cpp` | REQ-04 | 人为制造内存压力，降级链 0→5 单调，恢复 5→0 单调 |
| `tools/event_sync_test.cpp` | REQ-05 | `--no-event-sync` 下吞吐 ≤ 事件模式 95%，正确性一致 |
| `tools/kv_pipeline_test.cpp` | REQ-06 | 32K context decode 延迟 P99 降低 ≥ 20% |
| `tools/greedy_dual_test.cpp` | REQ-07 | 驱逐字节数 ≥ target，protected_set 零驱逐，heat*cost 单调 |
| `tools/prefetcher_test.cpp` | REQ-08 | 预取命中率 > 60%，带宽不超 `max_prefetch_bps` |
| `tools/cluster_gguf_test.cpp` | REQ-09 | 首次加载时间 ≤ 旧版 1.2×，后续加载 ≥ 2× 加速 |

---

## 7. 交付物

| 产出 | 路径 |
|------|------|
| 需求文档 | `.monkeycode/specs/arch-optimization-2026/requirements.md` |
| 设计文档 | `.monkeycode/specs/arch-optimization-2026/design.md` |
| 任务拆解 | `.monkeycode/specs/arch-optimization-2026/tasklist.md` |
| Phase 1 实现 | `src/core/memory_arbiter.hpp/cpp`、`src/core/degradation.hpp`、`serve/server.py` 修改、`src/platform/memory.cpp` 修改 |
| 测试用例 | `tools/test_zfs_arc.py`、`tools/memory_arbiter_test.cpp` 等 |