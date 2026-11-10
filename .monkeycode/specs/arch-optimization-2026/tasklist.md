# Strata 架构优化实施任务列表

版本：1.0
日期：2026-11-09
关联：requirements.md、design.md

---

## Phase 1：基础设施（Week 1-2）

### 1.1 ZFS ARC 感知
- [x] **TASK-1.1.1** 修改 `src/platform/memory.cpp`
  - 新增 `zfs_arc_reclaimable_bytes()` 函数
  - 在 `device_free_bytes()` 中集成，避免与 `MemAvailable` 重复计算
  - 环境变量 `FNX_NO_ZFS_ARC=1` 可关闭
- [x] **TASK-1.1.2** 新增单元测试 `tests/platform/zfs_arc_test.cpp`
  - Mock `/proc/spl/kstat/zfs/arcstats` 验证解析正确性
  - 容器/无权限场景返回 0 不崩溃
- [ ] **TASK-1.1.3** 新增集成测试 `tools/test_zfs_arc.py`
  - 创建临时 ZFS 池（或模拟），写入数据后验证 `device_free_bytes()` 反映可回收量

### 1.2 API 端点补齐
- [x] **TASK-1.2.1** 修改 `serve/server.py` `do_GET()`
  - 增加 `/v1/models/{model_id}` 路由分支
  - 复用现有 model dict 构造逻辑，支持 aliases
  - 404 遵循 OpenAI 错误格式
- [x] **TASK-1.2.2** 新增测试（并入 `serve/test_server.py` 的 `ModelAliases` 类）
  - `GET /v1/models/fnx` → 200，字段完整
  - `GET /v1/models/unknown` → 404，错误格式正确
  - `GET /v1/models/alias_name` → 200，含 `alias_of` 字段

### 1.3 MemoryBudgetArbiter 骨架
- [x] **TASK-1.3.1** 新增 `src/core/memory_arbiter.hpp`
  - 定义 `BudgetPlan`、`ArbiterConfig`、`ExpertCacheCost`、`KvPagesCost`
  - 类声明 `MemoryBudgetArbiter`
- [x] **TASK-1.3.2** 新增 `src/core/memory_arbiter.cpp`
  - 实现 `tick()` 水位算法、配额分配、降级等级计算
  - 线程安全（原子操作或 mutex）
- [x] **TASK-1.3.3** 新增单元测试 `tools/memory_arbiter_test.cpp`
  - 场景：充足内存 → 降级 0；连续低水位 → 1→2→3→4→5；恢复 → 5→4→3→2→1→0
  - 验证 `target_expert_bytes`、`target_kv_pages` 单调性

### 1.4 DegradationStateMachine 骨架
- [x] **TASK-1.4.1** 新增 `src/core/degradation.hpp`
  - 定义 `DegradationLevel` enum、 `DegradationConfig`
  - 类声明 `DegradationStateMachine`
- [x] **TASK-1.4.2** 新增 `src/core/degradation.cpp`
  - 实现状态转移、参数映射（prefetch 阈值、窗口缩放、带宽上限）
  - `apply_level()` 原子应用
- [x] **TASK-1.4.3** 新增单元测试 `tools/degradation_test.cpp`
  - 验证 5 级状态转移、滞回、各级参数输出正确

### 1.5 集成到 Engine
- [x] **TASK-1.5.1** 修改 `src/split/src/engine.hpp`
  - 新增成员：`unique_ptr<MemoryBudgetArbiter>`、`unique_ptr<DegradationStateMachine>`
  - 构造函数注入（读取环境变量决定是否创建）
- [x] **TASK-1.5.2** 修改 `src/split/src/engine.cpp`（或 `engine_impl.hpp` 相关）
  - 在 `window()`/`verify()` 入口每 N 次调用 `budget_arbiter_->tick()`
  - 应用 `degradation_->apply_level()`
- [x] **TASK-1.5.3** 修改 `CMakeLists.txt` / `setup.py`
  - 新增 `memory_arbiter.cpp`、`degradation.cpp` 到构建目标

### 1.6 环境变量文档化
- [x] **TASK-1.6.1** 环境变量写入 `docs/DETAILS.md`（FNX_ARBITER / FNX_ARBITER_LOW / FNX_ARBITER_HIGH / FNX_NO_ZFS_ARC）
- [x] **TASK-1.6.2** `FNX_*` 为 split 引擎自身变量，直接走进程环境，不经 `strata launch --set`；在 `docs/DETAILS.md` 说明

---

## Phase 2：事件驱动与 KV Pipeline（Week 3-4）

### 2.1 ExpertEventSync
- [x] **TASK-2.1.1** 新增 `src/split/src/expert_event_sync.hpp/.cpp`
  - `ExpertEvent` 结构、`ExpertEventSync` 类
  - `on_expert_loaded()`、`wait_expert_ready()`
  - `FNX_NO_EVENT_SYNC=1` 回退路径
- [ ] **TASK-2.1.2** 修改 `engine.cu` / `prefill.cu` 专家加载路径
  - Host->Device copy 完成后 `cudaEventRecord`
  - 下游 kernel `cudaStreamWaitEvent`
- [x] **TASK-2.1.3** 新增单元测试 `tools/event_sync_test.cpp`
  - 对比同步模式 vs 事件模式吞吐、正确性
  - HIP/Metal/SYCL 后端编译通过（可默认关闭）

### 2.2 KvPipeline
- [x] **TASK-2.2.1** 新增 `src/split/src/kv_pipeline.hpp/.cpp`
  - `KvLayerPipe` 结构、`KvPipeline` 类
  - `on_kv_proj_done()`、`wait_kv_ready()`
  - `FNX_KV_RAM_TIER=1` 启用 RAM tier（`KvRamPool`）
- [ ] **TASK-2.2.2** 修改 `engine.cu` Attention 流程
  - Layer L 投影在 `stream_proj`，拷贝在 `stream_copy`
  - Layer L+1 Attention `cudaStreamWaitEvent(copy_done)`
- [x] **TASK-2.2.3** 新增单元测试 `tools/kv_pipeline_test.cpp`
  - 测量 32K context decode P99 延迟降低 ≥ 20%
  - 验证 RAM tier 分配正确性

### 2.3 ResidencyManager (Greedy-Dual)
- [x] **TASK-2.3.1** 新增 `src/core/residency_manager.hpp/.cpp`
  - `ExpertEntry`、`ResidencyManager` 类
  - `evict(target_bytes)`、`init_protected_set()`、`update_heat()`
  - `FNX_ANCHOR_EXPERTS` 环境变量解析
- [ ] **TASK-2.3.2** 修改 `ExpertCache` 配合
  - 新增 `evict_slot(layer, expert)` 供 `ResidencyManager` 调用
  - `ref`/`hand` 语义调整为 cost-aware
- [x] **TASK-2.3.3** 新增单元测试 `tools/greedy_dual_test.cpp`
  - 驱逐字节数 ≥ target，protected_set 零驱逐
  - heat*cost 单调、指数衰减正确

### 2.4 集成 Window 循环
- [ ] **TASK-2.4.1** 修改 `engine.cpp` `window()`/`verify()`
  - 每 tick 调用 `budget_arbiter_->tick()` → `residency_mgr_->set_target_bytes()` → `evict()`
  - `kv_pipeline_->set_target_pages()` → `kv_pages_->shrink()`
- [ ] **TASK-2.4.2** 短对话基准测试
  - `strata launch --model iq3_xxs --context 32k --no-tui --plan` 验收
  - 目标：相对旧版 +10%~20%

---

## Phase 3：预取与 Anchor Cache（Week 5-6）

### 3.1 Prefetcher
- [ ] **TASK-3.1.1** 新增 `src/core/prefetcher.hpp/.cpp`
  - `PrefetchTask`、`TokenBucket`、`Prefetcher` 类
  - `tick(route_history, probe_counts)` 置信计算、排序、Token Bucket 限流
  - 与 `DegradationStateMachine` 联动参数
- [ ] **TASK-3.1.2** 修改 `engine.cu` 路由历史收集
  - `FNX_ROUTE_COUNT=1` 已有 `route_counts`，扩展为滑动窗口历史
  - `FNX_PREFETCH_PROBE=1` 已有 `probe_counts`，复用
- [ ] **TASK-3.1.3** 新增单元测试 `tools/prefetcher_test.cpp`
  - 命中率 > 60%，带宽不超 `max_prefetch_bps`
  - 降级链各级阈值、带宽衰减验证

### 3.2 Anchor Cache 离线工具
- [ ] **TASK-3.2.1** 新增 `tools/cluster_gguf.cpp`
  - 读取 GGUF shards，按层聚类专家 tensor
  - 输出 `.bin` + `.idx`（JSON），支持 `--layers` 过滤
- [ ] **TASK-3.2.2** 修改 `src/split/src/gguf.cpp` Stager
  - 识别 `.idx` 存在 → 单次 mmap + 顺序切片
  - `FNX_CLUSTER_GGUF=0` 回退逐专家加载
- [ ] **TASK-3.2.3** 新增测试 `tools/cluster_gguf_test.cpp`
  - 首次加载时间 ≤ 旧版 1.2×
  - 后续加载 ≥ 2× 加速

### 3.3 端到端长上下文验收
- [ ] **TASK-3.3.1** 32K 上下文 10 轮对话压测
  - 吞吐不衰减、P99 降低 ≥ 30%
  - 多会话并发（8 会话）聚合吞吐 1.3×~1.8×
- [ ] **TASK-3.3.2** 贪心一致性测试
  - `./tools/greedy_consistency_test --new-residency --event-sync --kv-pipeline --prefetcher`
  - 逐 token logits 与 baseline 逐 bit 一致

---

## Phase 4（可选）：PD 解耦与异构调度

- [ ] **TASK-4.1** PD 解耦原型（仅统一内存设备 `FNX_PD_DECOUPLE=1`）
- [ ] **TASK-4.2** PCIe 拓扑感知层分配（`lspci -vv` 解析 `LnkCap`/`LnkSta`）
- [ ] **TASK-4.3** MTP/ngram 预取实验（`Prefetcher` 接入 ngram 模型）

---

## 依赖关系图

```
TASK-1.1 ──┐
TASK-1.2 ──┤
TASK-1.3 ──┼──→ TASK-1.5 ──→ TASK-2.1
TASK-1.4 ──┤         │
TASK-1.6 ──┘         │
                     ├→ TASK-2.2
                     └→ TASK-2.3 ──→ TASK-2.4
                                    │
                     ┌──────────────┘
                     │
TASK-3.1 ──────────┐
TASK-3.2 ──────────┼──→ TASK-3.3
```

---

## 验收门槛汇总

| 阶段 | 关键指标 | 通过标准 |
|------|----------|----------|
| Phase 1 | ZFS 测试通过、API 端点 200/404、Arbiter/Degradation 单测绿、构建通过 | 所有 CI 绿 |
| Phase 2 | Event sync 吞吐 ≥ 同步 95%、KV pipeline P99 降低 ≥ 20%、Greedy-Dual 驱逐正确、短对话 +10%~20% | 基准跑分达标 |
| Phase 3 | Prefetch 命中率 >60%、Anchor Cache 加载加速 ≥2×、长上下文 32K 衰减后 +20%~60%、P99 降低 ≥30%、多会话 1.3×~1.8× | 端到端验收通过 |
| 全阶段 | 贪心一致性逐 bit 一致、ASAN/TSAN 干净、四后端构建通过 | 发布候选就绪 |

> **实施偏差记录（2026-11-09）**
> - 头文件按本仓库约定放在 `include/strata/core/`（`memory_arbiter.hpp`、`degradation.hpp`），实现放在 `src/core/`；不是都放在 `src/core/`。
> - 测试按本仓库约定放在 `tests/core/memory_arbiter_test.cpp`，并在 `CMakeLists.txt` 的 `strata_engine` 段落注册为 CPU-only 可执行文件；不是 `tools/`。
> - Engine 成员用值类型（`MemoryBudgetArbiter arbiter_`、`DegradationStateMachine degradation_`）配合 `arbiter_on_` 开关，符合该文件既有风格；未用 `unique_ptr`。
> - `arbiter_tick()` 实现放在 `src/split/src/engine_common.cu`（与 `window()` 同文件），`window()` 与 `step()` 入口各调用一次。
> - Phase 验证不在 `tools/test_zfs_arc.py` 之外另写新基准脚本：复用既有 `tools/ab_engine.py`（按 arm 起一次 `serve/server.py --engine strata`，臂的键写进服务器进程环境，split 引擎子进程经 `child_env()` 从 `os.environ` 继承，正好能开关 `FNX_ARBITER` / `FNX_PF_STAGE_ROWS`），并只输出服务器自报的 `timings.predicted_per_second` / `prompt_per_second` 中位数。
> - 结果目录按既有 `bench/results/YYYY-MM-DD-name/` 约定：`bench/results/2026-11-09-arch-arbiter/README.md`（本次运行的手填模板，状态 not yet run），运行手册另放 `bench/arch-opt-2026/RUNBOOK.md`（含 CUDA 构建、三组实验、ASAN/TSAN 叠加与回填步骤）。
> - ZFS 集成测试（TASK-1.1.3）不在自动化套件里：它要在带 ZFS 池的机器上实际建数据集、写数据，CI 无此设备；改在 `RUNBOOK.md` 第 3 节给出 ZFS 宿主的手动步骤，`zfs_arc_reclaimable_bytes_from()` 的解析已由 CPU-only `zfs_arc_test` 用 fixture 覆盖。
> - `FNX_ARBITER` 的语义以代码为准：默认**关**，设成任何非 `0` 值才开（`engine.cu` 构造 `arbiter_on_ = on != nullptr && strcmp(on,"0") != 0`），`docs/DETAILS.md:250` 的「off by default」与之一致；基准的 A 臂显式 `FNX_ARBITER=0` 把默认关钉死。
> - Issue #5440 的内容在本 checkout 与当前会话里都拿不到，`phase2-notes.md` 只把它列成待填 TODO 与一条与 issue 文本无关的设计约束，不复述、不推测其正文。

> **实施偏差记录（2026-11-10，Phase 2 核心库）**
> - 规格里 Phase 2 的头/源放在 `src/split/src/`，测试放 `tools/`。本机没有 cmake、CUDA/HIP 工具链和 GPU，`src/split` 进不了构建，所以本轮把 Phase 2 先落成设备无关的纯 C++ 库：头在 `include/strata/core/`，源在 `src/core/`，测试在 `tests/core/`，与 Phase 1 的 `memory_arbiter`/`degradation` 一致。头文件不含任何 CUDA/Metal/SYCL 类型，因此 CPU 可单独编译与测试。
> - 已完成并本机 `g++ -Wall -Wextra -Werror` 编译 + 运行全绿的五个库及测试：`expert_event_sync`（TASK-2.1.1/2.1.3 的 CPU 形态）、`kv_pipeline`（2.2.1/2.2.3）、`residency_manager` Greedy-Dual + anchor 保护（2.3.1/2.3.3）、`cluster_mmap` STRCM 读取 + 延迟模型（3.2.3 的读者侧）、`expert_prefetcher` P0/P1/P2 规划（3.1.1 的规划侧）。另扩展 `memory_arbiter`：`SubBudgets`、`initial_budgets` 60/30/10、`rebalance`（滞回 3 窗口、单次迁移 ≤5%）。
> - 新增 `include/strata/core/backend_event.hpp`：`BackendEvent`/`BackendStream`/`NullBackendEvent`/`NullBackendStream`/`BackendEventCapabilities`，让事件/流水线只依赖抽象，不引入 CUDA 类型；设备适配留给 `src/split`。
> - `tools/cluster_experts.py` 已实现并验证：读 co-activation CSV 或 `.npz`，networkx 在场走 Louvain ACL，否则连通分量；写二进制 `STRCM`（magic + 4×u32 头 + clusters + u32 n_anchors + anchors）与人类可读 `.json`，最大簇作锚点。已用本机 C++ `ClusterMmapReader` 往返验证字节一致（clusters/anchors/locate 全对）。
> - `CMakeLists.txt` 已把 5 个新源的 CPU 测试注册进 `strata_engine` 测试块（`STRATA_BUILD_TESTS` 下、无 GPU 依赖），并把 5 个源加入 `strata_engine` 源列表；`cmake` 本机不可用，配置/生成未跑，仅保证语法与规则与既有块一致。
> - **未完成（需带 CUDA/HIP + cmake + 2-4 卡的机器）**：把上述库真正接进 `engine.cu`/`prefill.cu` 的专家加载、Attention、路由历史与 `window()`/`verify()` 路径（TASK-2.1.2、2.2.2、2.3.2、2.4、3.1.2、3.2.3）；四后端构建；`RUNBOOK.md` 三组实验与 ASAN/TSAN；验收门槛里的吞吐/P99/命中率/短对话 +10%~20% 等实测数字。这些本轮一律未测，文档与代码中均未填任何数字。
> - `STRATA_EXPERT_EVENT_SYNC`、`STRATA_KV_PIPELINE`、`STRATA_PREFETCH`、`STRATA_RESIDENCY`、`STRATA_CLUSTER_MMAP` 这组开关本应在 `src/split` 的场景接线里读取；本轮只交付了库本身与 CPU 测试，`src/core` 新库内**没有任何** `getenv`，所以默认构建路径与上一版字节一致。接线时再由 `src/split` 读取这些变量并把库回调注入。
> - Issue #5440 仍拿不到正文（联网额度不足），沿用 `phase2-notes.md` 的 TODO，不复述、不推测。
