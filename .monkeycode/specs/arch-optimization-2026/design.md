# Strata 架构优化技术设计规格

版本：1.0
日期：2026-11-09
状态：草案

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    接入层                                        │
│  OpenAI/Anthropic API、SSE、会话管理、限流、取消                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    推理编排层                                     │
│  请求状态机、Micro-batch、QoS、水位保护                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              韧性调度层  【新增核心】                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ DegradationState │  │ MemoryBudget     │  │ Residency    │  │
│  │ Machine (5级)    │  │ Arbiter          │  │ Manager      │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│  ┌──────────────────┐                                           │
│  │ Prefetcher       │  (置信分级 + 全局带宽 Token Bucket)      │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              预测式执行层  【新增核心】                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ ExpertEventSync  │  │ KvPipeline       │  │ Intent/Route │  │
│  │ (per-expert事件) │  │ (跨层异步KV传输) │  │ Predictor    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    存储/缓存层                                    │
│  Anchor Cache、聚类 mmap 顺序读取、Pinned RAM 池                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    现有推理核心                                   │
│  llama.cpp/ggml Kernel（通过回调注入决策，尽量不改）               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块设计

### 2.1 MemoryBudgetArbiter (`src/core/memory_arbiter.hpp/cpp`)

#### 职责
统一管理 VRAM/RAM 预算，仲裁 Expert Cache 与 KV Pool 的配额竞争，输出降级等级。

#### 接口
```cpp
struct BudgetPlan {
    size_t target_expert_bytes;   // 本轮专家驻留上限
    int target_kv_pages;          // KV pool 目标页数
    int degradation_level;        // 0~5
    bool kv_ram_tier_enabled;     // 是否启用 KV RAM tier
};

class MemoryBudgetArbiter {
public:
    explicit MemoryBudgetArbiter(const ArbiterConfig& cfg);
    BudgetPlan tick(size_t free_vram_bytes,
                    const ExpertCacheCost& expert_cost,
                    const KvPagesCost& kv_cost);
    void set_water_marks(size_t low, size_t high);  // 低/高水位
private:
    ArbiterConfig cfg_;
    int degradation_level_ = 0;
    int low_tick_count_ = 0;
    int high_tick_count_ = 0;
};
```

#### 算法
```
tick(free_vram, expert_cost, kv_cost):
  1. 计算总需求 = expert_cost.working_set + kv_cost.working_set
  2. 计算可用预算 = free_vram * (1 - headroom_ratio)  // headroom_ratio 默认 0.1
  3. IF 总需求 ≤ 可用预算:
        degradation_level = max(0, degradation_level - 1)  // 逐级恢复
        low_tick_count = 0
        high_tick_count += 1
     ELSE:
        low_tick_count += 1
        high_tick_count = 0
        IF low_tick_count >= 3: degradation_level = min(5, degradation_level + 1)
  4. 按 degradation_level 计算 target_expert_bytes、target_kv_pages
  5. 返回 BudgetPlan
```

#### 配置 (`ArbiterConfig`)
```cpp
struct ArbiterConfig {
    double headroom_ratio = 0.10;       // 预留 10%
    double expert_weight = 1.0;         // 专家权重
    double kv_weight = 1.2;             // KV 权重略高（关键路径）
    int low_tick_threshold = 3;         // 连续低水位触发降级
    int high_tick_threshold = 5;        // 连续高水位触发恢复
    size_t min_expert_bytes = 2ull << 30;  // 最小专家预算 2 GiB
    int min_kv_pages = 1000;            // 最小 KV 页数
};
```

---

### 2.2 DegradationStateMachine (`src/core/degradation.hpp`)

#### 状态定义
```cpp
enum class DegradationLevel : uint8_t {
    Normal = 0,         // 全功能
    ShrinkPrefetch = 1, // 预取窗口 50%
    HighConfOnly = 2,   // 仅 confidence > 0.8
    NoNgram = 3,        // 关闭 ngram/MTP 预取
    NoPrefetch = 4,     // 全关预取
    FreezeLowPrio = 5   // 冻结低优会话
};
```

#### 状态转移
```
Normal ←→ ShrinkPrefetch ←→ HighConfOnly ←→ NoNgram ←→ NoPrefetch ←→ FreezeLowPrio
   ↑                      ↑                   ↑             ↑              ↑
高水位恢复              高水位恢复          高水位恢复      高水位恢复      高水位恢复
```

#### 接口
```cpp
class DegradationStateMachine {
public:
    explicit DegradationStateMachine(const DegradationConfig& cfg);
    void apply_level(DegradationLevel level);  // 原子应用
    DegradationLevel current() const;
    // 运行时查询给 Prefetcher/ResidencyManager/SessionManager
    bool prefetch_enabled() const;
    double prefetch_confidence_threshold() const;
    double prefetch_window_scale() const;
    bool ngram_prefetch_enabled() const;
    bool freeze_low_priority_sessions() const;
private:
    DegradationLevel level_ = DegradationLevel::Normal;
    DegradationConfig cfg_;
};
```

#### 配置
```cpp
struct DegradationConfig {
    double shrink_prefetch_window_scale = 0.5;
    double high_conf_threshold = 0.8;
    int freeze_priority_threshold = 50;  // priority < 50 被冻结
};
```

---

### 2.3 ResidencyManager (`src/core/residency_manager.hpp/cpp`)

#### 职责
实现 Greedy-Dual 代价驱逐，管理 `protected_set`（锚点专家），提供 `evict(target_bytes)` 供 `MemoryBudgetArbiter` 调用。

#### 核心数据结构
```cpp
struct ExpertEntry {
    int layer;
    int expert;
    size_t gate_up_bytes;
    size_t down_bytes;
    double heat;           // 指数衰减：heat = prev_heat * decay + route_count
    double cost;           // (gate_up + down) / estimated_bandwidth
    double score;          // heat * cost（越大越热）
    bool protected_ = false;
};
```

#### 算法
```cpp
void evict(size_t target_bytes) {
    // 1. 收集所有非 protected 专家
    std::vector<ExpertEntry*> candidates;
    for (auto& e : experts_) if (!e.protected_) candidates.push_back(&e);
    
    // 2. 按 score 升序排序（最冷最先驱逐）
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](auto a, auto b) { return a->score < b->score; });
    
    // 3. 累积驱逐直到释放 ≥ target_bytes
    size_t freed = 0;
    for (auto* e : candidates) {
        if (freed >= target_bytes) break;
        evict_expert(e->layer, e->expert);  // 调用 ExpertCache::evict_slot
        freed += e->gate_up_bytes + e->down_bytes;
    }
    
    // 4. 更新 heat（指数衰减）
    for (auto& e : experts_) e.heat = e.heat * 0.95 + e.recent_route_count;
    for (auto& e : experts_) e.score = e.heat * e.cost;
}
```

#### 锚点专家配置
```cpp
// 环境变量：FNX_ANCHOR_EXPERTS="0:5,0:12,1:3,..." (layer:expert 逗号分隔)
// 或自动推断：启动后前 N 个 window 统计路由频率 Top-K
void init_protected_set(const std::string& env_value, int top_k = 32);
```

---

### 2.4 Prefetcher (`src/core/prefetcher.hpp/cpp`)

#### 职责
置信分级预取，全局带宽 Token Bucket 限流，与 `DegradationStateMachine` 联动。

#### 核心数据结构
```cpp
struct PrefetchTask {
    int layer;
    int expert;
    double confidence;  // P(route_next | history)
    size_t bytes;       // gate_up + down
    int priority;       // confidence * bytes
};

class TokenBucket {
    double tokens_ = 0;
    double capacity_;
    double refill_rate_;  // bytes/sec
public:
    TokenBucket(double capacity_bps, double refill_bps);
    bool consume(size_t bytes);  // 原子扣减，成功返回 true
    void refill(double dt_sec);
};
```

#### 主循环
```cpp
void tick(const RouteHistory& history, const ProbeCounts& probes) {
    // 1. 计算每专家 confidence
    std::vector<PrefetchTask> tasks;
    for (int l = 0; l < NL; ++l) {
        for (int e = 0; e < NEXP; ++e) {
            double conf = calc_confidence(l, e, history, probes);
            if (conf > degradation_.prefetch_confidence_threshold()) {
                tasks.push_back({l, e, conf, expert_size(l, e), 
                                 int(conf * expert_size(l, e))});
            }
        }
    }
    
    // 2. 按 priority 降序排序
    std::sort(tasks.begin(), tasks.end(),
              [](auto a, auto b) { return a.priority > b.priority; });
    
    // 3. Token Bucket 限流执行
    for (auto& t : tasks) {
        if (!bucket_.consume(t.bytes)) break;  // 带宽用尽
        if (degradation_.prefetch_enabled()) {
            issue_prefetch(t.layer, t.expert);  // 异步 host->device
        }
    }
}
```

#### 降级联动
```cpp
// DegradationLevel -> 参数映射
prefetch_confidence_threshold():
  Normal: 0.3
  ShrinkPrefetch: 0.4
  HighConfOnly: 0.8
  NoNgram: 0.8 (ngram 单独关闭)
  NoPrefetch: 1.0 (永不满足)
  FreezeLowPrio: 1.0

prefetch_window_scale():
  Normal: 1.0
  ShrinkPrefetch: 0.5
  others: 1.0 (窗口内专家数不变，仅阈值变)

max_prefetch_bps():
  Normal: 20 GB/s
  ShrinkPrefetch: 10 GB/s
  HighConfOnly: 5 GB/s
  others: 0
```

---

### 2.5 ExpertEventSync (`src/split/src/expert_event_sync.hpp`)

#### 目标
替代粗粒度 `cudaStreamSynchronize` per window，改为 per-expert `cudaEvent_t` 细粒度同步。

#### 设计
```cpp
// 每卡每层每专家一个 Event
struct ExpertEvent {
    cudaEvent_t ready;      // host->device copy 完成
    cudaEvent_t consumed;   // kernel 读取完成（可选，用于回收）
};

// Engine 持有
std::vector<std::vector<ExpertEvent>> expert_events_;  // [layer][expert]

// 专家加载完成时
void on_expert_loaded(int layer, int expert) {
    cudaEventRecord(expert_events_[layer][expert].ready, stream_load);
}

// 下游 kernel 启动前
void wait_expert_ready(int layer, int expert, cudaStream_t stream_compute) {
    cudaStreamWaitEvent(stream_compute, expert_events_[layer][expert].ready, 0);
}
```

#### 回退
```cpp
// 环境变量 FNX_NO_EVENT_SYNC=1 或 --set no_event_sync=1
// 退回：window 级 cudaStreamSynchronize(stream_load); 所有 expert 同步后再启动 compute
```

---

### 2.6 KvPipeline (`src/split/src/kv_pipeline.hpp`)

#### 目标
跨层异步 KV 传输，实现 Layer L 的 K/V 投影与 Layer L+1 的 Attention 重叠。

#### 设计
```cpp
// 每层独立 stream + event
struct KvLayerPipe {
    cudaStream_t stream_proj;   // K/V 投影
    cudaStream_t stream_copy;   // host->device 或 VRAM->VRAM
    cudaEvent_t proj_done;      // 投影完成
    cudaEvent_t copy_done;      // 拷贝完成
};

// Engine 持有
std::vector<KvLayerPipe> kv_pipes_;  // [layer]

// Layer L 投影完成后
void on_kv_proj_done(int layer) {
    cudaEventRecord(kv_pipes_[layer].proj_done, kv_pipes_[layer].stream_proj);
    // 立即在 stream_copy 上发起拷贝到 KvPages
    cudaStreamWaitEvent(kv_pipes_[layer].stream_copy, kv_pipes_[layer].proj_done, 0);
    cudaMemcpyAsync(kv_pages.dst_ptr(layer), kv_proj_src(layer), ..., kv_pipes_[layer].stream_copy);
    cudaEventRecord(kv_pipes_[layer].copy_done, kv_pipes_[layer].stream_copy);
}

// Layer L+1 Attention 启动前
void wait_kv_ready(int layer, cudaStream_t stream_attn) {
    cudaStreamWaitEvent(stream_attn, kv_pipes_[layer].copy_done, 0);
}
```

#### 统一内存设备的 RAM Tier
```cpp
// FNX_KV_RAM_TIER=1 启用
// 冷 KV page（LRU 尾部）落到 page-locked host mem (KvRamPool)
// KvPages::alloc() 优先 VRAM，不足则从 KvRamPool 分配
// Attention kernel 统一通过 page table 访问，无感知
```

---

### 2.7 Anchor Cache 与聚类读取 (`tools/cluster_gguf.cpp`)

#### 离线工具
```bash
# 用法
./cluster_gguf --input /path/model --output /path/model_clustered --layers 0-47
```

#### 文件格式
```
model_clustered/
  layer_00.experts.bin   # 连续存储：expert_0_gate_up, expert_0_down, expert_1_gate_up, ...
  layer_00.experts.idx   # JSON: {"experts": [{"offset":0,"gate_up":123456,"down":654321}, ...]}
  ...
```

#### 运行时加载
```cpp
// Stager 识别 .idx 文件存在时，改为单次 mmap + 顺序读取
void load_layer_clustered(Stager& st, int layer, const std::string& cluster_dir) {
    auto idx = read_json(cluster_dir + "/layer_" + fmt::format("{:02d}", layer) + ".experts.idx");
    auto fd = open(cluster_dir + "/layer_" + fmt::format("{:02d}", layer) + ".experts.bin", O_RDONLY);
    // 单次 mmap 整个文件
    void* base = mmap(nullptr, idx.total_size, PROT_READ, MAP_PRIVATE, fd, 0);
    // 按 offset 切片给各专家
    for (auto& e : idx.experts) {
        Tensor gate_up = {base + e.offset, e.gate_up_size, ...};
        Tensor down = {base + e.offset + e.gate_up_size, e.down_size, ...};
        // ... 原有加载流程
    }
}
```

---

### 2.8 ZFS ARC 感知 (`src/platform/memory.cpp`)

#### 实现
```cpp
// 新增函数
size_t zfs_arc_reclaimable_bytes() {
    // 仅 Linux
    FILE* f = fopen("/proc/spl/kstat/zfs/arcstats", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long arc_size = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "size %llu", &arc_size) == 1) break;
    }
    fclose(f);
    // ARC 可回收约 80%（保留 20% 给热数据）
    return arc_size > 0 ? (size_t)(arc_size * 0.8) : 0;
}

// 修改 device_free_bytes()
size_t device_free_bytes() {
    size_t free_b = cudaMemGetInfo().free;
    // ... 现有 UMA MemAvailable 逻辑 ...
    
    // 新增：ZFS ARC 可回收
    #if defined(__linux__)
    size_t arc_reclaimable = zfs_arc_reclaimable_bytes();
    if (arc_reclaimable > 0) {
        // 避免与 MemAvailable 重复计算：仅当 arc_reclaimable 不在 MemAvailable 中时加上
        // 简化：直接取 max(free_b, arc_reclaimable) 作为可用上限的参考
        free_b = std::max(free_b, arc_reclaimable);
    }
    #endif
    
    return free_b;
}
```

---

### 2.9 API 端点补齐 (`serve/server.py`)

#### 修改点：`do_GET` 增加 `/v1/models/{model_id}`
```python
elif path.startswith("/v1/models/") and path != "/v1/models":
    if self._authorized():
        model_id = path[len("/v1/models/"):]
        loaded = svc.loaded()
        model = svc.model  # 当前服务的模型名
        aliases = svc.aliases
        
        # 匹配主模型或 aliases
        if model_id == model or model_id in aliases:
            model_info = {
                "id": model_id,
                "object": "model",
                "status": {"value": "loaded" if loaded else "unloaded"},
                "meta": {"n_ctx": svc.reported_ctx()},
                "architecture": {
                    "input_modalities": ["text", "image"] if svc.vision else ["text"],
                    "output_modalities": ["text"]
                }
            }
            if model_id in aliases:
                model_info["alias_of"] = model
            self._json(200, model_info)
        else:
            self._json(404, {"error": {"message": f"model '{model_id}' not found", "type": "not_found_error"}})
```

---

## 3. 集成点与回调

### 3.1 Engine 构造注入
```cpp
// engine.hpp 新增成员
std::unique_ptr<MemoryBudgetArbiter> budget_arbiter_;
std::unique_ptr<DegradationStateMachine> degradation_;
std::unique_ptr<ResidencyManager> residency_mgr_;
std::unique_ptr<Prefetcher> prefetcher_;
std::unique_ptr<ExpertEventSync> event_sync_;
std::unique_ptr<KvPipeline> kv_pipeline_;

// 构造时初始化（可配置关闭）
Engine::Engine(Model& m, int max_ctx, int lanes) {
    // ... 现有初始化 ...
    if (!env_flag("FNX_NO_BUDGET_ARBITER")) budget_arbiter_ = make_unique<MemoryBudgetArbiter>(cfg);
    if (!env_flag("FNX_NO_DEGRADATION")) degradation_ = make_unique<DegradationStateMachine>(cfg);
    if (!env_flag("FNX_NO_RESIDENCY_MGR")) residency_mgr_ = make_unique<ResidencyManager>(m);
    if (!env_flag("FNX_NO_PREFETCHER")) prefetcher_ = make_unique<Prefetcher>(degradation_.get());
    if (!env_flag("FNX_NO_EVENT_SYNC")) event_sync_ = make_unique<ExpertEventSync>(NCARD, NL, NEXP);
    if (!env_flag("FNX_NO_KV_PIPELINE")) kv_pipeline_ = make_unique<KvPipeline>(NCARD, NL);
}
```

### 3.2 Window 循环集成
```cpp
// Engine::window() / verify() / step() 入口
std::vector<int> Engine::window(const std::vector<int>& tokens) {
    // 1. 预算 tick（每 10 window 一次）
    if (budget_arbiter_ && window_count_ % 10 == 0) {
        auto plan = budget_arbiter_->tick(
            free_vram_mib()[0] << 20,  // 取卡 0 为代表
            residency_mgr_->compute_cost(),
            kv_pages_->compute_cost()
        );
        degradation_->apply_level(static_cast<DegradationLevel>(plan.degradation_level));
        residency_mgr_->set_target_bytes(plan.target_expert_bytes);
        kv_pages_->set_target_pages(plan.target_kv_pages);
    }
    
    // 2. 预取 tick
    if (prefetcher_) prefetcher_->tick(route_history_, probe_counts_);
    
    // 3. 事件驱动同步 window（或回退同步）
    if (event_sync_) return run_window_event_driven(tokens);
    else return run_window_sync(tokens);  // 现有逻辑
}
```

### 3.3 会话优先级 (`strata_control/session.hpp`)
```cpp
struct Session {
    int priority = 100;  // 0~100，默认 100
    // ... 现有字段 ...
    bool is_frozen() const { 
        return degradation_ && degradation_->freeze_low_priority_sessions() && priority < 50; 
    }
};
```

---

## 4. 环境变量与配置旗标汇总

| 变量 | 说明 | 默认 |
|------|------|------|
| `FNX_NO_BUDGET_ARBITER` | 关闭统一预算仲裁 | 0 |
| `FNX_NO_DEGRADATION` | 关闭降级状态机 | 0 |
| `FNX_NO_RESIDENCY_MGR` | 关闭 Greedy-Dual 驱逐 | 0 |
| `FNX_NO_PREFETCHER` | 关闭预取器 | 0 |
| `FNX_NO_EVENT_SYNC` | 关闭 per-expert 事件同步 | 0 |
| `FNX_NO_KV_PIPELINE` | 关闭 KV 跨层流水线 | 0 |
| `FNX_NO_ZFS_ARC` | 关闭 ZFS ARC 感知 | 0 |
| `FNX_ANCHOR_EXPERTS` | 锚点专家列表 "layer:expert,..." | 空 |
| `FNX_KV_RAM_TIER` | 启用 KV RAM tier（统一内存设备） | 0 |
| `FNX_CLUSTER_GGUF` | 启用聚类读取（需离线转换） | 0 |
| `STRATA_UMA_HEADROOM_GIB` | UMA 设备预留 GiB（现有） | 6 |
| `FNX_UMA_HEADROOM_GIB` | 同上，别名 | 6 |

---

## 5. 测试策略

### 5.1 单元测试 (C++)
| 文件 | 覆盖模块 |
|------|----------|
| `tools/memory_arbiter_test.cpp` | `MemoryBudgetArbiter::tick` 水位触发、配额分配 |
| `tools/degradation_test.cpp` | `DegradationStateMachine` 状态转移、参数映射 |
| `tools/greedy_dual_test.cpp` | `ResidencyManager::evict` 驱逐字节数、protected_set 零驱逐 |
| `tools/prefetcher_test.cpp` | `Prefetcher::tick` 置信阈值、Token Bucket 限流 |
| `tools/event_sync_test.cpp` | `ExpertEventSync` 与同步模式吞吐对比、正确性 |
| `tools/kv_pipeline_test.cpp` | `KvPipeline` 跨层重叠延迟测量 |
| `tools/zfs_arc_test.cpp` | `zfs_arc_reclaimable_bytes` 解析正确性 |

### 5.2 集成测试 (Python)
| 文件 | 场景 |
|------|------|
| `tests/terminal/test_phase1.py` | `strata launch --plan --model iq3_xxs --context 32k` 验收新 flags |
| `tests/terminal/test_api_model_detail.py` | `GET /v1/models/fnx`、`GET /v1/models/unknown` |
| `tests/terminal/test_degradation.py` | 人为制造内存压力（`stress-ng --vm`），观察降级链与恢复 |
| `tests/terminal/test_long_context.py` | 32K 上下文 10 轮对话，吞吐不衰减、P99 降低 |

### 5.3 贪心一致性测试
```bash
# 现有测试扩展
./tools/greedy_consistency_test --new-residency --event-sync --kv-pipeline
# 要求：逐 token logits 与 baseline 逐 bit 一致（fp32 累加路径）
```

---

## 6. 分阶段实施计划

### Phase 1 (Week 1-2) — 基础设施
| 任务 | 产出 | 验收 |
|------|------|------|
| ZFS ARC 感知 | `src/platform/memory.cpp` 修改 | `tools/test_zfs_arc.py` 通过 |
| API 端点补齐 | `serve/server.py` 修改 | `test_api_model_detail.py` 通过 |
| MemoryBudgetArbiter 骨架 | `src/core/memory_arbiter.hpp/cpp` | 单测通过 |
| DegradationStateMachine 骨架 | `src/core/degradation.hpp` | 单测通过 |
| 集成到 Engine 构造 | `engine.hpp/cpp` 注入 | `strata launch --plan` 无报错 |

### Phase 2 (Week 3-4) — 事件驱动与 KV Pipeline
| 任务 | 产出 | 验收 |
|------|------|------|
| ExpertEventSync | `src/split/src/expert_event_sync.hpp` | `event_sync_test.cpp` 吞吐 ≥ 同步 95% |
| KvPipeline | `src/split/src/kv_pipeline.hpp` | `kv_pipeline_test.cpp` P99 降低 ≥ 20% |
| ResidencyManager (Greedy-Dual) | `src/core/residency_manager.hpp/cpp` | `greedy_dual_test.cpp` 通过 |
| 集成到 window 循环 | `engine.cpp` tick 调用 | 短对话 +10%~20% |

### Phase 3 (Week 5-6) — 预取与 Anchor Cache
| 任务 | 产出 | 验收 |
|------|------|------|
| Prefetcher | `src/core/prefetcher.hpp/cpp` | `prefetcher_test.cpp` 命中率 >60% |
| Anchor Cache 离线工具 | `tools/cluster_gguf.cpp` | `cluster_gguf_test.cpp` 加载加速 ≥2× |
| 聚类运行时识别 | `gguf.cpp` Stager 修改 | 首次加载 ≤ 1.2× |
| 端到端长上下文验收 | - | 32K 衰减后 +20%~60%，P99 降低 ≥30% |

### Phase 4 (可选) — PD 解耦与异构调度
- 仅在统一内存设备（Apple Silicon、APU）启用
- 非对称 PCIe 层分配需硬件实测

---

## 7. 回滚矩阵

| 功能 | 回滚开关 | 回滚后行为 |
|------|----------|------------|
| BudgetArbiter | `FNX_NO_BUDGET_ARBITER=1` | 退回固定 `FNX_EXPERT_RAM_GIB` + `FNX_ARENA` |
| Degradation | `FNX_NO_DEGRADATION=1` | 无降级，内存压力直接 Refused |
| ResidencyManager | `FNX_NO_RESIDENCY_MGR=1` | 退回 `expert_residency.hpp` 旧 score 排序 |
| Prefetcher | `FNX_NO_PREFETCHER=1` | 无预取，纯被动 LRU |
| EventSync | `FNX_NO_EVENT_SYNC=1` | 退回 window 级 `cudaStreamSynchronize` |
| KvPipeline | `FNX_NO_KV_PIPELINE=1` | 退回同步 KV 投影+拷贝 |
| ZFS ARC | `FNX_NO_ZFS_ARC=1` | 仅用 `MemAvailable` |
| Anchor Cache | `FNX_CLUSTER_GGUF=0` | 退回逐专家 mmap |

所有开关可在 `strata launch --set FNX_NO_xxx=1` 或 run config JSON 中设置，无需重编译。

---

## 8. 风险登记

| 编号 | 风险 | 可能性 | 影响 | 缓解措施 |
|------|------|--------|------|----------|
| R-01 | Event sync 在 HIP/Metal/SYCL 后端不稳定 | 中 | 高 | 先在 CUDA 后端验证，其他后端默认关闭，CI 逐个启用 |
| R-02 | Greedy-Dual score 计算开销大 | 低 | 中 | 仅在 `budget_arbiter_->tick()` 时计算（每 10 window），增量更新 heat |
| R-03 | KvPipeline 增加 stream/event 资源耗尽 | 低 | 中 | 每卡 NL 个 stream + event，NL=48 远低于驱动上限 |
| R-04 | ZFS ARC 读取权限问题（容器/非 root） | 中 | 低 | 读取失败静默返回 0，不阻塞启动 |
| R-05 | 聚类文件格式不兼容旧版本 | 低 | 中 | 版本号在 .idx 中，运行时检查不匹配则回退逐专家加载 |
| R-06 | 降级链抖动 | 中 | 中 | 引入滞回：低水位需连续 3 tick，高水位需连续 5 tick |

---

## 9. 文档更新清单

| 文档 | 更新内容 |
|------|----------|
| `docs/AI_REFERENCE.md` | 新增环境变量表、架构分层图 |
| `docs/TERMINAL_LAUNCH.md` | `--set` 新 flags 说明 |
| `docs/DETAILS.md` | 专家缓存、KV cache、内存探测章节更新 |
| `docs/COMMUNITY_BENCHMARKS.md` | 新增 Phase 1-3 基准测试方法 |
| `CHANGELOG.md` | 版本发布记录 |

---

## 10. 里程碑

| 里程碑 | 日期 | 交付 |
|--------|------|------|
| M1: Phase 1 代码冻结 | T+2 周 | `memory_arbiter`、`degradation`、`ZFS ARC`、`API endpoint` 合入 main |
| M2: Phase 2 代码冻结 | T+4 周 | `event_sync`、`kv_pipeline`、`residency_manager` 合入 |
| M3: Phase 3 代码冻结 | T+6 周 | `prefetcher`、`anchor_cache` 合入，长上下文验收通过 |
| M4: 发布候选 | T+7 周 | 全后端构建通过、ASAN/TSAN 干净、社区基准跑分 |