目前 通过bash tools_zjh/z_train.sh &>06041446.log训练的模型 能解决行人横穿或前车慢速 躲避绕行，但是无法躲避他车右边插入（发生撞击）
要让策略学会遇见障碍物减速之后让行 或者 绕行，请修改奖励的config 并给出建议

---

## 本次已做的配置修改

本次直接修改了 `tools_zjh/202605280011_HUGSM_reinforcepp_closed_loop_closeCloseloop_NoGRPOCraft.yaml`，`tools_zjh/z_train.sh` 继续指向这个配置即可。

### 1. 奖励主分支从 CRAFT close-loop 切到 step_path + safety

- 将 `env.reward.CRAFT.enable` 从 `true` 改为 `false`
- 保留 `reward.mode: step_path`

这样做的原因是：当前 `CRAFT close loop` 更偏向“继续向前走 + 发生碰撞后再重罚”，对**右侧插入这种尚未完全进入本车正前方、但已经需要预防性减速/让行**的场景，前瞻性不够；而 `step_path` 分支里已经有现成的 `safety` 逻辑，会根据前方障碍物距离、TTC、横向重叠程度，主动削弱 progress reward，并额外增加风险 cost，更适合学“先减速避让，再决定绕行还是跟车”。

### 2. 降低“硬顶着往前走”的动机

将路径进度奖励调弱：

- `w_progress: 1.0 -> 0.6`
- `progress_forward_cap_m: 2.0 -> 1.5`
- `progress_backward_cap_m: 0.5 -> 0.3`

目的：减少 policy 在 cut-in 场景下“为了吃 progress 分数而抢行”的倾向。

### 3. 放宽轻微绕行动作的惩罚

将横向/航向惩罚调得更宽松，让策略在安全前提下允许小幅横移和转向：

- `w_lateral: 0.25 -> 0.12`
- `lateral_free_m: 0.3 -> 0.8`
- `lateral_huber_delta_m: 0.5 -> 0.9`
- `w_yaw: 0.1 -> 0.04`
- `yaw_free_deg: 5.0 -> 8.0`
- `yaw_huber_delta_deg: 10.0 -> 15.0`

同时保留严重偏移保护：

- `severe_lateral_error_m: 3.5`
- `severe_lateral_cost: 2.0`
- `severe_yaw_error_deg: 55.0`
- `severe_yaw_cost: 2.0`

目的：允许“合理绕行”，但不允许大幅失控偏航。

### 4. 强化前方障碍物的让行/减速信号

新增并强化 `safety`：

- `enable: true`
- `lookahead_m: 18.0`
- `corridor_half_width_m: 3.2`
- `safe_gap_m: 10.0`
- `safe_ttc_s: 4.5`
- `w_clearance: 3.5`
- `w_ttc: 5.0`
- `progress_gate_strength: 1.35`
- `min_progress_gate: 0.05`

思路：

1. `corridor_half_width_m` 加宽，是为了更早覆盖“右侧插入、还没完全切到正前方”的车。
2. `safe_gap_m`、`safe_ttc_s` 变大，是为了让策略更早把这类场景视为风险，而不是等到快撞上才反应。
3. `progress_gate_strength` 提高后，风险一上来，正向 progress reward 会明显被压制，策略更容易学会先刹车让行。
4. `min_progress_gate=0.05` 不是完全锁死前进，而是保留一点点推进空间，避免策略学成“永远站住不动”。

### 5. 提高真实碰撞的代价

- `collision.mode: constraint_gate -> dense_penalty`
- `w_static: 5.0 -> 10.0`
- `w_dynamic: 5.0 -> 14.0`
- `terminal.penalty: 0.0 -> -20.0`

其中动态碰撞罚得更重，是因为当前问题主要就是**与插入车辆发生动态碰撞**。这样策略会更清楚地学到：右侧强插不能硬顶。

### 6. 避免新旧实验结果混在一起

同步修改了：

- `train.wandb.group -> HUGSIM_StepPathYieldSafety_NoGRPOCraft`
- `train.actor_learner.buffer_dir -> /root/clone/ReconDreamer-RL/checkpoints/actor_learner/HUGSIM_StepPathYieldSafety_NoGRPOCraft`
- `train.grpo.debug_dir -> checkpoints/visualize/HUGSIM_StepPathYieldSafety_NoGRPOCraft`

避免和旧的 close-loop 奖励实验混淆。

## 为什么这套改法更适合“右侧插入”

当前问题本质上不是“碰撞罚得不够大”这么简单，而是**缺少碰撞前的连续风险信号**。

右侧插入场景通常有三个阶段：

1. 对方刚开始切入，本车还没撞上，但继续维持原速度已经不安全。
2. 此时如果奖励仍然鼓励“多走一点就有分”，策略就会选择继续抢行。
3. 等真正碰撞时再给大罚，学习信号已经太晚、太稀疏。

这次改成 `step_path + safety` 后，reward 会在**障碍物进入前方风险走廊**时，提前通过：

- 压制 progress reward
- 增加 clearance / TTC cost
- 保留适度绕行自由度

来引导策略学习“减速让行优先，其次才是绕过去”。

## 训练建议

### 1. 先观察这几个指标是否真的动起来

重点看 wandb / rollout 里的：

- `front_obstacle_cost`
- `safe_progress_gate`
- `dynamic_collision`
- `terminal_failure_rate`

理想现象：

- cut-in 出现时，`safe_progress_gate` 明显下降
- `front_obstacle_cost` 明显上升
- `dynamic_collision` 和 `terminal_failure_rate` 逐步下降

如果 cut-in 来了但 `safe_progress_gate` 基本不掉，说明风险走廊还不够敏感，需要继续增大 `corridor_half_width_m` 或 `safe_ttc_s`。

### 2. 如果训练后变得“太保守”

表现：见车就停，不敢通过。

优先回调：

- `progress_gate_strength: 1.35 -> 1.1`
- `w_ttc: 5.0 -> 3.5`
- `w_clearance: 3.5 -> 2.5`

### 3. 如果训练后还是会撞右侧插入车

优先继续增强：

- `corridor_half_width_m: 3.2 -> 3.5`
- `safe_ttc_s: 4.5 -> 5.0`
- `w_dynamic: 14.0 -> 18.0`
- `terminal.penalty: -20.0 -> -25.0`

### 4. 如果只会刹车，不会绕行

说明安全信号有了，但绕行动作仍被 path tracking 压制，可以再轻一点：

- `w_lateral: 0.12 -> 0.08`
- `w_yaw: 0.04 -> 0.02`
- `lateral_free_m: 0.8 -> 1.0`

但这一步建议在“先学会不撞”之后再做，不要一开始就放太松。

## 总结

这次调参的核心不是单纯把 collision penalty 拉大，而是把奖励从“撞了才知道错”改成“看到风险就先别抢、先减速、必要时允许小幅绕行”。  
对右侧插入问题，最关键的是：

1. 让 risk 信号更早出现（`safety`）
2. 降低盲目前进的收益（`w_progress`、progress cap）
3. 给合理绕行留空间（放宽 lateral/yaw）
4. 真撞上时足够疼（dynamic collision + terminal penalty）