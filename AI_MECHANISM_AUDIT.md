# AI 机制审计报告

审计日期：2026-09-26

审计对象：`generals_ai.py`、`generals_training.py`、`generals_core.py`

## 审计结论

- 此前提出并最终保留的 AI 决策机制均已进入代码和自动化测试。
- 发现并修正了一个训练集兼容性问题：旧 `manifest.json` 没有 `training_mode` 时，会错误清空历史最佳适应度。
- 100 局 `20 x 20 + 6 AI` 测试全部完成，100 局都发生接触和实际兵力抵消，96 局在 900 回合内决出胜者。
- 100 局共 600 个 AI 样本已写入 `training_data/history.jsonl`，并触发一次进化批次；训练世代由 9 提升到 10。
- 原训练集已备份到 `work/training_data_backup_20260925`。
- 训练进化改为每累计 5 局更新一次；冠军改为最近 100 局窗口内适应度最高的个体，避免机制调整后旧冠军永久占位。
- 删除旧战略标签、旧强制吞并、旧回撤和旧补兵分支后，完整单元测试为 `212 tests OK`；困难档 `100 x 100` 双 AI 基准 1 至 10 全部通过，接触/抵消回合为 `124/125、63/83、117/119、127/128、160/161、162/163、97/104、85/96、55/57、115/116`，全部在困难档 200 回合指标内。
- 修正并行混合训练小地图的共享内存步长：槽位按最大 `55 x 55` 混战地图预留，快照记录实际宽高；短测试验证了 `25 x 25` 与 `55 x 55` 混用不会写越界。
- 训练小地图固定显示用户选中的并行对局，用户点击地图才切换槽位；任务评估间隔统一为 2 回合，快照改为每 60 回合写入。24 进程、96 局基准开启快照约 `230.1` 回合/秒，关闭快照约 `235.4` 回合/秒，小地图同步开销约 `2.3%`。
- 修正和平期主动索敌衰减：搜索压力会随未接触回合和领地停滞上升，搜索任务可覆盖普通的守城维护与局部扩地；敌方搜索目标和全局搜索航点都会排除已搜索格并定期推进。`20 AI / 75x75 / 300 回合` 探针中，中位领地由第 50 回合约 `32.5` 格增长到第 300 回合 `130` 格，停滞 AI 仅 `0-4` 个，每 50 回合均检测到兵力交换。
- 同一探针扩展到最大阵营 `37 方 / 75x75 / 300 回合`：存活数由 37 降至 17，中位领地由第 50 回合约 `27` 格增长到第 300 回合 `173` 格，每 50 回合仍有实际兵力交换，证明大混战和平期不再集体固守主城。
- 本轮地形与 AI 机制更新后完整单元测试为 `232 tests OK`；困难档 `100 x 100` 双 AI 基准种子 1 至 5 的接触/抵消回合为 `172/175、133/137、126/128、138/139、154/156`，全部通过 200 回合指标。45 张 `20 x 20` 至 `100 x 100` 地图审计为 0 失败；扩展后的 90 张地形审计同样为 0 失败，密度 `16.96%–20.25%`、最长连续山脊 29 格，全部通过可通行连通和无 `2 x 2` 聚团检查。
- 地图生成不再依赖最多 400 次随机重试。`20 x 20` 至 `100 x 100`、2 至 36 人、种子 1 至 5 的生成审计全部通过；最坏 `100 x 100 + 36 AI` 单次生成由约 `12.9s` 降至 `0.22s`，第 108 项机制由同一测试覆盖。
- 额外机制审计运行 `30` 局 `20 x 20 + 6 AI`：接触和实际兵力抵消均为 `30/30`，平均 `286.97` 回合，六类任务、主城守卫、纵深任务、重大战役和禁止挂机全部成立；攻击、防守、探索状态在实战中持续出现，发育状态由新增的四状态分支测试覆盖。
- 和平期内 `军N` 不再因补给状态围着主城驻守；已形成规模的军队会优先转向持久敌区、敌将战役或全局搜索航点，独立继续索敌。
- 独立军队专项审计覆盖 `20 x 20 + 6 AI` 种子 `1、2、7`，所有军队连续停摆数均为 `0`，驻留己方主城的军队数均为 `0`。军队现在只使用 `进攻 / 防守 / 补兵` 三态，丘陵延迟等待仍不计作挂机。
- 实际 `GameApp` 20 x 20 困难档回合循环已运行到第 182 回合正常结束，AI 自动军队头衔成功创建并参与行动。
- 90 张 `20 x 20` 至 `100 x 100`、2 至 36 玩家地图的地形审计全部通过：山脉密度 `16.96%–20.25%`，最长连通山脊 `29` 格，无 `2 x 2` 实心山脉，所有可通行格连通，丘陵数量始终不超过平地。
- 山脉生成不再按四个象限强制布种；随机起点、八方向游走和受控随机分支允许横、竖、斜向山脊及局部疏密差异，同时保持随机尺寸可复现。
- 主城拱卫不再使用固定候选半径。所有普通己方可移动地块统一比较守军缺口、威胁兵力、紧急度、距离、兵力/距离比、局部战斗价值、反攻价值和前线风险；远处高价值兵堆、近处援军、本地反击和继续打更好局部战斗都由同一评分决定，独立军队格始终排除。
- 独立军队与主 AI 的位置隔离扩大到“当前格、丘陵 transit 起点/终点、待执行目标”。主 AI 候选和兜底移动都会避开这些位置；守军上限过滤也已前移到候选阶段，固定的可移动野战兵不会因为一个非法回城候选而被整体判空。
- 后期动态军队上限：公式已修正为 `4 + floor(总兵力 / 8000)`。总兵力达到 `97,333` 时上限为 `16`，自动化测试验证 AI 会在候选地块充足时实际填满全部新增名额。
- 末期战争节奏新增重大兵力抵消记忆。存活势力不超过 `8` 个时，若只有小交换、连续 `30–90` 回合没有达到 `max(12, 总兵力 * 0.3%)` 的重大交换，AI 会提高索敌与进攻压力；该分支不参与早期 36 AI 阶段，避免拖慢开局性能。
- 本轮固定回归矩阵全部通过：`75 x 75 / 36 AI` 五局各 1000 回合、同规格一局跑到结束、`20 x 20 / 4 AI` 五局各 1000 回合、`51 x 51 / 16 AI` 五局各 1000 回合。非法移动、主 AI 挂机、军队挂机、主 AI 误指挥军队格和重复移动源均为 `0`；同规格终局局在第 `3,299` 回合结束，最大动态军队上限 `27`、最大同时军队数 `3`。
- 最新完整单元测试为 `253 tests OK`（最终一次耗时 `66.087s`）。正式终局验证 `75 x 75 / 36 AI / 困难` 通过，非法移动、AI 挂机、军队挂机和重复移动源均为 `0`；第 `200` 回合后达到至少 `75%` 动态军队上限的活跃 AI 回合占比为 `68.75%`，前 `200` 回合最多同时 3 支军队。报告保存在 `training_data/regression_completion_latest.json`，历史综合矩阵报告保存在 `training_data/regression_matrix_latest.json`。
- 本轮斩首与长期绕圈检测修复后，`IndependentArmyTests` 小规模回归为 `60 tests OK`（`5.941s`），新增的 4 个针对性用例全部通过；`100 x 100 / 12 AI / 24 回合` 性能门槛为 `2.170s`，低于 `6.0s` 上限，没有出现显著性能回退。
- 本轮速度优化将 `51 x 51 / 16 AI / 300 回合` 的无剖析探针从 `15.75s` 降至 `14.39s`，约快 `8.6%`；同一探针在 `cProfile` 下从 `53.1s` 降至 `33.5s`。回归矩阵本轮墙钟总耗时为 `1,215.951s`，其中终局局计算耗时 `818.486s`；并行大混战单项计算的墙钟噪声较大，核心性能结论以探针和矩阵通过性为准。

## 机制清单

| 编号 | 机制 | 当前实现 | 验证 |
| --- | --- | --- | --- |
| 1 | 自动过回合 | 自动倒数并同步结算玩家与 AI 命令 | 主流程测试 |
| 2 | 每回合两阶段 | 每个地块每阶段最多行动一次，同一部队普通地形一回合最多两格 | `test_ai_can_move_the_same_stack_in_two_turn_phases` |
| 3 | AI 操作公平 | AI 也遵守每阶段每格一次移动，不再一回合操作多个格子 | 训练模拟和移动规则测试 |
| 4 | AI 视野限制 | AI 始终使用自身视野半径，关闭人类迷雾也不会让 AI 获得全图 | `test_disabling_fog_does_not_disable_ai_vision_radius` |
| 5 | 默认视野半径 | 真人与 AI 默认为每块占据格半径 2，设置范围 0 至 5 | 视野测试 |
| 6 | 丘陵阻挡视野 | 丘陵自身可见，正后方沿直线的格子不可见 | `test_hill_blocks_vision_behind_it_but_not_itself` |
| 7 | 丘陵移动 | 平地出入丘陵耗时两回合，丘陵到丘陵一回合，己方丘陵无额外延迟 | 丘陵规则测试 |
| 8 | 丘陵等待显示 | 延迟第一回合士兵仍留在原格，第二回合才真正移动 | `test_pending_hill_move_reserves_source_and_can_be_cancelled` |
| 9 | AI 丘陵寻路 | 使用实际 1/2 回合代价的桶队列寻路 | AI 距离场和丘陵测试 |
| 10 | 六类任务评分 | 防守主城、防守领土、攻敌城、攻敌土、攻中立城、探索扩地 | `test_mission_candidates_cover_all_six_tasks` |
| 11 | 任务承诺与切换惩罚 | 当前任务有承诺加成，替换必须超过难度阈值 | `test_ai_switches_mission_only_after_hysteresis_margin` |
| 12 | 集结 A | 己方普通土地超过一半达到 4 兵时动员 | `test_land_gather_triggers_only_when_majority_has_four_army` |
| 13 | 集结 B | 城堡兵力足够时按周边压力选择半兵或全兵 | `test_city_gather_uses_pressure_and_chooses_largest_capacity` |
| 14 | 集结模式选择 | 比较 A/B 可动员容量，选择更快或更多的一种 | 集结模式和出征测试 |
| 15 | 禁止挂机 | 有可移动兵力时，每回合至少产生一个行动 | `test_ai_never_idles_while_it_has_a_movable_army` |
| 16 | 强制脱离无限集结 | 达到集结阈值后取消继续囤兵并重新执行最高分任务 | `test_ai_must_leave_after_reaching_muster_threshold` |
| 17 | 无敌人时主动搜索 | 向地图中心、全域汇合点和未侦察区域持续推进；长期未接触或领地停滞会提升搜索压力 | `test_ai_actively_searches_when_no_enemy_is_visible`、`test_sustained_peace_search_outranks_home_area_expansion` |
| 18 | 后方调兵增援 | 前线停滞或先锋不足时，从至少 4 格外、至少 4 兵的后方地块调援 | `test_ai_moves_rear_hill_troops_toward_frontline` |
| 19 | 远距预备队 | 先锋弱化时先调后方预备队，而不是继续送兵消耗 | `test_ai_calls_far_reserve_before_a_weakened_vanguard_advances` |
| 20 | 进攻绕路 | 进攻路线遇强敌时提高风险格代价并尝试绕路 | 风险寻路测试 |
| 21 | 防守不绕路 | 守城和防守任务使用普通最短路径 | 风险寻路分支测试 |
| 22 | 持久消耗分兵 | 对某对手累计损耗超过己方总兵力两倍时触发发育分兵 | `test_ai_triggers_development_when_attrition_exceeds_twice_its_army` |
| 23 | 消耗计数清零 | 90 回合没有新损耗则清除该对手累计值 | `test_ai_resets_stale_attrition_instead_of_remembering_it_forever` |
| 24 | 发育任务并行 | 发育任务只占一个移动阶段，另一阶段继续战争 | 分兵发育测试 |
| 25 | 自身状态判断 | AI 判断进攻、防守、发育、探索四种状态 | `test_strategy_state_changes_mission_weights` |
| 26 | 状态影响任务权重 | 四种状态分别放大或压低六类任务评分 | `STATE_MISSION_WEIGHTS` 与状态测试 |
| 27 | 压倒优势梭哈 | 可机动兵力达到目标敌军估算的 1.25 倍且不在防守时，有概率冻结发育并全军进攻 | `test_overwhelming_force_freezes_development_and_attacks` |
| 28 | 边境集结点 | 进攻敌军时先在边境建立集结区 | `test_attack_starts_when_local_enemy_is_half_of_mustered_army` |
| 29 | 取消 40% 全国兵力门槛 | 集结区周围 3 格敌军不超过己方当地兵力的 50% 即可出击 | `BORDER_MUSTER_LOCAL_ENEMY_RATIO` |
| 30 | 重大入侵防守 | 某敌军在己方附近投入超过全国兵力 20% 时，防守评分乘 2 | `test_major_invasion_doubles_defense_weights` |
| 31 | 敌主城暴露 | 发现敌将后，梭哈与主城任务权重乘 1.5，并优先该势力 | `EXPOSED_GENERAL_ALL_IN_MULTIPLIER` |
| 32 | 守城硬底线 | 首次接触后至少保留全国兵力 10% 驻扎主城 | `test_home_guard_move_keeps_hard_ten_percent_floor` |
| 33 | 守城偏好比例 | 常规目标约 20%，但可被明确斩首窗口覆盖 | `test_ai_maintains_twenty_percent_home_guard` |
| 34 | 首次接触前不拱卫 | 未接触敌人前不提前触发主城拱卫任务 | `test_home_guard_priority_starts_after_first_contact` |
| 35 | 主城视野方形优先占领 | 主城周围 `(2v+1) x (2v+1)` 非己方可通行地块以高权重夺取，`v` 为当前视野半径；近身 3x3 防守警戒圈保持独立 | `test_ai_prioritizes_occupying_home_vision_square` |
| 36 | 先守城后反攻 | 主城有守军缺口时先补军，再追击附近敌军 | `test_ai_garrisons_home_before_chasing_nearby_intruder` |
| 37 | 纵深阶段任务 | 第 1 至 100 回合要求 2；之后每 100 回合升 1，最高 5；按缺口获得递增奖励，但不具备绝对最高优先 | `test_capital_depth_requirement_ramps_after_hundred_turns`、`test_capital_depth_rewards_deficit_without_absolute_priority` |
| 38 | 纵深与守军解耦 | 提高纵深不再直接改变守军比例 | 守军比例测试 |
| 39 | 动态守军比例 | 主城到最近已发现敌区超过 8 格时降至 5%，超过 15 格时降至 2% | `test_capital_guard_ratios_follow_nearest_enemy_territory_distance` |
| 40 | 区域补兵权重 | 区域平均兵力除以最近敌对距离超过 3 时，优先把该区域兵力调往前线；未接触敌人时距离按 1 | `test_regional_reinforcement_ratio_uses_distance_to_nearest_enemy` |
| 41 | 重兵远距集结倾向 | 集结评分同时参考兵力和到战线/集结点距离 | `test_reinforcement_prefers_large_distant_stack_over_small_nearby_tile` |
| 42 | 局部压倒吞并 | 兵力差超过 `max(4, 1.2 * 目标兵力)` 时，仅把该次吞并操作权重乘 1.5，不强制最高优先 | `test_local_engulf_is_a_weight_bonus_not_a_forced_priority` |
| 43 | 排除等待集结与独立军队 | 局部吞并权重不作用于等待集结的兵、主城硬守军或 `军N` 独立军队格 | `test_local_engulf_weight_excludes_waiting_gather_stack`、`test_local_engulf_weight_excludes_independent_army_cell` |
| 44 | 相邻主城即时斩首 | 己方兵力高于相邻敌主城守军时，立即吞并主城 | `test_adjacent_capturable_enemy_capital_is_taken_immediately` |
| 45 | 斩首窗口压过守城 | 明确能攻破敌将时，允许压倒性进攻覆盖被动守城 | `test_ai_all_in_enemy_general_beats_home_threat_mission` |
| 46 | 敌区情报持久化 | 敌方格重新进入迷雾后，仍保留已发现的敌方控制区域 | `test_ai_remembers_enemy_territory_after_it_returns_to_fog` |
| 47 | 找主城战略 | 未发现敌将时优先搜索尚未侦察的敌方控制区 | `test_ai_persists_enemy_general_campaign_after_losing_sight` |
| 48 | 发现主城后停止搜索 | 一旦发现敌将，停止该势力区域搜索，转为斩首战役 | `test_known_enemy_general_switches_off_territory_search` |
| 49 | 中立城堡发育 | 容易取得且条件好的中立城会提高发育任务得分 | `test_good_neutral_city_can_outrank_home_guard_maintenance` |
| 50 | 长期消耗不拖死战争 | 没有强制胜者回合，但战争节奏、探索和发育分支持续运行 | 100 局实测 |
| 51 | 四类 AI 头衔比例 | 每个普通对局按标准 40%、进攻 20%、打堡 20%、龟缩 20% 分配 | `test_ai_archetype_assignment_uses_fixed_weight_quotas` |
| 52 | 进攻头衔 | 攻敌城和攻敌领土权重乘 1.5，梭哈奖励乘 2，突击期间取消守军上限等梭哈限制 | `test_attack_archetype_boosts_attacks_and_all_in` |
| 53 | 打堡头衔 | 探索权重乘 1.5，中立城和弱敌城任务权重乘 2 | `test_fort_archetype_boosts_exploration_and_weak_castles` |
| 54 | 龟缩头衔 | 前期压低探索和进攻并囤兵主城，达到目标后全军探索与进攻，斩首权重乘 2 | `test_turtle_archetype_accumulates_then_releases_all_in` |
| 57 | 战争与和平状态 | 发生兵力交换进入战争状态，连续 60 回合无交换恢复和平 | `test_attrition_marks_war_and_stale_exchange_returns_to_peace` |
| 58 | 和平期排名倾向 | 和平且全球兵力排名低者优先发育，排名高者优先探索 | `test_peace_rank_prefers_development_when_low_and_exploration_when_high` |
| 59 | 战争期强弱修正 | 弱于交战方时防守权重乘 1.2；不弱于对方时进攻权重乘 1.2 | `test_war_state_scales_weaker_defense_or_stronger_attack` |
| 60 | 平地集结路径 | 选出具体起点与到集结点的路径，最大化沿途可收集兵力，并优先经过城堡 | `test_land_gather_selects_concrete_path_and_prefers_castle` |
| 61 | 城堡集结路径 | 城堡兵力超过相邻非城堡地兵总和且兵力/前线距离大于 2 时，以城堡为起点运兵到集结点 | `test_castle_gather_uses_ratio_path_from_castle_to_rally` |
| 62 | 山地邻地折算 | 山地按周围非山地、非城堡地块的平均兵力计入城堡邻地压力 | `test_castle_neighbor_pressure_uses_surrounding_average_for_mountain` |
| 63 | 待机区域集结倾向 | 区域长时间未出兵时按分段函数缩短对敌距离，套入区域补兵比例公式；实际出兵立即重置 | `test_idle_region_shortens_effective_distance_in_reinforcement_ratio`、`test_idle_distance_discount_uses_piecewise_steps` |
| 64 | 独立军队数量与冷却 | 真人、AI 上限均为 `4 + floor(总兵力 / 8000)`；真人冷却 10 回合，AI 冷却 35 回合 | `test_army_limit_scales_with_total_army_for_human_and_ai`、`test_creation_respects_limit_cooldown_and_general_restriction` |
| 65 | 主城建军限制 | 真人、AI 都不能在主城创建独立军队；城堡属于合法建军地并获得建军评分加成 | `test_creation_respects_limit_cooldown_and_general_restriction`、`test_castle_can_host_an_army` |
| 66 | 军队兵目标 A | A 为全国最高非军队地块兵力的两倍，最低 50；军队头衔格不参与 A 计算 | `test_target_strength_excludes_tiles_that_become_armies` |
| 67 | 独立行动预算 | 每支军队每阶段各有独立行动槽；主 AI 从候选阶段排除军队格并保留自己的普通行动 | `test_army_move_is_separate_from_human_route_action`、`test_main_ai_keeps_its_action_when_multiple_armies_act` |
| 68 | 真人双击右键建军 | 双击己方非主城地块创建独立 AI 军队，冷却或达到当前动态上限时拒绝 | `test_double_right_click_creates_human_army_and_shows_cooldown` |
| 69 | 真人右键接管 | 右键单击己方军队立即取消头衔并恢复普通手动指挥 | `test_right_click_releases_human_army_and_restores_control` |
| 70 | 独立军队防守优先（已合并） | 由机制 86 的移动敌军距离分级取代，保留单一防守判定 | `test_moving_enemy_forces_defense_at_ten_cells` |
| 71 | AI 军队头衔阈值 | AI 对超过对应 `A` 或任意非主城 `3` 兵堆授予头衔，尽可能填满当前兵力预算；训练循环和实战共用该入口 | `test_ai_does_not_title_weak_spread_stacks`、`test_ai_is_eager_to_title_leading_stack`、`test_ai_titles_any_three_stack_with_ai_budget` |
| 72 | AI 释放评分限制 | AI 只能释放低于 100 兵且表现持续不合格的军队；100 兵及以上不可自动取消 | `test_ai_can_release_poorly_performing_army_below_100`、`test_ai_cannot_release_army_with_strength_100_or_more` |
| 73 | 军队头衔撤销 | 主锚点被攻占且没有存活续接目标时自动撤销整支军队头衔 | `test_army_title_is_removed_when_formation_anchor_is_captured` |
| 74 | 找主城视野土地记忆 | AI 永久记录所有视野内看过的可通行土地；搜索敌主城时跳过已看区域，迷雾恢复后记忆不丢失 | `test_ai_remembers_all_seen_land_for_capital_search` |
| 75 | 找主城全局未探索搜索 | 已知敌区外沿搜完后，沿已侦察边界选择新的未探索区域；不重复旧巡逻区域 | `test_ai_global_capital_search_moves_beyond_seen_land`、`test_ai_global_capital_search_does_not_repeat_seen_target`、`test_enemy_search_falls_back_to_global_unseen_region` |
| 76 | 军队强制移动（已合并） | 由机制 91 的每阶段强制移动与三态选择取代 | `test_army_moves_every_phase_even_without_gather_target` |
| 77 | 军队丘陵跟随 | 延迟移动期间头衔留在起始格，完成登丘时头衔与兵力同步转移 | `test_army_title_follows_delayed_hill_transit` |
| 78 | 军队格隔离 | 真人或主 AI 都不能把军队格选作普通行动起点/终点；普通兵不会并入军队格，军队也不会进入另一军队格 | `test_human_commands_cannot_select_or_route_through_army_cells`、`test_main_ai_never_uses_army_cells_as_source_or_target`、`test_army_movement_rejects_other_army_cells` |
| 79 | 移动友军补兵隔离 | 军队补兵排除 `pending` 移动、当前阶段 incoming 移动和已预订的友军来源格；新建军时清除该格旧移动 | `test_army_resupply_ignores_pending_and_incoming_moving_sources`、`test_new_army_creation_cancels_its_pending_move` |
| 80 | 残编自动解散（已合并） | 由机制 90 的 `<= 2` 自动解散统一覆盖，不再维护 1 兵特例 | `test_army_with_two_soldiers_is_disbanded_before_planning` |
| 81 | 无效驻军与停摆清理 | 军队状态位于己方主城时立即清理；不存在任何合法移动时立即撤销头衔 | `test_army_is_released_if_it_ends_up_on_own_general`、`test_unable_army_is_released_immediately` |
| 82 | 军队编号 | 第一支军队编号为 `军1`，后续依次为 `军2`、`军3`，低缩放也保留编号 | `test_creation_respects_limit_cooldown_and_general_restriction` |
| 83 | 主城排除出 A | 主城即使有大量守军，也不参与军队补兵目标 `A` 的最高兵力计算 | `test_target_strength_excludes_tiles_that_become_armies` |
| 84 | 军队三态 | 每支军队始终处于进攻、防守、补兵三态之一，不存在旧的 `moving` 临时状态 | `test_forced_move_keeps_one_of_three_army_states` |
| 85 | 进攻转补兵阈值 | 进入进攻后，兵力高于相邻最小敌军两倍时不能转补兵；下降到阈值内才允许 | `test_attack_state_stays_until_adjacent_enemy_reaches_hold_ratio` |
| 86 | 移动敌军防守判定 | 大股敌军移动部队进入本土，距离不超过 10 格强制防守；11 至 20 格且兵力不悬殊时优先防守 | `test_moving_enemy_forces_defense_at_ten_cells`、`test_moving_enemy_prefers_defense_between_eleven_and_twenty_cells` |
| 87 | 军队自主补兵 | 军队自行移动到局部高兵堆、己方城堡、安静区域或可回撤边疆，不等待主 AI 向军队格增援 | `test_resupply_army_moves_itself_toward_castle_or_local_stack` |
| 88 | 补兵完成转攻 | 兵力达到 `A` 且没有防守压力时，军队由补兵状态转为进攻状态 | `test_resupply_switches_to_attack_after_reaching_a` |
| 89 | 失联军队解散 | 军队无法通过可通行路径回到己方边疆时立即取消编制 | `test_unable_army_is_released_immediately` |
| 90 | 残编自动解散（后续修订） | 普通状态兵力小于等于 2 时解散；第 200 回合后的编制保存模式按机制 130 保留 2 兵骨架用于补兵，低于 2 兵仍解散 | `test_army_with_two_soldiers_is_disbanded_before_planning`、`test_army_is_released_when_failed_attack_leaves_one_soldier` |
| 91 | 每回合强制移动 | 除丘陵延迟外，每个存在且可移动的军队阶段内必须产生一个合法 Move；被包围时优先撞向敌军抵消 | `test_army_moves_every_phase_even_without_gather_target`、`test_forced_move_keeps_one_of_three_army_states` |
| 92 | 跨 AI 移动观察 | 同阶段先前玩家已规划的移动会传给后续 AI，使军队能识别正在深入的敌军 | `test_moving_enemy_forces_defense_at_ten_cells` |
| 93 | 真人路线时间队列 | 多条真人路线按录入时间排队；每个移动阶段只执行队首一条可行动路线，执行后轮到队尾，暂停预填也不会并发执行 | `test_paused_commands_execute_in_chronological_phase_order`、`test_scheduled_route_rotates_behind_other_commands` |
| 94 | 37 方独立颜色 | 调色板扩展到 37 个互不相同的颜色，任意两个玩家或 AI 不再复用同一色块 | `test_all_players_have_distinct_colors` |
| 95 | 主城与低兵军队清理（后续修订） | `军N` 不能锚定或显示在主城、非己方地块及低兵地块；普通状态清理 1–2 兵，后期保存模式按机制 130 保留 2 兵骨架，旧战略标签路径已删除 | `test_army_is_released_if_it_ends_up_on_own_general`、渲染层硬过滤 |
| 96 | 建军最低兵力 | 只有至少 3 兵的地块可以创建独立军队，2 兵直接拒绝 | `test_army_creation_rejects_two_soldiers` |
| 97 | 军队互不合并 | 同一控制器为每支军队预留目标格，多支军队即使追击同一目标也不能在同阶段汇入同一格 | `test_two_armies_never_choose_the_same_target_cell` |
| 98 | 扩兵轮军队 AI 激活 | 每个扩兵轮后的下一回合开始，逐支同步并重置 `军N` 的独立 AI 状态、清理异常状态并刷新 `A` | `test_growth_round_reactivates_each_numbered_army_ai` |
| 99 | 单一编队系统 | AI 只使用后加入的 `ArmyController`/`军N` 管理独立编制，不再维护旧战略主战标签 | `test_legacy_strategic_army_label_is_removed` |
| 100 | 单一局部吞并规则 | 局部吞并只保留后加入的 `1.5` 倍权重规则，旧强制吞并决策函数已删除 | `test_local_engulf_weight_requires_confirmed_margin`、`test_local_engulf_is_a_weight_bonus_not_a_forced_priority` |
| 101 | 单一军队补兵/回撤路径 | 独立军队只通过后加入的三态自主规划补兵或回撤，旧 `_gather_target`、`_retreat_target` 已删除 | `test_army_resupply_ignores_pending_and_incoming_moving_sources`、`test_resupply_army_moves_itself_toward_castle_or_local_stack` |
| 102 | 城堡建军与建军欲望 | 城堡可作为建军起点；AI 与真人一样至少有 4 支，并每 `8000` 兵增加一槽；AI 冷却 35 回合、任意非主城 `3` 兵堆即可建军，并对城堡加权 | `test_castle_can_host_an_army`、`test_ai_titles_any_three_stack_with_ai_budget` |
| 103 | 军队生存权重 | 军队进入会被整建制吃掉的敌军地或相邻强敌地时降权，存在安全候选时优先绕开而不是抵消兵力；防守判定不受影响 | `test_army_survival_weight_avoids_equal_enemy_counterattack` |
| 104 | 受击回补权重 | 军队累计损失达到阈值且当前处于危险地时，强制转入补兵并优先撤向安全城堡、后方兵堆或远离敌区的己方地 | `test_damaged_army_retreats_to_resupply`、`test_resupply_prefers_safe_rear_stack` |
| 105 | 位移距离量 | 每 30 回合结算一次军队累计行驶距离，超过 20 格记为一次位移奖励，并计入训练适应度与表现评审 | `test_displacement_reward_requires_twenty_cells_per_interval`、`test_displacement_reward_ignores_short_trips` |
| 106 | 索敌目标推进 | 敌方搜索目标和全局航点会排除已搜索候选，并在 `45` 回合或失效后重新选择，防止原地循环 | `test_enemy_search_target_rotates_instead_of_repeating_visited_cell`、`test_ai_global_capital_search_does_not_repeat_seen_target` |
| 107 | 和平期主动出击 | 长时间无接触时，主 AI 的战争节奏可直接建立高权重搜索任务；形成规模的 `军N` 会离开补给循环向外索敌 | `test_war_tempo_creates_an_active_search_mission_without_enemy_intel`、`test_peace_army_leaves_resupply_to_search_outward` |
| 108 | 随机游走山脊 | 全图随机种子、八方向随机游走，目标密度 `18.75%`；山脊可横、竖、斜向绵延，单格厚、无 2x2 实心块，收尾用最少山格开凿隘口并连通所有可通行组件 | `test_mountains_form_ridges_with_increased_density`、45 张地图审计 |
| 109 | 地形配色 | 山脉改为石头色背景与山脊标记；丘陵背景与平地一致，只保留丘陵线稿标识 | 渲染检查、地图预览和训练/实战小地图共用配色 |
| 110 | 动态军队上限与冷却 | 真人、AI 上限均为 `4 + floor(总兵力 / 8000)`；真人冷却 10 回合，AI 冷却 35 回合 | `test_army_limit_scales_with_total_army_for_human_and_ai`、`test_creation_respects_limit_cooldown_and_general_restriction` |
| 111 | 主城远距调兵 | 回城调兵评分考虑 `地块兵力 / 到主城距离`，达到 `8` 后按比例加权，优先抽调远处高兵地块 | `test_ai_calls_far_high_ratio_stack_for_capital_defense` |
| 112 | 纵深奖励非绝对优先 | 纵深任务按缺口获得递增奖励，但基础评分低于斩首战役和主动索敌上限，不再无条件抢最高任务 | `test_capital_depth_rewards_deficit_without_absolute_priority` |
| 113 | 长征先锋不频繁换锚 | 只有先锋兵力降至难度探索阈值以下才从后方换主力；形成规模的先锋不会被后方多 2 兵的地块反复替换 | `test_ai_keeps_a_formed_vanguard_instead_of_swapping_to_home_reserve` |
| 114 | 中心搜索点避山 | 双 AI 长距离搜索时，如果地图中心是山脉，选择中心附近最近的可通行格，避免距离场把中心视为不可达 | `test_two_player_search_uses_passable_tile_when_center_is_mountain` |
| 115 | 扩地边界形状 | 日常扩地使用多分支前沿奖励和走廊降权，先分兵铺开再由同一先头完成第二格；主动索敌与强制进攻不受降权影响 | `test_expansion_shape_prefers_branching_frontier_over_corridor`、`test_local_expansion_splits_troops_but_forced_offensive_does_not` |
| 116 | 山脉随机有机分布 | 不使用固定象限布种；随机起点和八方向游走生成横/竖/斜向山脊，保留局部疏密差异，但禁止 `2 x 2` 实心块并限制单连通山脊长度 | `test_mountain_distribution_is_random_organic_not_quadrant_tiled`、`test_mountains_form_ridges_with_increased_density` |
| 117 | 丘陵数量与散射 | 丘陵按山脉 `1–2` 倍随机生成，主要聚集在山脉附近，也保留远离山脉的零散丘陵；最终丘陵不超过平地 | `test_hill_counts_follow_mountain_multiplier`、`test_hills_allow_sparse_cells_away_from_mountains` |
| 118 | 山脉与丘陵渲染及视野 | 山脉使用石头色，丘陵保留与平地一致的底色；丘陵自身可见但阻挡其正后方直线视野 | `test_mountain_and_hill_use_required_terrain_rendering`、`test_hill_blocks_vision_behind_it_but_not_itself` |
| 119 | 点击反馈防泄露 | 只有真人、真人托管 AI 和真人军队 AI 会生成占格反馈；其他 AI 的占格动画被过滤 | `test_ai_captures_do_not_create_player_action_animations` |
| 120 | 训练快照防越界 | 共享内存快照读取前校验槽位、缓冲长度、地图宽高与瓦片数量，不完整快照回退缓存 | `test_live_view_rejects_a_truncated_shared_buffer`、`test_training_board_ignores_an_incomplete_snapshot` |
| 121 | 主城援军统一评分 | 不使用固定半径；守军缺口、威胁、距离、兵力/距离比、局部战斗和反攻价值共同决定援军、换防或本地继续作战 | `test_home_support_uses_score_without_a_fixed_candidate_radius`、`test_home_support_keeps_clearly_better_local_fight_when_capital_is_safe`、`test_ai_calls_far_high_ratio_stack_for_capital_defense` |
| 122 | 军队位置全量隔离 | 主 AI 同时避开军队当前格、丘陵延迟起点/终点和待执行目标；守军上限过滤不会让可移动野战兵整回合空过 | `test_main_ai_does_not_duplicate_a_delayed_army_move_source`、`test_home_guard_cap_does_not_idle_a_movable_field_stack`、`test_ai_never_idles_while_it_has_a_movable_army` |
| 123 | 后期动态军队扩张 | 总兵力每满 `8000` 增加一个军队槽，AI 会实际创建新军队 | `test_ai_creates_extra_armies_when_total_army_scales_past_eight_thousand`、终局矩阵 |
| 124 | 重大战斗节奏升级 | 存活势力不超过 `8` 时跟踪 `max(12, 总兵力 * 0.3%)` 的重大兵力抵消；长期只有小摩擦时提高索敌进攻压力，早期大混战不触发 | `test_war_tempo_reacts_to_a_gap_in_significant_battles`、`test_significant_battle_updates_war_tempo_memory`、`test_significant_battle_escalation_is_late_game_only` |
| 125 | 全屏显示 | 设置页可切换窗口/全屏，游戏内 `F11` 随时切换且保留当前视角与比例尺 | 启动参数、显示标志测试、冒烟测试 |
| 126 | 隐藏主军队标签 | 真人或主 AI 操作己方部队时建立隐藏标签；跟随移动、可进入主城、保留主 AI/真人控制，`10` 回合未操作自动撤销，转正式军队时立即撤销 | `test_main_army_tag_follows_moves_and_expires_after_idle_turns`、`test_main_ai_move_activity_creates_the_hidden_main_army_tag`、`test_main_army_tag_allows_army_to_enter_own_general`、`test_main_army_tag_is_removed_when_the_stack_becomes_an_army` |
| 127 | 主军队标签隔离与失效回收 | 隐藏主军队标签有效时保留主 AI/真人控制，独立军队避开该格；标签连续 10 回合无操作失效后，该格恢复普通补兵候选 | `test_main_army_tag_can_be_reinforced_by_an_independent_army`、`test_main_army_tag_follows_moves_and_expires_after_idle_turns` |
| 128 | AI 计算热路径优化 | 稀疏距离场直接分派，扩地形状、邻域、区域均值改走本地网格访问，减少百万级 Python 调用 | `253 tests OK`、固定矩阵、独立速度探针 |
| 129 | 前 200 回合渐进建军 | AI 第 0 回合最多 1 支，之后随回合线性增长，第 200 回合恢复到 `4 + floor(总兵力 / 8000)` 动态上限；AI 冷却仍为 35 回合 | `test_ai_army_creation_limit_ramps_until_turn_two_hundred`、`test_early_ai_army_limit_trims_extra_armies` |
| 130 | 后期编制保存 | 第 200 回合后，当前军队少于有效上限 `75%` 时进入保守保存模式，优先补兵与规避无把握战斗，并保留 2 兵骨架直至补兵；低于 2 兵仍撤销 | `test_late_game_low_roster_army_is_preserved_for_resupply`、`test_ai_creates_extra_armies_when_total_army_scales_past_eight_thousand` |
| 131 | 军队同步顺序 | 主 AI 规划普通行动前先同步 `军N` 的丘陵延迟、残编、位置和隐藏标签，避免主 AI 与军队 AI 在同一格重复取得控制权 | `253 tests OK`、正式终局报告 |
| 132 | 同回合反向移动保护 | 同一回合第二阶段不能把第一阶段刚移动的军队直接移回原格；低兵军队优先在己方后方安全区域补兵，暂时无落点只累计停摆而不撤销编制 | `253 tests OK`、正式终局报告 |
| 133 | 相邻主城必胜斩首 | 军队相邻敌方主城且 `当前兵力 - 1 > 主城守军` 时直接攻城主城，绕过普通敌军地的 `3` 倍安全门槛；兵力不足时仍不会送死 | `test_army_immediately_captures_adjacent_enemy_general_when_stronger`、`test_army_does_not_suicide_into_stronger_enemy_general` |
| 134 | 长期小区域绕圈检测 | 每 `30` 回合采样位置，连续 3 次采样的曼哈顿直径不超过 `2` 即判定为绕圈挂机，并强制向区域外突破；正常防守和必胜斩首优先于突破 | `test_army_is_marked_loitering_after_sustained_tiny_area_motion`、`test_army_keeps_moving_when_loitering_is_detected` |

头衔机制只在正常人机对局创建 AI 时启用。训练引擎、AI 机制审计、接触基准和单元测试默认都使用“标准”头衔，避免随机类型污染自进化样本的可复现性。

## 冲突与最终口径

编号为历史稳定 ID，已合并或已删除的机制保留原编号，不重新编号，便于对照旧版本。

| 历史表述冲突 | 最终处理 |
| --- | --- |
| 最大玩家曾要求 12、24、36 | 最终规则为最多 36 个 AI，加真人的总阵营上限 37 |
| 主城纵深曾要求前 100 回合直接调整到大于 5 | 最终按最后修正执行：前 100 回合要求 2，之后每 100 回合升 1，最高 5 |
| 守军比例曾直接由纵深距离决定 | 最终改为只看主城到最近已发现敌区的距离；纵深不再直接参与守军比例 |
| 曾要求全国 40% 兵力才能进攻 | 已取消，改为边境集结区周围敌军不超过己方当地兵力 50% 即可出击 |
| “禁止挂机”与“等待集结”可能冲突 | 有可移动兵力时仍执行合并、扩地、调兵或防御；只有被战略目标明确保留的兵力可不出发 |
| 主城 20% 守军与梭哈斩首可能冲突 | 10% 为硬底线，20% 为常规偏好；明确斩首窗口和相邻必胜窗口可以覆盖偏好 |
| AI 迷雾与人类关闭迷雾可能冲突 | 关闭迷雾只对真人显示生效，AI 始终使用自身视野 |
| 曾要求自动寻路，后又取消 | 最终没有 AI 自动寻路；真人只保留相邻格连续路线，AI 使用多回合持久任务和动态调兵 |
| A/B 集结与边境集结优先级 | 边境集结和压倒性进攻优先级更高；A/B 仍由单元测试验证，小地图高压局面可能不会进入 A/B |
| 旧“战略主战标签”与后加入的 `军N` 独立军队重复 | 删除旧标签、跟随、目标分配和 UI 分支，只保留后来加入的 `ArmyController` 与 `军N` |
| 旧“局部吞并必须强制最高优先”与后加入的 `1.5` 倍权重规则冲突 | 删除旧强制吞并函数，只保留权重倍率；命中规则时不强制抢占其他更高任务 |
| 旧军队 `_gather_target`/`_retreat_target` 与后加入的三态自主补兵/回撤重复 | 删除两个旧目标选择器，统一由进攻、防守、补兵三态规划处理 |
| 旧 1 兵残编特例与后加入的 `<= 2` 统一解散规则重复 | 删除 1 兵特例，AI 和真人统一使用 `<= 2` 自动解散 |
| 旧“战略标签”与后加入的独立军队使用 `_make_move` 辅助重复 | 删除仅测试使用的旧辅助路径，生产路径统一由 `_move_toward` 加 `_record_army_move` 记录 |
| 旧 `_all_spawns_connected`（只要求出生点互通）与后加入的 `_all_passable_connected`（要求全可通行区连通）重复 | 删除旧校验，地图生成统一要求所有非山格子处于同一连通区且全部出生点可达 |
| 主城拱卫旧固定半径与新评分机制冲突 | 最终统一为无固定半径的连续评分，局部战斗、远距高兵比、守军缺口和梭哈窗口在同一评分体系内比较 |
| 军队延迟移动与主 AI 行动源的旧隔离不完整 | 最终统一使用当前格、transit、pending destination 的占用集合，主 AI 候选和兜底都排除 |

本轮再次复核全部历史口径，没有发现需要修改代码的新矛盾；上述两项是历史实现与新后加入机制之间的重复/遗漏，已按“后加入规则覆盖旧实现”的原则统一。

## 100 局测试结果

测试配置：

```text
对局数：100
每局 AI：6
地图：20 x 20
每局上限：900 回合
随机种子：20260925
```

结果：

```text
视野内接触率：100 / 100
实际兵力抵消率：100 / 100
900 回合内决出胜者：96 / 100
平均回合：337.89
平均每名 AI 大战回合：8.138
采样 AI 回合：108847
可行动却空过：16 次
空过率：0.0147%
```

任务和状态覆盖：

```text
四种自身状态全部出现：进攻、防守、发育、探索
六类任务全部出现
主城守卫：600 / 600 个 AI 样本出现
纵深高权重任务：600 / 600 个 AI 样本出现
重大入侵/防守分支出现
消耗分兵发育：375 个 AI 样本出现
边境集结：426 个 AI 样本出现
全军突击：563 个 AI 样本出现
大战：100 局均出现符合兵力规模的兵力抵消
```

未被高压小地图选中但仍通过单元测试的机制：

- A 模式土地集结。
- B 模式城堡集结。

原因是 20 x 20、6 AI 局面很快进入边境集结或压倒性进攻分支，这两个分支的优先级高于普通 A/B 集结。该现象符合既定优先级，不判定为机制缺失。

## 训练集回灌

```text
新增历史条目：100
训练集来源标签：mechanism_audit_6ai_20x20
回灌前已完成对局：72
回灌后已完成对局：172
回灌前世代：9
回灌后世代：10
冠军口径：最近 100 局窗口内适应度最高的个体
```

原始训练集备份：

```text
work/training_data_backup_20260925
```

新增汇总文件：

```text
training_data/mechanism_evaluation_6ai_20x20.json
```

## 固定回归矩阵

运行：

```powershell
python benchmark_regression_matrix.py
```

本轮结果：

```text
75 x 75 / 36 AI / 5 局 x 1000 回合：5/5 通过
75 x 75 / 36 AI / 1 局跑到结束：通过，AI 21 在第 3,299 回合获胜
20 x 20 / 4 AI / 5 局 x 1000 回合：5/5 通过
51 x 51 / 16 AI / 5 局 x 1000 回合：5/5 通过

非法移动：0
主 AI 挂机：0
军队挂机：0
主 AI 误指挥军队格：0
重复移动源：0

终局局最大总兵力：189,209
终局局动态军队上限：27
终局局实际同时军队数：3
终局局军队上限扩张：成立（后续保留逻辑会提高后续运行的可见上限利用率）
```

原始报告：`training_data/regression_matrix_latest.json`。
