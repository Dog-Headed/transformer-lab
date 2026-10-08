# Transformer Lab · 看见注意力

面向初学者的中文 Transformer 交互式教学网站。单层双向 Post-LN Encoder，React 19 + TypeScript + Next.js App Router API（Vinext / Vite 运行），SVG 与 CSS 动画，无外部模型 API。

## 本地运行

需要 Node.js 22.13+（推荐 Node.js 24）和 npm。

```bash
npm run install:ci
npm run dev
```

打开终端显示的地址，默认 http://localhost:5173/。

```bash
npm run check:types
npm run check:model
npm run check:generation
npm run build
npm start
```

`npm start` 使用本地 Cloudflare Workers 运行生产构建；请打开终端显示的地址。没有 npm 的受限环境也可以在依赖安装后直接执行 `node scripts/run-framework.mjs dev` 或 `node scripts/run-framework.mjs build`。

## 已实现

- 输入、简化分词、Embedding、正弦位置编码、Q/K/V、缩放点积、Softmax 与 AV、多头拼接与投影、Add & Norm、FFN、第二次 Add & Norm 与输出表示。
- 上一步、下一步、自动播放、暂停、速度切换、重置、直接跳转。
- token 查询切换、2 个注意力头对比、动态连线、热力图、数值提示、权重条形图。
- 中文说明、公式、实时张量维度、键盘操作、手机布局与减少动态效果支持。
- 空输入提示，最多计算前 10 个 token，并明确提示截断；最多输入 240 个字符。

## 教学约定

本项目使用固定随机种子的演示参数，**未经过训练**。数值计算真实、可复现，但不代表已经学到的语义关系。

- 空格分词；无空格中文按字拆分，拉丁字母连续串作为一个 token。不是 BPE。
- token ID 来自当前句子的临时词表；重复 token 共享 ID 和嵌入。
- `d_model=8`、`heads=2`、`d_k=d_v=4`、`d_ff=16`、`batch=1`。
- 正弦位置编码；后置 LayerNorm；ReLU FFN；两次残差与归一化。
- γ=1、β=0、ε=1e-5；省略 dropout、嵌入缩放和 batch 维展示。
- Encoder 可以查看前后所有 token。没有因果遮罩；不是 GPT Decoder。
- 最后输出上下文向量，不生成下一词。
- 输入只用于浏览器内计算，不发送给模型服务。

结构依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

## 结构与扩展

```text
app/page.tsx                         页面状态、播放控制、交互编排
app/globals.css                      设计变量、响应式样式和动画
components/lab/visualization.tsx     SVG 连线、向量、矩阵、逐阶段可视化
lib/transformer/engine.ts            无 UI 依赖的确定性计算引擎
lib/transformer/steps.ts             中文课程、公式和维度元数据
lib/modules.ts                      模块注册表（LoRA / RAG / KV Cache 扩展点）
scripts/verify-model.mjs             数值和边界验证
```

后续模块可各自建立 `lib/<module>/engine.ts` 与 `steps.ts`，复用播放控制与可视化组件。LoRA、RAG 尚未实现。LLM 生成模式及 KV Cache 已实现，入口为 `/generation`，首页顶部与“输出表示”之后均有入口。

## 检查结果

已通过 TypeScript 检查与生产构建；覆盖 6 类输入的数值不变量检查，包括 Softmax 稳定性、每行权重和、归一化、位置区分、上下文敏感性与截断。浏览器验证包括 11 个步骤、输入边界、头与 token 切换、播放终点、键盘、对话框、桌面/390px 手机布局与减少动态效果；未发现页面脚本异常。

WebMCP 渐进增强工具 `navigate_transformer_step` 已通过模拟注册接口的有效/无效输入检查。当前浏览器无原生 WebMCP，因此未验证原生浏览器集成，不影响正常交互。

## 发布

本项目可通过 Sites 发布；`.openai/hosting.json` 保存站点身份，不包含密钥。开发预览的本地模拟登录来自 Sites starter，生产发布不使用模拟登录。



## LLM 生成模式（/generation）

独立于双向 Encoder 的单层因果 Decoder。生成过程为：因果层内计算 → 最后一个 hidden state → LM Head → logits → 温度与 Softmax → 候选筛选 → 选出 token → 追加上下文 → 下一轮。

- 固定的 24 项教学词表（含 EOS），真实矩阵乘法得到 logits；不是预写答案或连接在线模型。
- 采样支持 Greedy、Top-k、Top-p 和全词表 Temperature。温度先作用于 logits；过滤后再归一化。Greedy 忽略温度。
- 默认固定种子，输入/参数相同可复现。更改采样参数会重开实验；后退与前进回放同一个采样结果。
- “自动播放本轮”追加一个 token 后暂停；“连续生成”自动进入下一轮，直到 EOS 或上限。都可以随时暂停。
- 起始提示 1–12 个 token，最多新生成 12 个，上下文不超过 24；过长提示明确报错，不静默截断。
- KV Cache 是实际增量计算：首次 Prefill 计算提示的全部 K/V；后续轮只投影新增 token。开关缓存保留教学进度与已生成结果，并重建对应计算轨迹。
- 因果遮罩矩阵展示全部行；最后一行注意力重点高亮，未来 token 单独标注为不可见。
- 缓存之外保留的 hidden state 与旧注意力行仅供教学复盘。计数对比表示算法操作次数，不声称是真实延迟或内存测量。

新增源码：

```text
app/generation/page.tsx              生成模式路由与元数据
components/lab/generation-lab.tsx    生成流程、图表、缓存对照、采样控制
lib/generation/engine.ts             因果 Decoder、增量 KV、LM Head、采样
lib/generation/session.ts            可回退的生成帧、停止条件与缓存切换
lib/generation/steps.ts              生成教学步骤与公式
scripts/verify-generation.mjs        因果不变性、缓存等价性、采样与回退验证
```

数值测试覆盖 6 轮完整/增量推理的 hidden state、attention 与 logits 等价；改变未来 token 不影响前文表示；400 次种子采样均只落在保留候选中；包括 top-p 最短前缀、温度、EOS、上限、上下文回退与缓存切换。浏览器检查覆盖全部 8 步、四种策略、单轮播放与连续生成、输入边界、桌面与手机、原 Encoder 入口。

参考：[生成策略](https://huggingface.co/docs/transformers/main/en/generation_strategies)、[KV Cache](https://huggingface.co/docs/transformers/main/en/kv_cache)。

## 模型训练实验室（/training）

新模块与 Encoder、LLM 生成模式共用导航和设计。全部句子与对话均为预设，无需用户输入、模型 API 或服务器训练。

- 7 章：训练材料、随机初始化、一次参数更新、训练与 checkpoint、SFT、RLHF、训练与生成对照。
- 字符级固定词表与从语料生成的真实前缀/目标；微型因果 Transformer 使用交叉熵、完整反向传播与 Adam，在浏览器实际更新共享参数。
- 手动拆解预测、loss、梯度、优化器更新与更新后的检查；自动循环最多 2400 步；曲线使用固定训练集的实际平均 loss。
- 每 400 步保存深拷贝权重，历史查看不覆盖当前状态。数据和权重仅保存在页面会话中，刷新重置。
- 训练后的真实续写与初始化权重对比共用模型结构；SFT 回复对比仍为清楚标注的预设教学样例，不运行 token 级微调，也不伪造 loss 数字。
- RLHF 收集真实点击偏好，训练四个人工教学特征上的线性奖励模型；在四个候选上实际优化期望奖励减 KL。不是完整 PPO。用户可撤销标注，也可比较准确性与奉承评分规则。
- temperature 只属于原有生成模块，不参与训练参数更新；普通生成不触发反向传播。

数值验证：`npm run check:training`。验证目标对齐、有限差分梯度、单样本更新、2400 步训练、checkpoint 拷贝、相反偏好产生相反评分及 KL 策略优化。

新增实现：`lib/training/engine.ts`、`components/lab/training-lab.tsx`、`app/training/page.tsx`、`app/training/training.css`、`scripts/verify-training.mjs`。

## 泛化专题（/training/grokking）

从模型训练模块的「泛化专题」入口进入。以 49 道模 7 加法题为材料，固定划分 21 道训练题与 28 道验证题。

- 教学概率回放，不执行神经网络训练；27 个预设存档，step 仅为教学刻度。
- 训练与验证 loss 均由每道题目标答案的概率计算，准确率由最大概率预测计算；曲线、方格、概率条、同题存档共享相同数据。
- 自动回放在训练题满分之后暂停，先考一道新题，再继续体验较晚的验证表现改善；提供暂停、重置、存档滑块、预设节点与回放速度。
- 同一问题可以固定观察，也能同屏比较六个存档；切换「较晚泛化」「只背答案」保持当前存档和题目。
- 独立十位数样例比较 exact-match 和逐位准确率，说明评分门槛造成的突变外观；不宣称所有涌现都是测量假象。
- 手机布局、键盘操作与减少动态效果；状态仅保存在页面会话。

验证：`npm run check:grokking`。检查题目划分、49 题 × 27 存档 × 两条轨迹的概率归一化与指标一致性、延迟泛化反差、固定新题前后答案、回放暂停门槛与两种评分。

主要源码：`lib/training/grokking.ts`、`components/lab/grokking-lab.tsx`、`app/training/grokking/page.tsx`、`app/training/grokking/grokking.css`。

### Trainable Transformer connection

The first four chapters at `/training` now train a real one-layer, one-head causal Transformer: character and learned position embeddings; pre-LayerNorm attention with Q/K/V and output projection; residual connections; an 8→16→8 ReLU feed-forward network; final LayerNorm and vocabulary projection. All 1,273 parameters are shared across prefixes and updated by full analytic backpropagation and Adam (global gradient norm clipping at 1; default learning rate 0.003). Each teaching update uses the last position's next-character target.

Intermediate activations, masked attention, parameter-group gradients, actual deltas, full weight/optimizer snapshots and greedy autoregressive output all come from the same model. Inference never changes the weights. Numerical checks validate every parameter gradient against finite differences, causal invariance, deterministic initialization/training, checkpoint independence and a real training-loss reduction. This tiny corpus is a training demonstration, not a claim of held-out generalization. SFT, finite-candidate RLHF and grokking remain clearly labeled separate teaching modules; the original Encoder and generation demonstrations retain their independent fixed example weights.

## 从零造模工作台（/build-model）

十二步连续项目：命名与目标、组合预设语料、清洗、整句划分、分词、动态模型结构、分步/自动预训练、验证诊断、实际 assistant-only SFT、独立的有限候选偏好实验、留出测试、模型卡与实验文件导入导出。所有文本与测试提示均预设，仅名字允许自由输入。

- 复用实际可训练因果 Transformer，增加 8/12/16 维及 16/24 token 上下文配置；默认结构保持原训练页兼容。
- 数据选择、去重、空白处理和乱码过滤影响实际训练材料；整句及其所有重复版本只属于同一数据集合。
- 字符词表固定公开，另可从训练集频次建立最多八个双字合并；明确不是完整 BPE。训练集、验证集、测试集指标来自真实预测。
- 默认预训练最多 1,600 步；SFT 继续同一权重和 Adam 状态，只有示范回答 token 提供目标，最多 800 步。曲线允许真实过拟合和遗忘；不捏造智能或泛化成功。
- 偏好环节训练独立线性评分器和四候选概率策略，明确不执行完整 PPO，也不更新 Transformer。新增/撤销标注会重置附加策略。
- 模型权重、Adam 状态、初始化与历史存档、配置、词表、选择、日志和源码随 JSON 实验文件导出；经结构/数值检查后可导入并保持相同后续训练行为。测试报告导入时重算。页面状态临时，必须下载实验文件才能在刷新后恢复；不声称跨设备自动同步。
- `scripts/generate-journey-source.mjs` 从真正运行的四份源码生成显示内容。片段标记只指导显示，教学暂停发生在函数调用之间；不是浏览器调试器。生产构建重新生成源码视图，避免展示代码与执行代码脱节。

验证：`npm run check:journey` 覆盖划分泄漏、清洗和分词、六种维度/分词配置的梯度与训练、真实 SFT、独立偏好策略、留出评测、显示源码一致性以及完整导入后相同的继续训练。

从零造模的修复流程：第 8 步可预览并恢复检查点，连同 Adam 状态和阶段计数恢复，恢复前保留权重备份并让旧验收报告失效。第 9 步支持 0/25/50% 原训练文本混入 SFT，以及预训练学习率的 15/35/70% 微调学习率。第 8 步比较预训练检查点和当前权重的留出续写、已训练示范问答、未训练改写问答；后者只用于评测。问题提示依据实际分数和数据覆盖，链接到对应检查步骤；阈值只是教学线索。旧版 JSON 实验档案仍可导入，导入时补齐旧检查点的阶段计数。
