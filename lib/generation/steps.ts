export const generationSteps=[
  {title:'因果 Decoder',en:'Causal attention → output',formula:'A = softmax(QKᵀ / √dₖ + M)\nMᵢⱼ = 0 (j ≤ i), −∞ (j > i)',description:'先完成带因果遮罩的层内计算。每个位置只能看自己和左侧 token；输出表示已经包含可见的上下文。',tip:'最后一行能看到所有已有 token，因为它们都不是未来。右侧虚线框中的下一 token 还不存在。'},
  {title:'取最后一个表示',en:'Last hidden state',formula:'h_last = H[n − 1, :]\n[n, 8] → [1, 8]',description:'整层产生 n 个 hidden state。生成下一个 token 时，只把最后一个位置的向量交给 LM Head。',tip:'这里的 hidden state 来自双向 Encoder 的因果版本，模型参数固定、未训练。'},
  {title:'词表投影',en:'LM Head',formula:'logits = h_last W_vocab + b\n[1, 8] × [8, 24] → [1, 24]',description:'LM Head 为词表中的每个 token 计算一个分数。候选词来自固定的 24 项教学词表，不限于输入里已经出现的词。',tip:'数值由矩阵乘法计算，不是预写好的句子；未训练模型可能输出重复或不通顺的文本。'},
  {title:'查看原始分数',en:'Logits',formula:'zᵢ ∈ ℝ\n分数可为负，不要求总和为 1',description:'logits 是未归一化的偏好分数。绝对值不是置信度，比较分数的相对大小才有意义。',tip:'同一个词表位置在每轮都表示相同的 token。EOS 也是一个可被选中的候选。'},
  {title:'转为概率',en:'Temperature & Softmax',formula:'pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)\nΣᵢ pᵢ = 1',description:'温度控制概率分布的尖锐程度。较低温度更集中，较高温度更平坦；Greedy 直接取最大分数，忽略温度。',tip:'概率图显示筛选前的完整 Softmax 分布。后面的 top-k / top-p 会过滤并重新归一化。'},
  {title:'筛选候选',en:'Top-k / Nucleus',formula:'top-k：保留概率最大的 k 项\ntop-p：保留累计概率 ≥ p 的最短前缀\np′ᵢ = pᵢ / Σ保留项 pⱼ',description:'将候选按概率排序。Top-k 固定数量，Top-p 固定累计概率阈值；Temperature 模式保留全部候选。',tip:'榜单同时展示原概率与筛选后的采样概率。被移除的候选采样概率为 0。'},
  {title:'选出下一个 token',en:'Sample next token',formula:'Greedy: argmax(z)\nSampling: token ∼ Categorical(p′)',description:'Greedy 总是取第一名；其他模式按最终分布抽样，不保证选中概率最高的词。固定随机种子便于复现。',tip:'“上一步”会回到采样之前，再向前会恢复同一个结果，不会偷偷重抽。'},
  {title:'追加并进入下一轮',en:'Append → repeat',formula:'context ← context + [next_token]\nrepeat until EOS or token limit',description:'选中的 token 追加到上下文。下一轮把它作为最后一个位置，重新计算分布并继续生成。',tip:'缓存保留的是已经经过 Decoder 的 token 的 K/V。刚追加的 token 要到下一轮才会被投影并写入缓存。'},
];
