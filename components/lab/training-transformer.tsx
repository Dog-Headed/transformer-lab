"use client";
import {useMemo,useState} from 'react';
import {GROUPS,PARAMETER_COUNT,WIDTH,CONTEXT,VOCAB,inspect,generate,type Learner} from '@/lib/training/engine';

export function TransformerTrace({model,prefix,previous}:{model:Learner;prefix:string;previous?:Learner}){
  const [stage,setStage]=useState('attention');
  const [cell,setCell]=useState<{r:number;c:number}|null>(null);
  const result=useMemo(()=>inspect(model,prefix),[model,prefix]);
  const earlier=useMemo(()=>previous?inspect(previous,prefix):null,[previous,prefix]);
  const tokens=Array.from(prefix);
  const row=Math.min(cell?.r??tokens.length-1,tokens.length-1),col=Math.min(cell?.c??0,row);
  const stages=[['embedding','Embedding'],['q','Q / K / V'],['attention','因果注意力'],['ffn','前馈网络'],['hidden','最终表示'],['logits','词表投影']];
  const matrix=stage==='logits'?[result.logits]:stage==='q'?result.q:stage==='embedding'?result.embedding:stage==='ffn'?result.ffn:stage==='hidden'?result.hidden:result.attention;
  const renderMatrix=(values:number[][],label:string,attention=false)=><div className="tiny-matrix-scroll"><table className="tiny-matrix"><caption>{label}</caption><thead><tr><th>位置</th>{values[0].map((_,c)=><th key={c}>{attention?tokens[c]:stage==='logits'?VOCAB[c]:`d${c+1}`}</th>)}</tr></thead><tbody>{values.map((values,r)=><tr key={r}><th>{stage==='logits'?'最后位置':`${r+1} · ${tokens[r]}`}</th>{values.map((v,c)=><td key={c} className={attention&&c>r?'masked':''} style={attention&&c<=r?{background:`rgba(89,107,231,${.04+v*.65})`}:undefined} title={v.toFixed(6)}>{attention?(c>r?'屏蔽':<button className={row===r&&col===c?'active':''} aria-pressed={row===r&&col===c} aria-label={`位置 ${r+1} ${tokens[r]} 读取位置 ${c+1} ${tokens[c]}，注意力 ${(v*100).toFixed(2)}%`} onClick={()=>setCell({r,c})}>{(v*100).toFixed(0)}%</button>):v.toFixed(2)}</td>)}</tr>)}</tbody></table></div>;
  return <div className="tiny-trace"><div className="tiny-spec"><b>同一套可训练权重 · Step {model.step}</b><span>1 层 · 1 个头 · {WIDTH} 维 · {PARAMETER_COUNT.toLocaleString()} 个参数 · 上下文 {CONTEXT}</span></div><nav className="tiny-flow" aria-label="Transformer 前向计算">{stages.map(([id,label])=><button key={id} className={stage===id?'selected':''} aria-pressed={stage===id} onClick={()=>setStage(id)}>{label}</button>)}</nav>
    {stage==='q'?<>{renderMatrix(result.q,'Query · 当前字符在寻找什么')}{renderMatrix(result.k,'Key · 每个字符提供的索引')}{renderMatrix(result.v,'Value · 被汇总的信息')}</>:renderMatrix(matrix,stage==='attention'?'每一行是一个查询位置，每一列是它能读取的字符':stage==='embedding'?'字符 Embedding + 可训练位置 Embedding':stage==='ffn'?'ReLU 前馈网络的实际输出':stage==='hidden'?'残差连接和 LayerNorm 后的最终表示':'最后位置的表示投影到整个词表',stage==='attention')}
    {stage==='attention'&&<output className="tiny-attention-readout">位置 {row+1}「{tokens[row]}」读取位置 {col+1}「{tokens[col]}」：<b>{(result.attention[row][col]*100).toFixed(2)}%</b>{earlier&&<span>更新前 {(earlier.attention[row][col]*100).toFixed(2)}% · 变化 {((result.attention[row][col]-earlier.attention[row][col])*100).toFixed(2)} 个百分点</span>}</output>}
    <p className="train-muted">{stage==='attention'?'未来位置的注意力严格为零。点一个格子查看精确数值；最后一行用于本次预测。注意力是信息汇总权重，不能单独代表一个字符对答案的全部贡献。':stage==='logits'?'这些 logits 经过 Softmax，就是本页的下一个 token 概率。':stage==='ffn'?'每个位置分别经过 8 → 16 → 8 维的前馈网络，再通过残差连接保留输入。':'这里显示的是实际中间结果。改换前缀或训练权重后，所有结果会重新计算。'}</p>
    {previous&&<p className="train-muted">已使用更新后的权重重新执行整条前向计算。点击注意力格子，可以比较更新前后的数值。</p>}
  </div>;
}

export function ParameterView({model,grad,previous}:{model:Learner;grad?:number[];previous?:Learner}){
  const [name,setName]=useState('q');const group=GROUPS.find(g=>g.name===name)!;
  return <div className="tiny-parameters"><label className="train-select-label">查看参数组<select value={name} onChange={e=>setName(e.target.value)}>{GROUPS.map(g=><option key={g.name} value={g.name}>{g.label} · {g.rows} × {g.cols}</option>)}</select></label>
    {grad&&<div className="tiny-gradient-groups">{GROUPS.map(g=>{const norm=Math.sqrt(grad.slice(g.start,g.start+g.size).reduce((s,v)=>s+v*v,0));const delta=previous?Math.sqrt(model.weights.slice(g.start,g.start+g.size).reduce((s,v,i)=>s+(v-previous.weights[g.start+i])**2,0)):null;return <button key={g.name} className={g.name===name?'selected':''} aria-pressed={g.name===name} onClick={()=>setName(g.name)}><b>{g.label}</b><span>梯度范数 {norm.toFixed(4)}</span>{delta!==null&&<small>实际更新幅度 {delta.toFixed(4)}</small>}</button>;})}</div>}
    <p className="train-muted">{group.label} · 前 {Math.min(18,group.size)} 个参数{grad?'；梯度为当前样本 loss 对这些权重的导数':''}。零梯度可能来自未使用的字符、位置或 ReLU，不表示没有参与计算。</p><div className="train-weights">{model.weights.slice(group.start,group.start+Math.min(18,group.size)).map((v,i)=><div key={i}><span>[{Math.floor(i/group.cols)},{i%group.cols}]</span><code>{v.toFixed(4)}</code>{grad&&<small>∇ {grad[group.start+i].toFixed(4)}</small>}{previous&&<small>原 {previous.weights[group.start+i].toFixed(4)}</small>}</div>)}</div>
  </div>;
}

export function TrainedGeneration({model,initial}:{model:Learner;initial:Learner}){
  const [prefix,setPrefix]=useState('小猫喜欢');
  const result=useMemo(()=>generate(model,prefix),[model,prefix]);
  const untrained=useMemo(()=>generate(initial,prefix),[initial,prefix]);
  return <section className="train-card"><div className="train-card-heading"><h2>用刚才训练的 Transformer 续写</h2><span className="train-badge actual">真实生成 · Step {model.step}</span></div><p className="train-copy">选择预设开头。模型用当前权重逐字预测，每次取概率最大的字符，再把它接回上下文；这里没有预设回复。</p><div className="train-choice-row">{['小猫喜欢','小狗喜欢','今天下雨，我们带','你好，'].map(p=><button key={p} className={prefix===p?'selected':''} aria-pressed={prefix===p} onClick={()=>setPrefix(p)}>{p}</button>)}</div><div className="sft-comparison"><div><span>初始化权重 · Step 0</span><p>{untrained.text}</p></div><div><span>正在查看的权重 · Step {model.step}</span><p>{result.text}</p></div></div><div className="tiny-generated-tokens">{result.steps.map((s,i)=><span key={i}>{s.token}<small>{(s.probability*100).toFixed(1)}%</small></span>)}</div><p className="train-muted">上方百分比是每步选中字的概率。遇到句号或生成 12 个字符就停止。生成只读取权重，不更新参数；小语料可能被记住，也可能混搭出错误句子。</p></section>;
}
