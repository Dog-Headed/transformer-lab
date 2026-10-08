"use client";
import {useMemo,useState} from 'react';
import {compareCapabilities,diagnoseExperiment,generateText,pretrainingCheckpoint,type Experiment} from '@/lib/journey/engine';
import {DIALOGUES,UNSEEN_DIALOGUES,GOALS} from '@/lib/journey/data';

export default function JourneyDiagnostics({experiment,onGo,onRestore,full=false}:{experiment:Experiment;onGo:(stage:number)=>void;onRestore:(id:number)=>void;full?:boolean}){
  const [probe,setProbe]=useState('completion');
  const result=useMemo(()=>{
    const checkpoint=pretrainingCheckpoint(experiment);
    const current=compareCapabilities(experiment);
    const before=checkpoint?compareCapabilities(experiment,checkpoint.model):undefined;
    return {checkpoint,current,before,advice:diagnoseExperiment(experiment,current,before)};
  },[experiment]);
  const {current,before,checkpoint,advice}=result;
  const goal=GOALS.find(g=>g.id===experiment.config.goal)!;
  const options=[{id:'completion',label:'目标主题续写',prefix:goal.prompt,expected:''},
    ...DIALOGUES.map((d,i)=>({id:`seen-${i}`,label:`示范题：${d.question}`,prefix:`问：${d.question}答：`,expected:d.answer})),
    ...UNSEEN_DIALOGUES.map((d,i)=>({id:`unseen-${i}`,label:`未训练题：${d.question}`,prefix:`问：${d.question}答：`,expected:d.answer}))];
  const selected=options.find(p=>p.id===probe)??options[0];
  const output=(model:Experiment['model'])=>{const text=generateText(experiment,selected.prefix,model).text;return selected.expected?text.slice(selected.prefix.length):text;};
  return <section className="journey-card journey-diagnostics" aria-label="模型问题诊断与修复"><h3>{full?'能力对比与修复建议':'遇到问题，下一步检查哪里？'}</h3>
    <p>根据实际评测给出线索。命中率指下一个 token；问答分数按完整回答完全匹配计数。提示阈值是教学参考。</p>
    {advice.slice(0,full?advice.length:2).map(a=><div className="journey-advice" key={a.id}><b>{a.title}</b><p>{a.detail}</p><button onClick={()=>onGo(a.stage)}>{a.action}</button></div>)}
    {!full&&<button onClick={()=>onGo(7)}>打开能力对比和检查点恢复</button>}
    {full&&<><div className="journey-comparison"><table><caption>预训练检查点与当前权重 · 同一组评测输入</caption><thead><tr><th>评测项目</th><th>预训练 Step {checkpoint?.id??0}</th><th>当前 Step {experiment.model.step}</th></tr></thead><tbody>
      <tr><th>留出句子续写<br/><small>loss / token 命中</small></th><td>{before?`${before.continuation.loss.toFixed(2)} / ${(before.continuation.accuracy*100).toFixed(1)}%`:'无检查点'}</td><td>{current.continuation.loss.toFixed(2)} / {(current.continuation.accuracy*100).toFixed(1)}%</td></tr>
      <tr><th>3 道 SFT 示范题<br/><small>完整回答匹配</small></th><td>{before?.seen.exact??'—'} / 3</td><td>{current.seen.exact} / 3</td></tr>
      <tr><th>3 道未训练改写题<br/><small>完整回答匹配</small></th><td>{before?.unseen.exact??'—'} / 3</td><td>{current.unseen.exact} / 3</td></tr>
    </tbody></table></div><p>改写题仅用于评测，不进入训练循环。反复用这些题调参后，它们就成了开发验证题；真实开发还要另留新的最终测试题。</p>
    <label className="journey-field">看同一个输入的实际输出<select value={selected.id} onChange={e=>setProbe(e.target.value)}>{options.map(p=><option key={p.id} value={p.id}>{p.label}</option>)}</select></label>
    {selected.expected&&<p>参考回答：{selected.expected}（评分用答案，不传给生成函数）</p>}
    <div className="journey-probe"><div><b>预训练检查点</b><output>{output(checkpoint?.model??experiment.initial)}</output></div><div><b>当前模型</b><output>{output(experiment.model)}</output></div></div>
    {experiment.phase==='sft'&&checkpoint&&<button onClick={()=>onRestore(checkpoint.id)}>恢复预训练检查点，重新尝试微调</button>}
    <p>恢复会先保存当前权重备份，再恢复所选权重、Adam 状态和训练步数。恢复或训练后，旧验收报告失效。</p></>}
  </section>;
}
