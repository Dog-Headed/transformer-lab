"use client";
import {useEffect, useRef, useState} from 'react';
import {ArrowRight, Pause, Play, RotateCcw} from 'lucide-react';
import {createTrainingApi} from '@/lib/training/api';
import type {AttentionData, Prediction, Task} from '@/lib/training/api';

const visibleToken = (token: string) => token === '\n' ? '↵ 换行' : token === ' ' ? '␣ 空格' : token;
const percent = (value: number) => `${(value*100).toFixed(value < .001 ? 4 : 2)}%`;
// Display spacing only; the model history uses the exact token list from the server.
function append(text: string, token: string, word: boolean) {
  return text + (word && /^[A-Za-z0-9]/.test(token) && text && !/[\s(['"“]$/.test(text) ? ' ' : '') + token;
}

export default function NextTokenExplorer({task, base, prompt, displayPrompt, temperature, topK, layer, head, disabled, action, onAttention}: {
  task: Task | null; base: string; prompt: string; displayPrompt?:string; temperature: number; topK: number; layer: number; head: number;
  disabled: boolean; action: (work:()=>Promise<void>)=>Promise<void>; onAttention: (data:AttentionData|null)=>void;
}) {
  const [prediction,setPrediction] = useState<Prediction|null>(null);
  const [output,setOutput] = useState('');
  const [steps,setSteps] = useState(0);
  const [last,setLast] = useState<{token:string;probability:number}|null>(null);
  const [all,setAll] = useState(false);
  const [search,setSearch] = useState('');
  const [auto,setAuto] = useState(false);
  const sequenceVersion = useRef(0);
  const [speed,setSpeed] = useState('standard');
  const [seconds,setSeconds] = useState(4);
  const intervals: Record<string,[number,number]> = {fast:[.5,1],standard:[3,5],slow:[6,10],custom:[seconds,seconds]};
  const interval = intervals[speed];
  const intervalLabel = interval[0]===interval[1]?`${interval[0]} 秒`:`${interval[0]}–${interval[1]} 秒`;
  const word = task?.config.tokenizer !== 'character';
  const unit = word ? '词' : '字符';
  useEffect(()=>{
    sequenceVersion.current++;
    setAuto(false);setPrediction(null);setOutput('');setSteps(0);setLast(null);onAttention(null);
    // A changed model, prompt or sampling control starts a fresh explicit sequence.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[task?.id,task?.step,base,prompt,temperature,topK,layer,head]);
  const options = {prompt,temperature,top_k:topK,seed:42+steps,layer,head};
  async function reset() {
    setAuto(false);
    const version = sequenceVersion.current;
    const next = await createTrainingApi(base).predict(task!.id,options);
    if(version!==sequenceVersion.current) return;
    setPrediction(next);setOutput(displayPrompt??prompt);setSteps(0);setLast(null);onAttention(next.attention);
  }
  async function advance() {
    const version = sequenceVersion.current;
    const next = await createTrainingApi(base).step(task!.id,{...options,context_tokens:prediction!.context_tokens});
    if(version!==sequenceVersion.current) return;
    setOutput(text=>append(text,next.next_token,word).slice(-16000));setSteps(count=>count+1);
    setLast({token:next.next_token,probability:next.probability});
    setPrediction(next.prediction);onAttention(next.prediction.attention);
  }
  useEffect(()=>{
    if (!auto || disabled || !prediction) return;
    // Schedule after the prior response; the shared busy state prevents overlapping calls.
    const timer = setTimeout(()=>void action(async()=>{
      try { await advance(); }
      catch (error) { setAuto(false); throw error; }
    }),(interval[0]+Math.random()*(interval[1]-interval[0]))*1000);
    return ()=>clearTimeout(timer);
    // Prediction owns the exact context; a changed response starts the next interval.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  },[auto,disabled,prediction,speed,seconds]);
  const filtered = prediction?.candidates.filter(c=>c.token.toLocaleLowerCase().includes(search.toLocaleLowerCase())) ?? [];
  const shown = all || search ? filtered : filtered.slice(0,15);
  return <div className="pt-token-explorer"><div className="pt-card-heading"><h3>一次，生成一个{unit}。</h3><span>LIVE NEXT TOKEN</span></div>
    <p className="pt-note">使用上方提示和采样设置。{word?'词级模型一次预测一个词语或标点；英文词保持完整，中文使用结巴分词。':'当前存档是字符模型。想生成完整词语，请在左侧选择词级模式并重新训练。'}</p>
    <div className="pt-actions"><button disabled={disabled} onClick={()=>void action(reset)}><RotateCcw size={14}/>{prediction?'从提示重新开始':`查看下一${unit}候选`}</button>
      <button className="primary" disabled={disabled||!prediction||auto} onClick={()=>void action(advance)}>下一{unit} <ArrowRight size={16}/></button>
      <button aria-pressed={auto} disabled={!prediction||(!auto&&disabled)} onClick={()=>setAuto(value=>!value)}>{auto?<Pause size={14}/>:<Play size={14}/>} {auto?'暂停自动生成':`开启自动下一${unit}`}</button><span className="pt-step-count">已生成 {steps} 个词元{auto?` · 自动生成中，间隔 ${intervalLabel}`:''}</span></div>
    <div className="pt-auto-speed"><label>自动生成速度<select aria-label="自动生成速度" value={speed} onChange={e=>setSpeed(e.target.value)}><option value="fast">快 · 0.5–1 秒</option><option value="standard">标准 · 3–5 秒</option><option value="slow">慢 · 6–10 秒</option><option value="custom">自定义</option></select></label>{speed==='custom' && <label>间隔秒数<input aria-label="间隔秒数" type="number" min="0.3" max="30" step="0.1" value={seconds} onChange={e=>setSeconds(Math.min(30,Math.max(.3,Number(e.target.value)||.3)))}/></label>}<span className="pt-note">可随时调速；每次预测完成后再等待 {intervalLabel}。</span></div>
    {prediction && <><div className="pt-output pt-step-output"><pre>{output}</pre>{last && <p>刚生成：<b>{visibleToken(last.token)}</b> · 抽样时概率 {percent(last.probability)}</p>}</div>
      <div className="pt-candidate-heading"><h4>接下来可能是哪些{unit}？</h4><label>查找候选<input value={search} onChange={e=>setSearch(e.target.value)} placeholder="搜索整个模型词表"/></label></div>
      <p className="pt-note">采样概率经过 Temperature {temperature} 与 {topK?`Top-k ${topK}`:'完整词表'} 处理，所有候选采样概率之和为 100%。原始概率为温度 1、无 Top-k 的分布。两者均排除 &lt;unk&gt;；Top-k 外采样概率为 0。</p>
      <div className="pt-candidates"><table><thead><tr><th scope="col">候选词元</th><th scope="col">采样概率</th><th scope="col">原始概率</th></tr></thead><tbody>{shown.map(c=><tr key={c.token_id}><th scope="row">{visibleToken(c.token)}</th><td><div className="pt-probability"><i style={{width:`${c.probability*100}%`}}/><span>{percent(c.probability)}</span></div></td><td>{percent(c.model_probability)}</td></tr>)}</tbody></table>{!shown.length && <p className="pt-note">词表里没有匹配项。</p>}</div>
      {!search && <button className="pt-expand" onClick={()=>setAll(value=>!value)}>{all?'只看前 15 项':`展开全部 ${prediction.candidates.length} 项候选`}</button>}
      <p className="pt-note">每次点击后，模型用新的完整词元序列重新 forward，候选概率随上下文更新。上下文保留最近 {task?.config.sequence_length} 个词元；页面保留最近 16,000 字符。候选仅限本次训练词表。</p>
    </>}
  </div>;
}
