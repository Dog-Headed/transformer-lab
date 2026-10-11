"use client";
import {useEffect, useRef, useState} from 'react';
import {Activity, ArrowRight, Database, Download, FlaskConical, Layers3, Play, Square} from 'lucide-react';
import {createTrainingApi, DEFAULT_CONFIG, isActive} from '@/lib/training/api';
import type {AttentionData, Checkpoint, Generation, Hardware, Metric, Task, TrainConfig} from '@/lib/training/api';

import NextTokenExplorer from './next-token-explorer';

const statuses: Record<Task['status'], string> = {queued:'准备中', running:'训练中', stopping:'正在停止',
  stopped:'已停止', completed:'训练完成', loaded:'已恢复存档', failed:'训练失败'};
const visibleToken = (token: string) => token === '\n' ? '↵' : token === ' ' ? '␣' : token;

function LossChart({points}: {points: Metric[]}) {
  const maximum = Math.max(1, ...points.flatMap(p => [p.train_loss, p.validation_loss])) * 1.1;
  const lastStep = Math.max(1, points.at(-1)?.step ?? 1);
  const x = (step: number) => 48 + step / lastStep * 552;
  const y = (loss: number) => 210 - loss / maximum * 180;
  const path = (key: 'train_loss' | 'validation_loss') => points.map((p,i) => `${i?'L':'M'}${x(p.step)},${y(p[key])}`).join(' ');
  return <div className="pt-chart"><div className="pt-legend"><span><i/>Train loss</span><span><i/>Validation loss</span></div>
    {points.length ? <svg viewBox="0 0 640 250" role="img" aria-label="实际训练集与验证集交叉熵曲线">
      {[0,.25,.5,.75,1].map(f => <g key={f}><line x1="48" x2="600" y1={y(maximum*f)} y2={y(maximum*f)} stroke="#e5eaf3"/>
        <text x="38" y={y(maximum*f)+4} textAnchor="end">{(maximum*f).toFixed(1)}</text></g>)}
      <path d={path('train_loss')} fill="none" stroke="#596be7" strokeWidth="3"/>
      <path d={path('validation_loss')} fill="none" stroke="#139b89" strokeWidth="3"/>
      {points.length === 1 && <>{(['train_loss','validation_loss'] as const).map(key => <circle key={key} cx="48" cy={y(points[0][key])} r="4" fill={key==='train_loss'?'#596be7':'#139b89'}/>)}</>}
      <text x="48" y="238">0</text><text x="600" y="238" textAnchor="end">{lastStep} step</text>
    </svg> : <div className="pt-empty"><Activity size={30}/><p>开始训练后显示实测曲线</p><small>每个评估点均来自当前模型的真实预测</small></div>}
    <p className="pt-note">Train loss：固定的前 8 批训练窗口；Validation loss：完整留出窗口。两者均在 eval 模式计算。</p>
  </div>;
}

function AttentionMap({data, label}: {data: AttentionData | null; label: string}) {
  const [query, setQuery] = useState(0);
  const selected = Math.min(query, (data?.tokens.length ?? 1)-1);
  return <section className="pt-card pt-attention"><div className="pt-card-heading"><h2>看见已学习的注意力</h2><span>{data ? `Layer ${data.layer+1} · Head ${data.head+1}` : 'CAUSAL ATTENTION'}</span></div>
    {data ? <><p className="pt-note">{label} · 最多展示 32 个词元。点击行标签，观察这个词元关注了谁。</p>
      <div className="pt-matrix-scroll"><table className="pt-matrix"><caption className="sr-only">当前模型的因果注意力权重，列是 key，行是 query</caption><thead><tr><th scope="col">Q / K</th>{data.tokens.map((t,i) => <th scope="col" key={i}>{visibleToken(t)}</th>)}</tr></thead>
        <tbody>{data.weights.map((row,i) => <tr key={i} className={selected===i?'selected':''}><th scope="row"><button aria-pressed={selected===i} aria-label={`观察第 ${i+1} 个词元 ${visibleToken(data.tokens[i])}`} onClick={()=>setQuery(i)}>{visibleToken(data.tokens[i])}</button></th>
          {row.map((weight,j) => <td key={j} title={j>i?'未来字符：被因果遮罩屏蔽':`Q${i+1} → K${j+1}：${weight.toFixed(5)}`} style={{background:j>i?'#edf0f5':`rgba(89,107,231,${.06+weight*.94})`}}><span className="sr-only">{weight.toFixed(5)}</span></td>)}</tr>)}</tbody></table></div>
      <div className="pt-attention-row">{data.weights[selected].map((weight,i) => <span key={i}><b>{visibleToken(data.tokens[i])}</b><i style={{height:`${Math.max(2,weight*70)}px`}}/><small>{(weight*100).toFixed(1)}%</small></span>)}</div>
    </> : <div className="pt-empty"><Layers3 size={30}/><p>等待模型第一次评估</p><small>权重来自 PyTorch forward 的 QKᵀ / √d 与因果 Softmax</small></div>}
  </section>;
}

export default function PytorchTrainingLab() {
  const [address, setAddress] = useState('http://127.0.0.1:8000');
  const [base, setBase] = useState('http://127.0.0.1:8000');
  const [connected, setConnected] = useState(false);
  const [hardware, setHardware] = useState<Hardware | null>(null);
  const [config, setConfig] = useState<TrainConfig>({...DEFAULT_CONFIG});
  const [task, setTask] = useState<Task | null>(null);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [checkpoint, setCheckpoint] = useState('');
  const [custom, setCustom] = useState(false);
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [prompt, setPrompt] = useState('First Citizen:');
  const [temperature, setTemperature] = useState(.8);
  const [topK, setTopK] = useState(20);
  const [stepAttention, setStepAttention] = useState<AttentionData | null>(null);
  const [newTokens, setNewTokens] = useState(80);
  const [oneSentence,setOneSentence] = useState(false);
  const [questionMode,setQuestionMode] = useState(false);
  const inferencePrompt = questionMode?`User: ${prompt}\nAssistant:`:prompt;
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [generation, setGeneration] = useState<Generation | null>(null);
  const [resumeSteps, setResumeSteps] = useState(400);
  const active = isActive(task);
  const usable = !!task && ['completed','stopped','loaded'].includes(task.status);
  const metric = task?.metrics.at(-1);
  const pollingVersion = useRef(0);
  const serviceVersion = useRef(0);

  function selectTask(next: Task) {
    setQuestionMode(false);setTask(next); setConfig(next.config); setGeneration(null); setStepAttention(null); setLayer(0); setHead(0);
    setResumeSteps(Math.min(2000, Math.max(400,next.step+200)));
    if(next.config.dataset.startsWith('stories')) {setPrompt('Once upon a time');setTemperature(.7);setTopK(10);setOneSentence(true);}
    try {localStorage.setItem(`transformer-task:${base}`,next.id);} catch {}
  }
  async function refreshCheckpoints() {
    setCheckpoints(await createTrainingApi(base).checkpoints());
  }
  async function connect() {
    const api = createTrainingApi(address);
    if (address !== base) {setBase(address); return;}
    setHardware(await api.health());
    const [saved, tasks] = await Promise.all([api.checkpoints(), api.tasks()]);
    setConnected(true); setCheckpoints(saved);
    const recovered = tasks.find(t=>isActive(t)) ?? tasks.find(t=>t.id===task?.id);
    if (recovered) selectTask(recovered);
    else {setTask(null);setGeneration(null);}
    setNotice('训练服务已连接');
  }
  async function action(work: () => Promise<void>) {
    setBusy(true); setError(''); setNotice('');
    try {await work();} catch(e) {setError(e instanceof Error ? e.message : '操作失败，请重试');}
    finally {setBusy(false);}
  }
  // Recover the running task after navigation/refresh; checkpoint list survives service restarts.
  useEffect(() => {
    const version = ++serviceVersion.current;
    setConnected(false); setHardware(null); setTask(null); setGeneration(null); setCheckpoints([]); setCheckpoint('');
    void (async () => {
      try {
        const api = createTrainingApi(base);
        const hardware = await api.health();
        const [saved, tasks] = await Promise.all([api.checkpoints(), api.tasks()]);
        if (version !== serviceVersion.current) return;
        setConnected(true); setHardware(hardware); setError(''); setCheckpoints(saved);
        let previous: string | null = null;
        try {previous = localStorage.getItem(`transformer-task:${base}`);} catch {}
        const recovered = tasks.find(t=>isActive(t)) ?? tasks.find(t=>t.id===previous);
        if (recovered) selectTask(recovered);
      } catch(e) {
        if (version === serviceVersion.current) setError(e instanceof Error ? e.message : '无法连接服务');
      }
    })();
    return () => {serviceVersion.current++;};
    // selectTask only writes state and the base-specific local storage key.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  useEffect(() => {
    if (!task || !active) return;
    const version = ++pollingVersion.current;
    let timer: ReturnType<typeof setTimeout>;
    const api = createTrainingApi(base);
    async function poll() {
      try {
        const next = await api.task(task!.id);
        if (version !== pollingVersion.current) return;
        setTask(next); setError('');
        if (!isActive(next)) {await refreshCheckpoints(); return;}
      } catch(e) {
        if (version !== pollingVersion.current) return;
        setError(e instanceof Error ? e.message : '获取状态失败');
      }
      if (version === pollingVersion.current) timer = setTimeout(poll,1000);
    }
    timer = setTimeout(poll,500);
    return () => {clearTimeout(timer); pollingVersion.current++;};
    // The stable task id owns this polling lifecycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task?.id, active, base]);

  function field<K extends keyof TrainConfig>(key: K, value: TrainConfig[K]) {setConfig(c=>({...c,[key]:value}));}
  function storyPreset(identity=false) {
    setConfig({...DEFAULT_CONFIG,dataset:identity?'stories_identity':'stories',layers:4,hidden_size:128,sequence_length:64,
      learning_rate:.001,max_steps:2000,eval_interval:100});
    setCustom(false);setQuestionMode(false);setPrompt('Once upon a time');setTemperature(.7);setTopK(10);setOneSentence(true);
    setNotice('已应用英文故事配置，点击“开始真实训练”。GPU 推荐；CPU 达到时间预算后可加载存档继续。');
  }
  function chooseGenerationMode(question:boolean, example?:string) {
    setQuestionMode(question);setPrompt(example??(question?'你是谁？':'Once upon a time'));
    setTemperature(question ? .5 : .7);setTopK(question?1:10);setOneSentence(true);setGeneration(null);setStepAttention(null);
  }
  const values = [
    {key:'layers', label:'Transformer 层数', min:1,max:4,step:1},
    {key:'heads', label:'注意力头数', min:1,max:8,step:1},
    {key:'hidden_size',label:'Hidden size',min:16,max:128,step:8},
    {key:'sequence_length',label:'上下文长度',min:8,max:128,step:8},
    {key:'learning_rate',label:'Learning rate',min:.00001,max:.02,step:.00001},
    {key:'batch_size',label:'Batch size',min:1,max:32,step:1},
    {key:'max_steps',label:'最大训练 steps',min:1,max:2000,step:1},
    {key:'eval_interval',label:'每隔多少步评估',min:1,max:100,step:1},
    {key:'seed',label:'随机种子',min:0,max:2147483647,step:1},
  ] as const;

  return <div className="pt-lab"><header className="topbar"><a className="brand" href="/"><span className="brand-icon"><Layers3 size={23}/></span>Transformer <span className="brand-light">Training Lab</span></a>
    <nav className="mode-nav" aria-label="学习模式"><a href="/">Transformer 层内流程</a><a href="/generation">LLM 生成模式</a><a href="/training" aria-current="page">PyTorch 训练</a><a href="/build-model">从零造模</a></nav></header>
    <main className="pt-main"><div className="pt-heading"><div><div className="eyebrow"><span/> PYTORCH · TRAIN, EVALUATE, GENERATE</div><h1>亲手训练一个 Transformer。</h1><p>从真实文本到梯度更新，看见模型如何学习，再用自己的权重续写。</p></div><a className="pt-guide-link" href="/training/guide">回顾训练原理 <ArrowRight size={16}/></a></div>
      <form className="pt-connection" onSubmit={e=>{e.preventDefault(); void action(connect);}}>
        <span className={connected?'pt-online':'pt-offline'} title={hardware?.gpu_name??undefined}><i/>{connected?(hardware?.cuda_available?'CUDA GPU 已就绪':'CPU 训练服务已连接'):'连接训练服务'}</span>
        <label htmlFor="pt-address" className="sr-only">FastAPI 服务地址</label><input id="pt-address" type="url" value={address} onChange={e=>setAddress(e.target.value)} disabled={active||busy}/><button disabled={active||busy}>连接 / 刷新</button>
      </form>
      {error && <div className="pt-alert" role="alert">{error}</div>}{notice && <div className="pt-notice" role="status">{notice}</div>}
      {!connected && <div className="pt-start-help"><b>先启动本地训练服务</b><code>python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000</code><span>在项目根目录运行，然后点击“连接 / 刷新”。原有教学页面可直接使用。</span></div>}
      <div className="pt-layout"><aside><form className="pt-card pt-config" onSubmit={e=>{e.preventDefault();void action(async()=>{if(custom && text.length<256)throw new Error('自定义语料至少需要 256 个字符');const next=await createTrainingApi(base).create(config,custom?text:undefined);selectTask(next);});}}>
        <div className="pt-card-heading"><h2><FlaskConical size={18}/> 配置实验</h2><span>TINY DECODER</span></div><fieldset disabled={active||busy}><legend className="sr-only">模型和训练超参数</legend><button type="button" className="pt-story-preset" onClick={()=>storyPreset(false)}>应用英文句子实验配置</button><button type="button" className="pt-story-preset" onClick={()=>storyPreset(true)}>应用续写与自我介绍配置</button><p className="pt-note">TinyStories 简单英文故事 · 4 层 / 128 维 / 2,000 步；全部从随机权重训练。适合观察基本句子生成，推荐 CUDA。</p><label className="pt-device-label">训练设备<select value={config.device} onChange={e=>field('device',e.target.value as TrainConfig['device'])}><option value="auto">自动选择（优先 CUDA GPU）</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></label><label className="pt-device-label">预测单位<select value={config.tokenizer} onChange={e=>field('tokenizer',e.target.value as TrainConfig['tokenizer'])}><option value="word">词级：一次预测一个词语</option><option value="character">字符级：一次预测一个字符</option></select></label><div className="pt-fields">{values.map(f=><label key={f.key}>{f.label}<input type="number" required min={f.min} max={f.max} step={f.step} value={config[f.key]} onChange={e=>field(f.key,Number(e.target.value))}/></label>)}
          <label>Epoch 上限（可选）<input type="number" min="1" max="20" placeholder="仅使用 steps" value={config.epochs??''} onChange={e=>field('epochs',e.target.value?Number(e.target.value):null)}/></label>
          <label>验证集比例<select value={config.validation_ratio} onChange={e=>field('validation_ratio',Number(e.target.value))}><option value="0.1">10%</option><option value="0.2">20%</option><option value="0.3">30%</option></select></label>
        </div><label className="pt-corpus-label">训练语料<select value={custom?'custom':config.dataset} onChange={e=>{setCustom(e.target.value==='custom');if(e.target.value!=='custom')field('dataset',e.target.value as TrainConfig['dataset']);}}><option value="shakespeare">Tiny Shakespeare · 50,000 字符</option><option value="stories">TinyStories · 1,901 篇英文短故事</option><option value="stories_identity">TinyStories + 自我介绍 / 创造者问答</option><option value="custom">自定义文本</option></select></label>
        {custom && <label className="pt-corpus-label">文本语料<textarea value={text} onChange={e=>setText(e.target.value)} minLength={256} maxLength={2000000} required rows={7} placeholder="粘贴真实文本，256–2000000 个字符"/><small>{text.length.toLocaleString()} / 2,000,000 字符</small></label>}
        <p className="pt-note"><Database size={14}/> 按字符顺序划分训练 / 验证集；词表只从训练集建立。达到 steps 或 epoch 上限即停止。</p></fieldset>
        <button className="primary pt-start" type="submit" disabled={!connected||active||busy}><Play size={16}/> 开始真实训练</button><p className="pt-budget">单任务 · CPU / CUDA · 最多 300 万参数<br/>GPU 分配器限额 ≤ 2 GB · 每次运行最多 180 秒</p>
      </form><section className="pt-card pt-checkpoints"><div className="pt-card-heading"><h2>模型存档</h2><span>CHECKPOINT</span></div><p className="pt-note">完成或停止后自动保存权重、Adam 状态、词表、语料与训练进度。服务重启后可恢复。</p>
        <label htmlFor="pt-checkpoint">选择存档</label><select id="pt-checkpoint" value={checkpoint} onChange={e=>setCheckpoint(e.target.value)} disabled={active||busy}><option value="">{checkpoints.length?'请选择 checkpoint':'还没有存档'}</option>{checkpoints.map(cp=><option value={cp.id} key={cp.id}>Step {cp.step} · {cp.id.slice(0,8)} · {cp.config.layers}L/{cp.config.hidden_size}D</option>)}</select>
        <div className="pt-actions"><button disabled={!checkpoint||active||busy} onClick={()=>void action(async()=>{const next=await createTrainingApi(base).load(checkpoint,config.device);selectTask(next);setNotice(`已加载 Step ${next.step}，可以续写或继续训练`);})}>加载存档</button><button disabled={!connected||active||busy} onClick={()=>void action(refreshCheckpoints)}>刷新</button></div>
        {task?.checkpoint_id && <a className="pt-download" href={createTrainingApi(base).download(task.checkpoint_id)}><Download size={15}/> 下载当前 checkpoint (.pt)</a>}
      </section></aside>
      <div className="pt-results"><section className="pt-card"><div className="pt-card-heading"><h2>训练进行时</h2><span className={active?'pt-running':''} role="status">{task?statuses[task.status]:'等待实验'}</span></div>
        <div className="pt-progress"><div><b>Step {task?.step??0}</b><span>/ {task?.target_steps??config.max_steps} · Epoch {metric?.epoch.toFixed(2)??'—'}</span></div><progress max={task?.target_steps??config.max_steps} value={task?.step??0} aria-label="真实训练步数"/></div>
        <div className="pt-metrics">{[['Train loss',metric?.train_loss.toFixed(4)],['Validation loss',metric?.validation_loss.toFixed(4)],['Perplexity',metric?.perplexity.toFixed(2)],[task?.config.tokenizer==='word'?'词元准确率':'字符准确率',metric?`${(metric.accuracy*100).toFixed(1)}%`:undefined],['Learning rate',metric?.learning_rate.toFixed(5)],['训练耗时',task?`${task.elapsed_seconds.toFixed(1)} s`:undefined]].map(([label,value])=><div key={label}><span>{label}</span><b>{value??'—'}</b></div>)}</div>
        <LossChart points={task?.metrics??[]}/>
        {task && <p className="pt-note">运行设备：{task.device==='cuda'?`CUDA GPU · 峰值张量显存 ${task.peak_gpu_memory_mb.toFixed(1)} MB`:'CPU · 2 线程'}<br/>{task.parameter_count.toLocaleString()} 参数 · {task.data.train_characters.toLocaleString()} 训练字符 / {task.data.validation_characters.toLocaleString()} 验证字符 · 词表 {task.data.vocabulary_size} · 验证未知词元 {task.data.validation_unknown_tokens}<br/>任务 {task.id.slice(0,12)} · 约每秒刷新状态；模型评估间隔 {task.config.eval_interval} steps</p>}
        {(task?.error||task?.stop_reason) && <p className={task.error?'pt-alert':'pt-note'} role={task.error?'alert':undefined}>{task.error||task.stop_reason}</p>}
        <div className="pt-actions"><button disabled={!active||busy||task?.status==='stopping'} onClick={()=>void action(async()=>{setTask(await createTrainingApi(base).stop(task!.id));})}><Square size={14}/>停止训练</button>
          <button disabled={!usable||busy} onClick={()=>void action(async()=>{const saved=await createTrainingApi(base).save(task!.id);setTask(t=>t?{...t,checkpoint_id:saved.id}:t);await refreshCheckpoints();setNotice(`已保存 Step ${saved.step} 的 checkpoint`);})}>保存 checkpoint</button></div>
        <form className="pt-resume" onSubmit={e=>{e.preventDefault();void action(async()=>{selectTask(await createTrainingApi(base).resume(task!.id,resumeSteps));});}}><label htmlFor="pt-resume">继续训练到累计 steps</label><input id="pt-resume" type="number" required min={(task?.step??0)+1} max="2000" value={resumeSteps} onChange={e=>setResumeSteps(Number(e.target.value))} disabled={!usable||busy||task!.step>=2000}/><button disabled={!usable||busy||(task?.step??0)>=2000}>继续训练</button></form>
      </section>
      <section className="pt-card pt-generate"><div className="pt-card-heading"><h2>让你的模型继续写</h2><span>TRAINED WEIGHTS</span></div><p className="pt-note">推理使用上方任务的训练权重。词级模型生成完整词语与标点，字符级模型生成字符。小语料模型可能重复；输出质量需要独立评测。</p>
        <div className="pt-actions"><button disabled={!usable||busy} onClick={()=>chooseGenerationMode(false)}>英文续写</button><button disabled={!usable||busy||task?.config.dataset!=='stories_identity'} onClick={()=>chooseGenerationMode(true)}>问问模型：你是谁？</button><button disabled={!usable||busy||task?.config.dataset!=='stories_identity'} onClick={()=>chooseGenerationMode(true,'你的创造者是谁？')}>问问模型：创造者是谁？</button></div><p className="pt-note">自我介绍需要使用“续写与自我介绍”语料重新训练。少量中英文问答与故事混合学习，回答由模型实际生成；适合问身份、创造者和能力，未训练的问题可能答错。</p>
        <form onSubmit={e=>{e.preventDefault();void action(async()=>{setStepAttention(null);setGeneration(await createTrainingApi(base).generate(task!.id,{prompt:inferencePrompt,temperature,max_new_tokens:newTokens,top_k:topK,seed:42,layer,head,stop_after_sentence:oneSentence||questionMode}));});}}>
          <label htmlFor="pt-prompt">{questionMode?"输入简单问题":"输入提示文本"}</label><textarea id="pt-prompt" value={prompt} maxLength={questionMode?256:512} required rows={3} onChange={e=>setPrompt(e.target.value)} disabled={!usable||busy}/>
          <div className="pt-generation-settings"><label>生成长度<select value={oneSentence?"sentence":"tokens"} onChange={e=>setOneSentence(e.target.value==="sentence")}><option value="sentence">一句话 · 句末停止</option><option value="tokens">按词元上限</option></select></label><label>Temperature<input type="number" min="0.1" max="2" step="0.1" value={temperature} required onChange={e=>setTemperature(Number(e.target.value))}/></label><label>新生成词元<input type="number" min="1" max="128" value={newTokens} required onChange={e=>setNewTokens(Number(e.target.value))}/></label>
            <label>Top-k（0 为全部）<input type="number" min="0" max="50" value={topK} required onChange={e=>setTopK(Number(e.target.value))}/></label><label>Attention layer<select value={layer} onChange={e=>setLayer(Number(e.target.value))}>{Array.from({length:task?.config.layers??1},(_,i)=><option key={i} value={i}>Layer {i+1}</option>)}</select></label><label>Attention head<select value={head} onChange={e=>setHead(Number(e.target.value))}>{Array.from({length:task?.config.heads??1},(_,i)=><option key={i} value={i}>Head {i+1}</option>)}</select></label>
          </div><button className="primary" disabled={!usable||busy}>{questionMode?"让模型回答":"用当前模型生成"} <ArrowRight size={16}/></button>
        </form>{generation && <div className="pt-output"><span>MODEL OUTPUT · 固定 seed 42 / Top-k {topK}</span><pre><span>{questionMode?"":generation.prompt}</span>{generation.generated_text}</pre></div>}
      <NextTokenExplorer task={task} base={base} prompt={inferencePrompt} displayPrompt={questionMode?`你：${prompt}\n模型：`:undefined} temperature={temperature} topK={topK} layer={layer} head={head} disabled={!usable||busy} action={action} onAttention={setStepAttention}/>
      </section><AttentionMap data={stepAttention??generation?.attention??task?.attention??null} label={stepAttention?'逐次预测当前上下文的 attention（前 32 个词元）':generation?'输入提示的 attention（生成前）':'验证集第一个窗口的 attention'}/>
      </div></div><footer className="pt-footer"><p>PyTorch 训练与教学可视化共存。原课程的浏览器权重与这里的 checkpoint 各自独立。</p><div><a href="/training/guide">训练原理 / SFT / RLHF</a><a href="/training/grokking">泛化专题</a><a href="/build-model">从零造模课程</a></div></footer>
    </main></div>;
}
