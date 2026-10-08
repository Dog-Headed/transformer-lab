import {initialize,forward,derivatives,adam,parameterGroups,type TinyModel} from '../training/transformer.ts';
import {rawRecords,cleanRecords,splitRecords,buildTokenizer,makeSamples,tokenize,assistantSamples,dialogueSamples,DIALOGUES,UNSEEN_DIALOGUES,GOALS,type PackId,type Cleaning,type Tokenizer,type TrainingSample} from './data.ts';
import {trainReward,preferenceLoss,QUESTIONS,reward,policyUpdate,softmax,REFERENCE,type Preference} from '../training/engine.ts';
export type Config={name:string;goal:PackId;packs:PackId[];cleaning:Cleaning;excluded:string[];moreTraining:boolean;tokenizer:Tokenizer['mode'];width:8|12|16;context:16|24;lr:number;budget:number};
// @lesson config
export const DEFAULT_CONFIG:Config={
  name:'星火 01', goal:'animals', packs:['animals','weather'],
  cleaning:{trim:true,dedupe:true,garbage:true}, excluded:[],
  moreTraining:false, tokenizer:'char', width:8, context:24,
  lr:.003, budget:1600,
};
// @end config
export type Metrics={loss:number;accuracy:number;count:number;unknown:number};
export type Point={step:number;train:number;validation:number;phase:'pretrain'|'sft'};
export type SFTSettings={replay:0|.25|.5;rate:.15|.35|.7};
export type Checkpoint={id:number;label:string;model:TinyModel;phase:'pretrain'|'sft';pretrainSteps?:number;sftSteps?:number;sftSettings?:SFTSettings};
export type Experiment={schema:1;config:Config;model:TinyModel;tokenizer:Tokenizer;initial:TinyModel;history:Point[];checkpoints:Checkpoint[];logs:string[];phase:'pretrain'|'sft';pretrainSteps:number;sftSteps:number;sftSettings?:SFTSettings;preferences:Preference[];rewardWeights:number[];policyLogits:number[][];preferenceUpdates:number;test:Metrics|null};
export function sftSettings(e:Experiment):SFTSettings{return e.sftSettings??{replay:0,rate:.35};}
export function dataset(config:Config){const raw=rawRecords(config.packs);const clean=cleanRecords(raw,config.cleaning,config.excluded);return {raw,clean,...splitRecords(clean,config.moreTraining)};}
export function samples(experiment:Experiment,split:'train'|'validation'|'test'){return makeSamples(dataset(experiment.config)[split],experiment.tokenizer,experiment.config.context);}
// @lesson initialize
export function createExperiment(config:Config):Experiment{
  if(!config.name.trim())throw Error('给你的模型起一个名字。');
  const data=dataset(config);
  if(!data.train.length||!data.validation.length||!data.test.length)throw Error('训练、验证和测试都至少要保留一句材料。');
  const tokenizer=buildTokenizer(data.train,config.tokenizer);
  const model=initialize(tokenizer.vocab.length,config.width,config.context);
  const initial=structuredClone(model);
  const experiment:Experiment={schema:1,config:structuredClone(config),model,tokenizer,initial,history:[],checkpoints:[{id:0,label:'随机初始化',model:initial,phase:'pretrain',pretrainSteps:0,sftSteps:0,sftSettings:{replay:0,rate:.35}}],logs:[`创建 ${config.name}；${data.train.length} 句训练材料；${model.weights.length} 个参数。`],phase:'pretrain',pretrainSteps:0,sftSteps:0,preferences:[],rewardWeights:[0,0,0,0],policyLogits:QUESTIONS.map(()=>REFERENCE.map(Math.log)),preferenceUpdates:0,test:null};
  return measure(experiment);
}
// @end initialize
// @lesson evaluate
export function evaluate(model:TinyModel,targets:TrainingSample[],vocab:number):Metrics{
  let loss=0,correct=0,unknown=0;
  for(const sample of targets){const p=forward(model,sample.ids,vocab).probabilities;loss-=Math.log(Math.max(1e-15,p[sample.target]));correct+=Number(p.indexOf(Math.max(...p))===sample.target);unknown+=Number(sample.target===0);}
  return {loss:targets.length?loss/targets.length:0,accuracy:targets.length?correct/targets.length:0,count:targets.length,unknown};
}
export function measure(experiment:Experiment):Experiment{
  const train=evaluate(experiment.model,samples(experiment,'train'),experiment.tokenizer.vocab.length);
  const validation=evaluate(experiment.model,samples(experiment,'validation'),experiment.tokenizer.vocab.length);
  const point={step:experiment.model.step,train:train.loss,validation:validation.loss,phase:experiment.phase};
  return {...experiment,history:[...experiment.history.filter(p=>p.step!==point.step),point].slice(-120)};
}
// @end evaluate
// @lesson replay
export function trainingTarget(experiment:Experiment){
  if(experiment.phase==='pretrain'){const list=samples(experiment,'train');return list[experiment.pretrainSteps%list.length];}
  const n=experiment.sftSteps,replay=sftSettings(experiment).replay;
  // 25%: one original-text update in every four; 50%: one in every two.
  const period=replay===.25?4:replay===.5?2:0;
  const original=period>0&&n%period===period-1;
  const replayBefore=period?Math.floor(n/period):0;
  const list=original?samples(experiment,'train'):assistantSamples(experiment.tokenizer,experiment.config.context);
  return list[(original?replayBefore:n-replayBefore)%list.length];
}
// @end replay
export function prediction(experiment:Experiment,sample=trainingTarget(experiment)){return forward(experiment.model,sample.ids,experiment.tokenizer.vocab.length);}
// @lesson loss
export function targetLoss(experiment:Experiment,sample=trainingTarget(experiment)){
  const probability=prediction(experiment,sample).probabilities[sample.target];
  return -Math.log(Math.max(1e-15,probability));
}
// @end loss
export function gradients(experiment:Experiment,sample=trainingTarget(experiment)){return derivatives(experiment.model,sample.ids,experiment.tokenizer.vocab.length,sample.target);}
// @lesson train
export function trainSteps(experiment:Experiment,count:number):Experiment{
  let next=experiment;
  const limit=next.phase==='sft'?800:next.config.budget;
  for(let n=0;n<count;n++){
    const done=next.phase==='sft'?next.sftSteps:next.pretrainSteps;
    if(done>=limit)break;
    const sample=trainingTarget(next);
    const grad=gradients(next,sample);
    const model=adam(next.model,grad,next.phase==='sft'?next.config.lr*sftSettings(next).rate:next.config.lr);
    next={...next,model,pretrainSteps:next.pretrainSteps+Number(next.phase==='pretrain'),sftSteps:next.sftSteps+Number(next.phase==='sft'),test:null};
    if(model.step%40===0)next=measure(next);
    if((next.phase==='pretrain'&&next.pretrainSteps%400===0)||(next.phase==='sft'&&next.sftSteps%200===0))next=saveCheckpoint(next);
  }
  return next;
}
// @end train
export function saveCheckpoint(experiment:Experiment,label?:string):Experiment{
  const cp={id:experiment.model.step,label:label??`${experiment.phase==='sft'?'SFT':'预训练'} ${experiment.model.step}`,model:structuredClone(experiment.model),phase:experiment.phase,pretrainSteps:experiment.pretrainSteps,sftSteps:experiment.sftSteps,sftSettings:structuredClone(sftSettings(experiment))};
  const current=experiment.checkpoints.filter(c=>c.id!==cp.id);
  const all=[...current,cp],baseline=experiment.phase==='sft'?all.find(c=>c.phase==='pretrain'&&c.id===experiment.pretrainSteps):undefined;
  const checkpoints=all.length>10&&baseline?[baseline,...all.filter(c=>c!==baseline).slice(-9)]:all.slice(-10);
  return {...experiment,checkpoints,logs:[...experiment.logs,`保存 ${cp.label}。`].slice(-80)};
}
// @lesson restore
export function restoreCheckpoint(e:Experiment,id:number):Experiment{
  const cp=e.checkpoints.find(c=>c.id===id);
  if(!cp)throw Error('这个检查点已不在档案中。');
  // Old exported files did not store per-phase counters; their pretrain boundary is unchanged.
  const pretrainSteps=cp.pretrainSteps??(cp.phase==='pretrain'?cp.id:e.pretrainSteps);
  const sftSteps=cp.sftSteps??(cp.phase==='pretrain'?0:cp.id-pretrainSteps);
  if(pretrainSteps<0||sftSteps<0||pretrainSteps+sftSteps!==cp.model.step)throw Error('检查点训练步数不一致。');
  const backedUp=e.model.step===cp.id&&JSON.stringify(e.model)===JSON.stringify(cp.model)?e:saveCheckpoint(e,'恢复前备份');
  return measure({...backedUp,model:structuredClone(cp.model),phase:cp.phase,pretrainSteps,sftSteps,
    sftSettings:structuredClone(cp.sftSettings??sftSettings(e)),
    history:e.history.filter(p=>p.step<=cp.id),test:null,
    logs:[...backedUp.logs,`恢复 ${cp.label}：权重、Adam 状态与训练步数一起恢复；旧测试报告失效。`].slice(-80)});
}
// @end restore
export function startSFT(experiment:Experiment):Experiment{
  if(experiment.phase==='sft')return experiment;
  const saved=saveCheckpoint(measure(experiment),'预训练完成');
  return {...saved,phase:'sft',logs:[...saved.logs,'使用当前 Transformer 权重开始 SFT；问答样本只训练回答，可混入原预训练文本。'].slice(-80)};
}
export function pretrainingCheckpoint(e:Experiment){return e.checkpoints.filter(c=>c.phase==='pretrain'&&c.id<=e.pretrainSteps).sort((a,b)=>b.id-a.id)[0];}
// @lesson generate
export function generateText(experiment:Experiment,prefix:string,model=experiment.model,maxNew=14){
  let text=prefix;const steps:{token:string;probability:number}[]=[];
  const context=tokenize(prefix,experiment.tokenizer).map(t=>t.id);
  for(let n=0;n<maxNew;n++){
    const ids=context.slice(-experiment.config.context);
    const p=forward(model,ids,experiment.tokenizer.vocab.length).probabilities;
    const id=p.indexOf(Math.max(...p)),token=experiment.tokenizer.vocab[id];
    steps.push({token,probability:p[id]});text+=token;context.push(id);
    if(token.includes('。')||token==='[UNK]')break;
  }
  return {text,steps};
}
// @end generate
// @lesson preferences
export function learnPreferences(experiment:Experiment):Experiment{
  if(!experiment.preferences.length)return experiment;
  const rewardWeights=trainReward(experiment.rewardWeights,experiment.preferences,30);
  const policyLogits=experiment.policyLogits.map((logits,q)=>{const scores=QUESTIONS[q].candidates.map(c=>reward(c.features,rewardWeights));let next=logits.slice();for(let n=0;n<30;n++)next=policyUpdate(next,scores,.25);return next;});
  return {...experiment,rewardWeights,policyLogits,preferenceUpdates:experiment.preferenceUpdates+30,logs:[...experiment.logs,`用 ${experiment.preferences.length} 组偏好更新了附加评分器和四候选策略；Transformer 权重保持不变。`].slice(-80)};
}
export function preferenceSummary(experiment:Experiment,q:number){return {loss:preferenceLoss(experiment.rewardWeights,experiment.preferences),scores:QUESTIONS[q].candidates.map(c=>reward(c.features,experiment.rewardWeights)),probabilities:softmax(experiment.policyLogits[q])};}
// @end preferences
export function runTest(experiment:Experiment):Experiment{return {...experiment,test:evaluate(experiment.model,samples(experiment,'test'),experiment.tokenizer.vocab.length),logs:[...experiment.logs,'打开留出测试报告；测试只读取权重，不参与训练。'].slice(-80)};}
// @lesson diagnose
export function compareCapabilities(e:Experiment,model=e.model){
  const score=(dialogues:typeof DIALOGUES)=>({
    ...evaluate(model,dialogueSamples(dialogues,e.tokenizer,e.config.context),e.tokenizer.vocab.length),
    exact:dialogues.filter(d=>generateText(e,`问：${d.question}答：`,model).text===`问：${d.question}答：${d.answer}`).length,
  });
  const goal=GOALS.find(g=>g.id===e.config.goal)!;
  return {continuation:evaluate(model,samples(e,'validation'),e.tokenizer.vocab.length),
    seen:score(DIALOGUES),unseen:score(UNSEEN_DIALOGUES),text:generateText(e,goal.prompt,model).text};
}
export type Advice={id:string;title:string;detail:string;stage:number;action:string};
export function diagnoseExperiment(e:Experiment,current=compareCapabilities(e),before?:ReturnType<typeof compareCapabilities>):Advice[]{
  const advice:Advice[]=[];
  if(!e.config.packs.includes(e.config.goal))advice.push({id:'coverage',title:'目标主题没有进入预训练材料',detail:'先补上对应主题的材料包。修改语料后需要重新初始化模型；旧实验可以保留备份。',stage:1,action:'回到第 2 步：挑选语料'});
  if(samples(e,'train').some(s=>s.target===0||s.ids.includes(0)))advice.push({id:'unknown',title:'训练材料含未知 token',detail:'检查保留的乱码、空白和分词结果；清洗后重新建立模型。',stage:2,action:'回到第 3 步：清洗材料'});
  if(e.sftSteps>0&&before&&(current.continuation.loss>before.continuation.loss+1||current.continuation.accuracy<before.continuation.accuracy-.15))advice.push({id:'regression',title:'微调后，留出句子的预测退步了',detail:`续写验证 loss 从 ${before.continuation.loss.toFixed(2)} 到 ${current.continuation.loss.toFixed(2)}。先恢复「预训练完成」检查点，再尝试混入原文、降低微调学习率，每 40 步比较一次。这是能力退步的线索，不能单凭它确认遗忘原因。`,stage:7,action:'回到第 8 步：恢复并比较'});
  if(current.seen.exact>current.unseen.exact)advice.push({id:'memorized',title:'示范题答对更多，改写问题仍有困难',detail:`示范题 ${current.seen.exact}/3，未训练的改写题 ${current.unseen.exact}/3。三个示范回答可能被记住了；这里可尝试混合微调，真实开发还需增加多样示范和独立评测。`,stage:8,action:'回到第 9 步：调整监督微调'});
  const base=e.sftSteps>0&&before?before:current;
  if(base.continuation.accuracy<.5)advice.push({id:'base',title:e.sftSteps>0?'微调前的基础续写就还不稳':'基础续写还没学稳',detail:`留出句子的下一个 token 命中 ${(base.continuation.accuracy*100).toFixed(1)}%。先在检查页观察预训练检查点；尚未到上限可继续预训练。若训练题好、验证题差，回语料页检查覆盖，不能只增加步数。`,stage:e.phase==='pretrain'?6:7,action:e.phase==='pretrain'?'回到第 7 步：检查预训练':'回到第 8 步：查看预训练检查点'});
  const train=evaluate(e.model,samples(e,'train'),e.tokenizer.vocab.length);
  if(current.continuation.loss>train.loss+1)advice.push({id:'gap',title:'训练材料与留出材料的成绩差距较大',detail:`训练 loss ${train.loss.toFixed(2)}，验证 loss ${current.continuation.loss.toFixed(2)}。检查材料覆盖与重复，再看整句划分。重建会保留旧实验供下载。`,stage:1,action:'回到第 2 步：检查语料覆盖'});
  if(!advice.length)advice.push({id:'monitor',title:'这些小样本暂未触发明显问题提示',detail:'继续比较续写和未训练问题；提示阈值只是教学参考，小样本成绩不能保证通用能力。',stage:7,action:'进入第 8 步：能力对比'});
  return advice;
}
// @end diagnose
// @lesson archive
export function exportExperiment(experiment:Experiment,sources:unknown){
  return JSON.stringify({format:'transformer-lab-tiny-model',version:1,description:'浏览器微型 Transformer 教育实验档案；不是商业大语言模型权重。',experiment,sources},null,2);
}
export function importExperiment(text:string):Experiment{
  if(text.length>8_000_000)throw Error('实验文件超过 8 MB。');
  const parsed=JSON.parse(text),e=parsed.experiment as Experiment;
  if(parsed.format!=='transformer-lab-tiny-model'||parsed.version!==1||!e||e.schema!==1)throw Error('不是受支持的实验档案。');
  validateExperiment(e);
  const restored=structuredClone(e);
  restored.checkpoints=restored.checkpoints.map(cp=>({...cp,pretrainSteps:cp.pretrainSteps??(cp.phase==='pretrain'?cp.id:e.pretrainSteps),sftSteps:cp.sftSteps??(cp.phase==='pretrain'?0:cp.id-e.pretrainSteps)}));
  if(restored.test)restored.test=evaluate(restored.model,samples(restored,'test'),restored.tokenizer.vocab.length);
  return restored;
}
// @end archive
export function validateExperiment(e:Experiment){
  const c=e.config;
  if(!c||typeof c.name!=='string'||!c.name.trim()||c.name.length>40||!['animals','weather','stories'].includes(c.goal)||!Array.isArray(c.packs)||!c.packs.length||c.packs.length>3||new Set(c.packs).size!==c.packs.length||c.packs.some(p=>!['animals','weather','stories'].includes(p))||!Array.isArray(c.excluded)||c.excluded.length>40||c.excluded.some(p=>typeof p!=='string')||!c.cleaning||['trim','dedupe','garbage'].some(key=>typeof c.cleaning[key as keyof Cleaning]!=='boolean')||typeof c.moreTraining!=='boolean'||!['char','pair'].includes(c.tokenizer)||![8,12,16].includes(c.width)||![16,24].includes(c.context)||!Number.isFinite(c.lr)||c.lr<.001||c.lr>.01||![400,800,1600,2400].includes(c.budget))throw Error('实验配置不合法。');
  const expected=createExperiment(c);if(JSON.stringify(e.tokenizer)!==JSON.stringify(expected.tokenizer))throw Error('词表与实验材料不一致。');
  const size=parameterGroups(e.tokenizer.vocab.length,c.width,c.context).reduce((s,g)=>s+g.size,0);
  const checkModel=(m:TinyModel)=>{if(!m||m.width!==c.width||m.context!==c.context||!Number.isInteger(m.step)||m.step<0||m.step>3200||[m.weights,m.first,m.second].some(a=>!Array.isArray(a)||a.length!==size||a.some(v=>!Number.isFinite(v)||Math.abs(v)>1e8))||m.second.some(v=>v<0))throw Error('模型参数不合法。');};
  checkModel(e.model);checkModel(e.initial);
  if(JSON.stringify(e.initial)!==JSON.stringify(expected.initial))throw Error('初始化存档与配置不一致。');
  const checkSettings=(s:SFTSettings|undefined)=>{if(s!==undefined&&(!s||![0,.25,.5].includes(s.replay)||![.15,.35,.7].includes(s.rate)))throw Error('微调设置不合法。');};checkSettings(e.sftSettings);
  if(!Array.isArray(e.checkpoints)||e.checkpoints.length>10)throw Error('存档数量不合法。');e.checkpoints.forEach(cp=>{checkModel(cp.model);checkSettings(cp.sftSettings);const pre=cp.pretrainSteps??(cp.phase==='pretrain'?cp.id:e.pretrainSteps),sft=cp.sftSteps??(cp.phase==='pretrain'?0:cp.id-pre);if(cp.id!==cp.model.step||typeof cp.label!=='string'||cp.label.length>80||!['pretrain','sft'].includes(cp.phase)||!Number.isInteger(pre)||pre<0||pre>c.budget||!Number.isInteger(sft)||sft<0||sft>800||pre+sft!==cp.id||(cp.phase==='pretrain'&&sft!==0))throw Error('存档不合法。');});
  if(!['pretrain','sft'].includes(e.phase)||!Number.isInteger(e.pretrainSteps)||e.pretrainSteps<0||e.pretrainSteps>c.budget||!Number.isInteger(e.sftSteps)||e.sftSteps<0||e.sftSteps>800||e.model.step!==e.pretrainSteps+e.sftSteps||!Array.isArray(e.history)||e.history.length>120||e.history.some(p=>!Number.isInteger(p.step)||p.step<0||!Number.isFinite(p.train)||!Number.isFinite(p.validation)||p.train<0||p.validation<0||!['pretrain','sft'].includes(p.phase))||!Array.isArray(e.logs)||e.logs.length>80||e.logs.some(l=>typeof l!=='string'||l.length>400))throw Error('实验记录不合法。');
  if(!Array.isArray(e.preferences)||e.preferences.length>60||e.preferences.some(p=>!Number.isInteger(p.question)||p.question<0||p.question>=QUESTIONS.length||![0,1,2,3].includes(p.chosen)||![0,1,2,3].includes(p.rejected)||p.chosen===p.rejected)||!Array.isArray(e.rewardWeights)||e.rewardWeights.length!==4||e.rewardWeights.some(v=>!Number.isFinite(v))||!Array.isArray(e.policyLogits)||e.policyLogits.length!==QUESTIONS.length||e.policyLogits.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>!Number.isFinite(v)))||!Number.isInteger(e.preferenceUpdates)||e.preferenceUpdates<0||e.preferenceUpdates>6000)throw Error('偏好记录不合法。');
  if(e.test!==null){const test=e.test;if(!test||!Number.isFinite(test.loss)||test.loss<0||!Number.isFinite(test.accuracy)||test.accuracy<0||test.accuracy>1||!Number.isInteger(test.count)||test.count<1||!Number.isInteger(test.unknown)||test.unknown<0||test.unknown>test.count)throw Error('测试报告不合法。');}
}
export type {TinyModel};
