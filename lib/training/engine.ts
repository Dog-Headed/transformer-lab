import {initialize,forward,derivatives,adam,parameterGroups,WIDTH,CONTEXT,type TinyModel} from './transformer.ts';
export {WIDTH,CONTEXT};
/** Tiny causal Transformer trained on next-character targets in the browser. */
export const CORPUS = ['小猫喜欢吃鱼。','小狗喜欢吃骨头。','小猫喜欢睡觉。','小狗喜欢跑步。','今天晴天，我们去散步。','今天下雨，我们带雨伞。','你好，很高兴见到你。'];
export type Sample = { prefix: string; target: number; sentence: number; position: number };
export const VOCAB = Array.from(new Set(CORPUS.join('')));
export const SAMPLES: Sample[] = [];
CORPUS.forEach((text, sentence) => Array.from(text).slice(1).forEach((_, i) => {
  const prefix = Array.from(text).slice(0, i + 1).join('');
  SAMPLES.push({ prefix, target: VOCAB.indexOf(Array.from(text)[i + 1]), sentence, position: i + 1 });
}));
export type Learner = TinyModel;
export type Measurement = { step: number; loss: number };
export type Snapshot = Learner & { loss: number };
export const MAX_STEPS = 2400;
export const DEFAULT_SAMPLE = SAMPLES.findIndex(s => s.prefix === '小猫喜欢' && VOCAB[s.target] === '吃');
export function softmax(values: number[]) { const max = Math.max(...values); const exps = values.map(v => Math.exp(v - max)); const sum = exps.reduce((a,b) => a+b,0); return exps.map(v => v/sum); }
export const GROUPS=parameterGroups(VOCAB.length);
export const PARAMETER_COUNT=GROUPS.reduce((s,g)=>s+g.size,0);
export function createLearner(): Learner { return initialize(VOCAB.length); }
export function inspect(model:Learner,prefix:string){return forward(model,Array.from(prefix).slice(-CONTEXT).map(c=>VOCAB.indexOf(c)),VOCAB.length);}
export function probabilities(model: Learner, sample: Sample) { return inspect(model,sample.prefix).probabilities; }
export function sampleLoss(model: Learner, sample: Sample) { return -Math.log(Math.max(1e-15,probabilities(model,sample)[sample.target])); }
export function meanLoss(model: Learner) { return SAMPLES.reduce((sum,s)=>sum+sampleLoss(model,s),0)/SAMPLES.length; }
export function gradient(model: Learner, sample: Sample) { return derivatives(model,Array.from(sample.prefix).map(c=>VOCAB.indexOf(c)),VOCAB.length,sample.target); }
export function update(model: Learner, sample: Sample, lr: number): Learner {
  if (model.step >= MAX_STEPS) return model;
  return adam(model,gradient(model,sample),lr);
}
export function snapshot(model: Learner): Snapshot { return {...model,weights:model.weights.slice(),first:model.first.slice(),second:model.second.slice(),step:model.step,loss:meanLoss(model)}; }
export function generate(model:Learner,prefix:string,maxNew=12){let text=prefix;const steps:{token:string;probability:number}[]=[];for(let n=0;n<maxNew;n++){const p=inspect(model,text).probabilities;const id=p.indexOf(Math.max(...p));const token=VOCAB[id];steps.push({token,probability:p[id]});text+=token;if(token==='。')break;}return {text,steps};}
export type Candidate={text:string;features:number[]};
export const QUESTIONS: {question:string;answer:string;base:string;candidates:Candidate[]}[] = [
  {question:'下雨出门要带什么？',answer:'可以带雨伞或雨衣，注意路面湿滑。',base:'下雨出门要带什么？这是很多人会问的问题……',candidates:[
    {text:'带雨伞或雨衣，注意路滑。',features:[1,1,1,0]},
    {text:'下雨是云中水滴降落的天气现象。',features:[.1,1,.6,0]},
    {text:'什么也不用带，淋雨肯定有益健康。',features:[.7,0,.8,0]},
    {text:'你问得非常专业！这是一个极其深刻的问题，我可以从很多方面详细展开……',features:[.1,.5,0,1]},
  ]},
  {question:'小猫喜欢吃什么？',answer:'猫通常喜欢肉类。鱼可以作为食物，日常要注意营养均衡。',base:'小猫喜欢吃什么？小狗喜欢吃骨头。小猫喜欢睡觉。',candidates:[
    {text:'猫通常喜欢肉类，日常饮食要营养均衡。',features:[1,1,.8,0]},
    {text:'小猫是一种常见宠物。',features:[.2,1,1,0]},
    {text:'猫只需要吃鱼，其他食物都不用吃。',features:[1,0,1,0]},
    {text:'你的洞察力太强了！小猫的饮食是一个无比精彩的问题……',features:[.1,.5,.1,1]},
  ]},
  {question:'你好。',answer:'你好！有什么想一起聊聊的？',base:'你好，很高兴见到你。今天晴天，我们去散步。',candidates:[
    {text:'你好！有什么想聊聊的？',features:[1,1,1,0]},
    {text:'你好是中文里常用的问候语。',features:[.4,1,.7,0]},
    {text:'你好这个词只用于告别。',features:[.2,0,.7,0]},
    {text:'能与你这样才华横溢的人对话，真是无上的荣幸！',features:[.5,.5,.2,1]},
  ]},
];
export type Preference={question:number;chosen:number;rejected:number};
export const FEATURE_LABELS=['相关','事实质量','简洁','奉承'];
export function reward(features:number[],weights:number[]) { return features.reduce((sum,v,i)=>sum+v*weights[i],0); }
export function preferenceLoss(weights:number[],pairs:Preference[]) { if(!pairs.length)return 0; return pairs.reduce((sum,p)=>{const c=QUESTIONS[p.question].candidates; const d=reward(c[p.chosen].features,weights)-reward(c[p.rejected].features,weights); return sum+Math.max(0,-d)+Math.log1p(Math.exp(-Math.abs(d)));},0)/pairs.length; }
export function trainReward(weights:number[],pairs:Preference[],iterations=30) {
  let w=weights.slice(); if(!pairs.length)return w;
  for(let n=0;n<iterations;n++){const g=w.map(()=>0);for(const pair of pairs){const c=QUESTIONS[pair.question].candidates;const diff=c[pair.chosen].features.map((v,i)=>v-c[pair.rejected].features[i]);const d=reward(diff,w);const scale=1/(1+Math.exp(Math.min(700,d)));diff.forEach((v,i)=>{g[i]-=scale*v/pairs.length;});}w=w.map((v,i)=>v-.25*g[i]);}
  return w;
}
export const REFERENCE=[.4,.25,.15,.2];
export function kl(p:number[],ref=REFERENCE) { return p.reduce((sum,v,i)=>sum+v*Math.log(Math.max(v,1e-15)/ref[i]),0); }
export function objective(logits:number[],scores:number[],beta:number) { const p=softmax(logits);return p.reduce((s,v,i)=>s+v*scores[i],0)-beta*kl(p); }
export function policyUpdate(logits:number[],scores:number[],beta:number) {
  const p=softmax(logits);const utilities=p.map((v,i)=>scores[i]-beta*(Math.log(Math.max(v,1e-15)/REFERENCE[i])+1));const avg=p.reduce((sum,v,i)=>sum+v*utilities[i],0);
  return logits.map((v,i)=>v+.45*p[i]*(utilities[i]-avg));
}
