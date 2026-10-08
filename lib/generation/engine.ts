import { add, hash, layerNorm, matmul, random, softmax, weights, type Matrix } from '../transformer/engine.ts';

export const VOCAB = ['我','你','喜欢','学习','人工','智能','语言','模型','理解','世界','今天','我们','一起','探索','的','是','可以','让','文字','连接','知识','未来','。','[EOS]'] as const;
export const MAX_CONTEXT = 24;
export type KVCache = { tokens:string[]; keys:Matrix[]; values:Matrix[]; hidden:Matrix; attention:Matrix[] };
export type Sampling = { method:'greedy'|'top-k'|'top-p'|'temperature'; temperature:number; k:number; p:number; seed:number; maxNew:number };
export const DEFAULT_SAMPLING:Sampling = {method:'top-k',temperature:.9,k:5,p:.9,seed:42,maxNew:6};

/** Actual incremental single-layer causal decoder. Historical hidden/attention rows are retained only for teaching displays. */
export function decode(tokens:string[], cache?:KVCache) {
  if(!tokens.length||tokens.length>MAX_CONTEXT)throw Error('上下文长度必须为 1–24。');
  if(cache && (cache.tokens.length>=tokens.length || cache.tokens.some((t,i)=>tokens[i]!==t)))throw Error('KV Cache 必须匹配当前上下文的严格前缀。');
  const reused=cache?.tokens.length??0;
  const x=tokens.slice(reused).map((t,i)=>{const next=random(hash(t));const p=reused+i;return Array.from({length:8},(_,d)=>next()*1.5+(d%2===0?Math.sin(p/10000**(d/8)):Math.cos(p/10000**((d-1)/8))));});
  const heads=Array.from({length:2},(_,h)=>{
    const q=matmul(x,weights(8,4,101+h*101));
    const newK=matmul(x,weights(8,4,202+h*101));
    const newV=matmul(x,weights(8,4,303+h*101));
    const k=[...(cache?.keys[h]??[]),...newK];
    const v=[...(cache?.values[h]??[]),...newV];
    const scores=q.map((row,i)=>k.map((key,j)=>j>reused+i?-Infinity:row.reduce((s,val,d)=>s+val*key[d],0)/2));
    const a=scores.map(softmax);
    const allAttention=[...(cache?.attention[h]??[]).map(r=>[...r,...Array(tokens.length-r.length).fill(0)]),...a];
    return {q,k,v,scores,attention:allAttention,output:matmul(a,v)};
  });
  const concat=x.map((_,i)=>heads.flatMap(h=>h.output[i]));
  const multihead=matmul(concat,weights(8,8,808));
  const norm1=layerNorm(add(x,multihead));
  const ffnHidden=matmul(norm1,weights(8,16,916)).map(r=>r.map(v=>Math.max(0,v+.1)));
  const ffn=matmul(ffnHidden,weights(16,8,168)).map(r=>r.map(v=>v+.05));
  const newHidden=layerNorm(add(norm1,ffn));
  const hidden=[...(cache?.hidden??[]),...newHidden];
  const lastHidden=hidden.at(-1)!;
  const lmWeights=weights(8,VOCAB.length,2408);
  const logits=matmul([lastHidden],lmWeights)[0].map((v,i)=>v+(VOCAB[i]==='[EOS]'?-1.7:0));
  const nextCache:KVCache={tokens:[...tokens],keys:heads.map(h=>h.k),values:heads.map(h=>h.v),hidden,attention:heads.map(h=>h.attention)};
  return {tokens:[...tokens],hidden,lastHidden,logits,lmWeights,heads,cache:nextCache,reused,computed:tokens.length-reused,scoreCells:2*(tokens.length-reused)*tokens.length};
}
export type DecoderResult=ReturnType<typeof decode>;
export function distribution(logits:number[], settings:Sampling) {
  if(!Number.isFinite(settings.temperature)||settings.temperature<=0||!Number.isInteger(settings.k)||settings.k<1||settings.k>logits.length||!Number.isFinite(settings.p)||settings.p<=0||settings.p>1)throw Error('采样参数不合法。');
  const temperature=settings.method==='greedy'?1:settings.temperature;
  const probs=softmax(logits.map(v=>v/temperature));
  const ranked=probs.map((prob,id)=>({id,prob,logit:logits[id],token:VOCAB[id]})).sort((a,b)=>b.prob-a.prob||a.id-b.id);
  let count=ranked.length;
  if(settings.method==='greedy')count=1;
  if(settings.method==='top-k')count=settings.k;
  if(settings.method==='top-p'){let sum=0;count=0;do{sum+=ranked[count++].prob;}while(sum<settings.p&&count<ranked.length);}
  const kept=ranked.slice(0,count);
  const mass=kept.reduce((s,c)=>s+c.prob,0);
  return ranked.map((c,i)=>({...c,kept:i<count,samplingProbability:i<count?c.prob/mass:0}));
}
export function selectNext(result:DecoderResult, settings:Sampling, round:number) {
  const candidates=distribution(result.logits,settings);
  const rng=random((settings.seed+Math.imul(round+1,2654435761))>>>0);
  const draw=(rng()+1)/2;
  let sum=0;
  const selected=candidates.find(c=>{sum+=c.samplingProbability;return draw<sum;})??candidates.filter(c=>c.kept).at(-1)!;
  return {id:selected.id,token:selected.token,draw,probability:selected.samplingProbability};
}
