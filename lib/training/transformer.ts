/** A browser-sized, trainable causal Transformer. All derivatives are analytic. */
export const WIDTH = 8;
export const CONTEXT = 16;
export type TinyModel = {weights:number[]; first:number[]; second:number[]; step:number; width?:number; context?:number};
export type Group = {name:string; label:string; rows:number; cols:number; start:number; size:number};
// @lesson model
export function parameterGroups(vocab:number,width=WIDTH,context=CONTEXT):Group[] {
  let start=0;
  return [
    ['token','字符 Embedding',vocab,width],['position','位置 Embedding',context,width],
    ['ln1g','注意力前归一化 · γ',1,width],['ln1b','注意力前归一化 · β',1,width],
    ['q','WQ',width,width],['k','WK',width,width],['v','WV',width,width],['o','WO',width,width],
    ['ln2g','前馈前归一化 · γ',1,width],['ln2b','前馈前归一化 · β',1,width],
    ['up','前馈 · W1',width,width*2],['upb','前馈 · b1',1,width*2],
    ['down','前馈 · W2',width*2,width],['downb','前馈 · b2',1,width],
    ['lng','输出归一化 · γ',1,width],['lnb','输出归一化 · β',1,width],
    ['head','词表投影',width,vocab],['headb','词表偏置',1,vocab],
  ].map(([name,label,rows,cols])=>{const size=Number(rows)*Number(cols);const g={name:String(name),label:String(label),rows:Number(rows),cols:Number(cols),start,size};start+=size;return g;});
}
export function initialize(vocab:number,width=WIDTH,context=CONTEXT):TinyModel {
  let seed=2026;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  const weights=parameterGroups(vocab,width,context).flatMap(g=>Array.from({length:g.size},()=>g.name.endsWith('g')?1:g.name.endsWith('b')?0:(random()-.5)*2*(g.name==='token'||g.name==='position'?.2:Math.sqrt(3/g.rows))));
  return {weights,first:weights.map(()=>0),second:weights.map(()=>0),step:0,width,context};
}
// @end model
type Tensor={v:Float64Array;g:Float64Array;r:number;c:number;back:()=>void};
function graph(model:TinyModel,ids:number[],vocab:number,training:boolean) {
  const width=model.width??WIDTH,context=model.context??CONTEXT;
  if(!ids.length||ids.length>context||ids.some(id=>!Number.isInteger(id)||id<0||id>=vocab))throw Error('字符编号或上下文长度不合法');
  const tape:Tensor[]=[];
  const node=(r:number,c:number,values:ArrayLike<number>,back:()=>void=()=>{}):Tensor=>{const t={r,c,v:Float64Array.from(values),g:new Float64Array(r*c),back};if(training)tape.push(t);return t;};
  const groups=parameterGroups(vocab,width,context);
  const banks:Record<string,Tensor>={};groups.forEach(g=>banks[g.name]=node(g.rows,g.cols,model.weights.slice(g.start,g.start+g.size)));
  const mm=(a:Tensor,b:Tensor)=>{const out=new Float64Array(a.r*b.c);for(let i=0;i<a.r;i++)for(let k=0;k<a.c;k++)for(let j=0;j<b.c;j++)out[i*b.c+j]+=a.v[i*a.c+k]*b.v[k*b.c+j];const t=node(a.r,b.c,out,()=>{for(let i=0;i<a.r;i++)for(let k=0;k<a.c;k++)for(let j=0;j<b.c;j++){const d=t.g[i*b.c+j];a.g[i*a.c+k]+=d*b.v[k*b.c+j];b.g[k*b.c+j]+=d*a.v[i*a.c+k];}});return t;};
  const add=(a:Tensor,b:Tensor)=>{const t=node(a.r,a.c,a.v.map((v,i)=>v+b.v[b.r===1?i%a.c:i]),()=>{for(let i=0;i<t.g.length;i++){a.g[i]+=t.g[i];b.g[b.r===1?i%a.c:i]+=t.g[i];}});return t;};
  const transpose=(a:Tensor)=>{const out=new Float64Array(a.v.length);for(let i=0;i<a.r;i++)for(let j=0;j<a.c;j++)out[j*a.r+i]=a.v[i*a.c+j];const t=node(a.c,a.r,out,()=>{for(let i=0;i<a.r;i++)for(let j=0;j<a.c;j++)a.g[i*a.c+j]+=t.g[j*a.r+i];});return t;};
  const norm=(a:Tensor,gamma:Tensor,beta:Tensor)=>{
    const z=new Float64Array(a.v.length),inv=new Float64Array(a.r),out=new Float64Array(a.v.length);
    for(let r=0;r<a.r;r++){let mean=0;for(let c=0;c<a.c;c++)mean+=a.v[r*a.c+c]/a.c;let variance=0;for(let c=0;c<a.c;c++)variance+=(a.v[r*a.c+c]-mean)**2/a.c;inv[r]=1/Math.sqrt(variance+1e-5);for(let c=0;c<a.c;c++){const i=r*a.c+c;z[i]=(a.v[i]-mean)*inv[r];out[i]=z[i]*gamma.v[c]+beta.v[c];}}
    const t=node(a.r,a.c,out,()=>{for(let r=0;r<a.r;r++){let avg=0,avgz=0;for(let c=0;c<a.c;c++){const i=r*a.c+c;const d=t.g[i]*gamma.v[c];avg+=d/a.c;avgz+=d*z[i]/a.c;gamma.g[c]+=t.g[i]*z[i];beta.g[c]+=t.g[i];}for(let c=0;c<a.c;c++){const i=r*a.c+c;a.g[i]+=inv[r]*(t.g[i]*gamma.v[c]-avg-z[i]*avgz);}}});return t;
  };
  const relu=(a:Tensor)=>{const t=node(a.r,a.c,a.v.map(v=>Math.max(0,v)),()=>{for(let i=0;i<a.v.length;i++)if(a.v[i]>0)a.g[i]+=t.g[i];});return t;};
  const causalSoftmax=(a:Tensor)=>{
    const out=new Float64Array(a.v.length),scale=1/Math.sqrt(width);
    for(let r=0;r<a.r;r++){let max=-Infinity;for(let c=0;c<=r;c++)max=Math.max(max,a.v[r*a.c+c]*scale);let sum=0;for(let c=0;c<=r;c++){out[r*a.c+c]=Math.exp(a.v[r*a.c+c]*scale-max);sum+=out[r*a.c+c];}for(let c=0;c<=r;c++)out[r*a.c+c]/=sum;}
    const t=node(a.r,a.c,out,()=>{for(let r=0;r<a.r;r++){let dot=0;for(let c=0;c<=r;c++)dot+=t.g[r*a.c+c]*t.v[r*a.c+c];for(let c=0;c<=r;c++){const i=r*a.c+c;a.g[i]+=scale*t.v[i]*(t.g[i]-dot);}}});return t;
  };
  // @lesson forward
  const e=banks.token,p=banks.position;
  const embedded=node(ids.length,width,ids.flatMap((id,r)=>Array.from({length:width},(_,c)=>e.v[id*width+c]+p.v[r*width+c])),()=>{for(let r=0;r<ids.length;r++)for(let c=0;c<width;c++){const d=embedded.g[r*width+c];e.g[ids[r]*width+c]+=d;p.g[r*width+c]+=d;}});
  const normalized=norm(embedded,banks.ln1g,banks.ln1b);
  const q=mm(normalized,banks.q),k=mm(normalized,banks.k),v=mm(normalized,banks.v);
  const attention=causalSoftmax(mm(q,transpose(k)));
  const mixed=mm(mm(attention,v),banks.o);
  const residual=add(embedded,mixed);
  const ffnInput=norm(residual,banks.ln2g,banks.ln2b);
  const ffn=add(mm(relu(add(mm(ffnInput,banks.up),banks.upb)),banks.down),banks.downb);
  const hidden=norm(add(residual,ffn),banks.lng,banks.lnb);
  const logits=add(mm(hidden,banks.head),banks.headb);
  // @end forward
  const last=Array.from(logits.v.slice((ids.length-1)*vocab));
  const max=Math.max(...last);const exp=last.map(v=>Math.exp(v-max));const sum=exp.reduce((s,v)=>s+v,0);const probabilities=exp.map(v=>v/sum);
  const matrix=(t:Tensor)=>Array.from({length:t.r},(_,r)=>Array.from(t.v.slice(r*t.c,(r+1)*t.c)));
  return {logits:last,probabilities,attention:matrix(attention),embedding:matrix(embedded),q:matrix(q),k:matrix(k),v:matrix(v),ffn:matrix(ffn),hidden:matrix(hidden),
    // @lesson backward
    backward(target:number){
      logits.g.set(probabilities.map((v,i)=>v-(i===target?1:0)),(ids.length-1)*vocab);
      for(let i=tape.length-1;i>=0;i--)tape[i].back();
      return groups.flatMap(g=>Array.from(banks[g.name].g));
    }
    // @end backward
  };
}
export function forward(model:TinyModel,ids:number[],vocab:number){const {backward,...result}=graph(model,ids,vocab,false);return result;}
export function derivatives(model:TinyModel,ids:number[],vocab:number,target:number){return graph(model,ids,vocab,true).backward(target);}
// @lesson optimizer
export function adam(model:TinyModel,grad:number[],lr:number):TinyModel {
  if(!Number.isFinite(lr)||lr<=0)throw Error('学习率必须大于零');
  const step=model.step+1,norm=Math.sqrt(grad.reduce((s,v)=>s+v*v,0)),scale=Math.min(1,1/Math.max(1e-15,norm));
  const first=grad.map((v,i)=>.9*model.first[i]+.1*v*scale),second=grad.map((v,i)=>.999*model.second[i]+.001*(v*scale)**2);
  const weights=model.weights.map((v,i)=>v-lr*(first[i]/(1-.9**step))/(Math.sqrt(second[i]/(1-.999**step))+1e-8));
  return {weights,first,second,step,width:model.width,context:model.context};
}

// @end optimizer
