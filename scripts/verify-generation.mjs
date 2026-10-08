import assert from 'node:assert/strict';
import { decode, distribution, selectNext, DEFAULT_SAMPLING, VOCAB } from '../lib/generation/engine.ts';
import { startSession, forward, backward, toggleCache } from '../lib/generation/session.ts';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const equalMatrix=(a,b)=>{assert.equal(a.length,b.length);a.forEach((r,i)=>{assert.equal(r.length,b[i].length);r.forEach((v,j)=>near(v,b[i][j]));});};
let tokens=['我','喜欢','学习'];let previous=decode(tokens);
for(let r=0;r<6;r++){
 for(const h of previous.heads)h.attention.forEach((row,i)=>{near(row.reduce((a,b)=>a+b,0),1);row.forEach((v,j)=>{if(j>i)assert.equal(v,0);});});
 tokens=[...tokens,VOCAB[r]];
 const cached=decode(tokens,previous.cache);const full=decode(tokens);
 equalMatrix(cached.hidden,full.hidden);equalMatrix(cached.heads[0].attention,full.heads[0].attention);equalMatrix(cached.heads[1].attention,full.heads[1].attention);
 cached.logits.forEach((v,i)=>near(v,full.logits[i]));assert.equal(cached.computed,1);assert.equal(cached.reused,tokens.length-1);assert.equal(full.computed,tokens.length);
 equalMatrix(full.hidden.slice(0,-1),previous.hidden);previous=cached;
}
assert.throws(()=>decode(['other'],previous.cache));
const a=decode(['我','喜欢','猫']),b=decode(['我','喜欢','狗']);equalMatrix(a.hidden.slice(0,2),b.hidden.slice(0,2));assert.notDeepEqual(a.lastHidden,b.lastHidden);
const logits=Array.from({length:24},(_,i)=>i/3);
for(const method of ['greedy','top-k','top-p','temperature']){
 const settings={...DEFAULT_SAMPLING,method};const d=distribution(logits,settings);
 near(d.reduce((s,c)=>s+c.prob,0),1);near(d.reduce((s,c)=>s+c.samplingProbability,0),1);
 if(method==='top-k')assert.equal(d.filter(c=>c.kept).length,5);
 if(method==='greedy'){assert.equal(d[0].id,23);assert.equal(d[0].samplingProbability,1);}
 if(method==='top-p'){const kept=d.filter(c=>c.kept);assert.ok(kept.reduce((s,c)=>s+c.prob,0)>=.9);assert.ok(kept.slice(0,-1).reduce((s,c)=>s+c.prob,0)<.9);}
 const toy={logits};for(let seed=1;seed<=100;seed++){const selected=selectNext(toy,{...settings,seed},0);assert.ok(d.find(c=>c.id===selected.id).kept);}
}
const cold=distribution(logits,{...DEFAULT_SAMPLING,method:'temperature',temperature:.1});const hot=distribution(logits,{...DEFAULT_SAMPLING,method:'temperature',temperature:2});assert.ok(cold[0].prob>hot[0].prob);
assert.equal(distribution(logits,{...DEFAULT_SAMPLING,k:1})[0].samplingProbability,1);
assert.equal(distribution(logits,{...DEFAULT_SAMPLING,method:'top-p',p:1}).filter(c=>c.kept).length,24);
let s=startSession(['我','喜欢'],{...DEFAULT_SAMPLING,method:'greedy',maxNew:2});
for(let i=0;i<7;i++)s=forward(s);
assert.equal(s.frames[s.cursor].phase,7);assert.equal(s.frames[s.cursor].generated.length,1);
const picked=s.frames[s.cursor].selection.token;
s=backward(s);assert.equal(s.frames[s.cursor].generated.length,0);s=forward(s);assert.equal(s.frames[s.cursor].generated[0],picked);
s=forward(s);assert.equal(s.frames[s.cursor].phase,0);assert.equal(s.frames[s.cursor].result.computed,1);
const before=s.frames[s.cursor];const off=toggleCache(s,false);assert.equal(off.frames[off.cursor].result.computed,3);assert.deepEqual(off.frames[off.cursor].result.logits,before.result.logits);assert.deepEqual(off.frames[off.cursor].generated,before.generated);
s=toggleCache(off,true);assert.equal(s.frames[s.cursor].result.computed,1);
for(let i=0;i<7;i++)s=forward(s);assert.ok(s.frames[s.cursor].stop);assert.equal(s.frames[s.cursor].generated.length,2);assert.equal(forward(s),s);
let eos=startSession(['我']);for(let i=0;i<6;i++)eos=forward(eos);eos={...eos,frames:eos.frames.map((f,i)=>i===eos.cursor?{...f,selection:{...f.selection,token:'[EOS]',id:23}}:f)};eos=forward(eos);assert.match(eos.frames[eos.cursor].stop,/EOS/);assert.equal(forward(eos),eos);
console.log('PASS generation: causal invariance, all head masks, full vs incremental equality across 6 rounds, sampling filters and 400 draws, temperature, reversible append, cache toggles, limit and EOS stops.');
