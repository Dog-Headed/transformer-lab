import assert from 'node:assert/strict';
import {CORPUS,VOCAB,SAMPLES,DEFAULT_SAMPLE,createLearner,probabilities,sampleLoss,gradient,update,meanLoss,snapshot,softmax,trainReward,preferenceLoss,reward,QUESTIONS,REFERENCE,policyUpdate,objective,kl,inspect,generate,GROUPS,PARAMETER_COUNT,MAX_STEPS} from '../lib/training/engine.ts';
const close=(a,b,tolerance=1e-6)=>assert.ok(Math.abs(a-b)<tolerance, `${a} != ${b}`);
const model=createLearner();assert.deepEqual(model,createLearner());
for(const s of SAMPLES){assert.equal(Array.from(CORPUS[s.sentence]).slice(0,s.position).join(''),s.prefix);assert.equal(Array.from(CORPUS[s.sentence])[s.position],VOCAB[s.target]);const p=probabilities(model,s);close(p.reduce((a,b)=>a+b,0),1);assert.ok(p.every(v=>v>0&&Number.isFinite(v)));close(sampleLoss(model,s),-Math.log(p[s.target]));}
const s=SAMPLES[DEFAULT_SAMPLE];const g=gradient(model,s);const eps=1e-5;
// Check every trainable parameter, including the shared embeddings and Q/K/V.
let worst=0;
for(let i=0;i<PARAMETER_COUNT;i++){const hi=structuredClone(model),lo=structuredClone(model);hi.weights[i]+=eps;lo.weights[i]-=eps;const numeric=(sampleLoss(hi,s)-sampleLoss(lo,s))/(2*eps);worst=Math.max(worst,Math.abs(g[i]-numeric));close(g[i],numeric,2e-5);}
for(const name of ['token','position','q','k','v','up','down','head']){const group=GROUPS.find(g=>g.name===name);assert.ok(g.slice(group.start,group.start+group.size).some(v=>Math.abs(v)>1e-7),`${name} must receive a loss gradient`);}
const short=inspect(model,'小猫'),long=inspect(model,'小猫喜欢');
for(let r=0;r<short.hidden.length;r++)short.hidden[r].forEach((v,c)=>close(v,long.hidden[r][c]));
for(let r=0;r<long.attention.length;r++){close(long.attention[r].reduce((a,b)=>a+b,0),1);for(let c=r+1;c<long.attention.length;c++)assert.equal(long.attention[r][c],0);}
const next=update(model,s,.003);assert.equal(next.step,1);assert.ok(sampleLoss(next,s)<sampleLoss(model,s));assert.notDeepEqual(next.weights,model.weights);assert.deepEqual(model,createLearner());
const saved=snapshot(model);saved.weights[0]=99;saved.first[0]=99;saved.second[0]=99;assert.notEqual(model.weights[0],99);assert.notEqual(model.first[0],99);assert.notEqual(model.second[0],99);
let trained=model;const started=performance.now();for(let n=0;n<MAX_STEPS;n++)trained=update(trained,SAMPLES[trained.step%SAMPLES.length],.003);
const initialLoss=meanLoss(model),finalLoss=meanLoss(trained);assert.ok(finalLoss<initialLoss*.2);assert.ok(trained.weights.every(Number.isFinite));assert.equal(update(trained,s,.003),trained);
const savedWeights=trained.weights.slice();const output=generate(trained,'小猫喜欢');assert.ok(output.text.startsWith('小猫喜欢'));assert.ok(output.steps.length>0&&output.steps.length<=12);assert.deepEqual(trained.weights,savedWeights);assert.equal(trained.step,MAX_STEPS);
let prefix='小猫喜欢';for(const step of output.steps){const p=inspect(trained,prefix).probabilities;const id=p.indexOf(Math.max(...p));assert.equal(step.token,VOCAB[id]);close(step.probability,p[id]);prefix+=step.token;}assert.equal(prefix,output.text);
assert.notEqual(output.text,generate(model,'小猫喜欢').text);
assert.deepEqual(trained,Array.from({length:MAX_STEPS}).reduce((m,_,i)=>update(m,SAMPLES[i%SAMPLES.length],.003),createLearner()));
console.log(`Transformer: ${PARAMETER_COUNT} parameters; all finite differences agree (max error ${worst.toExponential(2)}); loss ${initialLoss.toFixed(3)} → ${finalLoss.toFixed(3)}; ${MAX_STEPS} steps took ${(performance.now()-started).toFixed(0)} ms; actual output: ${output.text}`);
assert.deepEqual(softmax([10000,10000]),[.5,.5]);
const pair=[{question:0,chosen:0,rejected:3}], reversed=[{question:0,chosen:3,rejected:0}];const zero=[0,0,0,0];const weights=trainReward(zero,pair);const reverseWeights=trainReward(zero,reversed);const c=QUESTIONS[0].candidates;
assert.ok(preferenceLoss(weights,pair)<preferenceLoss(zero,pair));assert.ok(reward(c[0].features,weights)>reward(c[3].features,weights));assert.ok(reward(c[0].features,reverseWeights)<reward(c[3].features,reverseWeights));assert.deepEqual(trainReward(zero,[]),zero);
const ref=REFERENCE.slice();close(kl(ref),0);let logits=ref.map(Math.log);const scores=c.map(v=>reward(v.features,weights));const before=objective(logits,scores,.25);const updated=policyUpdate(logits,scores,.25);assert.ok(objective(updated,scores,.25)>before);assert.deepEqual(REFERENCE,ref);
const p=softmax(logits);const u=p.map((v,i)=>scores[i]-.25*(Math.log(v/ref[i])+1));const average=p.reduce((sum,v,i)=>sum+v*u[i],0);
for(let i=0;i<4;i++){const hi=logits.slice(),lo=logits.slice();hi[i]+=eps;lo[i]-=eps;close(p[i]*(u[i]-average),(objective(hi,scores,.25)-objective(lo,scores,.25))/(2*eps));}
for(let n=0;n<100;n++)logits=policyUpdate(logits,scores,.25);close(softmax(logits).reduce((a,b)=>a+b,0),1);assert.ok(logits.every(Number.isFinite));
console.log(`PASS: ${SAMPLES.length} character targets; causal attention, full Transformer gradients, immutable Adam checkpoints, 2400 training steps and shared-weight generation, reversible preference learning and KL policy optimization.`);
