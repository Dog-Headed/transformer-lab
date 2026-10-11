import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {DEFAULT_CONFIG,dataset,createExperiment,trainSteps,startSFT,evaluate,samples,trainingTarget,targetLoss,gradients,generateText,learnPreferences,preferenceSummary,runTest,exportExperiment,importExperiment,saveCheckpoint,restoreCheckpoint,compareCapabilities,diagnoseExperiment,sftSettings} from '../lib/journey/engine.ts';
import {PACKS,buildTokenizer,tokenize,assistantSamples,DIALOGUES,UNSEEN_DIALOGUES} from '../lib/journey/data.ts';
import {forward,parameterGroups} from '../lib/training/transformer.ts';
import {QUESTIONS} from '../lib/training/engine.ts';
const close=(a,b,tol=1e-6)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
for(const packs of [['animals'],['weather'],['stories'],PACKS.map(p=>p.id)]){
  for(const moreTraining of [false,true])for(const cleaning of [{trim:true,dedupe:true,garbage:true},{trim:false,dedupe:false,garbage:false}]){
    const data=dataset({...DEFAULT_CONFIG,packs,moreTraining,cleaning});
    assert.equal(data.train.length+data.validation.length+data.test.length,data.clean.length);
    const sets=['train','validation','test'].map(key=>new Set(data[key].map(r=>r.text.trim())));
    for(let a=0;a<3;a++)for(let b=a+1;b<3;b++)for(const text of sets[a])assert.ok(!sets[b].has(text),'duplicate leakage');
  }
}
const clean=dataset(DEFAULT_CONFIG),dirty=dataset({...DEFAULT_CONFIG,cleaning:{trim:false,dedupe:false,garbage:false}});
assert.ok(clean.clean.length<dirty.clean.length);assert.ok(clean.train.length&&clean.validation.length&&clean.test.length);
const char=buildTokenizer(clean.train,'char'),pair=buildTokenizer(clean.train,'pair');
assert.ok(pair.merges.length>0);assert.ok(tokenize('小猫喜欢吃鱼。',pair).length<tokenize('小猫喜欢吃鱼。',char).length);
assert.equal(tokenize('🦄',char)[0].id,0);
for(const mode of ['char','pair'])for(const width of [8,12,16]){
  const e=createExperiment({...DEFAULT_CONFIG,tokenizer:mode,width,budget:400});
  assert.equal(e.model.weights.length,parameterGroups(e.tokenizer.vocab.length,width,e.config.context).reduce((s,g)=>s+g.size,0));
  const sample=trainingTarget(e),grad=gradients(e,sample),eps=1e-5;
  for(const group of parameterGroups(e.tokenizer.vocab.length,width,e.config.context)){
    const i=group.start+Math.floor(group.size/3),hi=structuredClone(e.model),lo=structuredClone(e.model);hi.weights[i]+=eps;lo.weights[i]-=eps;
    const loss=m=>-Math.log(forward(m,sample.ids,e.tokenizer.vocab.length).probabilities[sample.target]);
    close(targetLoss(e,sample),loss(e.model));
    close(grad[i],(loss(hi)-loss(lo))/(2*eps),2e-5);
  }
  const initial=evaluate(e.model,samples(e,'train'),e.tokenizer.vocab.length).loss;const next=trainSteps(e,400);
  assert.ok(evaluate(next.model,samples(next,'train'),next.tokenizer.vocab.length).loss<initial*.7);assert.equal(e.model.step,0);assert.equal(next.pretrainSteps,400);
  const out=generateText(next,'小猫喜欢');const ids=tokenize('小猫喜欢',next.tokenizer).map(t=>t.id);
  for(const step of out.steps){const p=forward(next.model,ids.slice(-next.config.context),next.tokenizer.vocab.length).probabilities;const id=p.indexOf(Math.max(...p));assert.equal(step.token,next.tokenizer.vocab[id]);close(step.probability,p[id]);ids.push(id);}
}
let e=createExperiment(DEFAULT_CONFIG);const initialLoss=evaluate(e.model,samples(e,'train'),e.tokenizer.vocab.length).loss;e=trainSteps(e,1600);
const pretrainLoss=evaluate(e.model,samples(e,'train'),e.tokenizer.vocab.length).loss;assert.ok(pretrainLoss<initialLoss*.2);
const beforeSft=evaluate(e.model,assistantSamples(e.tokenizer,e.config.context),e.tokenizer.vocab.length).loss;const startingWeights=e.model.weights.slice();
e=startSFT(e);assert.deepEqual(e.model.weights,startingWeights);const sftSamples=assistantSamples(e.tokenizer,e.config.context);
assert.equal(sftSamples.length,DIALOGUES.reduce((n,d)=>n+tokenize(d.answer,e.tokenizer).length,0));assert.ok(sftSamples.every(s=>s.source.startsWith('sft-')&&s.prefix.includes('答：')));
e=trainSteps(e,800);const afterSft=evaluate(e.model,sftSamples,e.tokenizer.vocab.length).loss;assert.ok(afterSft<beforeSft*.2);assert.equal(e.model.step,2400);assert.equal(e.pretrainSteps,1600);assert.equal(e.sftSteps,800);
const weights=e.model.weights.slice();e={...e,preferences:[{question:0,chosen:0,rejected:3}]};const baseline=preferenceSummary(e,0);e=learnPreferences(e);const learned=preferenceSummary(e,0);assert.ok(learned.loss<baseline.loss);assert.ok(learned.probabilities[0]>baseline.probabilities[0]);assert.deepEqual(e.model.weights,weights);
e=runTest(e);const actualTest=evaluate(e.model,samples(e,'test'),e.tokenizer.vocab.length);assert.deepEqual(e.test,actualTest);assert.deepEqual(e.model.weights,weights);
const sourceFiles=JSON.parse(await readFile(new URL('../lib/journey/generated/sources.json',import.meta.url),'utf8'));
for(const file of sourceFiles){assert.equal(file.source.replace(/\r\n/g,'\n'),(await readFile(new URL('../'+file.path,import.meta.url),'utf8')).replace(/\r\n/g,'\n'),'source view must match actual runtime code');for(const section of file.sections){assert.ok(section.start>0&&section.end>=section.start);assert.ok(file.source.split('\n')[section.start-2].includes('@lesson '+section.id));}}
const archive=exportExperiment(e,sourceFiles);assert.ok(archive.length<8_000_000);const restored=importExperiment(archive);assert.deepEqual(restored,e);
assert.equal(generateText(restored,'小猫喜欢').text,generateText(e,'小猫喜欢').text);assert.deepEqual(restored.model.first,e.model.first);restored.model.weights[0]+=1;assert.notEqual(restored.model.weights[0],e.model.weights[0]);
const early=trainSteps(createExperiment({...DEFAULT_CONFIG,budget:400}),80);const resumed=importExperiment(exportExperiment(early,sourceFiles));assert.deepEqual(trainSteps(resumed,40),trainSteps(early,40));
const corrupt=JSON.parse(archive);corrupt.experiment.model.weights.pop();assert.throws(()=>importExperiment(JSON.stringify(corrupt)),/参数/);assert.throws(()=>importExperiment('{"format":"other"}'),/档案/);
const forged=JSON.parse(archive);forged.experiment.test.loss=999;assert.deepEqual(importExperiment(JSON.stringify(forged)).test,actualTest);

// Recovery restores optimizer moments and counters, not just a picture of the weights.
const rolled=restoreCheckpoint(e,1600);
assert.equal(rolled.phase,'pretrain');assert.equal(rolled.pretrainSteps,1600);assert.equal(rolled.sftSteps,0);assert.equal(rolled.test,null);
const baseCp=e.checkpoints.find(c=>c.id===1600);assert.deepEqual(rolled.model,baseCp.model);assert.ok(rolled.checkpoints.some(c=>c.id===2400&&c.label==='恢复前备份'));
assert.ok(rolled.history.every(p=>p.step<=1600));assert.deepEqual(e.model.weights,weights);
assert.deepEqual(importExperiment(exportExperiment(rolled,sourceFiles)).model,rolled.model);
const legacy=JSON.parse(archive);for(const cp of legacy.experiment.checkpoints){delete cp.pretrainSteps;delete cp.sftSteps;delete cp.sftSettings;}
const migrated=importExperiment(JSON.stringify(legacy));const oldSft=migrated.checkpoints.find(c=>c.phase==='sft');
const oldRestored=restoreCheckpoint(migrated,oldSft.id);assert.equal(oldRestored.sftSteps,oldSft.id-1600);assert.equal(oldRestored.pretrainSteps,1600);
const checkpointed=saveCheckpoint(early,'resume point'),advanced=trainSteps(checkpointed,40),back=restoreCheckpoint(advanced,80);
assert.deepEqual(trainSteps(back,40).model,advanced.model);
assert.deepEqual(sftSettings(migrated),{replay:0,rate:.35});
for(const replay of [0,.25,.5]){
  let mixed={...startSFT(rolled),sftSettings:{replay,rate:.15}};
  const targets=[];for(let n=0;n<40;n++){const sample=trainingTarget(mixed);targets.push(sample);mixed=trainSteps(mixed,1);}
  assert.equal(targets.filter(s=>!s.source.startsWith('sft-')).length,40*replay);
  for(const target of targets.filter(s=>!s.source.startsWith('sft-')))assert.ok(samples(mixed,'train').some(s=>JSON.stringify(s)===JSON.stringify(target)));
  assert.deepEqual(trainSteps(importExperiment(exportExperiment(mixed,sourceFiles)),40).model,trainSteps(mixed,40).model);
}
const sameStep=startSFT(rolled);const backFromZero=restoreCheckpoint(sameStep,1600);assert.ok(backFromZero.checkpoints.some(c=>c.id===1600&&c.phase==='pretrain'));
const badSettings=JSON.parse(archive);badSettings.experiment.sftSettings={replay:1,rate:.35};assert.throws(()=>importExperiment(JSON.stringify(badSettings)),/微调/);
const badCounter=JSON.parse(archive);badCounter.experiment.checkpoints[0].sftSteps=10;assert.throws(()=>importExperiment(JSON.stringify(badCounter)),/存档/);
assert.ok(UNSEEN_DIALOGUES.every(d=>!DIALOGUES.some(seen=>seen.question===d.question)));
for(const d of UNSEEN_DIALOGUES)assert.ok(tokenize(d.question,e.tokenizer).every(t=>t.id!==0));
const comparison=compareCapabilities(e),baselineComparison=compareCapabilities(e,baseCp.model);
const modelBefore=structuredClone(e.model);compareCapabilities(e);assert.deepEqual(e.model,modelBefore);
const regression={...comparison,continuation:{...comparison.continuation,loss:baselineComparison.continuation.loss+2}};
assert.ok(diagnoseExperiment(e,regression,baselineComparison).some(a=>a.id==='regression'&&a.stage===7));
assert.ok(diagnoseExperiment({...e,config:{...e.config,packs:['animals'],goal:'weather'}},comparison,baselineComparison).some(a=>a.id==='coverage'&&a.stage===1));
console.log('PASS repair: full checkpoint/Adam recovery; legacy archive migration; backup and test invalidation; exact original-text replay ratios without held-out data; resumed mixed SFT; independent unseen probes and step-specific diagnosis.');

console.log(`PASS journey: no split leakage; real cleaning and tokenization; 6 architecture/tokenizer configurations; full pretraining ${initialLoss.toFixed(3)} → ${pretrainLoss.toFixed(3)}; real SFT ${beforeSft.toFixed(3)} → ${afterSft.toFixed(3)}; independent preference policy; held-out metrics; source equality; archive validation and identical resumed training (${archive.length} bytes).`);
