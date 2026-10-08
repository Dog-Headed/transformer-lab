import assert from 'node:assert/strict';
import {QUESTIONS,TRAINING,HELD_OUT,STEPS,LAST_FRAME,MEMORIZATION_FRAME,DEFAULT_QUESTION,distribution,prediction,metrics,frameMetrics,trajectory,replayReducer,initialReplay,SCORE_OUTPUTS,digitScore,exactScore} from '../lib/training/grokking.ts';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
assert.equal(TRAINING.length,21);assert.equal(HELD_OUT.length,28);assert.equal(new Set(QUESTIONS.map(q=>q.id)).size,49);
assert.ok(TRAINING.every(q=>!HELD_OUT.some(v=>v.id===q.id)));
assert.equal(DEFAULT_QUESTION.training,false);assert.equal(DEFAULT_QUESTION.target,4);
for(const q of QUESTIONS){assert.equal(q.target,(q.a+q.b)%7);for(const mode of ['generalize','memorize'])for(let frame=0;frame<STEPS.length;frame++){
  const p=distribution(q,frame,mode);assert.equal(p.length,7);close(p.reduce((a,b)=>a+b,0),1);assert.ok(p.every(v=>Number.isFinite(v)&&v>0&&v<1));assert.deepEqual(p,distribution(q,frame,mode));assert.equal(prediction(q,frame,mode),p.indexOf(Math.max(...p)));
}}
for(const mode of ['generalize','memorize'])for(let frame=0;frame<STEPS.length;frame++){
  const row=frameMetrics(frame,mode);assert.equal(row.step,STEPS[frame]);
  for(const [split,list] of [['training',TRAINING],['validation',HELD_OUT]]){
    const losses=list.map(q=>-Math.log(distribution(q,frame,mode)[q.target]));close(row[split].loss,losses.reduce((a,b)=>a+b,0)/list.length);
    const correct=list.filter(q=>prediction(q,frame,mode)===q.target).length;close(row[split].accuracy,correct/list.length);
  }
}
const early=frameMetrics(MEMORIZATION_FRAME,'generalize'),late=frameMetrics(LAST_FRAME,'generalize'),control=frameMetrics(LAST_FRAME,'memorize');
assert.equal(early.training.accuracy,1);assert.equal(early.validation.accuracy,0);assert.ok(early.training.loss<.05);assert.ok(early.validation.loss>2);
assert.equal(late.training.accuracy,1);assert.equal(late.validation.accuracy,1);assert.ok(late.validation.loss<.05);assert.equal(control.training.accuracy,1);assert.equal(control.validation.accuracy,0);
assert.notEqual(prediction(DEFAULT_QUESTION,MEMORIZATION_FRAME,'generalize'),DEFAULT_QUESTION.target);assert.equal(prediction(DEFAULT_QUESTION,LAST_FRAME,'generalize'),DEFAULT_QUESTION.target);
for(let i=0;i<STEPS.length;i++)close(metrics(TRAINING,i,'generalize').loss,metrics(TRAINING,i,'memorize').loss);
assert.ok(trajectory('generalize').every(row=>Number.isFinite(row.validation.loss)));
let state=replayReducer(initialReplay,{type:'toggle'});for(let i=0;i<MEMORIZATION_FRAME;i++)state=replayReducer(state,{type:'tick'});assert.equal(state.frame,MEMORIZATION_FRAME);assert.equal(state.playing,false);assert.equal(state.tested,false);
assert.equal(replayReducer(state,{type:'toggle'}).playing,false);state=replayReducer(state,{type:'test'});state=replayReducer(state,{type:'toggle'});for(let i=MEMORIZATION_FRAME;i<LAST_FRAME;i++)state=replayReducer(state,{type:'tick'});assert.equal(state.frame,LAST_FRAME);assert.equal(state.playing,false);
const switched=replayReducer(state,{type:'mode',mode:'memorize'});assert.equal(switched.frame,state.frame);assert.equal(switched.tested,state.tested);
const paused=replayReducer(initialReplay,{type:'scrub',frame:15});assert.equal(paused.playing,false);assert.equal(replayReducer(paused,{type:'tick'}),paused);assert.equal(replayReducer(paused,{type:'reset'}).frame,0);
assert.ok(SCORE_OUTPUTS.slice(0,-1).every(s=>exactScore(s)===0));assert.equal(exactScore(SCORE_OUTPUTS.at(-1)),1);assert.equal(digitScore(SCORE_OUTPUTS.at(-1)),1);
const scores=SCORE_OUTPUTS.map(digitScore);assert.ok(scores.every((v,i)=>i===0||v>=scores[i-1]));assert.ok(scores[4]>scores[0]);assert.ok(scores[4]<1);
console.log('PASS: 49 fixed questions, 27 normalized checkpoints, consistent loss/accuracy, delayed generalization vs memorization, gated replay and metric-threshold comparison.');
