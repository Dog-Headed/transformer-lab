/** Designed probability checkpoints for teaching; no neural network is trained here. */
export type ReplayMode = 'generalize' | 'memorize';
export type Question = { id: number; a: number; b: number; target: number; training: boolean };
export const MODULUS = 7;
export const QUESTIONS: Question[] = Array.from({length:49},(_,id)=>{
  const a=Math.floor(id/7),b=id%7;
  return {id,a,b,target:(a+b)%7,training:(a*3+b*5)%7<3};
});
export const TRAINING=QUESTIONS.filter(q=>q.training);
export const HELD_OUT=QUESTIONS.filter(q=>!q.training);
export const STEPS=[0,100,200,400,600,1000,2000,3000,5000,8000,10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,240000,300000,400000,600000,800000,1000000];
export const LAST_FRAME=STEPS.length-1;
export const MEMORIZATION_FRAME=6;
export const DEFAULT_QUESTION=QUESTIONS.find(q=>q.a===5&&q.b===6)!;
export const sigmoid=(x:number)=>1/(1+Math.exp(-x));
export function distribution(question:Question,frame:number,mode:ReplayMode):number[]{
  const f=Math.max(0,Math.min(LAST_FRAME,frame));
  const jitter=((question.a*11+question.b*3)%7-3)*.12;
  const fit=sigmoid((f-2.3)*1.65+jitter);
  let targetProbability:number;
  if(question.training)targetProbability=.125+.855*fit;
  else{
    const memorization=sigmoid((f-3)*1.4);
    const baseline=.125-.085*memorization;
    const onset=19+((question.a*11+question.b*3)%7)*.42;
    const generalization=mode==='generalize'?sigmoid((f-onset)*1.25):0;
    targetProbability=baseline+(.975-baseline)*generalization;
  }
  const wrong=(question.target+1+(question.a*2+question.b*3)%6)%7;
  const weights:number[]=Array.from({length:7},(_,i)=>i===question.target?0:i===wrong?8.5:1);
  const total=weights.reduce((a,b)=>a+b,0);
  return weights.map((w,i)=>i===question.target?targetProbability:(1-targetProbability)*w/total);
}
export function prediction(q:Question,frame:number,mode:ReplayMode){const p=distribution(q,frame,mode);return p.indexOf(Math.max(...p));}
export function metrics(questions:Question[],frame:number,mode:ReplayMode){
  const loss=questions.reduce((sum,q)=>sum-Math.log(distribution(q,frame,mode)[q.target]),0)/questions.length;
  const accuracy=questions.filter(q=>prediction(q,frame,mode)===q.target).length/questions.length;
  return {loss,accuracy};
}
export function frameMetrics(frame:number,mode:ReplayMode){return {frame,step:STEPS[frame],training:metrics(TRAINING,frame,mode),validation:metrics(HELD_OUT,frame,mode)};}
export function trajectory(mode:ReplayMode){return STEPS.map((_,i)=>frameMetrics(i,mode));}
export const STAGES=[{frame:0,label:'随机预测'},{frame:6,label:'训练题满分'},{frame:15,label:'新题仍不会'},{frame:19,label:'泛化开始改善'},{frame:22,label:'更多新题答对'},{frame:26,label:'掌握这组规律'}];
export function stageLabel(frame:number,mode:ReplayMode){if(frame<3)return '正在拟合训练题';if(mode==='memorize')return '只背答案：新题仍不会';if(frame<18)return '训练题已会，新题还不会';if(frame<23)return '新题表现明显改善';return '这组新题也能答对了';}
export type ReplayState={frame:number;playing:boolean;tested:boolean;mode:ReplayMode};
export type ReplayAction={type:'tick'|'toggle'|'test'|'reset'}|{type:'scrub';frame:number}|{type:'mode';mode:ReplayMode};
export const initialReplay:ReplayState={frame:0,playing:false,tested:false,mode:'generalize'};
export function replayReducer(state:ReplayState,action:ReplayAction):ReplayState{
  switch(action.type){
    case 'reset':return {...initialReplay,mode:state.mode};
    case 'mode':return {...state,mode:action.mode,playing:false};
    case 'scrub':{const frame=Math.max(0,Math.min(LAST_FRAME,Math.round(action.frame)));return {...state,frame,playing:false,tested:state.tested||frame>MEMORIZATION_FRAME};}
    case 'test':return {...state,tested:true,playing:false};
    case 'toggle':if(state.frame>=MEMORIZATION_FRAME&&!state.tested)return {...state,playing:false};if(state.frame===LAST_FRAME)return {...initialReplay,mode:state.mode,playing:true};return {...state,playing:!state.playing};
    case 'tick':{
      if(!state.playing)return state;
      const frame=Math.min(LAST_FRAME,state.frame+1);
      const pause=frame===LAST_FRAME||(!state.tested&&frame>=MEMORIZATION_FRAME);
      return {...state,frame,playing:!pause};
    }
  }
}
export const SCORE_TARGET='3141592653';
export const SCORE_OUTPUTS=['8247601980','3148701980','3141591980','3141592680','3141592650','3141592653'];
export function digitScore(text:string){return Array.from(SCORE_TARGET).filter((char,i)=>text[i]===char).length/SCORE_TARGET.length;}
export function exactScore(text:string){return text===SCORE_TARGET?1:0;}
