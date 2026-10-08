import { decode, selectNext, type DecoderResult, type Sampling, DEFAULT_SAMPLING, MAX_CONTEXT } from './engine.ts';
export type Frame={phase:number;generated:string[];result:DecoderResult;selection:ReturnType<typeof selectNext>|null;stop:string|null};
export type Session={prompt:string[];frames:Frame[];cursor:number;sampling:Sampling;cacheEnabled:boolean};
export function startSession(prompt:string[],sampling:Sampling=DEFAULT_SAMPLING,cacheEnabled=true):Session {
  if(prompt.length>12)throw Error('教学演示最多输入 12 个 token。');
  return {prompt:[...prompt],sampling:{...sampling},cacheEnabled,cursor:0,frames:[{phase:0,generated:[],result:decode(prompt),selection:null,stop:null}]};
}
export function forward(session:Session):Session {
  if(session.cursor<session.frames.length-1)return {...session,cursor:session.cursor+1};
  const f=session.frames[session.cursor];
  if(f.stop)return session;
  let next:Frame;
  if(f.phase<5)next={...f,phase:f.phase+1};
  else if(f.phase===5)next={...f,phase:6,selection:selectNext(f.result,session.sampling,f.generated.length)};
  else if(f.phase===6){const generated=[...f.generated,f.selection!.token];const stop=f.selection!.token==='[EOS]'?'遇到 EOS，生成结束。':generated.length>=session.sampling.maxNew?'达到本次生成 token 上限。':session.prompt.length+generated.length>=MAX_CONTEXT?'达到 24 个 token 的上下文上限。':null;next={...f,phase:7,generated,stop};}
  else {const tokens=[...session.prompt,...f.generated];next={phase:0,generated:f.generated,result:decode(tokens,session.cacheEnabled?f.result.cache:undefined),selection:null,stop:null};}
  return {...session,frames:[...session.frames,next],cursor:session.cursor+1};
}
export function backward(session:Session):Session{return {...session,cursor:Math.max(0,session.cursor-1)};}
/** Toggle caching without changing sampled history, RNG, context, or lesson position. */
export function toggleCache(session:Session,enabled:boolean):Session {
  const results=new Map<number,DecoderResult>();
  const frames=session.frames.map(f=>{const n=f.result.tokens.length;let result=results.get(n);if(!result){result=decode(f.result.tokens,enabled?results.get(n-1)?.cache:undefined);results.set(n,result);}return {...f,result};});
  return {...session,frames,cacheEnabled:enabled};
}
