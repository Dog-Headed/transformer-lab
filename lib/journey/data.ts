export type PackId='animals'|'weather'|'stories';
export const PACKS:{id:PackId;label:string;description:string;texts:string[]}[]=[
  {id:'animals',label:'动物日常',description:'吃什么、做什么，短句搭配',texts:['小猫喜欢吃鱼。','小狗喜欢吃骨头。','小猫喜欢睡觉。','小狗喜欢跑步。','小猫在窗边睡觉。','小狗在公园跑步。','小猫喜欢在公园散步。','小狗喜欢在窗边睡觉。']},
  {id:'weather',label:'天气与出行',description:'晴天、下雨和出门的关系',texts:['今天晴天，我们去散步。','今天下雨，我们带雨伞。','晴天适合去公园。','下雨记得带雨伞。','我们喜欢晴天。','我们下雨也去散步。','今天晴天，我们去公园。','今天下雨，我们在窗边看雨。']},
  {id:'stories',label:'小小故事',description:'人物、地点与动作的组合',texts:['小明去公园看花。','小红在窗边看书。','小明喜欢看书。','小红喜欢看花。','今天小明带书去公园。','今天小红在公园散步。','小明在窗边看花。','小红带书去公园。']},
];
export const DIALOGUES=[
  {question:'猫喜欢吃什么？',answer:'猫喜欢吃鱼。'},
  {question:'下雨带什么？',answer:'下雨带雨伞。'},
  {question:'小明喜欢什么？',answer:'小明喜欢看书。'},
];
// These paraphrases are evaluation probes only; assistantSamples never uses them.
export const UNSEEN_DIALOGUES=[
  {question:'小猫喜欢吃什么？',answer:'猫喜欢吃鱼。'},
  {question:'今天下雨带什么？',answer:'下雨带雨伞。'},
  {question:'小明喜欢看什么？',answer:'小明喜欢看书。'},
];
export const GOALS=[{id:'animals',label:'动物小助手',prompt:'小猫喜欢',question:0},{id:'weather',label:'天气小助手',prompt:'今天下雨，我们',question:1},{id:'stories',label:'故事小助手',prompt:'小明喜欢',question:2}] as const;
export type Cleaning={trim:boolean;dedupe:boolean;garbage:boolean};
export type RecordLine={id:string;pack:PackId;slot:number;text:string;issue:string};
export function rawRecords(packs:PackId[]):RecordLine[]{return PACKS.filter(p=>packs.includes(p.id)).flatMap(p=>[
  ...p.texts.map((text,slot)=>({id:`${p.id}-${slot}`,pack:p.id,slot,text,issue:''})),
  {id:`${p.id}-duplicate`,pack:p.id,slot:0,text:p.texts[0],issue:'重复文本'},
  {id:`${p.id}-space`,pack:p.id,slot:1,text:`  ${p.texts[1]}  `,issue:'首尾空白'},
  {id:`${p.id}-noise`,pack:p.id,slot:0,text:'��<乱码>###',issue:'乱码材料'},
]);}
// @lesson clean
export function cleanRecords(records:RecordLine[],cleaning:Cleaning,excluded:string[]){
  const seen=new Set<string>();const kept:RecordLine[]=[];
  for(const record of records){
    if(excluded.includes(record.id))continue;
    if(cleaning.garbage&&record.issue==='乱码材料')continue;
    const text=cleaning.trim?record.text.trim():record.text;
    if(cleaning.dedupe&&seen.has(text))continue;
    seen.add(text);kept.push({...record,text});
  }
  return kept;
}
// @end clean
// @lesson split
export function splitRecords(records:RecordLine[],moreTraining:boolean){
  // A sentence and its dirty/duplicate versions always belong to the same split.
  const cutoff=moreTraining?6:5;
  return {
    train:records.filter(r=>r.slot<cutoff),
    validation:records.filter(r=>r.slot>=cutoff&&r.slot<7),
    test:records.filter(r=>r.slot===7),
  };
}
// @end split
export type Tokenizer={mode:'char'|'pair';vocab:string[];merges:string[]};
// The character alphabet is public and fixed, not learned from held-out answers.
const ALPHABET=Array.from(new Set(PACKS.flatMap(p=>p.texts).join('')+DIALOGUES.map(d=>`问：${d.question}答：${d.answer}`).join('')));
// @lesson tokenizer
export function buildTokenizer(train:RecordLine[],mode:Tokenizer['mode']):Tokenizer{
  const counts=new Map<string,number>();
  for(const {text} of train){const chars=Array.from(text);for(let i=0;i<chars.length-1;i++){const pair=chars[i]+chars[i+1];if(ALPHABET.includes(chars[i])&&ALPHABET.includes(chars[i+1]))counts.set(pair,(counts.get(pair)??0)+1);}}
  const merges=mode==='pair'?Array.from(counts).filter(([,n])=>n>=2).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0],'zh')).slice(0,8).map(([text])=>text):[];
  return {mode,merges,vocab:['[UNK]',...ALPHABET,...merges]};
}
export function tokenize(text:string,tokenizer:Tokenizer){
  const chars=Array.from(text),tokens:string[]=[];
  for(let i=0;i<chars.length;i++){const pair=chars[i]+(chars[i+1]??'');if(tokenizer.merges.includes(pair)){tokens.push(pair);i++;}else tokens.push(chars[i]);}
  return tokens.map(token=>({token,id:Math.max(0,tokenizer.vocab.indexOf(token))}));
}
// @end tokenizer
export type TrainingSample={ids:number[];target:number;prefix:string;answer:string;source:string};
// @lesson samples
export function makeSamples(records:RecordLine[],tokenizer:Tokenizer,context:number):TrainingSample[]{
  return records.flatMap(record=>{
    const tokens=tokenize(record.text,tokenizer);
    return tokens.slice(1).map((target,i)=>({ids:tokens.slice(Math.max(0,i+1-context),i+1).map(t=>t.id),target:target.id,prefix:tokens.slice(0,i+1).map(t=>t.token).join(''),answer:target.token,source:record.id}));
  });
}
// @end samples
// @lesson sft
export function assistantSamples(tokenizer:Tokenizer,context:number):TrainingSample[]{
  return dialogueSamples(DIALOGUES,tokenizer,context);
}
export function dialogueSamples(dialogues:typeof DIALOGUES,tokenizer:Tokenizer,context:number):TrainingSample[]{
  return dialogues.flatMap((dialogue,index)=>{
    const prefix=`问：${dialogue.question}答：`;
    const input=tokenize(prefix,tokenizer),answer=tokenize(dialogue.answer,tokenizer);
    return answer.map((target,i)=>({ids:[...input,...answer.slice(0,i)].slice(-context).map(t=>t.id),target:target.id,prefix:prefix+answer.slice(0,i).map(t=>t.token).join(''),answer:target.token,source:`sft-${index}`}));
  });
}
// @end sft
