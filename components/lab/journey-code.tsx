"use client";
import {useEffect,useState} from 'react';
import {Code2,Copy,Download,Check,FileCode2} from 'lucide-react';
import sources from '@/lib/journey/generated/sources.json';
export {sources};
export function downloadText(text:string,name:string,type='text/plain'){
  const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const colorize=(line:string)=>line.split(/(\/\/[^\n]*|'[^']*'|"[^"]*"|\b(?:export|function|return|const|let|for|if|else|import|from|type|number|string|boolean|new|throw)\b|\b\d+(?:\.\d+)?\b)/g).map((part,i)=><span key={i} className={part.startsWith('//')?'code-comment':/^['"]/.test(part)?'code-string':/^\d/.test(part)?'code-number':/^(export|function|return|const|let|for|if|else|import|from|type|number|string|boolean|new|throw)$/.test(part)?'code-keyword':undefined}>{part}</span>);
export default function JourneyCode({section,stage,description,consoleLines}:{section:string;stage:number;description:string;consoleLines:string[]}){
  const matching=sources.find(f=>f.sections.some(s=>s.id===section))??sources[0];
  const [path,setPath]=useState(matching.path);const [full,setFull]=useState(false);const[copied,setCopied]=useState(false);const[error,setError]=useState('');
  useEffect(()=>{setPath(matching.path);setFull(false);setCopied(false);},[matching.path,section]);
  const file=sources.find(f=>f.path===path)??matching;
  const selected=file.sections.find(s=>s.id===section);
  const all=file.source.split('\n');const start=!full&&selected?selected.start:1;const end=!full&&selected?selected.end:all.length;
  const visible=all.slice(start-1,end);
  const copy=async()=>{try{await navigator.clipboard.writeText(visible.join('\n'));setCopied(true);setError('');}catch{setError('复制不可用，可以下载源码。');}};
  return <section className="journey-code" id="journey-source"><header><div><Code2 size={17}/><b>真实代码工作台</b></div><span>TypeScript · 浏览器执行</span></header><div className="journey-code-tabs" role="tablist" aria-label="源码文件">{sources.map((f,i)=><button role="tab" aria-selected={file.path===f.path} disabled={(i===2&&stage<5)||(i===3&&stage<9)} key={f.path} onClick={()=>{setPath(f.path);setFull(true);}}><FileCode2 size={13}/>{i===1?'journey/':i===3?'training/':''}{f.name}{((i===2&&stage<5)||(i===3&&stage<9))&&' · 未到此步'}</button>)}</div><div className="journey-code-caption"><div><b>{description}</b><span>{file.path} · 行 {start}–{end}</span></div><button aria-pressed={full} onClick={()=>setFull(v=>!v)}>{full?'当前片段':'完整文件'}</button></div>
    <div className="journey-code-scroll" tabIndex={0} aria-label="当前环节实际运行的源码"><pre>{visible.map((line,i)=><div key={`${file.path}-${start+i}`} className={selected&&start+i>=selected.start&&start+i<=selected.end?'source-line highlighted':'source-line'}><span className="line-number" aria-hidden="true">{start+i}</span><code>{colorize(line)||' '}</code></div>)}</pre></div>
    <div className="journey-code-actions"><span>源码只读 · 在实验区修改配置</span><button onClick={copy}>{copied?<Check size={14}/>:<Copy size={14}/>} {copied?'已复制':'复制片段'}</button><button onClick={()=>downloadText(file.source,file.path.replaceAll('/','_'))}><Download size={14}/>下载文件</button></div>{error&&<p className="journey-code-error">{error}</p>}
    <div className="journey-console"><div><span>运行记录</span><small>来自实际操作；分步暂停是教学控制</small></div><div role="log" aria-live="polite" aria-relevant="additions">{consoleLines.length?consoleLines.slice(-5).map((line,i)=><p key={line+i}><span>{String(i+1).padStart(2,'0')}</span>{line}</p>):<p>等待你执行当前步骤。</p>}</div></div>
  </section>;
}
