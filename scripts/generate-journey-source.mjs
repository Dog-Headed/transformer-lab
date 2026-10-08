import {readFile,writeFile} from 'node:fs/promises';
const paths=['lib/journey/data.ts','lib/journey/engine.ts','lib/training/transformer.ts','lib/training/engine.ts'];
const files=[];
for(const path of paths){const source=await readFile(new URL('../'+path,import.meta.url),'utf8');const lines=source.split('\n'),sections=[];for(let i=0;i<lines.length;i++){const match=lines[i].match(/\/\/ @lesson (\w+)/);if(!match)continue;const end=lines.findIndex((line,j)=>j>i&&line.includes('// @end '+match[1]));if(end<0)throw Error('Missing source marker '+match[1]);sections.push({id:match[1],start:i+2,end});}files.push({path,name:path.split('/').at(-1),source,sections});}
await writeFile(new URL('../lib/journey/generated/sources.json',import.meta.url),JSON.stringify(files,null,2)+'\n');
console.log('Generated authentic code views from '+files.length+' runtime source files.');
