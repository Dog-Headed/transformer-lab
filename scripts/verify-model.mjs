import assert from 'node:assert/strict';
import { encode, softmax, tokenize, CONFIG } from '../lib/transformer/engine.ts';
let cases=0;
for(const text of ['猫','我 喜欢 学习 人工 智能','The cat sat on the mat','猫 猫 猫','人工智能','😀 ✨ <script>']){
 const m=encode(text);const n=m.tokens.length;
 assert.ok(n>0&&n<=CONFIG.maxTokens);
 assert.deepEqual(m,encode(text));
 for(const h of m.heads){assert.equal(h.q.length,n);assert.equal(h.q[0].length,4);for(const row of h.attention){assert.ok(row.every(v=>Number.isFinite(v)&&v>=0&&v<=1));assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);} }
 for(const matrix of [m.norm1,m.output])for(const row of matrix){assert.equal(row.length,8);assert.ok(row.every(Number.isFinite));assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)/8)<1e-12);assert.ok(Math.abs(row.reduce((a,b)=>a+b*b,0)/8-1)<.001);}
 assert.ok(m.hidden.flat().every(v=>v>=0));cases++;
}
assert.deepEqual(tokenize(''),[]);assert.throws(()=>encode('   '));
assert.deepEqual(softmax([10000,10000]),[.5,.5]);
const repeated=encode('猫 猫');assert.deepEqual(repeated.embedding[0],repeated.embedding[1]);assert.notDeepEqual(repeated.x[0],repeated.x[1]);assert.equal(repeated.ids[0],repeated.ids[1]);
const single=encode('猫');assert.deepEqual(single.heads[0].attention,[[1]]);assert.deepEqual(single.heads[0].output,single.heads[0].v);
assert.notDeepEqual(encode('猫 喜欢 鱼').output[0],encode('猫 喜欢 狗').output[0]);
const long=encode('a b c d e f g h i j k');assert.equal(long.tokens.length,10);assert.ok(long.truncated);
console.log(`PASS: ${cases} inputs; deterministic outputs, stable softmax, row sums, normalization, positional distinction, context sensitivity and truncation.`);

