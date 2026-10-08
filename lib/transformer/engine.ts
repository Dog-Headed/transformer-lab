/** A deterministic, untrained, single-layer post-LN encoder. No network requests. */
export type Matrix = number[][];
export const CONFIG = { dModel: 8, heads: 2, dHead: 4, dFF: 16, maxTokens: 10 } as const;
export function tokenize(text: string): string[] {
  const clean = text.trim();
  if (!clean) return [];
  return /\s/u.test(clean) ? clean.split(/\s+/u) : (clean.match(/\p{Script=Han}|[\p{L}\p{N}]+|[^\s]/gu) ?? []);
}
export function hash(text: string) { let h = 2166136261; for (const c of text) h = Math.imul(h ^ c.codePointAt(0)!, 16777619); return h >>> 0; }
export function random(seed: number) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296 * 2 - 1; }; }
export function weights(rows: number, cols: number, seed: number): Matrix { const next = random(seed); return Array.from({length: rows}, () => Array.from({length: cols}, () => next() * Math.sqrt(3 / rows))); }
export function matmul(a: Matrix, b: Matrix): Matrix { return a.map(row => b[0].map((_, j) => row.reduce((sum, v, k) => sum + v * b[k][j], 0))); }
export function add(a: Matrix, b: Matrix): Matrix { return a.map((r, i) => r.map((v, j) => v + b[i][j])); }
export function softmax(row: number[]): number[] { const max = Math.max(...row); const exp = row.map(v => Math.exp(v - max)); const sum = exp.reduce((a, b) => a + b, 0); return exp.map(v => v / sum); }
export function layerNorm(matrix: Matrix): Matrix { return matrix.map(row => { const mean = row.reduce((a,b)=>a+b,0)/row.length; const variance = row.reduce((a,b)=>a+(b-mean)**2,0)/row.length; return row.map(v => (v-mean)/Math.sqrt(variance+1e-5)); }); }
export function encode(text: string) {
  const allTokens = tokenize(text);
  const tokens = allTokens.slice(0, CONFIG.maxTokens);
  if (!tokens.length) throw new Error('请输入至少一个 token。');
  const vocabulary = [...new Set(tokens)];
  const ids = tokens.map(t => vocabulary.indexOf(t));
  const embedding = tokens.map(t => { const next=random(hash(t)); return Array.from({length:8},()=>next()*1.5); });
  const position = tokens.map((_, p) => Array.from({length:8}, (_,d) => d%2===0 ? Math.sin(p/10000**(d/8)) : Math.cos(p/10000**((d-1)/8))));
  const x = add(embedding,position);
  const heads = Array.from({length:2},(_,h)=>{
    const q = matmul(x,weights(8,4,101+h*101));
    const k = matmul(x,weights(8,4,202+h*101));
    const v = matmul(x,weights(8,4,303+h*101));
    const scores = q.map(row=>k.map(key=>row.reduce((s,val,i)=>s+val*key[i],0)/2));
    const attention = scores.map(softmax);
    return {q,k,v,scores,attention,output:matmul(attention,v)};
  });
  const concat = tokens.map((_,i)=>heads.flatMap(h=>h.output[i]));
  const multihead = matmul(concat,weights(8,8,808));
  const residual1 = add(x,multihead);
  const norm1 = layerNorm(residual1);
  const hidden = matmul(norm1,weights(8,16,916)).map(row=>row.map(v=>Math.max(0,v+.1)));
  const ffn = matmul(hidden,weights(16,8,168)).map(row=>row.map(v=>v+.05));
  const residual2 = add(norm1,ffn);
  const output = layerNorm(residual2);
  return {tokens,ids,allTokens,truncated:allTokens.length>tokens.length,embedding,position,x,heads,concat,multihead,residual1,norm1,hidden,ffn,residual2,output};
}
export type EncoderResult = ReturnType<typeof encode>;

