/** Add future learning modules here; each module owns its engine and lesson steps. */
export const learningModules = [
  { id: 'transformer', label: 'Transformer', available: true },
  { id: 'lora', label: 'LoRA', available: false },
  { id: 'rag', label: 'RAG', available: false },
  { id: 'generation', label: 'LLM 生成 / KV Cache', available: true },
  { id: 'training', label: '模型训练实验室', available: true },
  { id: 'build-model', label: '从零造模', available: true },
  { id: 'grokking', label: '泛化与 Grokking', available: true },
] as const;

