/** API contracts for the independent PyTorch service. */
export type TrainConfig = {
  device: 'auto' | 'cpu' | 'cuda';
  tokenizer: 'character' | 'word';
  dataset: 'shakespeare' | 'stories' | 'stories_identity';
  layers: number; heads: number; hidden_size: number; sequence_length: number;
  learning_rate: number; batch_size: number; max_steps: number; epochs: number | null;
  eval_interval: number; validation_ratio: number; seed: number;
};
export const DEFAULT_CONFIG: TrainConfig = {
  device: 'auto',
  tokenizer: 'word',
  dataset: 'shakespeare',
  layers: 2, heads: 4, hidden_size: 64, sequence_length: 32, learning_rate: 0.003,
  batch_size: 16, max_steps: 200, epochs: null, eval_interval: 20, validation_ratio: 0.1, seed: 42,
};
export type Metric = {
  step: number; epoch: number; train_loss: number; validation_loss: number;
  perplexity: number; accuracy: number; learning_rate: number; elapsed_seconds: number;
  optimization_loss: number | null; train_eval_tokens: number; validation_tokens: number;
};
export type AttentionData = { tokens: string[]; layer: number; head: number; weights: number[][] };
export type Task = {
  device: 'cpu' | 'cuda'; peak_gpu_memory_mb: number;
  id: string; status: 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'loaded' | 'failed';
  config: TrainConfig; step: number; target_steps: number; elapsed_seconds: number;
  parameter_count: number; metrics: Metric[]; attention: AttentionData | null;
  checkpoint_id: string | null; error: string | null; stop_reason: string | null;
  data: { characters: number; train_characters: number; validation_characters: number;
    vocabulary_size: number; validation_unknown_characters: number; validation_unknown_tokens: number;
    train_tokens: number; validation_tokens: number; sha256: string };
};
export type Checkpoint = {
  id: string; step: number; created_at: number; size_bytes: number;
  config: TrainConfig; metrics: Metric | null;
};
export type Generation = { text: string; generated_text: string; prompt: string; attention: AttentionData };
export type PredictionOptions = {prompt: string; temperature: number; top_k: number; seed: number; layer: number; head: number; context_tokens?: string[]};
export type Prediction = {tokenizer: 'character' | 'word'; temperature: number; top_k: number; context_tokens: string[]; attention: AttentionData;
  candidates: {token: string; token_id: number; probability: number; model_probability: number}[]};
export type TokenStep = {next_token: string; probability: number; seed: number; prediction: Prediction};
export type Hardware = {cuda_available: boolean; devices: string[]; gpu_name: string | null; gpu_total_memory_mb: number; gpu_allocator_limit_mb: number; pytorch_version: string};
export const isActive = (task: Task | null) => !!task && ['queued', 'running', 'stopping'].includes(task.status);

export function createTrainingApi(baseUrl: string, fetcher: typeof fetch = fetch) {
  const url = new URL(baseUrl);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('请输入有效的 HTTP/HTTPS 训练服务地址');
  }
  const base = url.href.replace(/\/$/, '');
  async function call<T>(path: string, body?: unknown): Promise<T> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetcher(base + path, {
        method: body === undefined ? 'GET' : 'POST', signal: controller.signal,
        ...(body === undefined ? {} : {headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
      });
      let data: unknown;
      try { data = await response.json(); }
      catch { throw new Error(`训练服务返回无效响应 (${response.status})，请查看服务日志`); }
      if (!response.ok) {
        const detail = typeof data === 'object' && data !== null && 'detail' in data ? data.detail : undefined;
        throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
          ? detail.map((item: {msg: string}) => item.msg).join('；') : `请求失败 (${response.status})`);
      }
      return data as T;
    } catch (error) {
      if (error instanceof TypeError) throw new Error('无法连接训练服务，请检查地址并启动 FastAPI 后端。');
      if (error instanceof Error && error.name === 'AbortError') throw new Error('训练服务响应超时，请稍后刷新状态。');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  return {
    health: () => call<Hardware & {status: string}>('/health'),
    tasks: () => call<Task[]>('/tasks'),
    create: (config: TrainConfig, text?: string) => call<Task>('/tasks', {config, ...(text ? {text} : {})}),
    task: (id: string) => call<Task>(`/tasks/${id}`),
    stop: (id: string) => call<Task>(`/tasks/${id}/stop`, {}),
    resume: (id: string, max_steps: number) => call<Task>(`/tasks/${id}/resume`, {max_steps}),
    save: (id: string) => call<Checkpoint>(`/tasks/${id}/checkpoint`, {}),
    checkpoints: () => call<Checkpoint[]>('/checkpoints'),
    load: (id: string, device: TrainConfig['device'] = 'auto') => call<Task>(`/checkpoints/${id}/load`, {device}),
    generate: (id: string, options: {prompt: string; max_new_tokens: number; temperature: number; top_k: number; seed: number; layer: number; head: number; stop_after_sentence?: boolean}) =>
      call<Generation>(`/tasks/${id}/generate`, options),
    predict: (id: string, options: PredictionOptions) => call<Prediction>(`/tasks/${id}/predict`, options),
    step: (id: string, options: PredictionOptions) => call<TokenStep>(`/tasks/${id}/step`, options),
    download: (id: string) => `${base}/checkpoints/${id}/download`,
  };
}
