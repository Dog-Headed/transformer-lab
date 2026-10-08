import ModelJourney from '@/components/lab/model-journey';
import './journey.css';
export const metadata={title:'从零造模 · Transformer Lab',description:'在真实代码工作台体验语料清洗、分词、Transformer 预训练、监督微调、偏好教学、留出评测与模型导出。'};
export default function BuildModelPage(){return <ModelJourney/>;}
