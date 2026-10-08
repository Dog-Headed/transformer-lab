import TrainingLab from '@/components/lab/training-lab';
import './training.css';
export const metadata={title:'模型训练实验室 · Transformer Lab',description:'在浏览器训练微型 Transformer，观察实际注意力、损失、梯度、参数更新与同一套权重的续写，再了解 SFT 和人类偏好训练。'};
export default function TrainingPage(){return <TrainingLab/>;}
