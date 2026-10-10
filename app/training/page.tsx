import PytorchTrainingLab from '@/components/lab/pytorch-training-lab';
import './pytorch.css';
export const metadata={title:'Transformer Training Lab｜Transformer 训练与可视化实验平台',description:'真实 PyTorch 字符级 Transformer 训练、留出验证、注意力可视化、checkpoint 保存恢复与模型续写。'};
export default function TrainingPage(){return <PytorchTrainingLab/>;}
