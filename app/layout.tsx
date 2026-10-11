import type { Metadata } from 'next';
import './globals.css';
export const metadata:Metadata={title:'Transformer Training Lab｜Transformer 训练与可视化实验平台',description:'从 Transformer 计算可视化到真实 PyTorch 训练、验证评估、模型存档与文本生成。',icons:{icon:'/favicon.svg'}};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="zh-CN"><body>{children}</body></html>}
