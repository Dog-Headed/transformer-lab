import type { Metadata } from 'next';
import './globals.css';
export const metadata:Metadata={title:'Transformer Lab · 看见注意力',description:'从分词到输出表示，用交互动画理解 Transformer 的每一步计算。',icons:{icon:'/favicon.svg'}};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>){return <html lang="zh-CN"><body>{children}</body></html>}
