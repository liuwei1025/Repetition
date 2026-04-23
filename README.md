# Gesture Repetition Web Demo

这个原型先不做健身动作识别，只验证 `repetition` 计数逻辑。

当前规则：
- 输入：浏览器摄像头
- 感知：MediaPipe Gesture Recognizer
- 关键帧：开合信号在局部极值处，变化率接近 0 的点
- 相似度：起点关键帧和返回关键帧做姿态向量余弦相似度
- 计数：满足 `A(关键帧) -> B(反向关键帧) -> A'(相似关键帧)` 时记 1 次

## 本地资源

为了优化本地调试速度，MediaPipe 运行时资源已经改成走本地静态文件：
- WASM 资源来自 npm 包 `@mediapipe/tasks-vision`，同步到 `public/vendor/mediapipe/wasm/`
- 手势模型文件位于 `public/models/gesture_recognizer.task`

安装依赖后会自动执行：

```bash
npm run sync:assets
```

这样开发环境不再依赖 jsDelivr 和 Google Cloud Storage 的实时下载速度。

## 运行

```bash
npm install
npm run dev
```

然后打开本地 Vite 地址，点击“启动摄像头”。

## 怎么测试

1. 单手进入画面。
2. 先张开手掌停留约 1 秒。
3. 做一次握拳，再张开。
4. 观察是否只增加 1 次。
5. 故意做半次动作或抖动，检查是否不会误计数。

## 这版验证了什么

- 重复动作可以抽象成关键帧闭环，而不是动作名
- 关键帧可以由“动作变化率接近 0”的极值点定义
- 是否完成一次重复，可以通过关键帧姿态向量的余弦相似度判断回归
- 后续把“手势开合”替换成“人体姿态主信号”即可迁移到健身场景
