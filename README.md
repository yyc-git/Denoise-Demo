# How to install

```js
npm install --registry https://registry.npmmirror.com
```


项目根目录下运行：
```js
npm run webpack:dev-server
```


## 运行说明

将会执行src/main.ts代码，该代码使用WebNN实现推理，显示了降噪后的结果（目前只进行了快速训练和推理，所以效果不是很好）


关于训练部分，请详见[wspk官方实现代码](https://github.com/Rendering-at-ZJU/weight-sharing-kernel-prediction-denoising)。它使用pytorch实现了训练，并将训练出来的weight保存到checkpoints/中。

本仓库的代码在推理时直接使用了src/checkpoints/中保存的weight，它来自上面的官方实现代码中的checkpoints/