// export let convertArrayTensorToFloat32Tensor = (arrayTensor):Float32Array => {

// export let buildImageData = () => {

// }

export let convertImageDataToFloat32Tensor = (imageData, inputDimensions): Float32Array => {
    let tensor = new Float32Array(
        inputDimensions.slice(1).reduce((a, b) => a * b));

    let [channels, height, width] = inputDimensions.slice(1);

    let mean = [0, 0, 0, 0];
    let std = [1, 1, 1, 1];
    let imageChannels = 4; // RGBA

    for (let c = 0; c < channels; ++c) {
        for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
                let value;
                // if (channelScheme === 'BGR') {
                //     value = imageData[h * width * imageChannels + w * imageChannels +
                //         (channels - c - 1)];
                // } else {
                value = imageData[h * width * imageChannels + w * imageChannels + c];
                // }
                // if (inputLayout === 'nchw') {
                tensor[c * width * height + h * width + w] =
                    (value - mean[c]) / std[c];
                // } 
                // else {
                //     tensor[h * width * channels + w * channels + c] =
                //         (value - mean[c]) / std[c];
                // }
            }
        }
    }

    return tensor
}