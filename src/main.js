import '@webmachinelearning/webnn-polyfill'
import { buildConstantByNpy, getInputTensor, setBackend, sizeOfShape } from "./common/utils"
import irradiance_img_path from './dataset/color0.png'
import albedo_img_path from './dataset/albedo0.png'
import depth_img_path from './dataset/depth0.png'
import normal_img_path from './dataset/shading_normal0.png'
import conv1Weight_path from './checkpoints/open-source-test/classroom/conv1Weight.npy'
import conv2Weight_path from './checkpoints/open-source-test/classroom/conv2Weight.npy'
import conv3Weight_path from './checkpoints/open-source-test/classroom/conv3Weight.npy'
import conv4Weight_path from './checkpoints/open-source-test/classroom/conv4Weight.npy'
import conv5Weight_path from './checkpoints/open-source-test/classroom/conv5Weight.npy'
import convFinalWeight_path from './checkpoints/open-source-test/classroom/convFinalWeight.npy'


//create state

function _createState() {
    return {
        context: null,
        builder: null,
        graph: null,
        output: null
    }
}





//load inputs


// function getInputFromCanvas() {
//     digitContext.clearRect(0, 0, digitCanvas.width, digitCanvas.height)
//     digitContext.drawImage(
//         visualCanvas, 0, 0, digitCanvas.width, digitCanvas.height)
//     let imageData =
//         digitContext.getImageData(0, 0, digitCanvas.width, digitCanvas.height)
//     let input = new Float32Array(digitCanvas.width * digitCanvas.height)
//     for (let i = 0 i < input.length i++) {
//         input[i] = imageData.data[i * 4]
//     }
//     return input
// }


function _loadImage(url) {
    let image = new Image()
    image.src = url

    return new Promise((resolve) => {
        image.onload = () => {
            resolve(image)
        }
    })
}

async function _loadInputs(state) {
    // TODO gamma correction
    let irradiance_tensor = getInputTensor(await _loadImage(irradiance_img_path), {
        // TODO error?
        inputDimensions: [1, 3, 720, 1280],
        inputLayout: 'nchw',
        norm: true
    })

    let albedo_tensor = getInputTensor(await _loadImage(albedo_img_path), {
        inputDimensions: [1, 3, 720, 1280],
        inputLayout: 'nchw',
        norm: true
    })

    let normal_tensor = getInputTensor(await _loadImage(normal_img_path), {
        inputDimensions: [1, 3, 720, 1280],
        inputLayout: 'nchw',
        norm: true
    })

    // TODO normal_tensor = normal_tensor.map(value => value * 0.5 + 0.5)?

    let depth_tensor = getInputTensor(await _loadImage(depth_img_path), {
        inputDimensions: [1, 1, 720, 1280],
        inputLayout: 'nchw',
        norm: true
    })

    // TODO depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))?

    // console.log(depth_tensor)


    return [irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor]


    ////padding
    //concat



    ////shape: [1,10,height, width]
}



////uniform crop

//build network

async function _buildConv(builder, input, weightNpyFilePath) {
    let weight = await buildConstantByNpy(builder, weightNpyFilePath)

    return builder.conv2d(
        input,
        weight,
        {
            activation: builder.leakyRelu(0.1),
            padding: [2, 2, 2, 2],
            strides: [1, 1]
        }
    )
}

function _buildConvS(input, i) {
    let kernel_base_size = 3
    let kernel_size_stride = 2

    let kernel_size = kernel_base_size + i * kernel_size_stride
    let padding = (kernel_size - 1) / 2

    let weightDimension = [1, 1, kernel_size, kernel_size]

    //shape: [1, 1, kernel_size, kernel_size]
    return builder.conv2d(
        input,
        builder.constant(
            { type: 'float32', dimensions: weightDimension },
            new Float32Array(sizeOfShape(weightDimension)).fill(1.0)
        ),
        {
            padding: [padding, padding, padding, padding],
            strides: [1, 1]
        }
    )
}

function _softmaxNCHW4DTensor(builder, tensor, axis) {
    let max_x = builder.reduceMax(tensor, { axes: [axis], keepDimensions: true });
    let exp_x = builder.exp(builder.sub(x, max_x));

    return builder.div(exp_x, builder.reduceSum(exp_x, { axes: [axis], keepDimensions: true }));
}

async function _kernelFusion(builder, x_irradiance, x_albedo, convFinalOutput) {
    let kernel_num = 6

    let x_guidemap = builder.exp(
        builder.slice(convFinalOutput, [0], [kernel_num], { axes: [2] })
    )
    let x_alpha = _softmaxNCHW4DTensor(builder, builder.slice(convFinalOutput, [kernel_num], [kernel_num], { axes: [2] }), 2)



    // TODO optimize
    // refer to https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840

    // let outputDimension = [1, 3, 720, 1280]
    let outputSingleChannelDimension = [1, 1, 720, 1280]
    let x_out_r = builder.constant(
        { type: 'float32', dimensions: outputSingleChannelDimension },
        new Float32Array(sizeOfShape(outputSingleChannelDimension)).fill(0.0)
    )

    for (i = 0; i++; i < kernel_num) {
        // [1,1,720,1280]
        let x_guidemap_windowsum = _buildConvS(builder.slice(x_guidemap, [i], [1], { axes: [2] }), i)

        x_out_r = builder.add(
            x_out_r,
            builder.mul(
                builder.slice(x_alpha, [i], [1], { axes: [2] }),
                builder.div(
                    _buildConvS(
                        builder.mul(
                            builder.slice(x_guidemap, [i], [1], { axes: [2] }),
                            builder.slice(x_irradiance, [0], [1], { axes: [2] })
                        ), i),
                    x_guidemap_windowsum
                )

            )
        )
    }


    let x_out_g = builder.constant(
        { type: 'float32', dimensions: outputSingleChannelDimension },
        new Float32Array(sizeOfShape(outputSingleChannelDimension)).fill(0.0)
    )

    for (i = 0; i++; i < kernel_num) {
        // [1,1,720,1280]
        let x_guidemap_windowsum = _buildConvS(builder.slice(x_guidemap, [i], [1], { axes: [2] }), i)

        x_out_g = builder.add(
            x_out_g,
            builder.mul(
                builder.slice(x_alpha, [i], [1], { axes: [2] }),
                builder.div(
                    _buildConvS(
                        builder.mul(
                            builder.slice(x_guidemap, [i], [1], { axes: [2] }),
                            builder.slice(x_irradiance, [1], [1], { axes: [2] })
                        ), i),
                    x_guidemap_windowsum
                )

            )
        )
    }




    let x_out_b = builder.constant(
        { type: 'float32', dimensions: outputSingleChannelDimension },
        new Float32Array(sizeOfShape(outputSingleChannelDimension)).fill(0.0)
    )

    for (i = 0; i++; i < kernel_num) {
        // [1,1,720,1280]
        let x_guidemap_windowsum = _buildConvS(builder.slice(x_guidemap, [i], [1], { axes: [2] }), i)

        x_out_b = builder.add(
            x_out_b,
            builder.mul(
                builder.slice(x_alpha, [i], [1], { axes: [2] }),
                builder.div(
                    _buildConvS(
                        builder.mul(
                            builder.slice(x_guidemap, [i], [1], { axes: [2] }),
                            builder.slice(x_irradiance, [2], [1], { axes: [2] })
                        ), i),
                    x_guidemap_windowsum
                )
            )
        )
    }

    //[1,3,720,1280]
    let x_out = builder.concat([x_out_r, x_out_g, x_out_b], 2)

    x_out = builder.mul(x_out, x_albedo)

    return x_out
}

async function _load(state, contextOptions, irradiance_tensor, albedo_tensor) {
    let context = await navigator.ml.createContext(contextOptions)

    let tf = context.tf
    //TODO really use webgpu? or just webgl?
    await tf.setBackend("webgpu")
    await tf.ready()

    let builder = new MLGraphBuilder(context)

    let input_irradianceShape = [1, 3, 720, 1280]
    let input_irradiance = builder.input('input_irradiance', { type: 'float32', dimensions: input_irradianceShape })

    let input_albedoShape = [1, 3, 720, 1280]
    let input_albedo = builder.input('input_albedo', { type: 'float32', dimensions: input_albedoShape })

    let input_normalShape = [1, 3, 720, 1280]
    let input_normal = builder.input('input_normal', { type: 'float32', dimensions: input_normalShape })

    let input_depthShape = [1, 1, 720, 1280]
    let input_depth = builder.input('input_depth', { type: 'float32', dimensions: input_depthShape })


    //shape: [1,10,720,1280]
    let input = builder.concat([input_irradiance, input_albedo, input_normal, input_depth], 2)


    let conv1 = await _buildConv(builder, input, conv1Weight_path)
    let conv2 = await _buildConv(builder, conv1, conv2Weight_path)
    let conv3 = await _buildConv(builder, conv2, conv3Weight_path)
    let conv4 = await _buildConv(builder, conv3, conv4Weight_path)
    let conv5 = await _buildConv(builder, conv4, conv5Weight_path)
    let convFinal = await _buildConv(builder, conv5, convFinalWeight_path)


    let x_out = _kernelFusion(builder, irradiance_tensor, albedo_tensor, convFinal)


    return {
        ...state,
        context,
        builder,
        output: x_out
    }
}



async function _build(state, outputOperand) {
    let graph = await state.builder.build({ 'output': outputOperand })

    return {
        ...state,
        graph
    }
}


async function _compute(state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, output) {
    let inputs = {
        'input_irradiance': irradiance_tensor,
        'input_albedo': albedo_tensor,
        'input_normal': normal_tensor,
        'input_depth': depth_tensor,
    }
    let outputs = { 'output': output }
    let results = await state.context.compute(state.graph, inputs, outputs);

    return results;
}




// load .npy

//inference with input


//write output to .png






window.onload = async () => {
    let state = _createState()

    let [irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor] = await _loadInputs()

    state = await _load(state, {
        deviceType: "gpu"
    }, irradiance_tensor, albedo_tensor)


    state = await _build(state, state.output)

    let outputBuffer = new Float32Array(utils.sizeOfShape([1, 3, 720, 1280]));
    let results = await _compute(state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, outputBuffer)


    //write results.outputs.output to .png
    // TODO finish
}