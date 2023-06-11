import '@webmachinelearning/webnn-polyfill'
import { buildConstantByNpy, sizeOfShape } from "./common/utils"

//create state

export let createState = () => {
    return {
        context: null,
        builder: null,
        graph: null,
        input_irradiance: null,
        input_albedo: null,
        input_normal: null,
        input_depth: null,
        convFinal: null,
        output: null
    }
}





//load inputs


// let getInputFromCanvas= () => {
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





////uniform crop

//build network

let _buildConv = (builder, input, weight) => {
    // let weight = await buildConstantByNpy(builder, weightNpyFilePath)

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

let _buildConvS = (builder, input, i) => {
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

let _softmaxNCHW4DTensor = (builder, tensor, axis) => {
    let max_x = builder.reduceMax(tensor, { axes: [axis], keepDimensions: true });
    let exp_x = builder.exp(builder.sub(tensor, max_x));

    return builder.div(exp_x, builder.reduceSum(exp_x, { axes: [axis], keepDimensions: true }));
}

let _computeForOneChannelInIrradiance = (builder, [width, height], channelIndex, kernel_num, x_guidemap, x_alpha, input_irradiance) => {
    let outputSingleChannelDimension = [1, 1, height, width]
    let x_out = builder.constant(
        { type: 'float32', dimensions: outputSingleChannelDimension },
        new Float32Array(sizeOfShape(outputSingleChannelDimension)).fill(0.0)
    )
    for (let i = 0; i < kernel_num; i++) {
        // [1,1,height,width]
        let x_guidemap_windowsum = _buildConvS(builder, builder.slice(x_guidemap, [i], [1], { axes: [1] }), i)

        x_out = builder.add(
            x_out,
            builder.mul(
                builder.slice(x_alpha, [i], [1], { axes: [1] }),
                builder.div(
                    _buildConvS(builder,
                        builder.mul(
                            builder.slice(x_guidemap, [i], [1], { axes: [1] }),
                            builder.slice(input_irradiance, [channelIndex], [1], { axes: [1] })
                        ), i),
                    x_guidemap_windowsum
                )

            )
        )
    }

    return x_out
}

let _kernelFusion = (builder, [width, height], input_irradiance, input_albedo, convFinalOutput) => {
    let kernel_num = 6

    let x_guidemap = builder.exp(
        builder.slice(convFinalOutput, [0], [kernel_num], { axes: [1] })
    )

    let x_alpha = _softmaxNCHW4DTensor(builder, builder.slice(convFinalOutput, [kernel_num], [kernel_num], { axes: [1] }), 1)

    // TODO optimize
    // refer to https://discuss.pytorch.org/t/applying-conv2d-filter-to-all-channels-seperately-is-my-solution-efficient/22840

    let x_out_r = _computeForOneChannelInIrradiance(builder, [width, height], 0, kernel_num, x_guidemap, x_alpha, input_irradiance)
    let x_out_g = _computeForOneChannelInIrradiance(builder, [width, height], 1, kernel_num, x_guidemap, x_alpha, input_irradiance)
    let x_out_b = _computeForOneChannelInIrradiance(builder, [width, height], 2, kernel_num, x_guidemap, x_alpha, input_irradiance)

    //[1,3,height,width]
    let x_out = builder.concat([x_out_r, x_out_g, x_out_b], 1)

    x_out = builder.mul(x_out, input_albedo)

    return x_out
}

export let init = async (state, contextOptions) => {
    let context = await (navigator as any).ml.createContext(contextOptions)

    let tf = context.tf
    //TODO really use webgpu? or just webgl?
    await tf.setBackend("webgpu")
    await tf.ready()

    let builder = new MLGraphBuilder(context)

    return {
        ...state,
        context,
        builder,
    }
}

export let createComputeGraphOfInput = (state, [width, height]) => {
    let { builder } = state

    let input_irradianceShape = [1, 3, height, width]
    let input_irradiance = builder.input('input_irradiance', { type: 'float32', dimensions: input_irradianceShape })

    let input_albedoShape = [1, 3, height, width]
    let input_albedo = builder.input('input_albedo', { type: 'float32', dimensions: input_albedoShape })

    let input_normalShape = [1, 3, height, width]
    let input_normal = builder.input('input_normal', { type: 'float32', dimensions: input_normalShape })

    let input_depthShape = [1, 1, height, width]
    let input_depth = builder.input('input_depth', { type: 'float32', dimensions: input_depthShape })

    return {
        ...state,
        input_irradiance,
        input_albedo,
        input_normal,
        input_depth,
    }
}

export let createComputeGraphOfAllConvs = (state, [conv1Weight, conv2Weight, conv3Weight, conv4Weight, conv5Weight, convFinalWeight]) => {
    let {
        builder,
        input_irradiance,
        input_albedo,
        input_normal,
        input_depth,
    } = state

    //shape: [1,10,height,width]
    let input = builder.concat([input_irradiance, input_albedo, input_normal, input_depth], 1)

    // console.log(input)
    // console.log(conv1Weight)

    let conv1 = _buildConv(builder, input, conv1Weight)
    let conv2 = _buildConv(builder, conv1, conv2Weight)
    let conv3 = _buildConv(builder, conv2, conv3Weight)
    let conv4 = _buildConv(builder, conv3, conv4Weight)
    let conv5 = _buildConv(builder, conv4, conv5Weight)
    let convFinal = _buildConv(builder, conv5, convFinalWeight)

    return {
        ...state,
        convFinal
    }
}

export let createComputeGraphOfKernelFusion = (
    state, [width, height]
) => {
    return {
        ...state,
        output: _kernelFusion(state.builder, [width, height], state.input_irradiance, state.input_albedo, state.convFinal)
    }
}

export let build = async (state, outputOperand) => {
    let graph = await state.builder.build({ 'output': outputOperand })

    return {
        ...state,
        graph
    }
}

export let compute = async (state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, output) => {
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

