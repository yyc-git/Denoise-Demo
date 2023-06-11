import { buildConstantByNpy, sizeOfShape } from "./common/utils"
import { build, compute, createComputeGraphOfInput, createComputeGraphOfAllConvs, createComputeGraphOfKernelFusion, createState, init } from "./wspk"
import { loadInputs } from "./input"
import conv1Weight_path from './checkpoints/open-source-test/classroom/conv1Weight.npy'
import conv2Weight_path from './checkpoints/open-source-test/classroom/conv2Weight.npy'
import conv3Weight_path from './checkpoints/open-source-test/classroom/conv3Weight.npy'
import conv4Weight_path from './checkpoints/open-source-test/classroom/conv4Weight.npy'
import conv5Weight_path from './checkpoints/open-source-test/classroom/conv5Weight.npy'
import convFinalWeight_path from './checkpoints/open-source-test/classroom/convFinalWeight.npy'


window.onload = async () => {
    let state = createState()

    let [irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor] = await loadInputs()

    state = await init(state, {
        deviceType: "gpu"
    })

    state = createComputeGraphOfInput(state,
        [1280, 720]
    )
    state = createComputeGraphOfAllConvs(state,
        [
            await buildConstantByNpy(state.builder, conv1Weight_path),
            await buildConstantByNpy(state.builder, conv2Weight_path), await buildConstantByNpy(state.builder, conv3Weight_path), await buildConstantByNpy(state.builder, conv4Weight_path), await buildConstantByNpy(state.builder, conv5Weight_path), await buildConstantByNpy(state.builder, convFinalWeight_path),
        ])
    state = createComputeGraphOfKernelFusion(state,
        [1280, 720]
    )

    state = await build(state, state.output)

    let outputBuffer = new Float32Array(sizeOfShape([1, 3, 720, 1280]));
    let results = await compute(state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, outputBuffer)

    console.log(results.outputs.output)


    //write results.outputs.output to .png
    // TODO finish
}