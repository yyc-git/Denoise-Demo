import { loadFeature, defineFeature } from 'jest-cucumber';
import { sizeOfShape } from '../../src/common/utils';
import { build, compute, createComputeGraphOfInput, createComputeGraphOfAllConvs, createComputeGraphOfKernelFusion, createState } from "../../src/wspk"
import { convertImageDataToFloat32Tensor } from '../utils/TensorUtils';

const feature = loadFeature('./test/features/wspk.feature');

defineFeature(feature, test => {
  let state
  let context, builder
  let results

  function _buildWeight(dimensions, value) {
    return builder.constant({ type: "float32", dimensions: dimensions },
      new Float32Array(sizeOfShape(dimensions)).fill(value)
    )
  }

  test('create compute graph of input and allConvs', ({
    given,
    and,
    when,
    then
  }) => {
    let width = 1
    let height = 2
    let irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor

    given('prepare fake input: irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor', () => {
      irradiance_tensor = convertImageDataToFloat32Tensor(
        [
          0.5, 1.0, 0.5, 0.1,
          1.0, 0.5, 0.5, 0.3,
        ],
        [1, 3, height, width]
      )
      albedo_tensor = convertImageDataToFloat32Tensor(
        [
          0.0, 0.5, 1.0, 0.1,
          1.0, 1.0, 1.0, 0.3,
        ],
        [1, 3, height, width]
      )
      normal_tensor = convertImageDataToFloat32Tensor(
        [
          1.0, 1.0, 0.0, 1.0,
          0.5, 0.5, 0.5, 1.0,
        ],
        [1, 3, height, width]
      )
      depth_tensor = convertImageDataToFloat32Tensor(
        [
          1.0, 0.0, 0.0, 1.0,
          0.5, 0.0, 0.0, 1.0
        ],
        [1, 1, height, width]
      )

      // console.log(irradiance_tensor)
    })

    and('create context', async () => {
      context = await (navigator as any).ml.createContext({
        deviceType: "cpu"
      })
    })

    and('set backend to cpu', async () => {
      let tf = context.tf
      await tf.setBackend("cpu")
      await tf.ready()
    })


    and('create builder', async () => {
      builder = new MLGraphBuilder(context)
      // console.log(builder.constant({
      //   type: "float32",
      //   dimensions: [1, 1, 2, 3]
      // },
      //   new Float32Array(6).fill(1.0)
      // ))
    })

    and('create state', async () => {
      state = createState()

      state = {
        ...state,
        context,
        builder
      }
    })

    when('create compute graph of input and allConvs', () => {
      state = createComputeGraphOfInput(state,
        [width, height]
      )
      state = createComputeGraphOfAllConvs(state,
        [
          _buildWeight(
            [14, 10, 5, 5],
            1.0
          ),
          _buildWeight(
            [14, 14, 5, 5],
            1.0
          ),
          _buildWeight(
            [14, 14, 5, 5],
            2.0
          ),
          _buildWeight(
            [14, 14, 5, 5],
            1.0
          ),
          _buildWeight(
            [14, 14, 5, 5],
            1.0
          ),
          _buildWeight(
            [12, 14, 5, 5],
            0.5
          ),
        ]
      )
    })

    and('build', async () => {
      state = await build(state, state.convFinal)
    })

    and('compute with input', async () => {
      let outputBuffer = new Float32Array(sizeOfShape([1, 12, height, width]));

      results = await compute(state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, outputBuffer)
    });

    then('get correct data', () => {
      expect(results.outputs.output).toEqual(new Float32Array([
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
        232339968,
      ]))
    });
  });

  test('create compute graph of kernel fusion', ({
    given,
    and,
    when,
    then
  }) => {
    let width = 1
    let height = 2
    let irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor
    let convFinal

    given('prepare fake input: irradiance_tensor, albedo_tensor', () => {
      irradiance_tensor = convertImageDataToFloat32Tensor(
        [
          0.5, 1.0, 0.5, 0.1,
          1.0, 0.5, 0.5, 0.3,
        ],
        [1, 3, height, width]
      )
      albedo_tensor = convertImageDataToFloat32Tensor(
        [
          0.0, 0.5, 1.0, 0.1,
          1.0, 1.0, 1.0, 0.3,
        ],
        [1, 3, height, width]
      )
      // normal_tensor = convertImageDataToFloat32Tensor(
      //   [
      //     1.0, 1.0, 0.0, 1.0,
      //     0.5, 0.5, 0.5, 1.0,
      //   ],
      //   [1, 3, height, width]
      // )
      // depth_tensor = convertImageDataToFloat32Tensor(
      //   [
      //     1.0, 0.0, 0.0, 1.0,
      //     0.5, 0.0, 0.0, 1.0
      //   ],
      //   [1, 1, height, width]
      // )

      // console.log(irradiance_tensor)

    })

    and('create context', async () => {
      context = await (navigator as any).ml.createContext({
        deviceType: "cpu"
      })
    })

    and('set backend to cpu', async () => {
      let tf = context.tf
      await tf.setBackend("cpu")
      await tf.ready()
    })


    and('create builder', async () => {
      builder = new MLGraphBuilder(context)
    })

    and('prepare convFinal', async () => {
      let dimensions = [1, 12, height, width]

      let value = new Float32Array(sizeOfShape([1, 12, height, width])).fill(1.0)
      value[0] = 0.5
      value[12] = 1.5

      convFinal = builder.constant({
        type: "float32",
        dimensions: dimensions
      }, value)
    })

    and('create state', async () => {
      state = createState()

      state = {
        ...state,
        context,
        builder,
        convFinal
      }
    })

    and('create compute graph of input', () => {
      state = createComputeGraphOfInput(state,
        [width, height]
      )
    })

    when('create compute graph of kernel fusion', () => {
      state = createComputeGraphOfKernelFusion(state,
        [width, height]
      )
    })

    and('build', async () => {
      state = await build(state, state.output)
    })

    and('compute with input', async () => {
      let outputBuffer = new Float32Array(sizeOfShape([1, 3, height, width]));

      // results = await compute(state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, outputBuffer)


      let inputs = {
        'input_irradiance': irradiance_tensor,
        'input_albedo': albedo_tensor,
        // 'input_normal': normal_tensor,
        // 'input_depth': depth_tensor,
      }
      let outputs = { 'output': outputBuffer }

      results = await state.context.compute(state.graph, inputs, outputs);
    });

    then('get correct data', () => {
      expect(results.outputs.output).toEqual(new Float32Array([
        0,
        0.7602049112319946,
        0.36740824580192566,
        0.7397950887680054,
        0.5,
        0.5,
      ]))
    });
  });
});
