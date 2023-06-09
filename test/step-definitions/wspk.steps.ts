import { loadFeature, defineFeature } from 'jest-cucumber';
import { sizeOfShape } from '../../src/common/utils';
import { build, compute, createState, load } from "../../src/wpsk"
import { convertImageDataToFloat32Tensor } from '../utils/TensorUtils';

const feature = loadFeature('./test/features/wspk.feature');

defineFeature(feature, test => {
  test('denoise', ({
    given,
    and,
    when,
    then
  }) => {
    let state
    let context, builder
    let width = 1
    let height = 2
    let irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor
    let results

    function _buildWeight(dimensions, value) {
      return builder.constant({ type: "float32", dimensions: dimensions },
        new Float32Array(sizeOfShape(dimensions)).fill(value)
      )
    }

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

      // console.log(depth_tensor)
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

    and('load', () => {
      state = load(state,
        [width, height],
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
      state = await build(state, state.output)
    })

    when('compute with input', async () => {
      let outputBuffer = new Float32Array(sizeOfShape([1, 3, height, width]));

      results = await compute(state, irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor, outputBuffer)
    });

    then('get denoised scene image data', () => {
      //TODO finish
      expect(results.outputs.output).toEqual(new Float32Array([]))
    });
  });
});
