import { getInputTensor } from "./common/utils"
import irradiance_img_path from './dataset/color0.png'
import albedo_img_path from './dataset/albedo0.png'
import depth_img_path from './dataset/depth0.png'
import normal_img_path from './dataset/shading_normal0.png'

let _loadImage = (url) => {
    let image = new Image()
    image.src = url

    return new Promise((resolve) => {
        image.onload = () => {
            resolve(image)
        }
    })
}

export let loadInputs = async () => {
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