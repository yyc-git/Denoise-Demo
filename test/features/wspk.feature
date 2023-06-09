Feature: WSPK
    As a WSPK algorithm
    I want to denoise 1 spp scene image with webnn api
    So that I can get denoised scene image data

    Scenario: create compute graph of input and allConvs
        Given prepare fake input: irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor
        And create context
        And set backend to cpu
        And create builder
        And create state
        And create compute graph of input and allConvs
        And build
        When compute with input
        Then get correct data

# TODO add 
#     Scenario: create compute graph of kernel fusion
#     first test simple data