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
        When create compute graph of input and allConvs
        And build
        And compute with input
        Then get correct data

    Scenario: create compute graph of kernel fusion
        Given prepare fake input: irradiance_tensor, albedo_tensor
        And create context
        And set backend to cpu
        And create builder
        And prepare convFinal
        And create state
        And create compute graph of input
        When create compute graph of kernel fusion
        And build
        And compute with input
        Then get correct data