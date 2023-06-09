Feature: WSPK
    As a WSPK algorithm
    I want to denoise 1 spp scene image with webnn api
    So that I can get denoised scene image data

    Scenario: denoise
        Given prepare fake input: irradiance_tensor, albedo_tensor, normal_tensor, depth_tensor
        And create context
        And set backend to cpu
        And create builder
        And create state
        And load
        And build
        When compute with input
        Then get denoised scene image data
