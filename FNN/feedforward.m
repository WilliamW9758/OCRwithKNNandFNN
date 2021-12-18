function y = feedforward(x, WeightsIn)
    InLayer = WeightsIn{1};
    layers = WeightsIn{2};
    OutLayer = WeightsIn{3};
    InLayerBias = WeightsIn{4};
    layersBias = WeightsIn{5};
    OutLayerBias = WeightsIn{6};
    x = double(x');
    
    y1 = InLayer * x + InLayerBias;
    y2 = y1;
    for i = 1:size(layers, 2)
        y2 = layers{i} * y2 + layersBias{i};
    end
    y = OutLayer * y2 + OutLayerBias;
end