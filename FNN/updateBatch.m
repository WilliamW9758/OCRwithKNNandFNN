function WeightsOut = updateBatch(X_batch, y_batch, eta, WeightsIn, map)
    InLayer = WeightsIn{1};
    layers = WeightsIn{2};
    OutLayer = WeightsIn{3};
    InLayerBias = WeightsIn{4};
    layersBias = WeightsIn{5};
    OutLayerBias = WeightsIn{6};
    
    grad_b_In = zeros(size(InLayerBias));
    grad_b_layers = cell(size(layersBias));
    for i = 1:size(layersBias, 2)
        grad_b_layers{i} = zeros(size(layersBias{i}));
    end
    grad_b_Out = zeros(size(OutLayerBias));
    
    grad_w_In = zeros(size(InLayer));
    grad_w_layers = cell(size(layers));
    for i = 1:size(layers, 2)
        grad_w_layers{i} = zeros(size(layers{i}));
    end
    grad_w_Out = zeros(size(OutLayer));
    
    for x = 1:size(X_batch, 1)
        grad_Weights = backprop(X_batch(x, :), y_batch(x, :), WeightsIn, map);
        
        grad_w_In = grad_w_In + grad_Weights{1};
        for i = 1:size(layers, 2)
            grad_w_layers{i} = grad_w_layers{i} + grad_Weights{2}{i};
        end
        grad_w_Out = grad_w_Out + grad_Weights{3};
        grad_b_In = grad_b_In + grad_Weights{4};
        for i = 1:size(layersBias, 2)
            grad_b_layers{i} = grad_b_layers{i} + grad_Weights{5}{i};
        end
        grad_b_Out = grad_b_Out + grad_Weights{6};
    end
    
    InLayer = InLayer - (eta/size(X_batch,1)) * grad_w_In;
    for i = 1:size(layers, 2)
        layers{i} = layers{i} - (eta/size(X_batch,1)) * grad_w_layers{i};
    end
    OutLayer = OutLayer - (eta/size(X_batch,1)) * grad_w_Out;
    InLayerBias = InLayerBias - (eta/size(X_batch,1)) * grad_b_In;
    for i = 1:size(layersBias, 2)
        layersBias{i} = layersBias{i} - (eta/size(X_batch,1)) * grad_b_layers{i};
    end
    OutLayerBias = OutLayerBias - (eta/size(X_batch,1)) * grad_b_Out;
    
    WeightsOut = {InLayer, layers, OutLayer, InLayerBias, layersBias, OutLayerBias};
end