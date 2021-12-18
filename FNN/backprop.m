function grad_WeightsOut = backprop(x, y, WeightsIn, map)
    InLayer = WeightsIn{1};
    layers = WeightsIn{2};
    OutLayer = WeightsIn{3};
    InLayerBias = WeightsIn{4};
    layersBias = WeightsIn{5};
    OutLayerBias = WeightsIn{6};
    yIdx = strfind(map, y);
    y = zeros(strlength(map), 1);
    y(yIdx) = 1.0;
    
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
    
    % feedforward
    activation = double(x');
    activations = cell(1,1);
    activations{1} = double(x');
    z = InLayer * activation + InLayerBias;
    zs = cell(1,1);
    zs{1} = z;
    activation = sigmoid(z)';
    activations = [activations, activation];
    for i = 1:size(layers,2)
        z = layers{i} * activation + layersBias{i};
        zs = [zs z];
        activation = sigmoid(z)';
        activations = [activations activation];
    end
    z = OutLayer * activation + OutLayerBias;
    zs = [zs z];
    activation = sigmoid(z)';
    activations = [activations activation];
    
    % backward
    delta = (activations{end} - double(y)) .* sigmoidPrime(zs{end})';
    grad_b_Out = delta;
    grad_w_Out = delta * activations{end-1}';
    for i = 1:size(layers,2)
        z = zs{end-i};
        sp = sigmoidPrime(z)';
        if i == 1
            delta = OutLayer' * delta .* sp;
        else
            delta = layers{end-i+2} * delta .* sp;
        end
        grad_b_layers{end-i+1} = delta;
        grad_w_layers{end-i+1} = delta * activations{end-i-1}';
    end
    z = zs{1};
    sp = sigmoidPrime(z)';
    delta = layers{1}' * delta .* sp;
    grad_b_In = delta;
    grad_w_In = delta * activations{1}';
    
    grad_WeightsOut = {grad_w_In, grad_w_layers, grad_w_Out, grad_b_In, ...
        grad_b_layers, grad_b_Out};
end