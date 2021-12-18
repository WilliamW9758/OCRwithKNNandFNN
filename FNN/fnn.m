function [Out, WeightsOut] = fnn(varargin)
    if nargin == 2
        % passed in X_test and weights to test
        X_test = varargin{1};
        Weights = varargin{2};
        y_est = evaluate(X_test, Weights);
        Out = y_est;
        WeightsOut = Weights;
    elseif nargin == 3
        % passed in X_train, y_train, and map to train
        Out = 2;
        WeightsOut = 2;
    elseif nargin == 5
        % passed in X_train, y_train, X_test, y_test, and map to train and comp result
        X_train = varargin{1};
        [N, In] = size(X_train);
        y_train = varargin{2};
        X_test = varargin{3};
        [M, ~] = size(X_test);
        y_test = varargin{4};
        map = varargin{5};
        Out = strlength(map);
        batch_size = 32;
        epoch = 100;
        num_layer = 1; % exclude input and output
        num_layer_node = 64;
        
        % Init weights to random numbers [-1,1]
        InLayer = 2*rand(num_layer_node, In)-1;
        InLayerBias = 2*rand(num_layer_node, 1)-1;
        layers = cell(1, num_layer);
        layersBias = cell(1, num_layer, 1);
        for n = 1:num_layer
            layers{n} = 2*rand(num_layer_node, num_layer_node)-1;
            layersBias{n} = 2*rand(num_layer_node, 1)-1;
        end
        OutLayer = 2*rand(Out, num_layer_node)-1;
        OutLayerBias = 2*rand(Out, 1)-1;
        
        Weights = {InLayer, layers, OutLayer, InLayerBias, layersBias, OutLayerBias};
        
        % train
        for i = 1:epoch
            randPerm = randperm(size(X_train, 1));
            X_train_loc = X_train(randPerm, :);
            y_train_loc = y_train(randPerm);
%             X_batches = zeros(batch_size, In, N/batch_size);
%             y_batches = zeros(batch_size, 1, N/batch_size);
            for j = 1:N/batch_size 
%                 X_batches(:,:,j) = X_train_loc((j-1)*batch_size+1: j*batch_size, :);
%                 y_batches(:,1,j) = y_train_loc((j-1)*batch_size+1: j*batch_size, :);
                Weights = updateBatch(X_train_loc((j-1)*batch_size+1: j*batch_size, :), ...
                    y_train_loc((j-1)*batch_size+1: j*batch_size, :), 0.9, Weights, map);
            end
            
            y_est = evaluate(X_test, Weights);
            mapped_y_est = double(map(y_est))';
            correct = mapped_y_est == y_test;
%             correctRate = sum(correct(:))/M;
            
            disp(['Epoch ', int2str(i), ': ', int2str(sum(correct(:))), '/', int2str(M)])
        end
        
        Out = y_est;
        WeightsOut = Weights;
    else
        Out = 0;
        WeightOut = 0;
    end
end