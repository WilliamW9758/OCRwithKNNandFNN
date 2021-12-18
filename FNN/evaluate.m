function y_est = evaluate(X_test, WeightsIn)
    y_est = zeros(size(X_test, 1), 1);
    for i = 1:size(X_test, 1)
        [~, y_est(i, 1)] = max(feedforward(X_test(i, :), WeightsIn));
    end
end