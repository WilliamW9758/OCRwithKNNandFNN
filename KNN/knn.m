function y_est = knn(X_train, y_train, X_test, k)
    [N, ~] = size(X_test);
    [M, ~] = size(X_train);
    X_train = double(X_train);
    X_test = double(X_test);
    y_est = zeros(N, 1);
    dist = zeros(N, M);
    k_nestest_labels = zeros(k, 1);
    for n = 1:N
        for m = 1:M
            dist(n, m) = norm(X_train(m, :) - X_test(n, :));
        end
        [~, index] = sort(dist(n, :));
        for idx = 1:k
            k_nestest_labels(idx) = y_train(index(idx));
        end
        y_est(n) = mode(k_nestest_labels);
    end
end