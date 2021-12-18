function y = sigmoidPrime(x)
    y = sigmoid(x).*(1-sigmoid(x));
end