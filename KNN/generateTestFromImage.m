function generateTestFromImage(dir)
    I = imread(dir);
    I = rgb2gray(I);
    Value(1,:) = I(:);
    Index = uint8(dir(1));
    save('TestingCus.mat','Value','Index');
end