function generateTrainingLetters(caps, nums, isize, dir, N, disp, asData)
    if disp
        figure;
    end
    if asData
        Value = uint8(zeros(N, isize^2));
        Index = uint8(zeros(N, 1));
    end
    
    for n = 1:N
        I = zeros(isize);
        Map = 'abcdefghijklmnopqrstuvwxyz';
        if (caps)
            Map = append(Map, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ');
        end
        if (nums)
            Map = append(Map, '0987654321');
        end

%         fonts = {'Arial', 'Arial Black', 'Arial Bold', ...
%                 'Calibri', 'Calibri Light', 'Calibri Bold', ...
%                 'Times New Roman', 'Times New Roman Bold', 'Agency FB', ...
%                 'Arial Rounded MT Bold', 'Courier', 'Courier New', ...
%                 'Microsoft Himalaya', 'Microsoft JhengHei', 'Microsoft JhengHei Light', ...
%                 'Microsoft JhengHei UI', 'Microsoft JhengHei UI Light', 'Microsoft New Tai Lue', ...
%                 'Microsoft PhagsPa', 'Microsoft Sans Serif', 'Microsoft Tai Le', ...
%                 'Microsoft YaHei', 'Microsoft YaHei Light', 'Microsoft YaHei UI', ...
%                 'Microsoft YaHei UI Light', 'Microsoft Yi Baiti'};
        fonts = {'Bahnschrift', 'Baskerville Old Face', 'Bell MT', ...
            'Berlin Sans FB', 'Berlin Sans FB Demi', ...
            'Bodoni MT', 'Bodoni MT Black', ...
            'Book Antiqua', 'Bookman'};
%         fonts = listfonts;
        letterIdx = ceil(rand()*strlength(Map));

        size = uint8(isize+2);
        loc = uint8([floor(isize/2)+2, floor(isize/2)+1]);
        success = false;
        while ~success
            try
                ret = insertText(I,loc,Map(letterIdx), ...
                    'AnchorPoint','Center', 'Font', fonts{ceil(rand()*length(fonts))}, ...
                    'FontSize',size, 'BoxColor', 'Black', 'TextColor', 'White');
                ret = imrotate(ret, rand()*30-15, 'bilinear', 'crop');
                success = true;
            catch
                success = false;
            end
        end
            
        ret = rgb2gray(uint8(ret*255));
        
        if disp
            image(ret); pbaspect([1 1 1]);
            colormap(gray(256));
%             title(Map(letterIdx))
        end
        if asData
            Value(n, :) = ret(:);
            Index(n) = Map(letterIdx);
        else
            imwrite(ret, append(dir, int2str(n), '_', Map(letterIdx),'.bmp'));
        end
    end
    if asData
        save('..\LetterTrainingDataset\Testing28.mat','Value','Index','Map');
    end
end