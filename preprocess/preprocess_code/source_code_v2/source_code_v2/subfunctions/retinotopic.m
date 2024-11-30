%% Calculate the eccentricity and polar angle of a location given its index in 2D image space
% inputs:
%   idx: indeces of locations in the 2D image

function retin = retinotopic(idx,s)
    if isempty(s)
        s = 55; % size of the 2D image
    end
    ct = round(s/2); % distance to the center location
    retin = zeros(size(idx,1)/2,2); % the first column is eccentricity, the second is polar angle 
    for i = 1 : size(retin,1)
        num = sum(idx((i-1)*2+1,:)>0);
        xy = idx((i-1)*2+1:i*2,1:num);
        xy = bsxfun(@minus,xy,[ct;ct]);
        xy(2,:) = -xy(2,:);
        d = sqrt(sum(xy.^2)); % relative distance to the screen center

        % eccentricity
        retin(i,1) = atan(mean(d)*0.1854/ct); % mean eccentricity of locations

        % quantify polar angle: use sin value with respect to the vertical axis 
        c = xy(1,:) ./ d;
        c(isnan(c)) = [];
        retin(i,2) =   mean(c);
    end
end