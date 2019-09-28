% FASTRADIAL - Loy and Zelinski's fast radial feature detector
%
% An implementation of Loy and Zelinski's fast radial feature detector
%
% Usage: S = fastradial(im, radii, alpha, beta)
%
% Arguments:
%            im    - Image to be analysed
%            radii - Array of integer radius values to be processed
%                    suggested radii might be [1 3 5]
%            alpha - Radial strictness parameter.
%                    1 - slack, accepts features with bilateral symmetry.
%                    2 - a reasonable compromise.
%                    3 - strict, only accepts radial symmetry.
%                        ... and you can go higher
%            beta  - Gradient threshold.  Gradients below this threshold do
%                    not contribute to symmetry measure, defaults to 0.
%
% Returns    S     - Symmetry map.  Bright points with high symmetry are
%                    marked with large positive values. Dark points of
%                    high symmetry marked with large -ve values.
%
% To localize points use NONMAXSUPPTS on S, -S or abs(S) depending on
% what you are seeking to find.

% Reference:
% Loy, G.  Zelinsky, A.  Fast radial symmetry for detecting points of
% interest.  IEEE PAMI, Vol. 25, No. 8, August 2003. pp 959-973.

% Copyright (c) 2004-2010 Peter Kovesi
% www.peterkovesi.com/matlabfns/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% November 2004  - original version
% July     2005  - Bug corrected: magitude and orientation matrices were
%                  not zeroed for each radius value used (Thanks to Ben
%                  Jackson) 
% December 2009  - Gradient threshold added + minor code cleanup
% July     2010  - Gradients computed via Farid and Simoncelli's 5 tap
%                  derivative filters

function S = fastradial(im, radii, alpha, beta, mode, orient)
    
    if ~exist('beta','var'), beta = 0;  end
    if ~exist('mode','var'), mode = 'both'; end
    if ~exist('orient','var'), orient = 0; end
    
    if any(radii ~= round(radii)) || any(radii < 1)
        error('radii must be integers and > 1')
    end
    
    bright = false;
    dark = false;
    if (strcmp(mode, 'bright'))
        bright = true;
    elseif (strcmp(mode, 'dark'))
        dark = true;
    elseif (strcmp(mode, 'both'))
        bright = true;
        dark = true;
    else
        error('invalid mode');
    end
    
    [rows,cols] = size(im);
    
    % Compute derivatives in x and y via Farid and Simoncelli's 5 tap
    % derivative filters
    [imgx, imgy] = derivative5(im, 'x', 'y');
    mag = sqrt(imgx.^2 + imgy.^2)+eps; % (+eps to avoid division by 0)
    
    % Normalise gradient values so that [imgx imgy] form unit 
    % direction vectors.
    imgx = imgx./mag;   
    imgy = imgy./mag;
    
    S = zeros(rows,cols);  % Symmetry matrix    
    
    [x,y] = meshgrid(1:cols, 1:rows);
    
    for n = radii
        if ~orient, M = zeros(rows,cols); end  % Magnitude projection image
        O = zeros(rows,cols);  % Orientation projection image
        
        if (bright)
            % Coordinates of 'positively' and 'negatively' affected pixels
            posx = x + round(n*imgx);
            posy = y + round(n*imgy);
            
            % Clamp coordinate values to range [1 rows 1 cols]
            posx( posx<1 )    = 1;
            posx( posx>cols ) = cols;
            posy( posy<1 )    = 1;
            posy( posy>rows ) = rows;
        end
        
        if (dark)
            % Coordinates of 'positively' and 'negatively' affected pixels
            negx = x - round(n*imgx);
            negy = y - round(n*imgy);
        
            % Clamp coordinate values to range [1 rows 1 cols]
            negx( negx<1 )    = 1;
            negx( negx>cols ) = cols;
            negy( negy<1 )    = 1;
            negy( negy>rows ) = rows;
        end
        
        % Form the orientation and magnitude projection matrices
        for r = 1:rows
            for c = 1:cols
                if mag(r,c) > beta
                    if (bright)
                        O(posy(r,c),posx(r,c)) = O(posy(r,c),posx(r,c)) + 1;
                        if ~orient, M(posy(r,c),posx(r,c)) = M(posy(r,c),posx(r,c)) + mag(r,c); end
                        
                    end
                    
                    if (dark)
                        O(negy(r,c),negx(r,c)) = O(negy(r,c),negx(r,c)) - 1;
                        if ~orient, M(negy(r,c),negx(r,c)) = M(negy(r,c),negx(r,c)) - mag(r,c); end
                    end
                end
            end
        end
        
        % Clamp Orientation projection matrix values to a maximum of 
        % +/-kappa,  but first set the normalization parameter kappa to the
        % values suggested by Loy and Zelinski
        if n == 1, kappa = 8; else kappa = 9.9; end
        
        if bright, O(O >  kappa) =  kappa; end
        if dark, O(O < -kappa) = -kappa; end
        
        % Unsmoothed symmetry measure at this radius value
        if (~orient)
            F = M./kappa .* (abs(O)/kappa).^alpha;
        else
            F = sign(O) .* (abs(O)/kappa).^alpha; % Orientation only based measure
        end  
        
        % Smooth and spread the symmetry measure with a Gaussian proportional to
        % n.  Also scale the smoothed result by n so that large scales do not
        % lose their relative weighting.
        S = S + gaussfilt(F, 0.25*n) * n; 

        
    end  % for each radius
    
    S  = S /length(radii);  % Average