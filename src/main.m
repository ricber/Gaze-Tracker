%% Initializations

close all
clear
clc

% ###### YOU CAN MODIFY THE FOLLOWING VARIABLES INITIALIZATIONS ######

% DEBUG
debugImage = false; % show debug images and video
debugAllFrames = false; % this option will show debug images for each frame. 
                        % It's very computational heavy and will make matlab 
                        % unresponsive after the first frames. 
                        % Also, when this flag is true the spacial continuity 
                        % check is disabled whichever state of 
                        % the checkCorruptedCalibration flag is set.  
debugInfo = true;   % print debug info

% Webcam object
% cam = webcam(); % put here the name of your webcam
 cam = webcam('HD Pro Webcam C920');
% cam = webcam('Logitech HD Pro Webcam C920');
cam.Resolution = '640x480'; % if you change resolution modify the spacialContinuityThrsh accordingly 

% Corrupted calibration check
checkCorruptedCalibration = false;

% Tracker
reinitializationThreshold = 2; % the minimum (>=2) number of valid points below which the tracker needs to be reintialized.

% Eyes detection
spacialContinuityThrsh = 25; % maximum accepted pixel distance of same side eye center between successive frames

% Sounds for calibration
amp = 0.25;
fs = 250;  % sampling frequency
duration = 0.5;
freq = 280;
val = 0:1/fs:duration;
bip1 = amp*sin(2*pi* freq*val);
bip2 = amp*sin(2*pi* 2*freq*val);


%% Start the Loop

% DO NOT MODIFY THESE VARIABLES!
% Global variables
global leftEyeBbox;
global rightEyeBbox;
global bboxFace;
global runLoop;
runLoop = true;
firstTime = true;
calibrationCounter = 1;
testWindowDrawn = false;
hPredictors = zeros(10,3);
vPredictors = zeros(10,2);
recording = false;
calibrationDone = false;
reinitializationTrackerCounter = 0;
spacialContinuityCorrectionCounter = 0;
backToDetection = false;
framesCounter = 0;
reinitializedTracker = false;
recordingCounter = 1;
endOfPageCounter = 0;
frameSize = str2num(strrep(cam.Resolution,'x',' '));
calibrationPoints=[0 0.5; 0.25 0.5; 0.5 0.5; 0.75 0.5; 1 0.5; 0.5 0; 0.5 0.25; 0.5 0.5; 0.5 0.75; 0.5 1];

% Video Player object
if debugImage, videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]); end

% Create detector
detectorFace = vision.CascadeObjectDetector();

while runLoop
    reinitializedTracker = false;
    
    % get the next frame
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    if debugInfo
        framesCounter = framesCounter + 1;
    end
    
    % display the first video frame using the video player object
    if debugImage, videoPlayer(videoFrame); end
    
    %%---------------------------- FEATURES DETECTION -----------------------------%%
    
    % bounding box for the face
    bboxFace = detectorFace(videoFrameGray);
    
    % check if a face has been found
    if ~isempty(bboxFace)
        
        % extract face region
        faceImage = imcrop(videoFrame, bboxFace(1,:));
        
        % heuristic eyes ROIs extraction
        x1 = 0.16;                  % lateral borders width wrt faceWidth
        x2 = 1-1/(3.5)*2-x1*2;      % space in between the two eyes ROIs
        y  = 0.3;                   % upper border height wrt faceHeight
        faceWidth = bboxFace(1,3);
        faceHeight = bboxFace(1,4);
        eyeRegionWidth = faceWidth/(3.5);
        eyeRegionHeight = faceHeight/(5);
       
        % eyes bounding boxes and cropped images
        leftEyeBbox = floor([x1*faceWidth, y*faceHeight, eyeRegionWidth, eyeRegionHeight]);
        leftEyeImage = imcrop(faceImage, leftEyeBbox);
        rightEyeBbox = floor([((x1+x2)*faceWidth+eyeRegionWidth), y*faceHeight, eyeRegionWidth, eyeRegionHeight]);
        rightEyeImage = imcrop(faceImage, rightEyeBbox);
        
        eyeCenters = zeros(2); % 2-by-2 matrix containing position of eye 
                               % centers with respect to the face box in
                               % the form [x,y] with left eye at row 1 and
                               % right eye at row 2
        
        % EYE LOCALIZATION: the goal is to obtain an
        % eye map which emphasizes the iris area.
        % This is done for both eyes separately. 
        for leftRightIndex = 1:2
            if leftRightIndex == 1
                eyeImage = leftEyeImage;
            else
                eyeImage = rightEyeImage;
            end
            
      
            % RGB -> YCbCr color space
            eyeImageYcbcr = rgb2ycbcr(eyeImage);
            % Y, Cb, Cr channels are isolated and normalized in a [0,1]
            % interval
            Y =  mat2gray(eyeImageYcbcr(:,:,1)); 
            Cb = mat2gray(eyeImageYcbcr(:,:,2));
            Cr = mat2gray(eyeImageYcbcr(:,:,3));
           
            
            % EyeMapC
            
            cbCrDivision = Cb./Cr;
            mapInf = isinf(cbCrDivision);
            % we want to saturate the cbCrDivision to the maximum number (infinity excluded)
            cbCrDivision(mapInf) = 0; % this is done to return the maximum number, otherwise we would get 'inf' as the max
            cbCrDivision(mapInf) = max(cbCrDivision, [], 'all');
            % in ther case of 0/0 divion we get NaN value and we substitute
            % it with 0
            cbCrDivision(isnan(cbCrDivision)) = 0; 
            
            eyeMapC = 1/3 * ( Cb.^2 + (1-Cr).^2 + cbCrDivision );
            
            % EyeMapI
            
            irisRad = eyeRegionWidth/10;            
            % Dilation of eyeMapC with the flat circular structuring
            % element B1
            B1Rad = floor(irisRad/2);
            B1 = strel('disk', double(B1Rad));
            eyeMapCDilated = imdilate(eyeMapC, B1);
            
            % Erosion of the luminance channel with the flat circular
            % structuring element B2
            B2Rad = floor(irisRad/2); % on the paper B2Rad = floor(B1Rad/2)
            B2 = strel('disk', double(B2Rad));
            YEroded = imerode(Y, B2);
            
            delta = mean(YEroded, 'all') ;
            
            eyeMapI = eyeMapCDilated ./ (YEroded + delta);
            
            % Fast radial symmetry computation
            
            n_min = ceil(irisRad/2);
            n_max = ceil(2*irisRad); % the paper proposes 5*irisRad
            radii = n_min : n_max;
            alpha = 2; % a higher alpha eliminates nonradially symmetric features such as lines;  choosing alpha=1 minimizes the computation
            beta = 0.1; % we ignore small gradients by introducing a gradient threshold parameter beta
     
            S_luminance = mat2gray(-fastradial(YEroded, radii, alpha, beta, 'dark', 0));
            S_eyeMapI = mat2gray(fastradial(eyeMapI, radii, alpha, beta, 'bright', 0));
                        
            if (debugImage && leftRightIndex == 1 && firstTime) || debugAllFrames % debug images only for left eye
                figure('Name', 'RGB'), imshow(leftEyeImage, 'InitialMagnification', 'fit');
                figure('Name', 'Ycbcr'), imshow(eyeImageYcbcr, 'InitialMagnification', 'fit');
                figure('Name', 'Cb'), imshow(Cb, 'InitialMagnification', 'fit');
                figure('Name', 'Cr'), imshow(Cr, 'InitialMagnification', 'fit');
                figure('Name', 'EyeMapC'), imshow(eyeMapC, 'InitialMagnification', 'fit');
                figure('Name', 'EyeMapCDilated'), imshow(eyeMapCDilated, 'InitialMagnification', 'fit');
                figure('Name', 'YEroded'), imshow(YEroded, 'InitialMagnification', 'fit');
                figure('Name', 'EyeMapI'), imshow(eyeMapI, 'InitialMagnification', 'fit');
                figure('Name', 'S_luminance'), imshow(S_luminance, 'InitialMagnification', 'fit');
                figure('Name', 'S_eyeMapI'), imshow(S_eyeMapI, 'InitialMagnification', 'fit');
            end
            
            sumS = S_luminance + S_eyeMapI;
            [maxvalue, argmax] = max(sumS(:));
            [max_row, max_col] = ind2sub(size(sumS), argmax);
            
            eyeCenters(leftRightIndex, :) = [max_col, max_row];
            
            % coordinates of eye centers in the face frame
            if leftRightIndex == 1
                leftCenterInEntireFrame =  coordinatesROIConverter(eyeCenters(1, :), 'eyeLeft', 'entireFrame'); 
            else
                rightCenterInEntireFrame = coordinatesROIConverter(eyeCenters(2, :), 'eyeRight', 'entireFrame');
            end
            
            % ensure spacial continuity of eyes centers
            if firstTime || debugAllFrames
                                
                % show first face and eyes detection
                if leftRightIndex == 1
                    % initialize variables for spacial continuity
                    oldLeftEyeCenter = eyeCenters(1,:);
                    
                    % Display a box around the detected face
                    videoFrameFigure = insertShape(videoFrame, 'Rectangle', bboxFace(1,:), 'LineWidth', 1, 'Color', 'red');
                    
                    % Display a box around the detected eyes
                    videoFrameFigure = insertShape(videoFrameFigure, 'Rectangle', [coordinatesROIConverter(leftEyeBbox(1,1:2), 'face', 'entireFrame'), leftEyeBbox(1,3), leftEyeBbox(1,4)], 'LineWidth', 1, 'Color', 'cyan');
                    videoFrameFigure = insertMarker(videoFrameFigure, leftCenterInEntireFrame,'+','color','green','size',4);
                    
                    figure(1), imshow(videoFrameFigure);
                    title('Face and eyes boxes');
                else
                    % initialize variables for spacial continuity
                    oldRightEyeCenter = eyeCenters(2,:);
                    
                    videoFrameFigure = insertShape(videoFrameFigure, 'Rectangle', [coordinatesROIConverter(rightEyeBbox(1,1:2), 'face', 'entireFrame'), rightEyeBbox(1,3), rightEyeBbox(1,4)], 'LineWidth', 1, 'Color', 'cyan');
                    videoFrameFigure = insertMarker(videoFrameFigure, rightCenterInEntireFrame,'+','color','green','size',4);
                    
                    figure(1), imshow(videoFrameFigure);
                    
                    if firstTime
                        % ask if detection is fine
                        choice = questdlg({'Are the Face and the Eyes Centers detected properly?'...
                        'If not, look at the camera and start Detection Mode again.'},...
                        'Detection Mode Check','Yes','No, back to Detection','No, back to Detection');
                        hold off;

                       if strcmp(choice, 'No, back to Detection')
                            % Back to detection
                            backToDetection = true; 
                       end
                    end 
                end  
                
            elseif leftRightIndex == 1 && norm(oldLeftEyeCenter-eyeCenters(1,:))<spacialContinuityThrsh
                oldLeftEyeCenter = eyeCenters(1,:);
            elseif leftRightIndex == 2 && norm(oldRightEyeCenter-eyeCenters(2,:))<spacialContinuityThrsh
                oldRightEyeCenter = eyeCenters(2,:);
            else
                spacialContinuityCorrectionCounter = spacialContinuityCorrectionCounter + 1;
                
                 if leftRightIndex == 1
                    if debugInfo
                        fprintf('left dist: %f, spacialContinuityCorrectionCounter: %d\n', norm(oldLeftEyeCenter-eyeCenters(1,:)), spacialContinuityCorrectionCounter);
                    end
                    eyeCenters(1,:) =  oldLeftEyeCenter;
                 else
                    if debugInfo
                        fprintf('right dist: %f, spacialContinuityCorrectionCounter: %d\n', norm(oldRightEyeCenter-eyeCenters(2,:)), spacialContinuityCorrectionCounter);
                    end
                    eyeCenters(2,:) = oldRightEyeCenter; 
                 end 
            end 
        end
    else
        % if no face has been detected drop the current frame and continue to the next iteration
        continue;
    end  
    
    if backToDetection
        disp('Back to Detection Mode...')

        firstTime = true;          
        backToDetection = false;
        
        continue;
    end 
    
    
    %%---------------------------- ANCHOR POINTS TRACKING -----------------------------%%
    
  
    
   % distance between eye centers
    interocularDistance = norm(leftCenterInEntireFrame-rightCenterInEntireFrame);
    
    % Initialization of anchor points patches
    if firstTime
        kWidthEye = 0.15;
        kHeightEye = 0.05;
        kHeightInt = 0.6;
        kWidthInt = 0.25;
        leftPatchRectangle = floor([eyeCenters(1,1)+kWidthEye*eyeRegionWidth, eyeCenters(1,2)-kHeightEye*eyeRegionWidth-kHeightInt*interocularDistance/2, kWidthInt*interocularDistance, kHeightInt*interocularDistance]);
        rightPatchRectangle = floor([eyeCenters(2,1)-kWidthEye*eyeRegionWidth-kWidthInt*interocularDistance, eyeCenters(2,2)-kHeightEye*eyeRegionWidth-kHeightInt*interocularDistance/2, kWidthInt*interocularDistance, kHeightInt*interocularDistance]);
        leftPatchROI = [coordinatesROIConverter(leftPatchRectangle(1,1:2), 'eyeLeft', 'entireFrame'), leftPatchRectangle(1,3), leftPatchRectangle(1,4)];
        rightPatchROI = [coordinatesROIConverter(rightPatchRectangle(1,1:2), 'eyeRight', 'entireFrame'), rightPatchRectangle(1,3), rightPatchRectangle(1,4)];
        
        % centers of the rectangular patches
        leftCenterTracked = floor([(leftPatchROI(1,1)+leftPatchROI(1,3)/2 -1), (leftPatchROI(1,2)+leftPatchROI(1,4)/2 -1)]);
        rightCenterTracked = floor([(rightPatchROI(1,1)+rightPatchROI(1,3)/2 -1), (rightPatchROI(1,2)+rightPatchROI(1,4)/2 -1)]);
        
        % keypoints detection
        leftPoints = detectMinEigenFeatures(videoFrameGray, 'ROI', leftPatchROI);
        rightPoints = detectMinEigenFeatures(videoFrameGray, 'ROI', rightPatchROI);
        
        if debugImage
            videoFrameFigure = insertShape(videoFrameFigure, 'Rectangle', leftPatchROI, 'LineWidth', 1, 'Color', 'blue');
            videoFrameFigure = insertShape(videoFrameFigure, 'Rectangle', rightPatchROI, 'LineWidth', 1, 'Color', 'blue');
            figure(1), imshow(videoFrameFigure), hold on, title('Detected features');
            plot(leftPoints);
            plot(rightPoints);
        end
        
        % Tracker initialization
        leftPointTracker = vision.PointTracker('MaxBidirectionalError', 2);
        rightPointTracker = vision.PointTracker('MaxBidirectionalError', 2);
        
        % Initialize the tracker with the initial point locations and the initial
        % video frame
        leftPoints = leftPoints.Location;
        initialize(leftPointTracker, leftPoints, videoFrame);
        rightPoints = rightPoints.Location;
        initialize(rightPointTracker, rightPoints, videoFrame);
        
        % Convert the ROIs into a list of 4 points
        % This is needed to be able to visualize the rotation of the object.
        leftTrackedBboxPoints = bbox2points(leftPatchROI);
        rightTrackedBboxPoints = bbox2points(rightPatchROI);
   
    end
    

    for leftRightIndex = 1:2
        if(leftRightIndex == 1)
            bboxPoints = leftTrackedBboxPoints;
            points = leftPoints;
            pointTracker = leftPointTracker;
            centerTracked = leftCenterTracked;
        else
            bboxPoints = rightTrackedBboxPoints;
            points = rightPoints;
            pointTracker = rightPointTracker;
            centerTracked = rightCenterTracked;
        end
        
        oldPoints = points;
        
        % Track the points. Note that some points may be lost.
        [points, isFound] = pointTracker(videoFrame);
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        if size(visiblePoints, 1) >= reinitializationThreshold % need at least reinitializationThreshold (>=2) points
            
            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            
            % Apply the transformation to the bounding box points
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            % Apply the transformation to the center
            centerTracked = transformPointsForward(xform, centerTracked);
            
            if debugImage
                % Insert a bounding box around the object being tracked
                bboxPolygon = reshape(bboxPoints', 1, []);
                videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
                    'LineWidth', 2);
            
                % Display tracked points
                videoFrame = insertMarker(videoFrame, visiblePoints, '+', ...
                    'Color', 'white');
                videoFrame = insertMarker(videoFrame, centerTracked, '+', 'Color', 'green');
            
                % Display the annotated video frame using the video player object
                videoPlayer(videoFrame);
            end
        else
            % --- TRACKER REINITIALIZATION --- %
            
            if leftRightIndex == 1
                % keypoints detection with last bboxes
                leftPoints = detectMinEigenFeatures(videoFrameGray, 'ROI', points2bbox(bboxPoints));
                points = leftPoints.Location;
                setPoints(leftPointTracker, points);
            else
                % keypoints detection with last bboxes
                rightPoints = detectMinEigenFeatures(videoFrameGray, 'ROI', points2bbox(bboxPoints));
                points = rightPoints.Location;
                setPoints(rightPointTracker, points); 
            end          
            
            if debugInfo
                reinitializationTrackerCounter = reinitializationTrackerCounter + 1;
                
                fprintf('reinitializationCounter: %d\n', reinitializationTrackerCounter);
            end
            
            reinitializedTracker = true;
            
        end
        
        
        if(leftRightIndex == 1)
            leftPoints = points;
            leftTrackedBboxPoints = bboxPoints;
            leftCenterTracked = centerTracked;
        else
            rightPoints = points;
            rightTrackedBboxPoints = bboxPoints;
            rightCenterTracked = centerTracked;
        end
    end
    
    if reinitializedTracker
        continue;
    end
    
     %%---------------------------- FEATURES VECTOR FORMATION -----------------------------%%

     
        % --- HORIZONTAL DIRECTION FEATURES --- %
     
     % horizontal (H) distance between each eye center (EC) and the respective
     % ipsilateral anchor point (AC) 
     leftECACHdistance = norm(leftCenterTracked(1,1)-leftCenterInEntireFrame(1,1));
     rightECACHdistance = norm(rightCenterTracked(1,1)-rightCenterInEntireFrame(1,1));
     
     % horizontal (H) distance of the right eye center and the left anchor 
     % point at the opposite side 
     rightECleftACHdistance = norm(rightCenterTracked(1,1)-leftCenterInEntireFrame(1,1));
     
        % --- VERTICAL DIRECTION FEATURES --- %
     
     % vertical (V) distances between the y-coordinates of each of the eye centers and 
     % the respective anchor points
     leftECACVdistance = norm(leftCenterTracked(1,2)-leftCenterInEntireFrame(1,2));
     rightECACVdistance = norm(rightCenterTracked(1,2)-rightCenterInEntireFrame(1,2));
     
     
     
     if ~calibrationDone
         %%---------------------------- CALIBRATION -----------------------------%%
           
         % The first iteration that we are in calibration  
         if firstTime

            disp('Starting Calibration Mode...')
             
            % Draw the calibration figure 
            calibFigure = figure;
            calibFigure.ToolBar = 'none';
            calibFigure.MenuBar = 'none';
            calibFigure.WindowState = 'maximized';
            calibFigure.Units = 'Normalized';
            calibFigure.Position = [0 0 1 1];
            calibFigure.Resize = 'off';
            calibFigure.CloseRequestFcn = @my_closereq;
            
            ax = gca;
            ax.Position = [0.1 0.1 0.8 0.8];
            ax.XLim = [0 1];
            ax.YLim = [0 1];
            ax.Visible = 'off';
            hold on
            
            plot(calibrationPoints(:,1), calibrationPoints(:,2),'k.','MarkerSize',30);
         end

         
         if recordingCounter < 4 
             if recordingCounter == 1
                 hold on
                 % color the active cirle red
                 plot(calibrationPoints(calibrationCounter,1),calibrationPoints(calibrationCounter,2),'r.','MarkerSize',35);
             end
             % make one of the three bips
             sound(bip1);
             pause(0.7)
             
             % allow recording after three bips
             recordingCounter = recordingCounter + 1;
         else   
             % final bip
             sound(bip2);
             % save predictors values
             hPredictors(calibrationCounter, :) = [leftECACHdistance, rightECACHdistance, rightECleftACHdistance];
             vPredictors(calibrationCounter, :) = [leftECACVdistance, rightECACVdistance];
             % color the circle green
             plot(calibrationPoints(calibrationCounter,1),calibrationPoints(calibrationCounter,2),'g.','MarkerSize',30);
             % wait a second
             pause(1)
             
             calibrationCounter = calibrationCounter + 1;
             recordingCounter = 1;
             
             if spacialContinuityCorrectionCounter > 0 && checkCorruptedCalibration
                % Clean up
                clear cam
                if debugImage, release(videoPlayer); end
                release(detectorFace);
                release(leftPointTracker);
                release(rightPointTracker);
                
                errordlg('Calibration corrupted! Please do not close eyelids during calibration phase and make sure ambient light condition is constant. Restart the program.')
            
                error('Calibration corrupted!')
             end
             
         end
         
         if calibrationCounter > 10
             calibrationDone = true;
             
             % --- LINEAR REGRESSION MODEL --- %
             Xh = [ones(size(hPredictors, 1), 1) hPredictors(:,1) hPredictors(:,2) hPredictors(:,3)];
             Xv = [ones(size(vPredictors, 1), 1) vPredictors(:,1) vPredictors(:,2)];
             
             Yh = calibrationPoints(:,1);
             Yv = calibrationPoints(:,2);
             
             % horizontal model parameters
             H = regress(Yh,Xh);
             % vertical model parameters
             V = regress(Yv,Xv);
         end
         
     else
         %%---------------------------- PREDICTION -----------------------------%%
       
        % Set screen the first time 
        if ~testWindowDrawn
           disp('Starting Final Test...')
           
           % Draw the prediction figure
           testFigure = figure;
           testFigure.ToolBar = 'none';
           testFigure.MenuBar = 'none';
           testFigure.WindowState = 'maximized';
           testFigure.Units = 'Normalized';
           testFigure.Position = [0 0 1 1];
           testFigure.Resize = 'off';
           testFigure.CloseRequestFcn = @my_closereq;

           
           ax = gca;
           ax.Position = [0.1 0.1 0.8 0.8];
           ax.XLim = [0 1];
           ax.YLim = [0 1];
           
           axis off
           
           % Insert end of page
           rectangle('Position',[0.75,0.175,0.1,0.15],'FaceColor','red')
           
           % Insert text
           text('String', 'Read this text to let the program', 'FontUnits', 'normalized', 'FontSize', 0.07, 'Position', [0.5 0.75], 'HorizontalAlignment', 'center')
           text('String', 'estimate your eyes gaze and check', 'FontUnits', 'normalized', 'FontSize', 0.07, 'Position', [0.5 0.5], 'HorizontalAlignment', 'center')
           text('String', 'if you reached the end of page or not.', 'FontUnits', 'normalized', 'FontSize', 0.07, 'Position', [0.5 0.25], 'HorizontalAlignment', 'center')
           hold on 
           
           testWindowDrawn = true;
        end
        
        % Estimates of the two outputs with the model found in calibration
        predH=sum([1, leftECACHdistance, rightECACHdistance, rightECleftACHdistance].*H');
        predV=sum([1, leftECACVdistance, rightECACVdistance].*V');
        
        % Plot the estimated screen point
        plot(predH,predV, 'b*','MarkerSize',3);
        
        % If the estimated gaze is inside the End of Page rectangle 5
        % times the program stops
        if predV>=0.175 && predV<=0.325 && predH>=0.75 && predH<=0.85  
            endOfPageCounter = endOfPageCounter+1;
            if endOfPageCounter == 3
                disp('End of page reached. Program closes.')
                rectangle('Position',[0.75,0.175,0.1,0.15],'FaceColor','green')
                text('String', 'End of page reached!', 'FontUnits', 'normalized', 'FontSize', 0.05, 'Position', [0.7 0.1], 'HorizontalAlignment', 'center')
                pause(3);
                runLoop = false;
                continue;
            end
        end
        
     end

if firstTime
    firstTime = false;
end

end

if debugInfo
    fprintf('framesCounter: %d\n', framesCounter);
end

% Clean up
clear cam;
if debugImage, release(videoPlayer); end
release(detectorFace);
release(leftPointTracker);
release(rightPointTracker);

% this function translates coordinates from the initial smaller ROI to the final
% bigger ROI. The possible values for the ROIs are: 'eyeLeft', 'eyeRight' 'face', 'entireFrame'
% I is a m-by-2 matrix whose rows are of the form [x,y]. F is a matrix of
% tge same size with the coordinates transformed
function F = coordinatesROIConverter(I, initial, final)
global leftEyeBbox;
global rightEyeBbox;
global bboxFace;

F = zeros(size(I,1),2);

if strcmp(initial, 'eyeLeft')
    if strcmp(final, 'face')
        for n = size(I,1)
            F(n,:) = I(n,:)+[leftEyeBbox(1,1)-1, leftEyeBbox(1,2)-1];
        end
    elseif strcmp(final, 'entireFrame')
        for n = size(I,1)
            F(n,:) = I(n,:)+[leftEyeBbox(1,1)-1, leftEyeBbox(1,2)-1]+[bboxFace(1,1)-1, bboxFace(1,2)-1];
        end
    else
        error('unespected value of final ROI frame');
    end
elseif strcmp(initial, 'eyeRight')
    if strcmp(final, 'face')
        for n = size(I,1)
            F(n,:) = I(n,:)+[rightEyeBbox(1,1)-1, rightEyeBbox(1,2)-1];
        end
    elseif strcmp(final, 'entireFrame')
        for n = size(I,1)
            F(n,:) = I(n,:)+[rightEyeBbox(1,1)-1, rightEyeBbox(1,2)-1]+[bboxFace(1,1)-1, bboxFace(1,2)-1];
        end
    else
        error('unespected value of final ROI frame');
    end
elseif strcmp(initial, 'face')
    if strcmp(final, 'entireFrame')
        for n = size(I,1)
            F(n,:) = I(n,:)+[bboxFace(1,1)-1, bboxFace(1,2)-1];
        end
    else
        error('unespected value of final ROI frame');
    end
else
    error('unespected value of initial ROI frame');
end
end

% A function that takes as input 4 points of a rectangle in a 4-by-2 matrix 
% and returns them in bbox format. The inputs points are expected to be 
% ordered points of a rectangle, starting from the top left point then going clockwise
% PAY ATTENTION: the function expects a rectangle with the sides
% parallel to the x and y axis. If not, the function will return a wrong
% result. 
function bbox = points2bbox(points)

% check well formatted input as 4 points of a rectangle
% cond = points(2,1)>points(1,1) && points(1,2)==points(2,2) && points(3,2)>points(2,2) && ...
%    points(3,1)==points(2,1) && points(4,2)==points(3,2) && points(4,1)==points(1,1);
% assert(cond , 'Error in input argument (wrong format)');
bbox(1,1) = points(1,1);
bbox(1,2) = points(1,2);
bbox(1,3) = norm(points(1,:)-points(2,:));
bbox(1,4) = norm(points(1,:)-points(4,:));

end

function my_closereq(src,callbackdata)
global runLoop;
% Close request function 
% to display a question dialog box 
   selection = questdlg('Close This Figure?',...
      'Close Request Function',...
      'Yes','No','Yes'); 
   switch selection 
      case 'Yes'
         delete(gcf)
         runLoop = false;
      case 'No'
      return 
   end
end



