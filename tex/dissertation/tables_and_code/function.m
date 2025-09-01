%   Copyright 2010 The MathWorks, Inc.
function ObjTrack(position)
% #codegen
% First, setup the figure
numPts = 300;              % Process and plot 300 samples
figure;hold;grid;          % Prepare plot window
% Main loop
for idx = 1: numPts
    z = position(:,idx);   % Get the input data
    y = kalmanfilter(z);   % Call Kalman filter to estimate the position
    plot_trajectory(z,y);  % Plot the results
end
hold;
end   % of the function