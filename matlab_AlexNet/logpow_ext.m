% Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,
%
% This file is part of Handbook of Biometric Anti-Spoofing 2.

function [xfeats] = logpow_ext(XX)

[X, fs] = audioread(XX);

[filepath, name, ext] = fileparts(XX);

% CONSTANTS ===============================================================
window_length = 16/1000; % 16 ms window
D = window_length*fs;
L = 454;
filter_length = L;
SP = 0.5; % Overlap factor
%==========================================================================

sg = buffer(X,D,ceil(SP*D),'nodelay').';
sg_filt = filter([1 -0.97],1,sg.').';

% Windowing and FFT
no_frames = size(sg_filt,1);
window = repmat(hamming(D).',no_frames,1);

segFFT = fft((sg_filt.*window),filter_length,2);
segFFT = segFFT(:,1:filter_length/2);

abssegFFT = max(abs(segFFT).^2,eps);
logsegFFT = log((abssegFFT));

feats = [logsegFFT];

feats = feats - mean(feats);
feats = feats./std(feats);

if size(feats,1) > 227
    feats = feats(1:227,:);
elseif size(feats,1) < 227/2
    temp = repmat(feats, ceil(227/size(feats,1)), 1);
    feats = temp(1:227,:);
else
    feats = repmat(feats, 2, 1);
    feats = feats(1:227,:);
end

normalizedImage = uint8(255*mat2gray(feats));
xfeats(:,:,1) = normalizedImage;
xfeats(:,:,2) = normalizedImage;
xfeats(:,:,3) = normalizedImage;

if strcmp(name(1),'T')
   [status, msg, msgID] = mkdir('train_features');
   imwrite(xfeats,strcat('train_features/',name,'.jpg'));
elseif strcmp(name(1),'E')
   [status, msg, msgID] = mkdir('eval_features');
   imwrite(xfeats,strcat('eval_features/',name,'.jpg'));
else
   [status, msg, msgID] = mkdir('dev_features');
   imwrite(xfeats,strcat('dev_features/',name,'.jpg'));
end
