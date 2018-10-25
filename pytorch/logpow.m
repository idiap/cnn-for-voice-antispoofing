% Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,
%
% This file is part of Handbook of Biometric Anti-Spoofing 2.

function [feats] = logpow(X, fs)
% Overlap-and-add

% CONSTANTS ===============================================================
window_length = 16/1000; % 16 ms window
D = window_length*fs; % samples_block (no samples per block) = L
L = 1536;
filter_length = L;
SP = 0.5; % Overlap factor
inc = L - ceil(SP*L); % Number of advance samples
%==========================================================================

% Segments -> no_frames x Length  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sg = buffer(X,D,ceil(SP*D),'nodelay').';
sg_filt = filter([1 -0.97],1,sg.').';

% Windowing and FFT
no_frames = size(sg_filt,1);
window = repmat(hamming(D).',no_frames,1);

segFFT = fft((sg_filt.*window),filter_length,2);

%After fft, only take half of it
segFFT = segFFT(:,1:filter_length/2);

abssegFFT = max(abs(segFFT).^2,eps);
logsegFFT = log((abssegFFT));

feats = [logsegFFT];
