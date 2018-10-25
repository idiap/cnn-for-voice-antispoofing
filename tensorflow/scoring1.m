% Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,
%
% This file is part of Handbook of Biometric Anti-Spoofing 2.

function scoring

warning off;
% add required libraries to the path
addpath(genpath('bosaris_toolkit'));

keyProtocolFile = fullfile('KEY_ASVspoof2017_eval_V2.trl.txt');
fileID = fopen(keyProtocolFile);
keyprotocol = textscan(fileID, '%s%s');
fclose(fileID);
labels = keyprotocol{2};

sco = csvread('temp.csv');
scores = log(sco+eps)-log(1-sco+eps);

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('EER is %.2f\n', EER);
exit
