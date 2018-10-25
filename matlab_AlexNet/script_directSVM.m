% Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,
%
% This file is part of Handbook of Biometric Anti-Spoofing 2.

% run on GPU, set CUDA_VISIBLE_DEVICE=0
% addpath(genpath('/home/himawan/alexnet'));

%1. Extract deep features using AlexNet
net = alexnet

load('asvtrain.mat')

trainingImages.Labels = asvtrain.Labels;

layer = 'fc7';
trainingFeatures = activations(net,asvtrain,layer);
trainingLabels = asvtrain.Labels;

%2. Train SVM classifier
classiferr = fitcsvm(trainingFeatures,trainingLabels);

%3. Classify Evaluation data
load('asvevalV2.mat');

fasvevalV2 = activations(net,asvevalV2,layer);

[predictedLabels, post] = predict(classiferr,fasvevalV2);

evalProtocolFile = 'ASVspoof2017_eval_v2_key.trl.txt';
% read evaluation protocol
fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

addpath(genpath('bosaris_toolkit'));
scores = post;

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('EER is %.2f\n', EER);

% --- Classify development data

fprintf('Now dev set ...');

load('asvdev.mat');
asvdevmat = activations(net,asvdev,layer);

devProtocolFile = 'ASVspoof2017_dev.trl';
% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

[predictedLabels_dev, postdev] = predict(classiferr,asvdevmat);
scores = postdev(:,1);
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('EER is %.2f\n', EER);

