% Copyright (c) 2017 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,

% Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
% Modified by - Srikanth Madikeri
% Modified by - Milos Cernak

% TRAIN
basedir = './'
filelist = 'filelist_train'
%
fileID = fopen(filelist);
tmp = textscan(fileID, '%s');
fclose(fileID);
%
warning off;
%% add required libraries to the path
addpath(genpath('functions'));
%
filenames = tmp{1};
%
j=0;
k=1;
cleared = 0;
for i = 1:length(filenames)
    [pathstr_w,name_w,ext_w] = fileparts(filenames{i});
    j = j + 1;
    [x,fs] = audioread(filenames{i});
    feats = logpow(x,fs)';
    total{j} = reshape(feats',1,[]);
    cleared = 1;
    if mod(j, 1000) == 0
        save(strcat(basedir, 'normallogpower_train_', num2str(k), '.mat'), '-v7.3','total');
        k = k+1;
        j = 0;  
        cleared = 0;
    end    
end

if cleared == 1
        save(strcat(basedir,'normallogpower_train_', num2str(k), '.mat'), '-v7.3','total');
end

% EVAL
filelist = 'filelist_eval'

fileID = fopen(filelist);
tmp = textscan(fileID, '%s');
fclose(fileID);

warning off;
% add required libraries to the path
addpath(genpath('functions'));

filenames = tmp{1};

j=0;
k=1;
cleared = 0;
for i = 1:length(filenames)
    [pathstr_w,name_w,ext_w] = fileparts(filenames{i});
    j = j + 1;
    [x,fs] = audioread(filenames{i});
    feats = logpow(x,fs)';
    total{j} = reshape(feats',1,[]);
    cleared = 1;
    if mod(j, 1000) == 0
        save(strcat(basedir,'normallogpower_eval_', num2str(k), '.mat'), '-v7.3','total');
        k = k+1;
        j = 0;  
        cleared = 0;
        clear total
    end    
end

if cleared == 1
        save(strcat(basedir,'normallogpower_eval_', num2str(k), '.mat'), '-v7.3','total');
end
clear


filelist = 'filelist_dev'

fileID = fopen(filelist);
tmp = textscan(fileID, '%s');
fclose(fileID);

warning off;
% add required libraries to the path
addpath(genpath('functions'));

filenames = tmp{1};

j=0;
k=1;
cleared = 0;
for i = 1:length(filenames)
    [pathstr_w,name_w,ext_w] = fileparts(filenames{i});
    j = j + 1;
    [x,fs] = audioread(filenames{i});
    feats = logpow(x,fs)';
    total{j} = reshape(feats',1,[]);
    cleared = 1;
    if mod(j, 1000) == 0
        save(strcat(basedir,'normallogpower_dev_', num2str(k), '.mat'), '-v7.3','total');
        k = k+1;
        j = 0;  
        cleared = 0;
        clear total
    end    
end

if cleared == 1
        save(strcat(basedir,'normallogpower_dev_', num2str(k), '.mat'), '-v7.3','total');
end
clear
%
