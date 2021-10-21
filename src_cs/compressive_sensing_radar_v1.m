%clear, close all, clc;
function compressive_sensing_radar_v1(scene,meas) 
myDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene', string(scene),'/radar'));
saveDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene', string(scene),'/radar-recons-sparse-40-', meas));
myFiles = dir(fullfile(myDir,'*.png'));
meas
saveDir
parallel = 12;
outer_loop = [1:parallel:length(myFiles)];


for p = 1:length(outer_loop)
recons_full = [];
snrs_full = [];
MAE_full = [];
saveFileNameFull = [];
%if p == length(outer_loop)
%    parallel = 2;
%end
parfor k = 1: parallel
    current = outer_loop(p) + (k-1);
    baseFileName = myFiles(current).name;
    fullFileName = fullfile(myDir, baseFileName);
    disp(fullFileName)
    A = imread(fullFileName);
    baseFileName = strrep(baseFileName, '.png', '.mat');
    meta = A(:, 1:11); 
    A = A(:,12:3779);
    [height, width] = size(A);
    %A = A([1:50],[12:87]);
    w = 25;
    h = 112 - 12; %87 - 12;
    rate = double(int16((w*h)*0.1));
    snrs = [];
    MAEs = [];
    rows = [1: w: 425];
    columns = [1: h: 3779];
    final_A = [];
    final_rate = 0;
    for c = 1:16
        c,rate;
        final_A_column = [];
        for d = 1:37
	    final_rate = final_rate + rate;
            %final_rate, rate
            A_ = A([rows(c):rows(c+1)-1],[columns(d):columns(d+1)-1]);
	    %continue
            x1 = compressed_sensing_example(A_, w, h, rate, meas);
            x1 = uint8(x1);
            peak = psnr(A_,x1);
            snrs = [snrs;peak];
            MAE=sum(abs(A_(:)-x1(:)))/numel(A_);
            MAEs = [MAEs; MAE];
            final_A_column = horzcat(final_A_column, x1);
            
        end
        final_A = vertcat(final_A, final_A_column);        
    end
    final_rate
    final_A_meta = horzcat(meta, final_A);
    recons_full = [recons_full ; final_A_meta];
    snrs_full = [snrs_full; snrs];
    MAE_full = [MAE_full; MAEs];
    fullFileNameRecons = fullfile(saveDir, baseFileName);
    saveFileNameFull = [saveFileNameFull; fullFileNameRecons];
end
for n = 1:parallel
%continue
final_A_meta = recons_full((n-1)*400 + 1: n*400, :);
snrs = snrs_full((n-1)*592 +1 : n*592);
MAEs = MAE_full((n-1)*592 +1 : n*592);
fullFileNameRecons = saveFileNameFull(n,:);
save(fullFileNameRecons, 'final_A_meta', 'snrs', 'MAEs')
end
end
end
