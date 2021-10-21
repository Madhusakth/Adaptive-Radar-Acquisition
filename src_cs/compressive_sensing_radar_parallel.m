
function compressive_sensing_radar_parallel(scene)


%rng(2) %change default seed
myDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene', string(scene),'/radar'));
saveDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene', string(scene),'/final-image-info-5/'));
x_point_rearDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene', string(scene),'/radar-x-points-rear-left18Close/'));
x_point_centreDir = char(strcat('/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/scene', string(scene),'/radar-x-points-centre-left18Close/'));


myFiles = dir(fullfile(myDir,'*.png'));
parallel = length(myFiles)-1; %10;  %%% total length of myFiles
outer_loop = [2:parallel:length(myFiles)];

%outer_loop = [1];
%parallel = 1;

for p = 1:length(outer_loop)
recons_full = [];
snrs_full = [];
MAE_full = [];
saveFileNameFull = [];
%if p == length(outer_loop)
%    parallel = 2;
%end
for k = 1: parallel
    %k = 1; 
    current = outer_loop(p) + (k-1);
    %current = 12;
    baseFileName = myFiles(current).name;
    fullFileName = fullfile(myDir, baseFileName);
    baseFileName = strrep(baseFileName, '.png', '.mat');
    x_point_rear = fullfile(x_point_rearDir, baseFileName);
    some = load(x_point_rear);
    object_type1 = some.object_type;
    x_point_rear = some.x_point_rear;
    x_point_centre = fullfile(x_point_centreDir, baseFileName);
    some = load(x_point_centre);
    object_type2 =some.object_type;
    x_point_centre = some.x_point_centre;
    x_point_centre(x_point_centre <0) = 360 + x_point_centre(x_point_centre <0);
    %x_point 0-180; add 90 to account for bottom part
    x_point_rear = x_point_rear + 90; 
    x_point = horzcat(x_point_rear, x_point_centre);
    x_point = x_point*(400/360); % convert to azimuth of 0.9 degree resolution
    x_point = floor(x_point); 

    object_type = horzcat(object_type1, object_type2);

    %x_point = unique(x_point); %remove repeated azimuths
    disp(fullFileName)
    A = imread(fullFileName);
    meta = A(:, 1:11); 
    A = A(:,12:3779);
    [height, width] = size(A);
    %A = A([1:50],[12:87]);
    w = 25;
    h = 112 - 12; %87 - 12;
    rate = int16((w*h)/5);
    snrs = [];
    MAEs = [];
    rows = [1: w: 425];
    columns = [1: h: 3779];
    sampler = zeros(16,1);
    for samp = 1:16
        above = x_point >= rows(samp);
        below = x_point <= rows(samp+1) -1;
        curr = and(below, above);
        if sum(curr(:)==1) >= 1
	    obj_idx = find(curr);
            obj_types = max(object_type(obj_idx));
	    if obj_types == 1
                obj_types = 0;
            end
            sampler(samp) = obj_types; 
        end
    end

    level2 = length(find(sampler==2));
    if level2 == 0
        %level2 = 1;
        samp_idx = find(sampler==0);
        rand_idx = randi(length(samp_idx));
        sampler(samp_idx(rand_idx)) = 2;
        level2 = length(find(sampler==2));
    end

    level3 = length(find(sampler==3));
    if level3 == 0
        %level3 = 1;
        samp_idx = find(sampler==0);
        rand_idx = randi(length(samp_idx));
        sampler(samp_idx(rand_idx)) = 3;
        level3 = length(find(sampler==3));        
    end

    
    level1 = 16 - (level2 + level3);
    
    A1 = optimvar('A1','LowerBound',0.05,'UpperBound',0.4);
    B1 = optimvar('B1','LowerBound',0.05,'UpperBound',0.4);
    C1 = optimvar('C1','LowerBound',0.05,'UpperBound',0.4);
    D1 = optimvar('D1','LowerBound',0.02,'UpperBound',0.025);
    prob = optimproblem('Objective',level3*18*25*100*A1 + level2*18*25*100*B1 + level1*18*25*100*C1 + 16*19*25*100*D1,'ObjectiveSense','max');
    prob.Constraints.c1 = level3*18*25*100*A1 + level2*18*25*100*B1 + level1*18*25*100*C1+ 16*19*25*100*D1 <= 148000;
    prob.Constraints.c2 = A1 == 3*C1;
    prob.Constraints.c3 = B1 == 2*C1;
    %prob.Constraints.c4 = D == (1/2)*C;

    problem = prob2struct(prob);
    [sol,fval,exitflag,output] = linprog(problem);
    sol
    

    find(sampler)
    final_A = [];
    final_rate = 0
    for c = 1:16
        %c
        final_A_column = [];
        for d = 1:37
            if (sampler(c) == 0 && d < 19) 
                rate = floor(sol(3)*2500);
            elseif (sampler(c) == 2 && d <19)
                rate = floor(sol(2)*2500);
            elseif (sampler(c) == 3 && d < 19)
                rate = floor(sol(1)*2500);  
            else
                rate = floor(sol(4)*2500);
            end
	    final_rate  = final_rate + rate;
            %continue %%%%%%%%%%%%
            c,d,rate
            A_ = A([rows(c):rows(c+1)-1],[columns(d):columns(d+1)-1]);
            x1 = compressed_sensing_example_parallel(A_, w, h, rate);
            x1 = uint8(x1);
            peak = psnr(A_,x1)
            snrs = [snrs;peak];
            MAE=sum(abs(A_(:)-x1(:)))/numel(A_)
            MAEs = [MAEs;MAE];
            final_A_column = horzcat(final_A_column, x1);
%        if d == 25
%exit
%end    
        end
        final_A = vertcat(final_A, final_A_column);
        row_samples = [];
    end
    final_rate
    %continue %%%%%%%%%%%%%

    final_A_meta = horzcat(meta, final_A);
    recons_full = [recons_full ; final_A_meta];
    snrs_full = [snrs_full; snrs];
    MAE_full = [MAE_full; MAEs];
    fullFileNameRecons = fullfile(saveDir, baseFileName);
    saveFileNameFull = [saveFileNameFull; fullFileNameRecons];
end
snrs_full
for n = 1:parallel
%continue%%%%%%%%%%%%
final_A_meta = recons_full((n-1)*400 + 1: n*400, :);
snrs = snrs_full((n-1)*592 +1 : n*592);
MAEs = MAE_full((n-1)*592 +1 : n*592);
fullFileNameRecons = saveFileNameFull(n,:);
save(fullFileNameRecons, 'final_A_meta', 'snrs', 'MAEs')
end
end
end
