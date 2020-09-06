clear; clc;
%% Inputs
folder_to_read = '1_KCF_fiber';

%% Function
file_name_to_write = strcat(folder_to_read,' Stone tracking.mp4');
folder_to_read = strcat('Data/',folder_to_read,'/');
folder_to_read_images = dir(strcat(folder_to_read,'*.jpg'));
 
%% define video writer
num_imgs = size(folder_to_read_images,1);
myVideo = VideoWriter(file_name_to_write,'MPEG-4');
myVideo.FrameRate = 20;
open(myVideo);

%% loop over all images and write video
for k = 1 : num_imgs
    image_name = folder_to_read_images(k).name;
    img_loc = strcat(folder_to_read,image_name);
    I =  imread(img_loc);
    writeVideo(myVideo,I);
end
close(myVideo);
fprintf('Done!\n');