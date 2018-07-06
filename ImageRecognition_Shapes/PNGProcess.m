#Process square images
cd Square/

#read in jpeg. Should result in 3D matrix. 
x = dir ("*.jpg");
square_mat = zeros(0, 400); #This will be the final feature matrix

for i = 1:length(x)
	#Load image
	ImageMatRGB = imread(x(i).name);

	#convert to grayscale (keeping only luminance value)
	ImageMatBW = rgb2ntsc(ImageMatRGB);
	ImageMatBW = ImageMatBW(:,:,1);

	#Verify that grayscale conversion worked. Commented out once verified.
	#imshow(ImageMatBW , [-1, 1])

	#Scale image to 20 x 20 pixels (no need to crop; all images are squares)
	height_scale_vect = floor(linspace(1+10, size(ImageMatBW, 1)-10, 20));
	width_scale_vect = floor(linspace(1+10, size(ImageMatBW, 2)-10, 20));
	Image_20x20 = ImageMatBW(height_scale_vect,width_scale_vect);

	#Verify that resolution reduction worked. Commented out once verified.
	#imshow(Image_20x20 , [-1, 1])

	#Unroll matrix
	Unrolled_Image=reshape(Image_20x20, 1,size(Image_20x20,1)* size(Image_20x20,2));

	#Add to feature matrix
	square_mat = [square_mat; Unrolled_Image];	
end

cd ../
save square_mat.mat square_mat;
clear;

#Process Triangle images
cd Triangle/

#read in jpeg. Should result in 3D matrix. 
x = dir ("*.jpg");
triangle_mat = zeros(0, 400); #This will be the final feature matrix

for i = 1:length(x)
	#Load image
	ImageMatRGB = imread(x(i).name);

	#convert to grayscale (keeping only luminance value)
	ImageMatBW = rgb2ntsc(ImageMatRGB);
	ImageMatBW = ImageMatBW(:,:,1);

	#Scale image to 20 x 20 pixels (no need to crop; all images are squares)
	height_scale_vect = floor(linspace(1+10, size(ImageMatBW, 1)-10, 20));
	width_scale_vect = floor(linspace(1+10, size(ImageMatBW, 2)-10, 20));
	Image_20x20 = ImageMatBW(height_scale_vect,width_scale_vect);

	#Unroll matrix
	Unrolled_Image=reshape(Image_20x20, 1,size(Image_20x20,1)* size(Image_20x20,2));

	#Add to feature matrix
	triangle_mat = [triangle_mat; Unrolled_Image];	
end

cd ../
save triangle_mat.mat triangle_mat;
clear;

#Process Plus images
cd Plus/

#read in jpeg. Should result in 3D matrix. 
x = dir ("*.jpg");
plus_mat = zeros(0, 400); #This will be the final feature matrix

for i = 1:length(x)
	#Load image
	ImageMatRGB = imread(x(i).name);

	#convert to grayscale (keeping only luminance value)
	ImageMatBW = rgb2ntsc(ImageMatRGB);
	ImageMatBW = ImageMatBW(:,:,1);

	#Scale image to 20 x 20 pixels (no need to crop; all images are squares)
	height_scale_vect = floor(linspace(1+10, size(ImageMatBW, 1)-10, 20));
	width_scale_vect = floor(linspace(1+10, size(ImageMatBW, 2)-10, 20));
	Image_20x20 = ImageMatBW(height_scale_vect,width_scale_vect);

	#Unroll matrix
	Unrolled_Image=reshape(Image_20x20, 1,size(Image_20x20,1)* size(Image_20x20,2));

	#Add to feature matrix
	plus_mat = [plus_mat; Unrolled_Image];	
end

cd ../
save plus_mat.mat plus_mat;
clear;


#Process Circle images
cd Circle/

#read in jpeg. Should result in 3D matrix. 
x = dir ("*.jpg");
circle_mat = zeros(0, 400); #This will be the final feature matrix

for i = 1:length(x)
	#Load image
	ImageMatRGB = imread(x(i).name);

	#convert to grayscale (keeping only luminance value)
	ImageMatBW = rgb2ntsc(ImageMatRGB);
	ImageMatBW = ImageMatBW(:,:,1);

	#Scale image to 20 x 20 pixels (no need to crop; all images are squares)
	height_scale_vect = floor(linspace(1+10, size(ImageMatBW, 1)-10, 20));
	width_scale_vect = floor(linspace(1+10, size(ImageMatBW, 2)-10, 20));
	Image_20x20 = ImageMatBW(height_scale_vect,width_scale_vect);

	#Unroll matrix
	Unrolled_Image=reshape(Image_20x20, 1,size(Image_20x20,1)* size(Image_20x20,2));

	#Add to feature matrix
	circle_mat = [circle_mat; Unrolled_Image];	
end

cd ../
save circle_mat.mat circle_mat;
clear;

#concatenate matrices into final feature vector
load square_mat.mat;
load triangle_mat.mat
load plus_mat.mat;
load circle_mat.mat;

Features = cat(1, square_mat, triangle_mat, plus_mat, circle_mat);
#Verify that fatures matrix looks reasonable. Commented out once verified.
#imshow(Features , [-1, 1])
save Features.mat Features

#labels matrix
y = zeros(400, 4);
y(1:100, 1) = 1; #square
y(101:200, 2) = 1; #triangle
y(201:300, 3) = 1;  #plus
y(301:400, 4) = 1;	#circle

save y.mat y

#Code to save each matrix. Can be used in loop. Unused in this code.
#matrix = Unrolled_Image; 
#filename = sprintf("%s.mat", x(i).name(1:end-4));
#save('-mat', filename, 'matrix'); ##name of every saved matrix will be "matrix once loaded"
