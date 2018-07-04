cd Square/

#read in jpeg. Should result in 3D matrix. 
x = dir ("*.jpg");

for i = 1:length(x)
	#Name = "Square1.jpg";
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

	%Unroll matrix
	Unrolled_Image=reshape(Image_20x20, 1,size(Image_20x20,1)* size(Image_20x20,2));
	matrix = Unrolled_Image;

	filename = sprintf("%s.mat", x(i).name(1:end-4));
	save('-mat', filename, 'matrix'); ##name of every saved matrix will be "matrix once loaded"
end
