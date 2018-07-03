#Code to process images to grayscale of desired size, resolution and contrast for downstream machine learning applications

#read in jpeg. Results in 450 x 845 x 3 matrix. 
ImageMatRGB = imread("puppy.jpg");

#convert to grayscale (keeping only luminance value)
ImageMatBW = rgb2ntsc(ImageMatRGB);
ImageMatBW = ImageMatBW(:,:,1);

#crop to square 
width = size(ImageMatBW)(2);
Cropped_ImageMatBW = ImageMatBW(:, floor(width/3):floor(width/3)+size(ImageMatBW)(1)-1);

#Scale image to 20 x 20 pixels
height_scale_vect = floor(linspace(1+10, size(Cropped_ImageMatBW)(1)-10, 20));
width_scale_vect = floor(linspace(1+10, size(Cropped_ImageMatBW)(2)-10, 20));
Processed_Image_20x20 = Cropped_ImageMatBW(height_scale_vect,width_scale_vect)

#Normalize all values to be between 0 and 1
#Note: this may not be necessary depending on training set used; also, for this particular image, all values are already in this range.
maxVal = max(Processed_Image_20x20(:));
minVal = min(Processed_Image_20x20(:));
range = maxVal - minVal;
norm_Processed_Image_20x20 = (Processed_Image_20x20 - minVal)/range;

%Show original and resulting image
setenv("GNUTERM","qt");
imshow(ImageMatRGB , [-1, 1]);
imshow(Processed_Image_20x20, [-1, 1]);

%Increase contrast of image
function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end
cont_norm_Processed_Image_20x20=sigmoid((Processed_Image_20x20-0.5)*10);

%Show contrasted image
imshow(cont_norm_Processed_Image_20x20, [-1, 1]);

