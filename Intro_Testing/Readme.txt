There are 20 PTOA subjects:

003
007
012
017
018
021
024
028
030
031
032
034
035
038
042
046
048
051
060
065

For each subject, there is information about the MRI scans in the files:

dicom_lst2.xlsx - spreadsheet for easy reading
dicom_lst2.mat - for use with Matlab

The MRI image data is in the MAT files:

T1rho_S*.mat, where * is the MRI series number.  The smaller series number is the left knee and the larger number is the right knee.

The image volume is in the variable v.  v is a 4-D matrix with dimensions of 512x512x64x4.  The first two dimensions are the image size, the third dimension is the slice number and the fourth (last) dimension is the spin lock time.  You will probably want to use the first spin lock time (0 ms).

v = squeeze(v(:,:,:,1));

The mask data is in the MAT files:

T1rho_S*_prois.mat, where * is the MRI series number.

The MRI slices with femur masks (cartilage) are listed in the variable rslf. The MRI slices with tibia masks (cartilage) are listed in the variable rslt.  The slices with masks for either the femur or tibia are listed in the variable rsl.

The femur masks are in the variable maskf.  The tibia masks are in the variable maskt. 

The first column in the logical masks is the mask for the image (262144 [512*512]), the second column is the cartilage layers (1 - superficial, and 2 - deep), and the third column is the slices in rsl.  To combine the superficial and deep layers, use or command (|):

maskf_combined_layers = squeeze(maskf(:,1,:)|maskf(:,2,:));
maskt_combined_layers = squeeze(maskt(:,1,:)|maskt(:,2,:));

Use reshape to make a mask the same dimensions as the image data.

mask = reshape(maskf_combined_layers(:,slice_number),512,512);
