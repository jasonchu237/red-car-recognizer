import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


'''===== Coding Block 1 ============================================================================================='''
'''=================================================================================================================='''
'''=============================================== Import Data ======================================================'''
'''=================================================================================================================='''

imgTrain_raw = np.load('data_train.npy')
imgTruth_raw = np.load('ground_truth.npy')
imgTest_raw = np.load('data_test.npy')


'''===== Coding Block 2 ============================================================================================='''
'''=================================================================================================================='''
'''================================================ Plot Image ======================================================'''
'''=================================================================================================================='''

plt.figure(figsize=(5, 5))
plt.imshow(imgTrain_raw)
plt.imshow(imgTest_raw)
# imgTruth[x-axis][y-axis]
markers_x = []
markers_y = []
for i in range(len(imgTruth_raw)):
    markers_x.append(imgTruth_raw[i][0])
    markers_y.append(imgTruth_raw[i][1])

# Label more red cars in the bottom of train image
imgTruth_x = markers_x
imgTruth_y = markers_y

add_x = [143, 176, 188, 281, 458, 494, 506, 525, 785, 801, 811, 817, 848, 862, 893, 842, 836, 875, 752, 866, 888, 902,
         718, 778, 784, 797, 808, 884, 898, 958, 699, 770, 929, 656, 662, 693, 736, 751, 868, 944, 1009, 1019, 1024,
         1046, 1048, 1069, 1092, 1129, 1116, 1094, 1124, 1118, 1200, 999, 1018, 1040, 1306, 1324, 1321, 1316, 1379,
         1384, 1413, 1429, 1460, 1452, 1434, 1422, 1434, 1378, 1385, 1406, 1384, 1373, 141, 245, 975, 1054, 948, 968,
         912, 2716, 3272, 4200, 4220, 4446, 4437, 4432, 4416, 4484, 4503, 4495, 4492, 3237, 4277, 4465, 4626, 5185,
         5142, 5790]
add_y = [2047, 2045, 2050, 1980, 2049, 2032, 2030, 2007, 1907, 1910, 1913, 1898, 1898, 1910, 1921, 1930, 1972, 1962,
         1992, 2007, 2005, 2010, 2048, 2060, 2043, 2057, 2066, 2067, 2047, 2046, 2104, 2107, 2097, 2145, 2144, 2154,
         2151, 2148, 2142, 2128, 2003, 1996, 2015, 1956, 1944, 1936, 1902, 1915, 1949, 1986, 2010, 2045, 2018, 2093,
         2092, 2134, 2095, 2093, 2116, 1989, 2015, 2022, 1953, 1942, 1946, 1973, 1972, 1982, 2004, 2015, 2023, 2055,
         2117, 2114, 4726, 5950, 5843, 5815, 6073, 6019, 6173, 4145, 4368, 4296, 4298, 5400, 5416, 5433, 5457, 5420,
         5420, 5436, 5445, 5746, 5588, 5522, 5617, 4869, 5307, 5370]

for i in range(len(add_x)):
    imgTruth_x.append(add_x[i])
    imgTruth_y.append(add_y[i])
# Plot labeled red cars
plt.title('Ground_Truth: labeled red cars')
plt.scatter(imgTruth_x, imgTruth_y, s=20, facecolors='none', edgecolors='r')
plt.show()


'''===== Coding Block 3 ============================================================================================='''
'''=================================================================================================================='''
'''================================================ Car Size   ======================================================'''
'''=================================================================================================================='''

# Locate first 4 red cars location in ground_truth, estimate the average pixels occupied by these 4 cars.
# [ 903 1186 2], [1112 1558 2], [1767 1361 2], [4865 964 2]
# The estimated width of car is 6 pixels, length of car is 13.
# Estimated number of pixels occupied by a car is 6 X 13 rectangular.
# So, I choose the 5 x 5 pixel square for the car


'''===== Coding Block 4 ============================================================================================='''
'''=================================================================================================================='''
'''==========================================       label assignment      ==========================================='''
'''==========================================              and            ==========================================='''
'''==========================================    Average pixel RGB value  ==========================================='''
'''=================================================================================================================='''

# First, assign every pixel in imgTrain to class one, then assign coordinate of labeled red car and its extension right
# square centered at specified coordinate in ground_truth (5 x 5 pixels )to class 2.
# The reason is that I assume each red car size will be at least 5 x 5 pixels range.
# So after get the RGB values, later when I train my model I can use validate data to learn
# the average red car RGB values should get.
# If we don't use average RGB value, I'm not learning red car; instead, I'm learning just red pixel.
# imgTrain[row : y][column : x],  row starts from top to down, column starts from left to right.


# Create a 6250 X 6250 ones array, first assigning all pixel to class 1.
one = np.ones((6250, 1))
imgTrain_label = []
for i in range(len(imgTrain_raw)):
    imgTrain_lab = np.append(imgTrain_raw[i], one, axis=1)
    imgTrain_label.append(imgTrain_lab)
# print(imgTest_raw[0].shape)
# print((imgTest_raw[0]))
# print(one)


# Now, imgTrain_label contain items originally from imgTrain_raw with class label.
imgTrain_label = np.array(imgTrain_label)

# Assign the square centered at coordinate provided by ground_truth to class 2
# Also calculate the average R value for red cars
# R_total_each : every R value contained inside a 5 X 5 square.
# R_total :　a list contain R values for index from 0 ~ 129 (27 ground_truth red car labels), each has 25 R values.
R_total = []
G_total = []
B_total = []
for i in range(len(imgTruth_y)):
    x = imgTruth_x[i]
    y = imgTruth_y[i]

    R_total_each = []
    G_total_each = []
    B_total_each = []
    ave_R_total_each = []
    ave_G_total_each = []
    ave_B_total_each = []

    # Assign nearby square to class2 (red cars)
    for j in range(x - 2, x + 3, 1):
        for k in range(y - 2, y + 3, 1):
            imgTrain_label[k][j][3] = 2
            R_total_each.append(imgTrain_label[k][j][0])
            G_total_each.append(imgTrain_label[k][j][1])
            B_total_each.append(imgTrain_label[k][j][2])

    # Used to calculate total 128 red cars' RGB values
    R_total.append(R_total_each)
    G_total.append(G_total_each)
    B_total.append(B_total_each)
    # Assign average calculated RGB value of single red car to all pixels nearby square of class2
    ave_R_total_each = np.average(R_total_each)
    ave_G_total_each = np.average(G_total_each)
    ave_B_total_each = np.average(B_total_each)

    # Assign average RGB values to these pixels inside this square
    for j in range(x - 2, x + 3, 1):
        for k in range(y - 2, y + 3, 1):
            imgTrain_label[k][j][0] = ave_R_total_each
            imgTrain_label[k][j][1] = ave_G_total_each
            imgTrain_label[k][j][2] = ave_B_total_each

# new average RGB values assigned array file name: imgTrain_label_newRGB
imgTrain_label_newRGB = imgTrain_label

# ave_R : average of each labeled red car with 5 X 5 square
# ave_R_total : average of total 128 red car
ave_R = []
ave_G = []
ave_B = []
for i in range(len(R_total)):
    aveR = sum(R_total[i]) / len(R_total[i])
    ave_R.append(aveR)

    aveG = sum(G_total[i]) / len(G_total[i])
    ave_G.append(aveG)

    aveB = sum(B_total[i]) / len(B_total[i])
    ave_B.append(aveB)

ave_R_total = sum(ave_R) / len(R_total)
ave_G_total = sum(ave_G) / len(G_total)
ave_B_total = sum(ave_B) / len(B_total)
print("\nThe average of 128 red cars R value (RGB): ", ave_R_total, '\n')
print("\nThe average of 128 red cars G value (RGB): ", ave_G_total, '\n')
print("\nThe average of 128 red cars B value (RGB): ", ave_B_total, '\n')

# Summary: 1. This coding block will assign new RGB values for those pixels
#             contain true red car location inside a square.
#          2. The above is store inside imgTrain_label_newRGB array.
#          3. It will print out the average RGB values of 128 red cars.


'''===== Coding Block 5 (PART A) ====================== Sub-Image 1 ================================================='''
'''=============================        Split imgTrain to training and validating          =========================='''
'''=============================  Train 66%[y-axis: 0~80];  Validate 33%[y-axis: 80~120]   =========================='''
'''=================================================================================================================='''
# In this part, I will use the same sub-image(a rectangle) to split into 2 part.
# Part A, I use left 66.66% of image as training data; right 33.33% as validate.
# Part B, I use right 33.33% of image as training data; left 66.66% as validate.
# Implementing Part B as cross-validation of Part A.


# imgTrain_label_newRGB : <class 'numpy.ndarray'>, shape: (6250, 6250, 4)
# For image range cover from: X axis = 4400 ~ 4520 and Y axis = 5380 ~ 5480
# Split above range into train and validate. Train: 66.66% & Validate:　33.33%


# lt_imgTrain (with new ave RGB): cast type to list
# YX_RGBs_tr01: row & col range for train01
# X_RGBs_tr01: col range for train01
# RGB_tr01: RGB values at each pixel for train01
# Train: X axis = 4400 ~ 4480 and Y axis = 5380 ~ 5480           => 80 X 100
# i: y-axis;     j: x-axis
lt_imgTrain = list(imgTrain_label_newRGB)
YX_RGBs_tr01 = []
for i in range(5380,5480):
    X_RGBs_tr01 = []
    for j in range(4400,4480):
        RGB_tr01 = []
        for k in range(4):
            RGB_tr01.append(lt_imgTrain[i][j][k])
        X_RGBs_tr01.append(RGB_tr01)
    YX_RGBs_tr01.append(X_RGBs_tr01)
YX_RGBs_tr01 = np.array(YX_RGBs_tr01)
print("Training data set 01 shape: ", YX_RGBs_tr01.shape, '\n')

# Validate: X axis = 4480 ~ 4520 and Y axis = 5380 ~ 5480     => 40 x 100
# Here generate validate data with class label
YX_RGBs_va01_label = []
for i in range(5380,5480):
    X_RGBs_va01 = []
    for j in range(4480,4520):
        RGB_va01 = []
        for k in range(4):
            RGB_va01.append(lt_imgTrain[i][j][k])
        X_RGBs_va01.append(RGB_va01)
    YX_RGBs_va01_label.append(X_RGBs_va01)
YX_RGBs_va01_label = np.array(YX_RGBs_va01_label)

# Count number of class2 assigned to Validate data set 01 after comparing with ground_truth.
count2 = 0
for i in range(YX_RGBs_va01_label.shape[0]):
    for j in range(YX_RGBs_va01_label.shape[1]):
        if YX_RGBs_va01_label[i][j][3] == 2:
            count2 = count2 + 1
print("The number of class 2 assigned before training: ", count2)


print("Validate data set 01 with label shape: ", YX_RGBs_va01_label.shape, '\n')

# print(YX_RGBs_va01_label.shape[0])
# print(YX_RGBs_va01_label.shape[1])

# Validate(without class label): X axis = 4480 ~ 4520 and Y axis = 5380 ~ 5480     => 40 x 100
# Here generate validate data without class label
# Use this to input function
YX_RGBs_va01 = []
for i in range(5380,5480):
    X_RGBs_va01 = []
    for j in range(4480,4520):
        RGB_va01 = []
        for k in range(3):
            RGB_va01.append(lt_imgTrain[i][j][k])
        X_RGBs_va01.append(RGB_va01)
    YX_RGBs_va01.append(X_RGBs_va01)
YX_RGBs_va01 = np.array(YX_RGBs_va01)
print("Validate data set 01 shape: ", YX_RGBs_va01.shape, '\n')
# print(YX_RGBs_va01.shape[0])
# print(YX_RGBs_va01.shape[1])


'''===== Coding Block 5 (PART B) ====================== Sub-Image 1 ================================================='''
'''=============================        Split imgTrain to training and validating          =========================='''
'''=============================  Train 66%[y-axis: 40~120];  Validate 33%[y-axis: 0~40]   =========================='''
'''=================================================================================================================='''

# imgTrain_label_newRGB : <class 'numpy.ndarray'>, shape: (6250, 6250, 4)
# For image range cover from: X axis = 4400 ~ 4520 and Y axis = 5380 ~ 5480
# Split above range into train and validate. Train: 66.66%(right) & Validate:　33.33%(left)


# crossTrain: X axis = 4440 ~ 4520 and Y axis = 5380 ~ 5480           => 80 X 100
# i: y-axis;     j: x-axis
cross_lt_imgTrain = list(imgTrain_label_newRGB)
cross_YX_RGBs_tr01 = []
for i in range(5380,5480):
    cross_X_RGBs_tr01 = []
    for j in range(4440,4520):
        cross_RGB_tr01 = []
        for k in range(4):
            cross_RGB_tr01.append(cross_lt_imgTrain[i][j][k])
        cross_X_RGBs_tr01.append(cross_RGB_tr01)
    cross_YX_RGBs_tr01.append(cross_X_RGBs_tr01)
cross_YX_RGBs_tr01 = np.array(cross_YX_RGBs_tr01)
print("Training data set 01 (cross-validate) shape: ", cross_YX_RGBs_tr01.shape, '\n')

# crossValidate: X axis = 4400 ~ 4440 and Y axis = 5380 ~ 5480        => 40 x 100
# Here generate crossvalidate data with class label
cross_YX_RGBs_va01_label = []
for i in range(5380,5480):
    cross_X_RGBs_va01 = []
    for j in range(4400,4440):
        cross_RGB_va01 = []
        for k in range(4):
            cross_RGB_va01.append(cross_lt_imgTrain[i][j][k])
        cross_X_RGBs_va01.append(cross_RGB_va01)
    cross_YX_RGBs_va01_label.append(cross_X_RGBs_va01)
cross_YX_RGBs_va01_label = np.array(cross_YX_RGBs_va01_label)

# Count number of class2 assigned to Validate data set 01 (cross-validate) after comparing with ground_truth.
cross_count2 = 0
for i in range(cross_YX_RGBs_va01_label.shape[0]):
    for j in range(cross_YX_RGBs_va01_label.shape[1]):
        if cross_YX_RGBs_va01_label[i][j][3] == 2:
            cross_count2 = cross_count2 + 1
print("The number of class 2 assigned before training (cross-validate): ", cross_count2)
print("Validate data set 01 with label shape (cross-validate): ", cross_YX_RGBs_va01_label.shape, '\n')


# crossValidate(without class label): X axis = 4400 ~ 4440 and Y axis = 5380 ~ 5480        => 40 x 100
# Here generate cross-validate data without class label
# Use this to input function
cross_YX_RGBs_va01 = []
for i in range(5380,5480):
    cross_X_RGBs_va01 = []
    for j in range(4400,4440):
        cross_RGB_va01 = []
        for k in range(3):
            cross_RGB_va01.append(cross_lt_imgTrain[i][j][k])
        cross_X_RGBs_va01.append(cross_RGB_va01)
    cross_YX_RGBs_va01.append(cross_X_RGBs_va01)
cross_YX_RGBs_va01 = np.array(cross_YX_RGBs_va01)
print("Validate data set 01 shape (cross-validate): ", cross_YX_RGBs_va01.shape, '\n')


'''===== Coding Block 6 ============================================================================================='''
'''=================================================================================================================='''
'''============================================= RGB value gap calculation  ========================================='''
'''=================================================================================================================='''
# a: training pixel array;                    size: (row, col, 4)
# b: validate pixel                           size: (row, col, 3)
# c: imgTrain_label  i: y-axis;   j: x-axis
# Function gap_rgb_val
# Input a whole training data array, and validate data array
# It will assign validate data to a class after training
# It will return an array b that has new label for validate data
#
def gap_rgb_val(a, b, k_val, x_coor, y_coor):
    k = k_val
    # a = YX_RGBs_tr01;    training                     size: (row, col, 4)
    # b = YX_RGBs_va01;    validate (without label)     size: (row, col, 3)
    # x_coor = x position at where I sliced validate set
    # y_coor = x position at where I sliced validate set
    # In this case:   Sliced origin = (x: 4480, y: 5380)
    # Calculate the gap of RGB distance between validate & train, store in RGBgap.
    print("The origin(x, y) of the slice validate dataset: ", "\nx: ",x_coor,"\ny: ", y_coor)
    one = np.ones((b.shape[1], 1))
    b_init_label = []
    for i in range(b.shape[0]):
        b_init_lab = np.append(b[i], one, axis=1)
        b_init_label.append(b_init_lab)

    # b_init_label: assign class 1 to every pixel in pass in array b (Initialization)
    b_init_label = np.array(b_init_label)

    # b_est_label: used for later assign estimated class label
    b_est_label = b_init_label

    # Vary (x,y) of validate dataset
    for t in range(b.shape[0]):
        for g in range(b.shape[1]):

            # Vary (x,y) of training dataset
            RGBgap_yx = []
            for i in range(a.shape[0]):
                RGBgap_x = []
                for j in range(a.shape[1]):
                    RGB_gap = ((b[t][g][0] - a[i][j][0]) ** 2 + (b[t][g][1] - a[i][j][1]) ** 2 + (
                                b[t][g][2] - a[i][j][2]) ** 2) ** 0.5

                    RGBgap_x.append(RGB_gap)
                RGBgap_yx.append(RGBgap_x)

            RGBgap = []
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    RGBgap.append(RGBgap_yx[i][j])

            RGBgap_sort = np.sort(RGBgap)
            RGBgap_sort_ind = np.argsort(RGBgap)
            #     print(RGBgap_sort)
            #     print(RGBgap_sort_ind)

            RGBgap_xloc = []
            RGBgap_yloc = []
            for i in range(k):
                RGBgap_xloc.append(RGBgap_sort_ind[i] % (a.shape[1]))
                RGBgap_yloc.append(RGBgap_sort_ind[i] // (a.shape[1]))

            # print("X (col) coordinate of ", k, " smallest gap: ", RGBgap_xloc)
            # print("Y (row) coordinate of ", k, " smallest gap: ", RGBgap_yloc)

            # imgTrain_label[y-axis][x-axis]
            # Check class1 > class2, or class2 > class1
            # Assign that validate data to majority class
            cls1 = 0
            cls2 = 0
            for i in range(k):
                if a[RGBgap_yloc[i]][RGBgap_xloc[i]][3] == 1:
                    cls1 = cls1 + 1
                else:
                    cls2 = cls2 + 1

            if cls1 > cls2:
                b_est_label[t][g][3] = 1
            else:
                b_est_label[t][g][3] = 2
        # Just count how many pixels are assigned to class2 after trained
        class2_cnt = 0
        for i in range(2, b_est_label.shape[0], 1):
            for j in range(2, b_est_label.shape[1], 1):
                if b_est_label[i][j][3] == 2:
                    class2_cnt = class2_cnt + 1

    return b_init_label, b_est_label, x_coor, y_coor, class2_cnt


'''===== Coding Block 7 ============================================================================================='''
'''=================================================================================================================='''
'''================================================== Get appropriate K ============================================='''
'''=================================================================================================================='''
# To get the appropriate K value of KNN
# We know that inside the validate data, we have around 4 cars,
# , and we also assume that each red car takes about 25 pixels.
# So, we loop run gap_rgb_val(), to get the k value.
# Pre-run the below coding block, I estimate that K = 48.
print("If run the code in coding block 7 (Get Appropriate K) part, getting the optimal K = 48")

# for k in range(3,60,5):
#     validate_trained01_init_label, validate_trained01_est_label, x_coor, y_coor, count_class2 = gap_rgb_val(YX_RGBs_tr01, YX_RGBs_va01, k, 4480,5380)
#     print("The # of class2 count after training in loop# = ", k, "count =>",count_class2)


'''===== Coding Block 8 (PART A)====================== Sub-Image 1 =================================================='''
'''=================================================================================================================='''
'''================================================== Call gapRGBval  ==============================================='''
'''=================================================================================================================='''
# Recall train: YX_RGBs_tr01
# Recall validate: YX_RGBs_va01
# Function name: gap_rgb_val(a, b, k_val, slice_origin_of_x, slice_origin_of_y )
# Use optimal k_val = 48 obtained from last step (In Get Appropriate K part)
# It will assign validate data to a class after training
# It will return an array b that has new label for validate data:   validate_trained01_est_label

validate_trained01_init_label, validate_trained01_est_label, x_coor01, y_coor01, count_class2_01 = gap_rgb_val(YX_RGBs_tr01, YX_RGBs_va01, 48, 4480, 5380)
print("\nPlease check again if inputing correct sliced origin coordinate. (Should input sliced origin of validate part)")
print("The # of class2 (sub-Image1) after training is ", count_class2_01, " at K = 48")
print("The shape of validate01  (sub-Image1) after trained: ", validate_trained01_est_label.shape)
print("!! Remember now that validate_trained01 file return new labels for validate dataset after training")
print("But this is not the original coordinate, be careful when using sliding window to get coordinate in next step.\n")


'''===== Coding Block 9 ============================================================================================='''
'''=================================================================================================================='''
'''===================================          Function Definition            ======================================'''
'''===================================   sortNearbyDatatoSameClass(estimate)   ======================================'''
'''=================================== Generate More precise red car locations ======================================'''
'''=================================================================================================================='''
# Parameter: estimate is the list (est_car_loc) that will be generated after next coding block (Validation)
# est_car_loc = estimate
# The est_car_loc will contain possible red car locations after validation process
# But will contain separate nearby locations but actually is the same red car location
# So, call this function to calculate more precise location of red cars


def sortNearbyDatatoSameClass(estimate):

    greater_10 = 0
    new_estimate = []
    remaining = []
    distance_sort = []
    distance_argsort = []
    new_estimate_list = []

    if (len(estimate) == 1):
        new_estimate.append(estimate[0])
    elif (len(estimate) == 0):
        print("Finish combining all nearby data to one class, and store possible red car locations in list.")
    else:

        init_dist = []
        for i in range(len(estimate)):
            distance = ((estimate[0][0] - estimate[i][0]) ** 2 + (estimate[0][1] - estimate[i][1]) ** 2) ** 0.5
            init_dist.append(distance)

        for i in range(len(init_dist)):
            if (init_dist[i] > 15):
                greater_10 = greater_10 + 1

        if (greater_10 == (len(init_dist) - 1)):
            new_estimate.append(estimate[0])
            for i in range(1, len(estimate)):
                remaining.append(estimate[i])
        else:
            distance_sort = np.sort(init_dist)
            distance_argsort = np.argsort(init_dist)

            for i in range(len(distance_sort)):
                if (distance_sort[i] <= 15):
                    new_estimate_list.append(estimate[distance_argsort[i]])
                else:
                    remaining.append(estimate[distance_argsort[i]])

            new_estimateXsum = 0
            new_estimateYsum = 0
            for i in range(len(new_estimate_list)):
                new_estimateXsum = new_estimateXsum + new_estimate_list[i][0]
                new_estimateYsum = new_estimateYsum + new_estimate_list[i][1]

            new_estimateX = new_estimateXsum // (len(new_estimate_list))
            new_estimateY = new_estimateYsum // (len(new_estimate_list))

            new_estimate.append([new_estimateX, new_estimateY])

    return new_estimate, remaining


'''===== Coding Block 10 (PART A)====================== Sub-Image 1 ================================================='''
'''=================================================================================================================='''
'''========================================           Validation            ========================================='''
'''========================================                &                ========================================='''
'''========================================    Estimate red car location    ========================================='''
'''=================================================================================================================='''

# For image range cover from: X axis = 4480 ~ 4520 and Y axis = 5380 ~ 5480  => 40 x 100
# For sliding window = 5(0~5) x 5(0~5) square, the center of the first will be (2,2)
# Validate data after trained01: validate_trained01   shape:(100,40,4)
# i&e: y-axis,    j&s: x-axis
# Sliced origin = (x: 4480, y: 5380)

est_car_loc = []
for i in range(2, validate_trained01_est_label.shape[0], 5):
    for j in range(2, validate_trained01_est_label.shape[1], 5):
        RR_val_each = []
        GG_val_each = []
        BB_val_each = []
        RR_ave = []
        GG_ave = []
        BB_ave = []
        cls2_counter = 0

        # First calculate if the majority of the whole sliding window is class2
        for e in range(i-2, i+3, 1):
            for s in range(j-2, j+3, 1):
                if validate_trained01_est_label[e][s][3] == 2:
                    cls2_counter = cls2_counter + 1
        #print(cls2_counter)

        # If majority is class 2, then we can calculate average RGB values
        if cls2_counter >= 3 :
            for e in range(i - 2, i + 3, 1):
                for s in range(j - 2, j + 3, 1):
                    RR_val_each.append(validate_trained01_est_label[e][s][0])
                    GG_val_each.append(validate_trained01_est_label[e][s][1])
                    BB_val_each.append(validate_trained01_est_label[e][s][2])
            RR_ave = np.average(RR_val_each)
            GG_ave = np.average(GG_val_each)
            BB_ave = np.average(BB_val_each)

        # Because I look at RGB panel, I can tell from R = 60 ~ 255 as red color.
        # And the average of R for red cars is 122, so I set range(122-60) = 60
        if (abs(RR_ave - ave_R_total) <= 60) and (abs(GG_ave - ave_G_total) <= 60) and (abs(BB_ave - ave_B_total) <= 60):
            est_car_loc.append([j, i])

for i in range(len(est_car_loc)):
    est_car_loc[i][0] = est_car_loc[i][0] + x_coor01
    est_car_loc[i][1] = est_car_loc[i][1] + y_coor01
print("=============================================================================================================")
print("\nThe possible location for red cars are list as below: \n")
print(est_car_loc)
print("\nNext these locations may overlap same car locations, ")
print("So, input est_car_loc list into function(to get more precise locations:  sortNearbyDatatoSameClass(estimate) ")


# Now we have estimated the possible location of red cars.
# However, we might have different locations but actually is the same car.
# So below is how I get more precise location for red cars.
# Calling function: sortNearbyDatatoSameClass,   and input est_car_loc as parameter

est = est_car_loc
new_loc = []
while ((len(est) - 1) >= 0):
    new, est = sortNearbyDatatoSameClass(est)
    # print(new)
    # print(est)
    new_loc.append(new)
print("The new locations of possible red cars as below:")
print(new_loc)


'''===== Coding Block 11 ============================================================================================'''
'''=================================================================================================================='''
'''==============================   Accuracy of this estimated validation dataset  =================================='''
'''=================================================================================================================='''
# Compare the new_loc obtained from previous step with the ground_truth location
# Check if distance is within 15
# Recall X-locations for true red cars are: imgTruth_x
# Recall Y-locations for true red cars are: imgTruth_y

correct_estloc = 0
for i in range(len(new_loc)):
    for j in range(len(imgTruth_x)):
        dist_grnd = 0
        dist_grnd = ((new_loc[i][0][0] - imgTruth_x[j])**2 + (new_loc[i][0][1] - imgTruth_y[j])**2)** 0.5
        if (dist_grnd <8):
            correct_estloc = correct_estloc + 1
print(correct_estloc)

# Recall that count2 = The number of class 2 assigned before training.
# So, divide it by 25 will be the number of true red car inside this image range.
accuracy_estloc = correct_estloc/(count2//25)

print("The accuracy of this model is (for validation dataset): ", accuracy_estloc*100,'%')
print("===================================== End of classification of this data set ===================================")

'''===== Coding Block 12 ============================================================================================'''
'''===== Actually recall Coding Block 8 (PART B)====== Sub-Image 1 =================================================='''
'''=================================================================================================================='''
'''================================================== Call gapRGBval  ==============================================='''
'''=================================================================================================================='''
# Recall cross-train: cross_YX_RGBs_tr01
# Recall cross-validate: cross_YX_RGBs_va01
# Function name: gap_rgb_val(a, b, k_val, slice_origin_of_x, slice_origin_of_y )
# Use optimal k_val = 48 obtained from last step (In Get Appropriate K part)
# It will assign validate data to a class after training
# It will return an array b that has new label for validate data:   cross_validate_trained01_est_label
# Sliced origin of validation part of sub-Image 1 = (x: 4400, y: 5380)

cross_validate_trained01_init_label, cross_validate_trained01_est_label, cross_x_coor01, cross_y_coor01,  cross_count_class2_01 = gap_rgb_val(cross_YX_RGBs_tr01, cross_YX_RGBs_va01, 48, 4400, 5380)
print("\nPlease check again if inputing correct sliced origin coordinate. (Should input sliced origin of validate part)")
print("The # of class2  (sub-Image1) after training is (cross-validate sub-Image1)", cross_count_class2_01, " at K = 48")
print("The shape of validate01  (sub-Image1) after trained (cross-validate sub-Image1): ", cross_validate_trained01_est_label.shape)
print("!! Remember now that cross_validate_trained01 file return new labels for validate dataset after training")
print("But this is not the original coordinate, be careful when using sliding window to get coordinate in next step.\n")



'''===== Coding Block 13 ============================================================================================'''
'''===== Actually recall Coding Block 10 (PART B)====== Sub-Image 1 ================================================='''
'''=================================================================================================================='''
'''========================================           Validation            ========================================='''
'''========================================                &                ========================================='''
'''========================================    Estimate red car location    ========================================='''
'''=================================================================================================================='''
# For crossValidate01 image range cover from: X axis = 4400 ~ 4440 and Y axis = 5380 ~ 5480        => 40 x 100
# For sliding window = 5(0~5) x 5(0~5) square, the center of the first will be (2,2)
# Validate data after trained01: validate_trained01   shape:(100,40,4)
# i&e: y-axis,    j&s: x-axis
# Sliced origin of validation part of sub-Image 1 = (x: 4400, y: 5380)

est_car_loc = []
for i in range(2, cross_validate_trained01_est_label.shape[0], 5):
    for j in range(2, cross_validate_trained01_est_label.shape[1], 5):
        RR_val_each = []
        GG_val_each = []
        BB_val_each = []
        RR_ave = []
        GG_ave = []
        BB_ave = []
        cls2_counter = 0

        # First calculate if the majority of the whole sliding window is class2
        for e in range(i-2, i+3, 1):
            for s in range(j-2, j+3, 1):
                if cross_validate_trained01_est_label[e][s][3] == 2:
                    cls2_counter = cls2_counter + 1
        #print(cls2_counter)

        # If majority is class 2, then we can calculate average RGB values
        if cls2_counter >= 3 :
            for e in range(i - 2, i + 3, 1):
                for s in range(j - 2, j + 3, 1):
                    RR_val_each.append(cross_validate_trained01_est_label[e][s][0])
                    GG_val_each.append(cross_validate_trained01_est_label[e][s][1])
                    BB_val_each.append(cross_validate_trained01_est_label[e][s][2])
            RR_ave = np.average(RR_val_each)
            GG_ave = np.average(GG_val_each)
            BB_ave = np.average(BB_val_each)

        # Because I look at RGB panel, I can tell from R = 60 ~ 255 as red color.
        # And the average of R for red cars is 122, so I set range(122-60) = 60
        if (abs(RR_ave - ave_R_total) <= 60) and (abs(GG_ave - ave_G_total) <= 60) and (abs(BB_ave - ave_B_total) <= 60):
            est_car_loc.append([j, i])

for i in range(len(est_car_loc)):
    est_car_loc[i][0] = est_car_loc[i][0] + cross_x_coor01
    est_car_loc[i][1] = est_car_loc[i][1] + cross_y_coor01
print("The possible location for red cars (cross-validate of  sub-Image1) are list as below: \n")
print(est_car_loc)
print("\nNext these locations may overlap same car locations, ")
print("So, input est_car_loc list into function(to get more precise locations:  sortNearbyDatatoSameClass(estimate) ")


# Now we have estimated the possible location of red cars.
# However, we might have different locations but actually is the same car.
# So below is how I get more precise location for red cars.
# Calling function: sortNearbyDatatoSameClass,   and input est_car_loc as parameter

est = est_car_loc
new_loc = []
while ((len(est) - 1) >= 0):
    new, est = sortNearbyDatatoSameClass(est)
    # print(new)
    # print(est)
    new_loc.append(new)
print("The new locations (cross-validate of  sub-Image1) of possible red cars as below:")
print(new_loc)


'''===== Coding Block 14 ============================================================================================'''
'''===== Actually recall Coding Block 11 (PART B)====== Sub-Image 1 ================================================='''
'''=================================================================================================================='''
'''==============================   Accuracy of this estimated validation dataset  =================================='''
'''==============================                 (cross-validate)                 =================================='''
'''=================================================================================================================='''
# Compare the new_loc obtained from previous step with the ground_truth location
# Check if distance is within 15
# Recall X-locations for true red cars are: imgTruth_x
# Recall Y-locations for true red cars are: imgTruth_y

correct_estloc = 0
for i in range(len(new_loc)):
    for j in range(len(imgTruth_x)):
        dist_grnd = 0
        dist_grnd = ((new_loc[i][0][0] - imgTruth_x[j])**2 + (new_loc[i][0][1] - imgTruth_y[j])**2)**0.5
        if (dist_grnd <8):
            correct_estloc = correct_estloc + 1
print(correct_estloc)

# Recall that cross_count2 = The number of class 2 assigned before training.
# So, divide it by 25 will be the number of true red car inside this image range.
accuracy_estloc = correct_estloc/(cross_count2//25)

print("The accuracy of this model is (for cross-validation of sub-Image1 dataset): ", accuracy_estloc*100,'%')
print("===================================== End of classification of this data set ===================================")


'''===== Coding Block 15 (PART C) ==================================================================================='''
'''=================================================  Sub-Image 2 ==================================================='''
'''=============================    Choose a sub-Image other that range of Sub-Image1   ============================='''
'''=============================     Use Sub-Image2 to further test my trained model    ============================='''
'''=================================================================================================================='''
# Review of previous steps from coding block 1 ~ coding block 2
# <Part A>
# I've trained a model by using left 70% of sub-image1, right 30% of sub-image1 as validation,
#  and calculating its accuracy.
# <Part B>
# I've also cross-validated my model by using right 70% of sub-image1 as training, left 30% of sub-image1
#  as validation, and calculating accuracy as well.
# Now for coding block 15 <Part C>
# I chose another sub image of imgTrain_raw to further test my trained model.
# The range of sub-image2 is as below:
# X axis = 1740 ~ 1840 and Y axis = 1280 ~ 1380           => 100 X 100
# i: y-axis;     j: x-axis

sub2_lt_imgTrain = list(imgTrain_label_newRGB)
sub2_YX_RGBs_va02_label = []
for i in range(1280,1380):
    sub2_X_RGBs_va02 = []
    for j in range(1740,1840):
        sub2_RGB_va02 = []
        for k in range(4):
            sub2_RGB_va02.append(sub2_lt_imgTrain[i][j][k])
        sub2_X_RGBs_va02.append(sub2_RGB_va02)
    sub2_YX_RGBs_va02_label.append(sub2_X_RGBs_va02)
sub2_YX_RGBs_va02_label = np.array(sub2_YX_RGBs_va02_label)

# Count number of class2 assigned to sub-image2 data set after comparing with ground_truth.
sub2_count2 = 0
for i in range(sub2_YX_RGBs_va02_label.shape[0]):
    for j in range(sub2_YX_RGBs_va02_label.shape[1]):
        if sub2_YX_RGBs_va02_label[i][j][3] == 2:
            sub2_count2 = sub2_count2 + 1
print("The initial number of class 2 assigned to sub-image2: ", sub2_count2)
print("Sub-image2 data set with label shape: ", sub2_YX_RGBs_va02_label.shape, '\n')
#
#
# sub-image2(without class label): X axis = 1740 ~ 1840 and Y axis = 1280 ~ 1380           => 100 X 100
# Here sub-image2 data without class label
# Use this to input function
sub2_YX_RGBs_va02 = []
for i in range(1280,1380):
    sub2_X_RGBs_va02 = []
    for j in range(1740,1840):
        sub2_RGB_va02 = []
        for k in range(3):
            sub2_RGB_va02.append(sub2_lt_imgTrain[i][j][k])
        sub2_X_RGBs_va02.append(sub2_RGB_va02)
    sub2_YX_RGBs_va02.append(sub2_X_RGBs_va02)
sub2_YX_RGBs_va02 = np.array(sub2_YX_RGBs_va02)
print("Sub-image2 data set without label shape: ", sub2_YX_RGBs_va02.shape, '\n')


'''===== Coding Block 16 ============================================================================================'''
'''===== Actually recall Coding Block 8 (PART C)====== Sub-Image 2 =================================================='''
'''=================================================================================================================='''
'''================================================== Call gapRGBval  ==============================================='''
'''=================================================================================================================='''
# Recall train: YX_RGBs_tr01
# Recall sub-image2 with label: sub2_YX_RGBs_va02_label
# Recall sub-image2 without label: sub2_YX_RGBs_va02
# Function name: gap_rgb_val(a, b, k_val, slice_origin_of_x, slice_origin_of_y )
# Use optimal k_val = 48 obtained from last step (In Get Appropriate K part)
# It will assign sub-image2 data to a class after training
# It will return an array b that has new label for sub-image2 data:   sub2_validate_trained02_est_label
# Sliced origin of validation part of sub-Image 2 = (x: 1740, y: 1280)

sub2_validate_trained02_init_label, sub2_validate_trained02_est_label, sub2_x_coor02, sub2_y_coor02,  sub2_count_class2_02 = gap_rgb_val(YX_RGBs_tr01, sub2_YX_RGBs_va02, 48, 1740, 1280)
print("\nPlease check again if inputing correct sliced origin coordinate. (Should input sliced origin of validate part)")
print("The # of class2  (sub-Image1) after training is (cross-validate sub-Image1)", sub2_count_class2_02, " at K = 48")
print("The shape of validate01  (sub-Image1) after trained (cross-validate sub-Image1): ", sub2_validate_trained02_est_label.shape)
print("!! Remember now that cross_validate_trained01 file return new labels for validate dataset after training")
print("But this is not the original coordinate, be careful when using sliding window to get coordinate in next step.\n")



'''===== Coding Block 17 ============================================================================================'''
'''===== Actually recall Coding Block 10 (PART C)====== Sub-Image 2 ================================================='''
'''=================================================================================================================='''
'''========================================    Estimate red car location    ========================================='''
'''=================================================================================================================='''
# The range of sub-image2 is as below:
# X axis = 1740 ~ 1840 and Y axis = 1280 ~ 1380           => 100 X 100
# For sliding window = 5(0~5) x 5(0~5) square, the center of the first will be (2,2)
# i&e: y-axis,    j&s: x-axis
# Sliced origin of validation part of sub-Image 2 = (x: 1740, y: 1280)
# sub2_validate_trained02_est_label


est_car_loc = []
for i in range(2, sub2_validate_trained02_est_label.shape[0], 5):
    for j in range(2, sub2_validate_trained02_est_label.shape[1], 5):
        RR_val_each = []
        GG_val_each = []
        BB_val_each = []
        RR_ave = []
        GG_ave = []
        BB_ave = []
        cls2_counter = 0

        # First calculate if the majority of the whole sliding window is class2
        for e in range(i-2, i+3, 1):
            for s in range(j-2, j+3, 1):
                if sub2_validate_trained02_est_label[e][s][3] == 2:
                    cls2_counter = cls2_counter + 1
        #print(cls2_counter)

        # If majority is class 2, then we can calculate average RGB values
        if cls2_counter >= 3 :
            for e in range(i - 2, i + 3, 1):
                for s in range(j - 2, j + 3, 1):
                    RR_val_each.append(sub2_validate_trained02_est_label[e][s][0])
                    GG_val_each.append(sub2_validate_trained02_est_label[e][s][1])
                    BB_val_each.append(sub2_validate_trained02_est_label[e][s][2])
            RR_ave = np.average(RR_val_each)
            GG_ave = np.average(GG_val_each)
            BB_ave = np.average(BB_val_each)

        # Because I look at RGB panel, I can tell from R = 60 ~ 255 as red color.
        # And the average of R for red cars is 122, so I set range(122-60) = 60
        if (abs(RR_ave - ave_R_total) <= 60) and (abs(GG_ave - ave_G_total) <= 60) and (abs(BB_ave - ave_B_total) <= 60):
            est_car_loc.append([j, i])

for i in range(len(est_car_loc)):
    est_car_loc[i][0] = est_car_loc[i][0] + sub2_x_coor02
    est_car_loc[i][1] = est_car_loc[i][1] + sub2_y_coor02
print("=============================================================================================================")
print("\nThe possible location of sub-image2 for red cars are list as below: \n")
print(est_car_loc)
print("\nNext these locations may overlap same car locations, ")
print("So, input est_car_loc list into function(to get more precise locations:  sortNearbyDatatoSameClass(estimate) ")


# Now we have estimated the possible location of red cars.
# However, we might have different locations but actually is the same car.
# So below is how I get more precise location for red cars.
# Calling function: sortNearbyDatatoSameClass,   and input est_car_loc as parameter

est = est_car_loc
new_loc = []
while ((len(est) - 1) >= 0):
    new, est = sortNearbyDatatoSameClass(est)
    # print(new)
    # print(est)
    new_loc.append(new)
print("The new locations of sub-image2 for possible red cars as below:")
print(new_loc)


'''===== Coding Block 18 ============================================================================================'''
'''===== Actually recall Coding Block 11 (PART C)====== Sub-Image 2 ================================================='''
'''=================================================================================================================='''
'''==============================   Accuracy of this estimated Sub-Image 2 dataset =================================='''
'''=================================================================================================================='''
# Compare the new_loc obtained from previous step with the ground_truth location
# Check if distance is within 15
# Recall X-locations for true red cars are: imgTruth_x
# Recall Y-locations for true red cars are: imgTruth_y

correct_estloc = 0
for i in range(len(new_loc)):
    for j in range(len(imgTruth_x)):
        dist_grnd = 0
        dist_grnd = ((new_loc[i][0][0] - imgTruth_x[j])**2 + (new_loc[i][0][1] - imgTruth_y[j])**2)**0.5
        if (dist_grnd <8):
            correct_estloc = correct_estloc + 1
print(correct_estloc)

# Recall that cross_count2 = The number of class 2 assigned before training.
# So, divide it by 25 will be the number of true red car inside this image range.
accuracy_estloc = correct_estloc/(sub2_count2//25)

print("The accuracy of this model is (for sub-Image2 dataset): ", accuracy_estloc*100,'%')
print("===================================== End of classification of this data set ===================================")







'''===== Coding Block 19 (PART D) ==================================================================================='''
'''===================================           Sub-Image of Test image         ===================================='''
'''===================================    Choose a sub-Image contain red cars    ===================================='''
'''=================================================================================================================='''
# In <Part D>, I chose a sub-image from given Test image that contain red cars
# Then, apply this sub-image to test my model, generating the predicted locations of red cars.
# I chose another sub image of imgTrain_raw to further test my trained model.
# The range of sub-image2 is as below:
# X axis = 5330 ~ 5430 and Y axis = 750 ~ 850           => 100 X 100
# i: y-axis;     j: x-axis
test_lt_imgTrain = list(imgTest_raw)
test_YX_RGBs_va03 = []
for i in range(750, 850):
    test_X_RGBs_va03 = []
    for j in range(5330, 5430):
        test_RGB_va03 = []
        for k in range(3):
            test_RGB_va03.append(test_lt_imgTrain[i][j][k])
        test_X_RGBs_va03.append(test_RGB_va03)
    test_YX_RGBs_va03.append(test_X_RGBs_va03)
test_YX_RGBs_va03 = np.array(test_YX_RGBs_va03)
print("Sub-image of Test Image data set without label shape: ", test_YX_RGBs_va03.shape, '\n')




'''===== Coding Block 20 ============================================================================================'''
'''===== Actually recall Coding Block 8 (PART D)====================================================================='''
'''===================================           Sub-Image of Test image         ===================================='''
'''================================================== Call gapRGBval  ==============================================='''
'''=================================================================================================================='''
# Recall train: YX_RGBs_tr01
# Recall sub-image of tes without label: test_YX_RGBs_va03
# Function name: gap_rgb_val(a, b, k_val, slice_origin_of_x, slice_origin_of_y )
# Use optimal k_val = 48 obtained from last step (In Get Appropriate K part)
# It will assign sub-image2 data to a class after training
# It will return an array b that has new label for sub-image2 data:   sub2_validate_trained02_est_label
# Sliced origin of validation part of sub-Image 2 = (x: 1740, y: 1280)

sub2_validate_trained02_init_label, sub2_validate_trained02_est_label, sub2_x_coor02, sub2_y_coor02,  sub2_count_class2_02 = gap_rgb_val(YX_RGBs_tr01, sub2_YX_RGBs_va02, 48, 1740, 1280)
print("\nPlease check again if inputing correct sliced origin coordinate. (Should input sliced origin of validate part)")
print("The # of class2  (sub-Image1) after training is (cross-validate sub-Image1)", sub2_count_class2_02, " at K = 48")
print("The shape of validate01  (sub-Image1) after trained (cross-validate sub-Image1): ", sub2_validate_trained02_est_label.shape)
print("!! Remember now that cross_validate_trained01 file return new labels for validate dataset after training")
print("But this is not the original coordinate, be careful when using sliding window to get coordinate in next step.\n")


