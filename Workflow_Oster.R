#########################################################################################
############### Workflow for mapping deadwood biomass in Ukrainian forests ##############
######################## Maksym MATSALA, matsala@nubip.edu.ua ###########################

### Some prior information: Ukraine is currently lacking tools and data for reliable
### estimation of forest ecosystem compartments those are hidden under trees' canopy. I.e.
### there are hitherto no LiDAR applications in Ukraine. Field-based data is limited
### by forest planning and management (FPAM) data (offen biased and unpresice, the last
### national-wide aggreagation of FPAM data was performed in 2011) and a few networks of
### sample plots established by scientists. Here we propose simple method to estimate and
### map deadwood (snags, i.e. dead stems, and logs, i.e. downed stems and stumps) biomass
### using Landsat 5 TM image, FPAM data for area of interest (AOI) and allometric models
### those are response of forest growth and yield. 
### Here we will build forest mask using Random Forest model and compare it with maps
### produced by support vector machine (SVM) models. And then we will use limited training
### dataset for building k-NN and gradient boosting models trying to map deadwood biomass
### in spatially explicit manner and as much consistently to FPAM data as we can.

### Such process gonna be performed in several steps:
### 1. Processing of Landsat image (obtained in Google Earth Engine);
### 2. Creation of land cover classes map and fores mask map;
### 3. Prediction of tree species within AOI.
### 4. Comparison of maps produced by Random Forest and SVM models;
### 5. Estimation of deadwood biomass using k-NN and GBM. Deadwood mapping;
### 6. Some exploratory computations and analysis.

### Install/run packages needed here:
library(tidyverse) # main framework
library(randomForest) # building RF models
library(yaImpute) # building k-NN models
library(kernlab) # building SVM models
library(gbm) # buidling gradient boosting models
library(sp) # reading .ascii files
library(caret) # here - for creating confusion matrixes
library(e1071) # here linked to caret package
library(raster) # processing of raster map data
library(rgdal) # here linked to raster package
library(viridis) # for visualization
library(MASS) # for Section 6

### 1. Processing of Landsat image
### Mosaic composite of Landsat 5 TM image for AOI was created in Google Earth Engine
### and obtained for summer period in northern hemisphere (22 May - 22 September).
### It was radiometrically corrected up to Top of Atmosphere (TOA) reflectance also there.
### This image preliminary contains 7 layers represented by 7 Landsat bands' reflectance:
### Coastal aerosol, Blue, Green, Red, Near-InfraRed (NIR), Short-Wave Infrared 1 (SWIR1)
### and SWIR2. 

### Let's download this pre-processed in such way image from following link:
### https://github.com/Janzeero-PhD/Geospatial_Oster/blob/master/Oster2011.tif
### Save it as "Oster2011.tif" in current folder.
Oster_2011 <- stack('Oster2011.tif') # create RasterLayer for image
plot(Oster_2011) # check layers of image

### AOI of this image encompasses ~ 1200 km of areas in Northern Ukraine, with two forest
### facilities: forest enterprise 'Oster' and regional landscape park. Approximately 40 %
### of area in 2010 was forested. Our goal here is to prepare spatial dataset for further
### extraction of spectral values for training dataset (classification purpose). So we
### need make some computations on this image, adding more layers.

X <- coordinates(Oster2011) [, 1] # latitude coordinates
Y <- coordinates(Oster2011) [, 2] # longtitude coordinates
x_raster <- y_raster <- Oster2011[[1]]
x_raster[] <- X
y_raster[] <- Y # thus two new layers with X and Y coordinates are created

### Spectral indices NDVI, IPVI and GRVI are based on spectral reflectance of actual
### satellite bands. So those can be created using simple computations:
NDVI_raster <- (Oster2011[[5]] - Oster2011[[4]]) / (Oster2011[[5]] + Oster2011[[4]])
IPVI_raster <- Oster2011[[5]] / (Oster2011[[5]] + Oster2011[[4]])
GRVI_raster <- (Oster2011[[3]] - Oster2011[[4]]) / (Oster2011[[3]] + Oster2011[[4]])

Oster_2011 <- addLayer(x_raster, y_raster, Oster_2011, IPVI_raster, 
                       NDVI_raster, GRVI_raster) # 12 layers in 1 object!

### Lets make easy understandable names for layers and check these:
names(Oster_2011) <- c("X", "Y", "Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
                      "NDVI", "IPVI", "GRVI")
plot(Oster_2011) # our spatial dataset for land cover classification is ready!

### 2. Creation of land cover classes map and fores mask map
### We can upload our dataset as image and extract all layers' values in software like
### Quantum GIS. We need to create random sample set of points within AOI and extract 
### 12 values (as 12 bands we have) for each point.
### Here we propose dataset of 1,000 points with already extracted spectral values done in
### QGIS. Points were previously interpreted in Google Earth software, with linking to
### one of 8 classes: bog, croplands, forest, grasslands, others, settlement, shrubland
### and water bodies. So our classification models will use differences among spectral 
### values of 12 layers those are apparent for different land cover (LC) classes.

### At first, we download dataset and clean it a little bit:
training.LC.dataset <- read_csv(
  "https://raw.githubusercontent.com/Janzeero-PhD/Geospatial_Oster/master/classification_buffer.csv")

head(training.LC.dataset) # lets check what columns we have
# Seemingly a lot needless variables. We need only land cover class interpreted in Google
# Earth (column "lc") and 12 spectral or ancillary variables. Others can be removed.
training.LC.dataset <- training.LC.dataset %>%
  select(-1:-14) # remove first 14 variables obviously remained as artifacts
head(training.LC.dataset)# We still have some needless variables:
training.LC.dataset <- training.LC.dataset %>%
  select(-2, -4, -7)

# We need to have same names of variables for modelling with layers in RasterStack object
training.LC.dataset$X <- training.LC.dataset$'_Xmean'
training.LC.dataset$Y <- training.LC.dataset$'_Ymean'
training.LC.dataset$Coastal <- training.LC.dataset$'_coastalmean'
training.LC.dataset$Blue <- training.LC.dataset$'_bluemean'
training.LC.dataset$Green <- training.LC.dataset$'_greenmean'
training.LC.dataset$Red <- training.LC.dataset$'_redmean'
training.LC.dataset$NIR <- training.LC.dataset$'_NIRmean'
training.LC.dataset$SWIR1 <- training.LC.dataset$'_SWIR1mean'
training.LC.dataset$SWIR2 <- training.LC.dataset$'_SWIR2mean'
training.LC.dataset$NDVI <- training.LC.dataset$'_NDVImean'
training.LC.dataset$IPVI <- training.LC.dataset$'_IPVImean'
training.LC.dataset$GRVI <- training.LC.dataset$'_GRVImean'
# Good! Lets see the table
head(training.LC.dataset)
# Time to remove last needless variables
training.LC.dataset <- training.LC.dataset %>%
  select(-5:-16)
# Great! And the last preparation is to transform "lc" from numeric to factor variable:
training.LC.dataset$lc <- as.factor(training.LC.dataset$lc)

### Now we can figure out whether our variables visually differ across land cover classes.
### In remote sensing of vegetation, infrared-related variables are common to capture
### differences among growing stocks and tree species.
theme_set(theme_bw())
ggplot(training.LC.dataset, aes(lc, SWIR1)) + 
  geom_boxplot()
### Median reflectance of band SWIR1 is closed to reflectance of "bog" and "water" classes,
### since those are also relatively dark (comparing to open lands like croplands).
### Class "other" has the highest reflectance since it represents sandy areas used as
### military test polygon near Desna river.
ggplot(training.LC.dataset, aes(lc, NDVI)) + 
  geom_boxplot() # Normalized Difference Vegetation Index also can good capture difference
ggplot(training.LC.dataset, aes(lc, Green)) + 
  geom_boxplot() # And even one of RGB bands shows good performance!
### Traditionally, RGB bands do not perform so good in distinguishing land cover classes
### etc., but here we can at least use all these variables to set "forest" aside from
### other types of land surface.
ggplot(training.LC.dataset, aes(lc, X)) + 
  geom_boxplot() # and what about geographical coordinates?
### Longtitude and latitude can enforce our classification efforts, since hidden rule of
### clumping (forests growth near other forests, rivers do not interrupt immediately etc.)
### works also here. So we can use X and Y variables as well in further modelling.

### Now it is time to build and implement our Random Forest model.
RF.mask <- randomForest(x = training.LC.dataset[ ,c(3:14)],
                              y = training.LC.dataset$lc, importance = T)
### As predictors we used columns from 3 to 14, and predicted classes are "lc".
### Here we used default set of model parameters: number of trees is 500 and parameter
### 'mtry' is ~ 3. Why 3? It is a number of variables randomly sampled as candidates at
### each split and here is a sqrt(X), where X is our set of 12 variables (mtry = 3.46 ~ 3).
RF.mask # summary of our model
### Out of bag error is 24.1 %. Seemingly not so good. However, here our main goal is to
### build forest mask, so quality of classification for such classes as "bog" or "shrubs"
### do not matter actually. Pre-validation here shows that class "forest" got only 10.36 %
### error, so almost 90 % accuracy we achive. But we will return there further.
varImpPlot(RF.mask) # plot of mean decrease accuracy
### According to the first plot, if we remove X, NIR or SWIR2 variables, we will have the
### largest decrease in accuracy. So there are the most important variables here. And Y
### with Coastal aerosol band are the least significant.
### Probably, we can remove these variables and run model again. But it will not result
### in increasing of accuracy, since we have 1,000 values and only 12 predictors, that is
### so-called 'curse of dimensionality' is not a case here actually.

### Lets implement our model and build map of land cover classes
RF_LC <- predict(Oster_2011, model = RF.mask, na.rm = T, # Oster_2011 is our spatial set
                     progress = "text", filename = "RF_LC.tif",
                     format = "GTiff", overwrite = T)
plot(RF_LC)
### Good! You can notice that class "8" is a river Desna, class "3" is a forest and it
### covers a quite large area. And the rest indicates to class "2" (croplands) mainly.

### Now we can transform out data and build forest mask that can be used for further
### modelling.
RF_matrix <- matrix(c(1, 0, 2, 0, 3, 1, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0),
                 ncol = 2, byrow = T) # reclassify class "3" to "1" and the rest to "0"
RF_reclassified <- reclassify(RF_LC, RF_matrix) 
RF_fact <- ratify(RF_reclassified) # implement reclassification
plot(RF_fact)
### Only two classes exist, "forest" (1) and "non-forest" (0).

### Now we can make the last post-processing step: filtering in 3x3 window.
### This operation will fill "empty" pixels without forest, if it is detected as noise.
RF_filtered <- focal(RF_fact,
                          w = matrix(1, nrow = 3, ncol = 3), fun = max)
plot(RF_filtered)
### We can notice that forested area slightly increased. Lets compute what happened.
zonal(RF_filtered, RF_filtered, fun = 'sum') # multiply this value by 0.09 to get area in ha
zonal(RF_fact, RF_filtered, fun = 'sum') # 597k vs. 452k
### Good! Now we have filtered forest mask and the last that should be done, it is to
### remove non-forest pixels, remaining raster mask of only forest cover.
RF_filtered[RF_filtered == 0] <- NA
RF_mask <- writeRaster(RF_filtered, filename = "RF_mask.tif",
            format = "GTiff", overwrite = T, progress = "text")
plot(RF_mask)

### 3. Prediction of tree species within AOI.
### Now we can build another Random Forest model for prediction of dominant tree species
### within our created forest mask. We have to prepare our spatial dataset giving to it
### masked forest covered AOI.
Oster_2011_masked <- mask(Oster_2011, RF_mask, filename = "Oster_2011_mask.tif",
                         format = "GTiff", overwrite = T, progress = "text")
plot(Oster_2011_masked) # good, now spatial variables are given only within forest cover
names(Oster_2011_masked) <- c("X", "Y", "Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
                       "NDVI", "IPVI", "GRVI") # change names of predictors again

### Now we have to download training dataset with tree species classified. It was simply
### obtained from forest planning and management (FPAM) data for 2010-2011.
training.species.dataset <- read_csv(
  "https://raw.githubusercontent.com/Janzeero-PhD/Geospatial_Oster/master/species_classification.csv")
### The last column represents intended tree species (acronyms): PISY - Pinus sylvestris,
### or Scots pine; ALGL - Alnus glutinosa, or Black alder; BEPE - Betula pendula, or Silver
### birch; POTR - Populus tremula, or Common aspen; OTHER - the rest, including common oak,
### black locust, Norway spruce and others. Dataset contains centroid points from polygonal
### layers (FPAM dapa) for AOI.

### Lets check how some variables differ among tree species:
ggplot(training.species.dataset, aes(poroda_short, NIR)) + 
  geom_boxplot()
### Such conifer as Scots pine (PISY) has visually different NIR reflectance. Difference
### between Silver birch (BEPE) and other species seems also to be significant.
ggplot(training.species.dataset, aes(poroda_short, Blue)) + 
  geom_boxplot()
### Herewith, one of RGB bands does not perform well in this case: median reflectance of 
### Scots pine and Black alder is almost same. And PISY here has much more outliers than
### in data provided by NIR band.
### We can check other predictors, but should consider that this data (obtained from FPAM)
### cannot be so consistent as data used for LC classification (obtained from Google Earth).
### Honestly saying, it is really hard and unlikely to real that validation of tree species
### dataset obtained from planning data can be validated using extra-high-resolution
### images or aerial photos. So we should prepare ourself that classification model
### produced here will be less precise than this built for distinguishing land cover.

### OK, we assume to use all 12 predictors again and lets build new RF model:
training.species.dataset$poroda_short <- as.factor(training.species.dataset$poroda_short)
RF.species <- randomForest(x = training.species.dataset[, 1:11], # without GRVI
                           y = training.species.dataset$poroda_short,
                           importance = T) # again default model parameters
print(RF.species) # summary of RF model
### Seemingly, only good presented classes (PISY and ALGL) have got nice (PISY 6 %) or
### moderate (ALGL 40 %) error rates. For less abundant species error rate can reach 94 %
### (for aspen). We will compare these results with validation dataset outputs further.
### We didnt use GRVI variable since it leads to biased outputs (total PISY cover).
varImpPlot(RF.species) # importance of predictors
### We can compare which bands and indices are more important for different purposes:
varImpPlot(RF.mask) # geographical coordinates are less important for tree species
# classification, since tree species in forest can be distributed in more fragmented and 
# even chaotic-like way than land cover classes in actively managed landscapes.

### Implement model:
RF_species_map <- predict(Oster_2011_masked, model = RF.species, na.rm = T,
                           progress = "text", filename = "RF_species.tif",
                           format = "GTiff", overwrite = T)
plot(RF_species_map) # Scots pine (4) totally dominates over landscape, alder (1) occurs
# in north-west where wet soils under protection regime are abundant.

### 4. Comparison of maps produced by Random Forest and SVM models
### Support Vector Machines is other type of machine learning models could be used for
### classification purposes in remote sensing applications. In many papers there are
### different accuracy outputs comparing RF and SVM-based classifications. Here we will
### produce one SVM with linear kernel and one SVM with polynomial kernel, then compare
### forested area with our RF-based image. The next step will be to provide robust
### validation comparison using special dataset with 964 new points interpeted in Google
### Earth. Also we will examine quality of our tree species classification.

### There are a lot of kernel types for SVM models. We will examine linear and polynomial
### ones. As shrinkage (learning rate) parameter lets set 1.
SVM.lin.mask <- ksvm(as.matrix(training.LC.dataset[ ,c(3:14)]), 
                     as.factor(training.LC.dataset$lc), 
                     kernel = 'vanilladot', 
                     C = 1) # 'vanilladot' = linear kernel, C = shrinkage
print(SVM.lin.mask) # training error is 21.0 %

### Now we can implement our SVM model 
SVM_lin_LC <- predict(Oster_2011, model = SVM.lin.mask, na.rm = T,
                 progress = "text", filename = "SVM_lin_LC.tif",
                 format = "GTiff", overwrite = T)
plot(SVM_lin_LC)
### First insight says this map is similar to the one produced by RF model. Lets build
### next SVM with polynomial kernel
SVM.poly.mask <- ksvm(as.matrix(training.LC.dataset[ ,c(3:14)]), 
                     as.factor(training.LC.dataset$lc), 
                     kernel = 'polydot', 
                     C = 1,
                     kpar=list(degree = 2)) # polynom with 2 degrees
print(SVM.poly.mask)

### Training error is indicated as 13.1 %.
SVM_poly_LC <- predict(Oster_2011, model = SVM.poly.mask, na.rm = T,
                      progress = "text", filename = "SVM_poly_LC.tif",
                      format = "GTiff", overwrite = T)
plot(SVM_poly_LC) # looks similar to previous map, but area of '7' and '8' increases

### Now we can compare these two maps with RF-based one. Notice that we will not compare
### here forest masks, but LC classifications. For comparison of directly forest covers
### you need to do same operations as done for RF model: reclassification to 0-1 and
### filtering in 3x3 window of forest mask (since such filtration enlarges forested area
### filling gaps considered as noises).
zonal(SVM_lin_LC, SVM_lin_LC, fun = 'count') # number of pixels of class '3' is 469k
zonal(SVM_poly_LC, SVM_poly_LC, fun = 'count') # 'forest' decreases - only 435 k pixels
### We can notice that class '1' (bog) produced by SVM with polynomial kernel 
### substantionally enlarged - from 278 up to 7905 pixels (multiply by 0.09 to get area).
zonal(RF_LC, RF_LC, fun = 'count') # number of 'forest' pixels is 444k
### Except number of forested pixels, RF-based other outputs are closer to linear SVM
### than to polynomial-based model.

### Using zonal statistics we can get only first impression on quality of classification
### performed. In statistics and analysis, the golden rule of "80/20" says that you have
### to split your dataset on training and validation parts with aim to check model 
### performance on data not used in model building.
### In geospatial modelling, there is a neccassary to have validation set as big as 
### possible. So here we will download dataset which contains 964 points (vs. 1000 points
### from training data):
validation.LC.dataset <- read_csv(
'https://raw.githubusercontent.com/Janzeero-PhD/Geospatial_Oster/master/vali_full.csv')
### Column 'map_class' means data produced by RF model. Other columns refer to different
### SVM models (linear and polynomial, with different learning rates and degrees).
validation.LC.dataset[validation.LC.dataset$lc == "bog", 'reference']<- 1 # recode classes
validation.LC.dataset[validation.LC.dataset$lc == "crops", 'reference']<- 2
validation.LC.dataset[validation.LC.dataset$lc == "forest", 'reference']<- 3
validation.LC.dataset[validation.LC.dataset$lc == "grassland", 'reference']<- 4
validation.LC.dataset[validation.LC.dataset$lc == "others", 'reference']<- 5
validation.LC.dataset[validation.LC.dataset$lc == "settlement", 'reference']<- 6
validation.LC.dataset[validation.LC.dataset$lc == "shrubland", 'reference']<- 7
validation.LC.dataset[validation.LC.dataset$lc == "water", 'reference'] <- 8
### Now we have column with reference (true) data visually interpreted in Google Earth.
validation.LC.dataset$reference <- as.factor(validation.LC.dataset$reference)

### Lets check performance of some models:
validation.LC.dataset$map_class <- as.factor(validation.LC.dataset$map_class)
confusionMatrix(validation.LC.dataset$map_class, validation.LC.dataset$reference, 
                positive = "1") # validate quaility of our RF classification
### Overall accuracy is 86.6 %, and user's accuracy for class 'forest' (3) is 91.2 %.

validation.LC.dataset$lin_1 <- as.factor(validation.LC.dataset$lin_1)
confusionMatrix(validation.LC.dataset$lin_1, validation.LC.dataset$reference, 
                positive = "1")
### Overall accuracy is 80.5 %, and user's accuracy for forested area is 94.0 %! Yes, in
### this case linear SVM outperforms Random Forest in term of creating forest mask.

validation.LC.dataset$poly_2_1 <- as.factor(validation.LC.dataset$poly_2_1)
confusionMatrix(validation.LC.dataset$poly_2_1, validation.LC.dataset$reference, 
                positive = "1")
### Overall accuracy is 78.9 %, and user's accuracy for class 'forest' is 89.1 %.

### We can conclude that in some cases SVM can outperform such ensemble method as RF. But
### if you check outputs provided by other SVM models (with increasing learning rate and
### number of degrees for polynom) you will notice substantial decreasing of accuracy rate.
### Notice that we did not tune RF parameters. Also notice that SVM with 2 degrees of
### polynom initially produced lower training error rate, but in result this model
### overfitted on validation dataset producing lower overall and user's accuracies.

### The last part of this step will be to validate quality of RF-based tree species 
### classification. For training RF model we used only ~ 500 points from FPAM data, but
### for validation we will use 4 times more! Dataset contains almost all centroid points
### of inventory units within AOI. So it can have some unconstrained biases and errors.
validation.species.data <- read_csv(
  'https://raw.githubusercontent.com/Janzeero-PhD/Geospatial_Oster/91cb2ee5a5e4372703a750dd39d34ae0b267b488/species_validate.csv'
)
validation.species.data$poroda_code <- as.factor(
  validation.species.data$poroda_code) # reference data obtained from FPAM
validation.species.data$species_RF <- as.factor(
  validation.species.data$species_RF) # data produced by RF model created above

confusionMatrix(validation.species.data$poroda_code, validation.species.data$species_RF, 
                positive = "1")
### Overall accuracy is 77.9 %. Scots pine (class 4) got the highest user's accuracy rate
### (90.0 %) while other tree species got much worse accuracies (Black alder - 51.3 %,
### Silver birch - 37.2 %, Common aspen - 43.1 %, other species - 51.7 %). As Scots pine
### totally dominates over this landscape, it gets so good accuracy thus influencing on
### overall accuracy rate. 
### To get better outputs for local broadleaved species, we need to have some better data
### that can be provided by FPAM (LiDAR, network of sample plots, etc.).

### 5. Estimation of deadwood biomass using k-NN and GBM. Deadwood mapping.
### k-Nearest Neighbours is a widely-used in scientific applications and national forest
### inventories data imputation technique. Here we will use limited (83 points) but at
### least consistent to real conditions dataset. As distance metric we will use matrix
### produced by Random Forest. The number of neighbours (k) was set as 10.

### Here we will try directly map biomass of deadwood. How could we do that avoiding
### modelling of growing stock vollume (GSV) and just using remote multispectral data - 
### see Section 6.
imputation.dataset <- read_csv(
  'https://raw.githubusercontent.com/Janzeero-PhD/Geospatial_Oster/07c5dabd5c2e8f4c19c05f976368543c671ed346/imputation.csv'
)
### Column 'dead' refers to the sum of deadwood biomass provided by columns 'snags' (dead
### standing stems of trees) and 'logs' (downed stems and stumps of trees). Lets create
### dataset of predictors:
imp.prediction.data <- imputation.dataset[, 15:24]
View(imp.prediction.data) # lattitude and longtitude will not help with imputation

### Create kNN model:
kNN.imp.model <- yai(x = imp.prediction.data, y = imputation.dataset$dead, 
                           k = 10, method = "randomForest")
# and implement it for training data:
kNN_imputed <- impute.yai(kNN.imp.model, 
                      ancillaryData = imputation.dataset$dead)

compare.yai(kNN_imputed) # root mean square difference 0.56
cor(kNN_imputed) # coefficient of correlation

# Visual analysis of kNN model performance
theme_set(theme_bw())
ggplot(kNN_imputed, aes(ancillaryData, ancillaryData.o)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  coord_fixed() +
  labs(x = c("Predicted values, t per ha"),
       y = c("Observed values, t per ha")) +
  scale_x_continuous(limits = c(0, 15)) +
  scale_y_continuous(limits = c(0, 15))

### We obtained moderate performance of kNN model. You can tune number of k and choose
### other methods for calculating distance metrics (from Euclidean to principal component
### analysis), but many studies found out RF-based kNN models outperform other produced
### alternatives.

### Lets try to map our deadwood biomass within AOI
plot(Oster_2011_masked) # check names of spatial dataset's layers
# Create set of .ascii files for each layer
writeRaster(Oster_2011_masked[[3]], filename = "Coastal_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[4]], filename = "Blue_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[5]], filename = "Green_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[6]], filename = "Red_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[7]], filename = "NIR_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[8]], filename = "SWIR1_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[9]], filename = "SWIR2_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[10]], filename = "NDVI_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[11]], filename = "IPVI_1.asc", format="ascii", overwrite = T)
writeRaster(Oster_2011_masked[[12]], filename = "GRVI_1.asc", format="ascii", overwrite = T)
# Create list of such files
xfiles.impute <- list(Coastal = "Coastal_1.asc", Blue = "Blue_1.asc", Green = "Green_1.asc",
               Red = "Red_1.asc", NIR = "NIR_1.asc", SWIR1 = "SWIR1_1.asc",
               SWIR2 = "SWIR2_1.asc", NDVI = "NDVI_1.asc", IPVI = "IPVI_1.asc",
               GRVI = "GRVI_1.asc")
# Create output of map
outfiles.impute <- list(dead = "imputation.asc")
# Impute values obtained from kNN model:
AsciiGridImpute(kNN.imp.model, xfiles.impute, outfiles.impute, 
                ancillaryData = imputation.dataset)
imputation_image <- read.asciigrid("imputation.asc") # read data
image(imputation_image) # visualize map
kNN_raster <- raster(imputation_image) # rasterize obtained image

### And finally we have raster map with wall-to-wall values of our kNN imputation
plot(kNN_raster) # deadwood biomass ranges from .ca 2 up to .ca 14 t per ha

### Further we will calculate some values from this map. Now we make such mapping using
### other method - ensemble gradient boosting models (GBM). It is similar to RF, but
### aggregation of decision trees' outputs (or other weak learners) is performing through
### 'boosting' of each tree 'above' previous one, while trees in RF perform independently,
### separately performing on each small subset of training data.

### Here we will tune parameters of GBM, obtaining: 1) model with similar error to our
### kNN model; 2) extra-good performing model which seems to overfit; 3) model with medium
### RMSD comparing to models No. 1 and No. 2.

GBM.prediction.data <- imputation.dataset[, c(15:24, 29)] # prepare dataset
View(GBM.prediction.data) # same as imp.prediction.data but with Y-variable column

GBM.1 <- gbm(dead ~ ., data = GBM.prediction.data, shrinkage = 0.002,
                    interaction.depth = 10, bag.fraction = 0.7, n.trees = 10000)
### If you set large number of trees (here is 10k), you have to set shrinkage as lower as
### possible (> 0.01). Interaction depth 10 and bagging fraction 70 % are found as 
### sufficient for good performance of model.
print(GBM.1)

# predict new values
GBM.prediction.data$predicted <- predict.gbm(GBM.1, GBM.prediction.data, n.trees = 10000)
# and visually compare it to reference data:
ggplot(GBM.prediction.data, aes(predicted, dead)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  coord_fixed() +
  labs(x = c("Predicted values, t per ha"),
       y = c("Observed values, t per ha")) +
  scale_x_continuous(limits = c(0, 15)) +
  scale_y_continuous(limits = c(0, 15))

### Lets compute RMSD for our GBM.1 output
rmsd.function <- function(predicted, observed, n) {
  sqrt(sum((observed - predicted) ^ 2) / n)
}
rmsd.function(predicted = GBM.prediction.data$predicted, 
              observed = GBM.prediction.data$dead, n = 83) # 0.76

### Now we will set larger learning rate (0.01):
GBM.prediction.data <- imputation.dataset[, c(15:24, 29)] # renew of training dataset

GBM.2 <- gbm(dead ~ ., data = GBM.prediction.data, shrinkage = 0.01,
             interaction.depth = 10, bag.fraction = 0.7, n.trees = 10000)
GBM.prediction.data$predicted <- predict.gbm(GBM.2, GBM.prediction.data, n.trees = 10000)

# and what we obtain:
ggplot(GBM.prediction.data, aes(predicted, dead)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  coord_fixed() +
  labs(x = c("Predicted values, t per ha"),
       y = c("Observed values, t per ha")) +
  scale_x_continuous(limits = c(0, 15)) +
  scale_y_continuous(limits = c(0, 15))
# So 'perfect' fitting! But such large shrinkage means that entire 1 % of our trees
# is used at one iteration, i.e. 100 trees. With such small training dataset it can easily
# lead to overfitting.
rmsd.function(predicted = GBM.prediction.data$predicted, 
              observed = GBM.prediction.data$dead, n = 83) # 0.17

### Lets put some moderate parameters into GBM.3:

GBM.prediction.data <- imputation.dataset[, c(15:24, 29)] # renew of training dataset

GBM.3 <- gbm(dead ~ ., data = GBM.prediction.data, shrinkage = 0.005,
             interaction.depth = 4, bag.fraction = 0.7, n.trees = 10000)
GBM.prediction.data$predicted <- predict.gbm(GBM.3, GBM.prediction.data, n.trees = 10000)

# We decreased shrinkage twice and changed interaction depth from 10 to 4.
ggplot(GBM.prediction.data, aes(predicted, dead)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, col = "red") +
  coord_fixed() +
  labs(x = c("Predicted values, t per ha"),
       y = c("Observed values, t per ha")) +
  scale_x_continuous(limits = c(0, 15)) +
  scale_y_continuous(limits = c(0, 15))
rmsd.function(predicted = GBM.prediction.data$predicted, 
              observed = GBM.prediction.data$dead, n = 83) # 0.37
### we have obtained medium RMSD comparing to GBM.1 and GBM.2

### Raster maps for GBM models are typically built using predict() function:
GBM_1_map <- predict(Oster_2011_masked, model = GBM.1, na.rm = T,
                   progress = "text", filename = "GBM_1_map.tif",
                   format = "GTiff", overwrite = T, n.trees = 10000)
GBM_2_map <- predict(Oster_2011_masked, model = GBM.2, na.rm = T,
                     progress = "text", filename = "GBM_2_map.tif",
                     format = "GTiff", overwrite = T, n.trees = 10000)
GBM_3_map <- predict(Oster_2011_masked, model = GBM.3, na.rm = T,
                     progress = "text", filename = "GBM_3_map.tif",
                     format = "GTiff", overwrite = T, n.trees = 10000)
# lets check one of these maps:
plot(GBM_2_map) # looks similar to kNN_raster

### Now we can compare four models and their outputs:
zonal(kNN_raster, RF_species_map, fun = 'mean') # for pine (3) is 7.27 t/ha
zonal(GBM_1_map, RF_species_map, fun = 'mean') # for pine is 6.91 t/ha
zonal(GBM_2_map, RF_species_map, fun = 'mean') # for pine is 7.06 t/ha
zonal(GBM_3_map, RF_species_map, fun = 'mean') # for pine is 6.96 t/ha

### Mean values look similar. The best way to validate imputation results is to have set
### of point pixels with not averaged but strict and consistent values of biomass (obtained
### from sample plots, for instance). Here we dont have such dataset, so can compare only
### predicted results.

### 6. Some exploratory computations and analysis.
### Here we will answer on some questions appeared. The first one will be why do we model
### deadwood biomass directly while it is actually hidden under canopy cover and is not
### visible for sensors of such satellites as Landsat?

### Lets download raw data for forest enterprise 'Oster' which encompasses .ca half of
### forested areas within our AOI:
FPAM.full.raw.dataset <- read_csv(
  'https://raw.githubusercontent.com/Janzeero-PhD/Geospatial_Oster/b92c067cbab84777e1f219488f3698de2528f013/Oster_full_data.csv'
)
### Now we can calculate deadwood biomass of two compartments: snags and logs. Here we will
### use local allometric linear models developed by Ukrainian researchers:
snag_biomass <- function(Species, M, D, H, P, A) {
  ifelse(Species == "PISY", # for Scots pine
         M * (0.157 * (D ^ 0.18) * (H ^ -1.001) * (P ^ -1.534)),
         ifelse(Species == "BEPE", # for Silver birch
                0.016 * (D ^ 0.971) * (H ^ 0.841) * (P ^ 0.817),
                ifelse(Species == "ALGL", # for Black alder
                       0.023 * (D ^ 0.587) * (H ^ 1.13) * (P ^ -0.29),
                       ifelse(Species == "POTR", # for Common aspen
                              0.287 * (A ^ 0.893) * (H ^ -0.114),
                              0 # for other species is not calculated
                       ))))
}
log_biomass <- function(Species, M, D, H, P, A) {
  ifelse(Species == "PISY",
         0.073 * (D ^ 0.245) * (H ^ 0.798) * (P ^ -0.447),
         ifelse(Species == "BEPE",
                0.01 * (A ^ 0.904) * (D ^ 0.789) * (P ^ 1.075),
                ifelse(Species == "ALGL",
                       0.429 * (D ^ 1.232) * (H ^ -0.482) * (P ^ 0.217),
                       ifelse(Species == "POTR",
                              1.505 * (D ^ 3.079) * (H ^ -2.96) * (P ^ -0.347),
                              0
                       ))))
}
### You can notice that outputs of these models are dependant on parameters of live trees
### stock: diameter at breist height (D), mean height (H), relative stocking (P), age (A)
### and, in case of Scots pine snags, growing stock vollume (M). So such modelling method
### assumes that stock of deadwood is strictly linked to stock of live tree biomass. As
### the last compartment mentioned above can be linked to spectral reflectance of its
### canopies, i.e. live vegetation, we can use these relationships to directly model
### deadwood biomass in spatially explicit manner, as we did in Section 5.

FPAM.full.raw.dataset <- FPAM.full.raw.dataset %>%
  mutate(snags = snag_biomass(Species = poroda_short, 
                              M = zap_1_ga, D = d, H = h, A = vik, P = Povnota)) %>%
  mutate(logs = log_biomass(Species = poroda_short, 
                            M = zap_1_ga, D = d, H = h, A = vik, P = Povnota)) %>%
  mutate(dead = snags + logs)

ggplot(FPAM.full.raw.dataset, aes(poroda_short, dead)) +
  geom_boxplot(aes(fill = poroda_short)) # compare median outputs to means of raster maps

### We have null values for species 'OTHERS' and some outliers for Common aspen. Lets
### remove them in order to visualize relationship between GSV and deadwood biomass.
FPAM.full.raw.dataset <- FPAM.full.raw.dataset %>%
  filter(dead > 0) %>%
  filter(dead < 22) # filter outliers

# build some assisting function:
get_density <- function(x, y, ...) { # create function for coloring density
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  return(dens$z[ii])
}
# create column with density of points for scatterplot
FPAM.full.raw.dataset$density <- get_density(FPAM.full.raw.dataset$zap_1_ga, # GSV
                                             FPAM.full.raw.dataset$dead, n = 100)
theme_set(theme_bw())
ggplot(FPAM.full.raw.dataset, aes(zap_1_ga, dead))+
  geom_point(aes(col = density)) + # apply column created by function above
  scale_color_viridis() +
  labs(x = "GSV, m3 per ha",
       y = "Deadwood biomass, t per ha") +
  theme(legend.position = "none")

### Yes! You can see explicit linear relationship between GSV (which is linked to TOA
### reflectance used in this study) and deadwood biomass, since the last one was modelled
### using allometric models linked to parameters of live tree biomass stock. So it is an
### answer why we obtain deadwood biomass directly modelling that using spectral reflectance
### and achieving medium precision results.