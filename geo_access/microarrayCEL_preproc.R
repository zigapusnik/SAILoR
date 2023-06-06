

################# data analysis -------------------

if (!requireNamespace("BiocManager", quietly = TRUE)) #Clariom_S_Mouse   clariomsmousecdf
  install.packages("BiocManager")

BiocManager::install("org.Hs.eg.db")

# load packages
library(oligo)
library(oligoClasses)
library(simpleaffy)
library(affyPLM)
library(gcrma)
library(limma)
library(pd.clariom.s.human)
library(arrayQualityMetrics)
library(affy)
library(affycoretools)
# gene ontology
library(diffEnrich)
library(GO.db)
library(org.Hs.eg.db) # Homo sapiens db
library(data.table)

# load data
setwd("D:\\PhDLjubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_data")
### clariom .CEL files, use oligo package
setwd("D:\\PhD_Ljubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_data")
celFiles <- list.celfiles()
rawData <- read.celfiles(celFiles)
rawData

# change working directory to save results in new folder
setwd("D:\\PhD_Ljubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_results")

hist(rawData)
boxplot(rawData)
class(rawData)
sampleNames(rawData)
image(rawData, which = 1)

pdf(file="reconstArray_images.pdf")
par(mfrow=c(8,2))
image(rawData, which = 1)
image(rawData, which = 2)
image(rawData, which = 3)
image(rawData, which = 4)
image(rawData, which = 5)
image(rawData, which = 6)
image(rawData, which = 7)
image(rawData, which = 8)
image(rawData, which = 9)
image(rawData, which = 10)
image(rawData, which = 11)
image(rawData, which = 12)
image(rawData, which = 13)
image(rawData, which = 14)
image(rawData, which = 15)
dev.off()

####################################################
# normalising data
rd_n = oligo::rma(rawData)
dim(rd_n)
rd_n
hist(rd_n)
boxplot(rd_n)
class(rd_n)
sum(is.na(data.frame(rd_n)))
#write.csv(rd_n, "cene_livercells_exprs_normalised.csv") #setwd("D:\\PhD_Ljubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_results")
rd_nNorm = data.frame(rd_n)

## QQ plot per array
### ========================== QQ plot per array
pdf(file="cene_liverCells_QQplots.pdf")
qqnorm(rd_nNorm[1,]) #qq plot per array
qqline(rd_nNorm[1,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[2,]) #qq plot per array
qqline(rd_nNorm[2,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[3,]) #qq plot per array
qqline(rd_nNorm[3,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[4,]) #qq plot per array
qqline(rd_nNorm[4,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[5,]) #qq plot per array
qqline(rd_nNorm[5,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[6,]) #qq plot per array
qqline(rd_nNorm[6,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[7,]) #qq plot per array
qqline(rd_nNorm[7,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[8,]) #qq plot per array
qqline(rd_nNorm[8,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[9,]) #qq plot per array
qqline(rd_nNorm[9,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[10,]) #qq plot per array
qqline(rd_nNorm[10,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[11,]) #qq plot per array
qqline(rd_nNorm[11,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[12,]) #qq plot per array
qqline(rd_nNorm[12,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[13,]) #qq plot per array
qqline(rd_nNorm[13,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[14,]) #qq plot per array
qqline(rd_nNorm[14,], col = "steelblue", lwd = 2)
qqnorm(rd_nNorm[15,]) #qq plot per array
qqline(rd_nNorm[15,], col = "steelblue", lwd = 2)
dev.off()
############ ----------------------------------------


## more QC
#fit1 = fitProbeLevelModel(rawData)
fit1 = fitProbeLevelModel(rawData, background=TRUE, normalize=TRUE, target="core", method="plm", verbose=TRUE, S4=TRUE)
fit1
#----------------------------------------- xx
signature(fit1)
image(fit1, which=2)
boxplot(fit1, main="Probe level model-fit1")
nprobes(fit1)
annotation(fit1)
sum(is.na(residuals(fit1)))

resd = residuals(fit1)
head(resd)
boxplot(resd, ylab="residuals", main="Residuals_ raw data")
write.csv(resd, "residuals_fit1.csv")
boxplot(resd)

resd_se = residualSE(fit1)
write.csv(resd_se, "residualSE_fit1.csv")
boxplot(resd_se)
# -------------------------------------------- xx

coef(fit1)[1:8, 1:5]
se(fit1)#[1:8, 1:5]
RLEqc = oligo::RLE(fit1, type='stats')
RLEqc
boxplot(RLEqc)
#write.csv(RLEqc, "RLE_probe_qc.csv")

RLE_values = oligo::RLE(fit1, type='values')
RLE_values
boxplot(RLE_values, ylab = "Relative log exp", main="RLE values")
write.csv(RLE_type, "RLE_values.csv")

# normalised unscaled standard errors
NUSE_stats = oligo::NUSE(fit1)

NUSE_values = oligo::NUSE(fit1, type='values')
NUSE_values[1:5,1:5]
boxplot(NUSE_values, main="NUSE_values")
write.csv(NUSE_values, "NUSE_values.csv")


# array quality metrics
arrayQualityMetrics(rawData, outdir = "ceneQCmetrics", 
   force = FALSE, do.logtransform = FALSE, intgroup = character(0), 
   grouprep, spatial = TRUE, reporttitle = paste("arrayQualityMetrics report for",
    deparse(substitute(expressionset))))



##################################################################################
############## differential expression nb: technical replicates

#setwd("D:\\PhD_Ljubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_results") #original run
setwd("D:\\PhDLjubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_results\\DiffEx_results_cene\\ceneDiffexpConfInts")
rd_nXX = read.csv("D:\\PhDLjubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_results\\cene_livercells_exprs_normalised.csv")
rd_nXX[1:5, 1:5]
#colnames(rd_n)


samp_inf = read.csv("D:\\PhDLjubljana\\ULlab_work\\cene_skubic\\cene_data\\liver_cells\\cene_results\\cene_sample_info.csv")
head(samp_inf)
dim(samp_inf)
#dim(rd_n)

gset = exprs(rd_n)   # expression data
dim(gset)
gset[1:5,1:5]
colnames(gset)
boxplot(rd_n)
#heatmap(gset) # too big

# =========================================================================
### %%%%%%%%%%%%%%% GETTING ANNOTATION FROM PACKAGE *****************
# Get gene symbols and gene names
# use affycoretools package. Use the normalised data
rd_n  # notice that this object has no feature data. So annotate it
ann = annotateEset(rd_n, pd.clariom.s.human) # GETTING THE ANNOTATION
ann # notice that it now has feature data
anne_exp = exprs(ann)
anne_exp[1:5,1:15]
dim(anne_exp)
fdata = fData(ann) # extract feature data and same it as data.frame
fdata[1:5,]
write.csv(fdata, "f_var_data.csv")
getwd()
# ==========================================================================


