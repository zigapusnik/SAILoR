#### for paired samples
####################### adwalakira@gmail.com ################################################


library(GEOquery)
library(arrayQualityMetrics)
library(affyio)
library(limma)
library(oligo)
library(oligoClasses)
library(simpleaffy)
library(affyPLM)
library(gcrma)
library(arrayQualityMetrics)
library(affy)
library(affycoretools)
library(AnnotationDbi)
library(factoextra)
library(matrixStats)
library(EnrichmentBrowser)
library(data.table)
library(factoextra)
library(clusterProfiler) #for geneset enrichment
library(org.Hs.eg.db) # Homo sapiens db
library(impute)

library(stringr)
###### --------------------------------------------

#setwd("D:\\PhDLjubljana\\PhD_thesis\\metaAnalysis\\analysisMeta\\limmaDE\\GSE39791")  # set your working directory
### read in the data from GEO and save it on hard drive
gset<-getGEO('GSE39791')
if (length(gset)>1) idx <- grep('GPL10558', attr(gset, 'names')) else idx<-1
gset <- gset[[idx]]
dim(gset)
head(gset)
class(gset)
#rmaD = rma(gset)
# get an idea of how the data was processed
pData(gset)$data_processing[1]

arrayQualityMetrics(gset, force=TRUE) #saved to working directory, look at QC plots

## get pData, fData, and exprData, pdata is phenotype data
pData = pData(gset)
dim(pData)
head(pData)
table(pData[, 37])
#colnames(pData)[37] ="tissue" #colnames(pData)[29] ="disease"

# added mmoskon
#patient_ID = str_split_fixed(pData[,1],"_",2)[,2]
# remove "T" and "N" from patient_ID
patient_ID <- gsub("T","",pData[,1])
patient_ID <- gsub("N","",patient_ID)
patient_ID
pData$patient_ID <- patient_ID
# eo added mmoskon

colnames(pData)

pData$geo_accession

colnames(pData)
write.csv(pData, "pDataGSE39791.csv")


fData = fData(gset) # fdata is feature data. It should have information about the genes
dim(fData)
head(fData)
write.csv(fData, "fDataGSE39791.csv")

data_n = exprs(gset) # get gene expression values
dim(data_n)
boxplot(data_n) # a bit crowded but appears like its already normalised
boxplot(data_n[,1:50]) # look at a few samples.
max(data_n)
data_n[1:5, 1:5]

#dst <- normalizeBetweenArrays(log2(data_n), method="quantile")
#***this data is already normalised and log transformed, look at the box plot (median at same level), and the max values
dst = data_n
dst[1:5, 1:5]
#boxplot(dst)
sum(is.na(dst))
dim(dst)
# remove rows with all NAs
#dstNA = dst[rowSums(is.na(dst)) != ncol(dst), ]
#dim(dstNA)
#dstNA[1:5, 1:5]
#data = impute.knn(dstNA,k = 10, rowmax = 0.005, colmax = 1.0, maxp = 1500, rng.seed=362436069) #impute missing expression data if you choose to
#sum(is.na(data))
#sum(is.na(dst))
#data[1:5, 1:5]
#head(data)
#names(data)
dataImp = dst
dataImp[1:5, 1:5]
dim(dataImp)


# summarising data
# read in fData
#fData = read.csv("D:\\PhDLjubljana\\PhD_thesis\\metaAnalysis\\analysisMeta\\limmaDE\\GSE39791\\fDataGSE39791.csv")
fData = read.csv("fDataGSE39791.csv")
fData[1:5,1:5]
fDataSub = fData[, c("ID", "Symbol")]
head(fDataSub)
dataU = setDT(as.data.frame(dataImp), keep.rownames=TRUE)
dataU[1:5,1:5]
colnames(dataU)[1] = "ID"
dataU[1:5,1:5]

# merge fdataSub with dataU to get gene names on data
datMerge = merge(fDataSub,dataU,by = "ID")
datMerge[1:5,1:5]
dim(datMerge)
datMergeSub = datMerge[, !(names(datMerge) %in% "ID")] # remove ID column
datMergeSub[1:5,1:5]
dim(datMergeSub)

# summarise probes by averaging probes per gene
dataSum = limma::avereps(datMergeSub, ID = datMergeSub$Symbol)
dim(dataSum)
dataSum[1:5,1:5]
dataSumq = noquote(dataSum)
dataSumq[1:5,1:5]
write.csv(dataSumq, "dataGSE39791.csv")





groups = as.factor(patient_ID)
############################################################################################
##*** read in the data. Call in the data afresh 
#expdata = read.csv("D:\\PhDLjubljana\\PhD_thesis\\metaAnalysis\\analysisMeta\\limmaDE\\GSE39791\\dataGSE39791.csv")
expdata = read.csv("dataGSE39791.csv")
expdata[1:5,1:5]

## call the pData
#pdata = read.csv("D:\\PhDLjubljana\\PhD_thesis\\metaAnalysis\\analysisMeta\\limmaDE\\GSE39791\\pDataGSE39791.csv")
pdata = read.csv("pDataGSE39791.csv")
disease = pdata$tissue
groups = as.factor(disease)
colnames(pdata)






#DatPCA = expdata[, 2:ncol(expdata)]
DatPCA = expdata[, 3:ncol(expdata)]
cmnDatpca = prcomp(t(DatPCA))
summary(cmnDatpca)
fviz_pca_ind(cmnDatpca,
             invisible = c("ind.sup", "quali", "var", "quanti.sup"), #remove center of ellipse i.e "ind"
             #axes = c(1, 80), #get plot on one dimension
             col.ind = groups, # color by groups
             geom = "point",
             #palette = c("red", "blue"), #Group
             #palette = c("red", "green", "blue"), #Group
             addEllipses = FALSE, # Concentration ellipses
             #ellipse.type = "confidence",
             legend.title = "Group",
             repel = TRUE # Avoid text overlapping
             )+ labs(title ="                    GSE39791")

fviz_pca_ind(cmnDatpca, repel = TRUE)  #  label = "none"






