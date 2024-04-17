## LogBTF (2023), Zhi-Ping Liu all rights reserved
##This program package is supported by the copyright owners and coders "as is" and without warranty of any kind, express or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. In no event shall the copyright owner or contributor be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, without limitation, procurement of substitute goods or services; loss of use, data, or profits; or business interruption), regardless of the theory of liability, whether in contract, strict liability or tort (including negligence or otherwise) for any use of the software, even if advised of the possibility of such damages.

source('./implementations/LogBTF/LogBTF.R')  

args = commandArgs(trailingOnly=TRUE)               

if(length(args) < 2) {
	message("Argument Error (Wrong number of arguments): Time series and output file must be specified! Exiting")  
	quit(save="no")
}

inputFileName  = args[1]
outputFileName = args[2] 

dataSeries = read.table(inputFileName,header=F,sep="\t")    
numGENES = ncol(dataSeries)     
l = nrow(dataSeries) 

set.seed(123) 
noDIAG = 1   
## features*times - A n x m matrix comprising m raw measurements of n features
datahatA <- as.matrix(dataSeries)   

n <- dim(datahatA)[1]
p <- dim(datahatA)[2]  

datahat <- PerMatrix(datahatA) 

## split train and test data
## for train
xglm <- as.data.frame(datahat)
yglm <- datahatA[-1,]   
## for validation
xglm01 <- as.data.frame(datahatA[1:(n-1),]) 

data <- datahatA
num_time_points <- dim(data)[1]
numGENES <- dim(data)[2]

## no scale 
DISTANCE_matrix <- as.matrix(data)
## with scale
# DISTANCE_matrix <- scale(as.matrix(data)) 

## penalty
X_matrix <- DISTANCE_matrix[1:(num_time_points-1),]
n <- dim(X_matrix)[1]
p <- dim(X_matrix)[2]

nfold <- dim(X_matrix)[1] #internal cross validation for obtaining optimal parameters  

#call logbtf function 
coefficient_matrix <- logBTFfunction(method=2, nfold, numGENES, X_matrix, DISTANCE_matrix)

#remove nan values
coefficient_matrix[is.na(coefficient_matrix)] <- 0    

#write down coefficient matrix to output file   
write.table(coefficient_matrix, file=outputFileName, row.names=FALSE, col.names=FALSE)