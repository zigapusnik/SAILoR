Context specific gene expression of Drosophila melanogaster female's response after mating

Sources: 
	- Time series transcriptome analysis implicates the circadian clock in the Drosophila melanogaster female’s response to sex peptide, Delbare et al., 2023
	- Reprogramming of regulatory network using expression uncovers sex-specific gene regulation in Drosophila, Wang et al., 2018 ->  Full female-fly network from matlab source 

Datasets:
	- Time Seres:
		- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198879

	- Male & Female Reference Networks (Ranked interractions)
		- Full female-fly network from matlab source 
		- Female no ovaries: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06382-z/MediaObjects/41467_2018_6382_MOESM4_ESM.txt
		- Male no testicles:   https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-06382-z/MediaObjects/41467_2018_6382_MOESM5_ESM.txt		


Data preprocessing: 
	- getIntersect.py: removes genes not in cpm_updated.txt and subnetwork_genes.txt, and creates file subnetwork_genes_intersect.txt
	- filterTimeSeries.py: extracts final CPMs for each experiment (cpm_V.txt, cpm_C.txt) 
	- interpolate.py interpolates time series to 1h intervals       

Files:
	- NetREX_female_prediciton_ranks.txt: prior female Drosophila melanogaster network 
	- cpm_V.txt, cpm_C.txt: cpm values 
	- cpm_A_C_interpolated.txt, cpm_B_C_interpolated.txt, ... : interpolated cpm values     