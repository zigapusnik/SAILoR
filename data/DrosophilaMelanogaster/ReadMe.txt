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
	- calculateTPM: converts CPM from cp,_updated to TPM based on raw counts (raw_corrected_counts_filtered_updated.txt), fragement lengths (fragment_lengths.txt) and transcripts lengths (FlyBase_transcripts.fasta)    

	- getFFGeneNames.py: writes unique gene names from NetREX_female_prediciton_ranks_full.txt to NetREX_female_prediciton_ranks_full_names.txt and creates file name with transcription factors NetREX_female_prediciton_ranks_TFs_names.txt 
	- https://www.biotools.fr/drosophila/fbgn_converter was used to get FlyBase IDs, IDs are in NetREX_female_prediciton_ranks_full_names_ids.txt 
	- getMissingFoundIDs.py: stores found IDs to file found.txt and missing IDs to missing.txt
	- IDs from missing.txt are obtained with https://flybase.org/batchdownload, data is in NetREX_female_prediciton_ranks_full_names_ids_validation_table.txt with duplicates
	- removeDuplicateIDs.py: removes conflict genes and appends IDs to found.txt   
	- getIntersect.py: removes TFs not in found.txt and in raw_corrected_counts_filtered_updated and creates files genes.txt and TFs.txt 
	- filterTimeSeries.py: extracts final CPMs and TPMs for each experiment (tpm_S.txt, tpm_V.txt, tpm_C.txt, cpm_S.txt, cpm_V.txt, cpm_C.txt) 
	- interpolate.py interpolates time series to 0.5h intervals    

To convert CPM (Count per million) numbers to TPM (transcripts per million) we obtained number of reads N from the raw gene count and CPM values, 
average fragment length was obtained from Sequence Read Archive, average gene-transcript length was obtained from exon counts and exon lengths obtained from FlyBase sequence downloader https://flybase.org/download/sequence/batch/   

Files:
	- genes.txt: list of genes (intersect from NetREX and Delbare et al., 2023)  
	- TFs.txt: list of transcriptor factors (intersect from NetREX and Delbare et al., 2023)  
	- NetREX_female_prediciton_ranks.txt: prior female Drosophila melanogaster network 
	- cpm_S.txt, cpm_V.txt, cpm_C.txt: cpm values
	- tpm_S.txt, tpm_V.txt, tpm_C.txt: tpm values
	- cpm_A_C_interpolated.txt, cpm_B_C_interpolated.txt, ... : interpolated cpm values 
	- tpm_A_C_interpolated.txt, tpm_B_C_interpolated.txt, ... : interpolated tpm values  