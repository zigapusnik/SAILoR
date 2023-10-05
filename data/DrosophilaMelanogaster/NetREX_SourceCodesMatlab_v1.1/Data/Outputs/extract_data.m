female_fly_data = load('..\Inputs\FemaleFlyData.mat');

Genesymbol = female_fly_data.Genesymbol;
TFsymbol = female_fly_data.TFsymbol; 
length(TFsymbol)

NetREX_struct_female = load('.\NetREX_Prediction_Female.mat');
NetREX_sparse_female = NetREX_struct_female.Network_Female_Top300000;

nnz(NetREX_sparse_female)


[i,j,s] = find(NetREX_sparse_female);

M = cat(2,j,i,s);
N = sortrows(M,3);

TFs = TFsymbol(N(:,1));
TFs = string(TFs)';
Genes = Genesymbol(N(:,2)); 
Genes = string(Genes)';
Ranks = N(:,3);

T = table(TFs, Genes, Ranks) 


writetable(T, "NetREX_female_prediciton_ranks.txt", Delimiter="\t")


