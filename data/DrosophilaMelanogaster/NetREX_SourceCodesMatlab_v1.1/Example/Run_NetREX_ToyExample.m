clear
clc

load ToyExampleData.mat

%% Input for NetREX
[Input.NumGene Input.NumTF] = size(S_prior); 
Input.NumExp = size(E,2);
Input.GEMatrix = E; % expression (oberserved) data
Input.GeneGraph = (abs(corrcoef(E'))>0.995).*abs(corrcoef(E')); % embeded graph (i.e. correlation network)
Input.TFGraph = zeros(Input.NumTF);
Input.S0 = (S_prior); % the prior network
Input.Exist = (S_prior~=0); % structure of the prior
Input.A0 = rand(Input.NumTF, Input.NumExp);
Input.mu = 1; % avoid ativity reaches to the boudnary
Input.kappa = 1; % paramter for graph embeding term
Input.xi = 0.4; % avoid ||S||_F^2 being to large
Input.IterNum = 1000; % max iteration times for NetREX
Input.C = 2; % bound for S
Input.M = 2; % bound for A


%% NetREX consensus
TotalEdge = [17];
KeepEdge = [10,11,13];
for ii = 1 : length(KeepEdge)
    for jj = 1 : length(TotalEdge)
        %%NetREX with bootstrap
        Input.KeepEdge = KeepEdge(ii); % number of kept edges in the prior
        Input.AddEdge = TotalEdge(jj)-KeepEdge(ii); % number of the edge that can be added
        [Sbt] = NetREX_Bootstrap(Input, 5);
        Temp_bt.S = Sbt;
        SDP{ii,jj} = sparse(Sbt);
              
    end
end

%% Integrate all results from different parameters
[SDPAll, SDPtimes] = RankingIntegal(SDP);
Temp_intg.S = SDPAll.*(SDPtimes>=2);
[idr idc val] = find(Temp_intg.S);
[Sval Orval] = sort(val, 'ascend');
Cutoff = 15;
Predicted_Network = sparse(idr(Orval(1:Cutoff)), idc(Orval(1:Cutoff)), 1:Cutoff, Input.NumGene, Input.NumTF); % weight is the reanking, the lower the better
