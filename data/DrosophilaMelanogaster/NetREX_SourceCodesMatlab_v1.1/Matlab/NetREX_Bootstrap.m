function [S_pred] = NetREX_Bootstrap(Input, NumBoot)

GE_T = Input.GEMatrix';
SP = sparse(Input.NumGene, Input.NumTF);
for II = 1 : NumBoot
    %% Bootstrap
    Input.GEMatrix = datasample(GE_T, Input.NumExp)';
    %% NetREX
    Input.Sold = 0;
    Input.Aold = 0;
    [Temp] = NetREX_EdgeControl(Input);
    %% Rank the edges
    Sp{II} = RankInteractions(Input.GEMatrix, Temp.S, Temp.A);%abs(Temp.S);%
    SP = SP + Sp{II};
    
end

SP = double(SP~=0);
SPP = sparse(Input.NumGene, Input.NumTF);
for II = 1 : NumBoot
    Rank0 = (sum(sum(SP~=0)) + sum(sum(Sp{II}~=0))+1)/2;
    Spp{II} = Rank0*((SP~=0) - (Sp{II}~=0)) + Sp{II};
    SPP = SPP + Spp{II};
end

SPPP = SPP / NumBoot;
S_pred = SPPP;