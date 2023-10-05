function [S_pred, Stimes] = RankingIntegal(Sp)

[m, n] = size(Sp{1,1});
[M, N] = size(Sp);
SP = sparse(m, n);
Stimes = sparse(m,n);
for II = 1 : M
    for JJ = 1 : N
        SP = SP + Sp{II,JJ};
        Stimes = Stimes + (Sp{II,JJ}~=0);
        
        %%rerank
        [idc idr val] = find(Sp{II,JJ});
        [sval sord] = sort(val, 'ascend');
        Sp{II,JJ} = sparse(idc(sord), idr(sord), 1:length(val), m, n);
    end
end

SP = double(SP~=0);
SPP = sparse(m,n);
for II = 1 : M
    for JJ = 1 : N
        Rank0 = (sum(sum(SP~=0)) + sum(sum(Sp{II,JJ}~=0))+1)/2;
        Spp{II,JJ} = Rank0*((SP~=0) - (Sp{II,JJ}~=0)) + Sp{II,JJ};
        SPP = SPP + Spp{II,JJ};
    end
end

SPPP = SPP / (M*N);
S_pred = SPPP;
