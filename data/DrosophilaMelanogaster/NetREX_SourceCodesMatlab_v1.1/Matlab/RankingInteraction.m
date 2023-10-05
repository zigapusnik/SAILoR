function [prediction_ps, S_pred] = RankingInteraction(Sp)

[m, n] = size(Sp{1,1});
[M, N] = size(Sp);
SP = sparse(m, n);
for II = 1 : M
    for JJ = 1 : N
        SP = SP + Sp{II,JJ};
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
[indr indc Val] = find(SPPP);
[sortV orderV] = sort(Val, 'ascend');
Len = length(Val);
prediction_ps = [];
for TT = 1 : length(Val)
    prediction_ps = [prediction_ps; TF_with_Interaction_ID(indc(orderV(TT))) indr(orderV(TT)) sortV(TT)];
end
S_pred = SPPP;