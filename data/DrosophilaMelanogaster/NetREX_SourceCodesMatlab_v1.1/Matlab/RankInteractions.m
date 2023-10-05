function RS = RankInteractions(E, S, A)
[m,n] = size(S);
[s t] = size(E);
[Indr Indc Val] = find(S);
EA = [E;A];
NumR = randperm(t);
for I = 1 : 20
    EAt = EA';
    EAtmp = datasample(EAt, NumR(I))';
    Et = EAtmp(1:m,:);
    TFAt = EAtmp(m+1:end,:);
    for i = 1 : length(Val)
        st1 = S(Indr(i),:);
        temp1 = norm(Et(Indr(i),:) - st1*TFAt, 'fro')^2;
        st1(Indc(i)) = 0;
        temp2 = norm(Et(Indr(i),:) - st1*TFAt, 'fro')^2;
        Confd(i) = (1 - temp1 / temp2);
    end
    [VSort Rankt] = sort(Confd, 'descend');
    Rank(I,Rankt) = 1:length(Val);
end

MeanRank = mean(Rank);
[SRank, ORank] = sort(MeanRank, 'ascend');
NewRank(ORank) = 1:length(Val);
RS = sparse(Indr, Indc, NewRank, m, n);
end
