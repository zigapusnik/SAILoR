function [output] = NetREX_EdgeControl(Input)

% optimization:
%              min_{S,A}: 1/2||E-SA||_F^2 + \kappa tr(S^TL_GS) + \mu ||A||_F^2
%                    s.t. ||A||_{\infty} \leq M
%                         ||S||_{\infty} \leq C
%                         ||S_0oS||_0 \leq # kept edges
%                         ||\bar{S_0}oS||_0 \leq # adding edges
%Input:
%Input.NumGene:  number of genes
%Input.NumTF:    number of TF
%Input.NumExp:   number of experiments
%Input.GEMatrix: the Gene expression data
%Input.GeneGraph: weighted adjancey matrix of Gene network
%Input.TFGraph:   weighted adjancey matrix of TF network
%Input.S0:        initial point for GENE-TF network
%Input.KeepEdge; Input.AddEdge; Input.kappa; Input.mu
%
%Output:
%Output.S:      GENE-TF network
%Output.A:      TF activity matrix
%
%Algorithm details can be find in "Research Diary: July07"

%process data
LapT = diag(sum(Input.TFGraph)) - Input.TFGraph;
% LapTmax = eigs(LapT, 1);
LapG = diag(sum(Input.GeneGraph)) - Input.GeneGraph;
% LapGmax = eigs(LapG, 1);

%% initilization
if(sum(Input.Sold(:)) == 0 || sum(Input.Aold(:)) == 0)
    [Sold, Aold] = NCA_1(Input, Input.S0, Input.A0, LapG, 1);
    Ecoli_Sold = Sold;
    Ecoli_Aold = Aold;
else
    Sold = Input.Sold;
    Aold = Input.Aold;
end

%% NetREX
ObjF = Objfunction(Input, Sold, Aold, LapG);
disp(['IterNum 0: OBJ ', num2str(ObjF(1)) ])
disp(sprintf('IterNum 0: OBJ: %10.5f Fitting: %10.5f Existing: %d Adding: %d ||S||: %f', ObjF(1), norm(Input.GEMatrix-Sold*Aold, 'fro'), sum(sum((Input.Exist).*Sold~=0)), sum(sum((1-Input.Exist).*Sold~=0)), norm(Sold,'fro')))
for k = 1 : Input.IterNum
    %% S first
    c(k) = norm(Aold*Aold', 'fro') + 2*norm(Input.kappa*LapG, 'fro');
    [Snew] = PALM_S_EdgeControl(Input, Sold, Aold, LapG, c(k));%PALM_S_ElasticNet(Input, Sold, Aold, LapG, c(k));%
    %% A next
    d(k) = norm(Snew*Snew', 'fro');
    [Anew] = PALM_A(Input, Snew, Aold, d(k));
    
    ObjF = [ObjF Objfunction(Input, Snew, Anew, LapG)];
    Aold = Anew;
    Sold = Snew;
    
    if(abs(ObjF(end)-ObjF(end-1)) < 1e-50 || isnan(ObjF(end)))
        break;
    end
    
    disp(sprintf('IterNum %d: OBJ: %10.5f Fitting: %10.5f Existing: %d Adding: %d ||S||: %f', k, ObjF(k+1), ...
        norm(Input.GEMatrix-Sold*Aold, 'fro'), sum(sum((Input.Exist).*Sold~=0)), ...
        sum(sum((1-Input.Exist).*Sold~=0)), norm(Sold,'fro') ))
end

output.A = Aold;
output.S = Sold;
output.kappa = Input.kappa;
output.mu = Input.mu;
output.C = Input.C;
output.M = Input.M;

end

function [val] = Objfunction(Input, S, A, LapG)
    Remaining = sum(sum((Input.Exist.*S~=0)));
    Adding = sum(sum((1-Input.Exist).*S~=0));
    val = 0.5*norm(Input.GEMatrix - S*A, 'fro')^2  + Input.kappa*trace(S'*LapG*S) + Input.mu*norm(A, 'fro')^2 + Input.xi*norm(S, 'fro')^2;%+ (Input.eta-Input.lambda)*Remaining + (Input.eta+Input.lambda)*Adding

end

function [Anew] = PALM_A(Input, Sold, Aold, dk)
    [m,n] = size(Aold);
    Vk = Aold - (1/dk)*(Sold'*Sold*Aold-Sold'*Input.GEMatrix);
    Anewt = (1/(1+((2*Input.mu)/dk)))*Vk;
    Anew = (abs(Anewt)<=Input.M).*Anewt + (abs(Anewt)>Input.M)*Input.M.*sign(Anewt);
end

function [Anew] = ClosedForm4A(Input, Sold)
    Anewt = inv(Sold'*Sold+Input.mu*eye(Input.NumTF))*Sold'*Input.GEMatrix;
    Anew = (abs(Anewt)<=Input.M).*Anewt + (abs(Anewt)>Input.M)*Input.M.*sign(Anewt);
end



function [Snew] = PALM_S_EdgeControl(Input, Sold, Aold, LapG, ck)
    [m,n] = size(Sold);
    Uk = Sold - (1/ck) * (Sold*Aold*Aold' + 2*Input.kappa*LapG*Sold + 2*Input.xi*Sold - Input.GEMatrix*Aold');
    UkP = (abs(Uk)>=Input.C)*Input.C.*sign(Uk) + (abs(Uk)<Input.C).*Uk;
    
    % pick up # of required edges
    SExist = UkP.*Input.Exist;
    SExist_ABS = abs(SExist);
    SAdding = UkP.*(1-Input.Exist);
    SAdding_ABS = abs(SAdding);
    
    % sort and pick up
    SExist_ABS_Sort = sort(SExist_ABS(:), 'descend');
    SExist_Pick = (SExist_ABS>=SExist_ABS_Sort(Input.KeepEdge)).*SExist;
    SAdding_ABS_Sort = sort(SAdding_ABS(:), 'descend');
    SAdding_Pick = (SAdding_ABS>=SAdding_ABS_Sort(Input.AddEdge)).*SAdding;
    Snew = SExist_Pick + SAdding_Pick;
    

end

function Tx = HardThreshold(x, p)
    if(abs(x) > p)
        Tx = x;
    elseif(abs(x) == p)
        t = rand(1);
        Tx = (t>=0.5)*p + (t<0.5)*0;
    else
        Tx = 0;
    end
end

function [S, A] = NCA_1(Input, Sold, Aold, LapG, IterNum)

Spport = Sold;
Ir = Input.xi*eye(Input.NumGene);
for i = 1 : IterNum
    [Anew] = ClosedForm4A(Input, Sold);
    [Snew] = bartelsStewart(2*Input.kappa*LapG+2*Ir, [], [], Anew*Anew', Input.GEMatrix*Anew');%RegressionWithFixSupport(Input, Spport, Anew);
    
    disp(sprintf('Itr %d: %f', i, norm(Input.GEMatrix - Snew*Anew, 'fro')))
    
    Aold = Anew;
    Aold = (abs(Aold)<=Input.M).*Aold + (abs(Aold)>Input.M)*Input.M.*sign(Aold);
    Sold = Snew;
    Sold = (abs(Sold)<=Input.C).*Sold + (abs(Sold)>Input.C)*Input.C.*sign(Sold);
end

S = Sold;
A = Aold;

end
