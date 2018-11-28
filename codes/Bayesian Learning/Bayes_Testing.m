function Bayes_Testing(test_data,p1, p2, pc1, pc2)
label = [];
numData = size(test_data,1);

for j = 1:numData
    pxc1 = 1;
    pxc2 = 1;
    for k = 1:(size(test_data,2)-1)
        pxc1 = pxc1 * power(p1(k),(1-test_data(j,k))) * power((1-p1(k)),test_data(j,k));
        pxc2 = pxc2 * power(p2(k),(1-test_data(j,k))) * power((1-p2(k)),test_data(j,k));
    end
    if pc1*pxc1 > pc2*pxc2
        label=[label, 1];
    else
        label=[label, 2];
    end
end

error_rate = [];
n = size(label,2);
error_count = 0;

for m=(1:n)
    if label(m)~=test_data(m,end)
        error_count=error_count+1;
    end
end

error_rate = error_count/n;

formatSpec = "%f       %f        %f\n";
str = sprintf(formatSpec, pc1,pc2, error_rate);
fprintf("BestPrior1     BestPrior2      Error Rate\n");
disp(str);

end