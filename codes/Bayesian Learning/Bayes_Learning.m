function [p1, p2, pc1, pc2] = Bayes_Learning(train_data,validation_data)
p1 = [];
p2 = [];

% Calculate p1 and p2
% p2-> Bernoulli paramter of the first class 
% p2-> Bernoulli paramter of the second class
for k=(1:2)
    class_data=train_data(train_data(:,end)==k,1:end-1);
    numData=size(class_data,1); % total number of data for this class.
    numDimensions = size(class_data,2);
    
    for i=(1:numDimensions)
        sum=0;
        for j=1:numData
            sum=sum+(1-class_data(j,i)); % According to problem 3(b).
        end
        
        if(k==1)
            p1=[p1,sum/numData];
        else
            p2=[p2,sum/numData];
        end
    end
end
% Done with p1 and p2

% Calculate best pc1 and pc2 over the validation set
min_error = 1; % 100 percent error.
best_pc1 = 0;

fprintf("Prior1       Prior2      Error Rate\n");
str = "";

for i = (-5:5)
    pc1 = 1/(1+exp(-i));
    pc2 = 1-pc1;
    
    label = [];
    
    for j=1:size(validation_data,1)
        pxc1 = 1;
        pxc2 = 1;
        for k=1:(size(validation_data,2)-1)
            pxc1 = pxc1 * power(p1(k),(1-validation_data(j,k))) * power((1-p1(k)),validation_data(j,k));
            pxc2 = pxc2 * power(p2(k),(1-validation_data(j,k))) * power((1-p2(k)),validation_data(j,k));
        end
        if pc1*pxc1 > pc2*pxc2
            label=[label, 1];
        else
            label=[label, 2];
        end
    end
    
    error_rate=[];
    n = size(label,2);
    error_count = 0;
    for m=(1:n)
        if label(m)~=validation_data(m,end)
            error_count = error_count+1;
        end
    end
    error_rate=error_count/n;
    
    formatSpec = "%f     %f     %f\n";
    str=str+sprintf(formatSpec,pc1,pc2,error_rate);
    
    if error_rate<min_error
        min_error=error_rate;
        best_pc1=pc1;
    end
end

% pc1-> The learned prior of the first class.
% pc2-> The learned prior of the second class.
pc1 = best_pc1;
pc2 = 1-pc1;
disp(str);
end
